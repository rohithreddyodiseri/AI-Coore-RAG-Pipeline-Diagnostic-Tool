"""Tenant-scoped read models for the diagnostic frontend.

Read captured telemetry directly; ingestion does not imply a judge has run.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone

from google.protobuf import json_format
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import Span

from rag_diagnostic.otlp_normalizer import attribute_values, normalize_execution

EXECUTION_ID = re.compile(r"[0-9a-f]{32}_[0-9a-f]{32}_[0-9a-f]{16}")


def timestamp(nanos):
    return datetime.fromtimestamp(nanos / 1_000_000_000, timezone.utc).isoformat()


def parse_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps require a timezone")
    return parsed


class UIReadModel:
    def __init__(self, store):
        self.store = store

    def records(self, tenant):
        with self.store.connection() as db:
            return db.execute(
                """SELECT e.*, s.span, s.resource, a.service_name, a.tenant,
                          d.job_id, d.state AS diagnosis_state,
                          d.created_at AS diagnosis_created_at, d.finished_at,
                          d.result, d.error_code, d.error_message
                   FROM executions e
                   JOIN apps a USING(app_id)
                   JOIN spans s ON s.app_id=e.app_id AND s.trace_id=e.trace_id
                               AND s.span_id=e.span_id
                   LEFT JOIN diagnostic_jobs d ON d.app_id=e.app_id AND d.trace_id=e.trace_id
                                             AND d.span_id=e.span_id
                   WHERE a.tenant=? ORDER BY e.app_id, e.trace_id, e.span_id""",
                (tenant,),
            ).fetchall()

    @staticmethod
    def execution(row):
        root = Span.FromString(row["span"])
        attrs = attribute_values(root)
        resource = attribute_values(Resource.FromString(row["resource"]))
        return {
            "execution_id": f"{row['app_id']}_{root.trace_id.hex()}_{root.span_id.hex()}",
            "trace_id": root.trace_id.hex(),
            "app_id": row["app_id"],
            "environment": resource.get("deployment.environment.name"),
            "started_at": timestamp(root.start_time_unix_nano),
            "duration_ms": (root.end_time_unix_nano - root.start_time_unix_nano) / 1_000_000,
            "status": attrs["rag.execution.status"],
            "flagged": attrs["rag.execution.status"] == "failed"
            or attrs.get("rag.failure.flagged") is True,
            "diagnosis_state": row["diagnosis_state"] or "not_requested",
        }

    def selected(self, tenant, filters):
        selected = []
        for row in self.records(tenant):
            item = self.execution(row)
            if any(
                key in filters and str(item[key]).lower() != filters[key]
                for key in ("app_id", "environment", "status", "flagged", "diagnosis_state")
            ):
                continue
            started = parse_time(item["started_at"])
            if "from" in filters and started < parse_time(filters["from"]):
                continue
            if "to" in filters and started >= parse_time(filters["to"]):
                continue
            selected.append((row, item))
        return selected

    def traces(self, tenant, filters, cursor, limit):
        items = [
            item for _, item in self.selected(tenant, filters) if item["execution_id"] > cursor
        ]
        return {
            "items": items[:limit],
            "next_cursor": items[limit - 1]["execution_id"] if len(items) > limit else None,
        }

    def find(self, tenant, execution_id):
        if not EXECUTION_ID.fullmatch(execution_id):
            return None
        return next(
            (
                row
                for row in self.records(tenant)
                if self.execution(row)["execution_id"] == execution_id
            ),
            None,
        )

    def spans(self, row):
        with self.store.connection() as db:
            spans = [
                Span.FromString(record["span"])
                for record in db.execute(
                    "SELECT span FROM spans WHERE app_id=? AND trace_id=?",
                    (row["app_id"], row["trace_id"]),
                )
            ]
        # Include only this execution's descendants, including untagged wrapper spans.
        owned = {row["span_id"]}
        changed = True
        while changed:
            before = len(owned)
            owned.update(span.span_id for span in spans if span.parent_span_id in owned)
            changed = len(owned) != before
        return sorted(
            (span for span in spans if span.span_id in owned),
            key=lambda span: (span.start_time_unix_nano, span.span_id),
        )

    def raw_spans(self, row):
        result = []
        for span in self.spans(row):
            item = json_format.MessageToDict(span, use_integers_for_enums=True)
            # OTLP JSON uses hexadecimal IDs; protobuf's generic JSON uses base64.
            item.update(traceId=span.trace_id.hex(), spanId=span.span_id.hex())
            if span.parent_span_id:
                item["parentSpanId"] = span.parent_span_id.hex()
            result.append(item)
        return {"resourceSpans": [{"scopeSpans": [{"spans": result}]}]}

    def envelope(self, row):
        return normalize_execution(
            row["span_id"], self.spans(row), row["service_name"], self.store.validator
        )

    def diagnosis(self, row):
        if not row["job_id"]:
            return None
        execution_id = self.execution(row)["execution_id"]
        result = {
            "diagnostic_run_id": row["job_id"],
            "execution_id": execution_id,
            "source": "live",
            "state": row["diagnosis_state"],
            "created_at": row["diagnosis_created_at"],
        }
        if row["finished_at"]:
            result["finished_at"] = row["finished_at"]
        if row["error_code"]:
            result["error"] = {"code": row["error_code"], "message": row["error_message"]}
        if row["result"]:
            from rag_diagnostic.diagnostic_jobs import present_report

            result.update(present_report(json.loads(row["result"])))
        return result

    def detail(self, row):
        spans = self.spans(row)
        envelope = normalize_execution(
            row["span_id"], spans, row["service_name"], self.store.validator
        )
        nodes, artifacts = [], {}

        def artifact(key, kind, content):
            artifacts[key] = {
                "artifact_id": key,
                "kind": kind,
                "availability": "available",
                "content": content,
            }
            return key

        execution_artifacts = []
        for field, kind in (
            ("query", "query"),
            ("final_answer", "answer"),
            ("inspection_errors", "inspection_note"),
        ):
            if field in envelope:
                execution_artifacts.append(artifact(field, kind, envelope[field]))
        for stage, value in envelope["stages"].items():
            ids = []
            for field, kind in (("input_data", "input"), ("output_data", "output")):
                if field in value:
                    ids.append(artifact(f"{stage}.{field}", kind, value[field]))
            node = {
                "node_id": value.get("span_id", stage),
                "stage": stage,
                "label": stage.replace("_", " ").title(),
                "status": value["status"],
                "duration_ms": value.get("duration_ms"),
                "artifact_ids": ids,
            }
            if "span_id" in value:
                node["span_id"] = value["span_id"]
            nodes.append(node)
        span_order = {span.span_id.hex(): index for index, span in enumerate(spans)}
        nodes.sort(key=lambda node: span_order.get(node["node_id"], len(spans)))
        node_ids = {node["node_id"] for node in nodes}
        edges = [
            {"from": span.parent_span_id.hex(), "to": span.span_id.hex(), "kind": "parent_span"}
            for span in spans
            if span.span_id.hex() in node_ids and span.parent_span_id.hex() in node_ids
        ]
        unavailable = {
            "available": False,
            "reason": "Replay is not exposed by the telemetry server.",
            "supported_overrides": [],
        }
        diagnosis = self.diagnosis(row)
        if row["result"]:
            report = json.loads(row["result"])
            for key, value in report["observations"]["input"]["artifacts"].items():
                kind = value["kind"]
                if kind not in {
                    "query",
                    "input",
                    "output",
                    "context",
                    "prompt",
                    "answer",
                    "manifest",
                    "inspection_note",
                }:
                    kind = "other"
                artifact(key, kind, value["text"])
        return {
            "execution": self.execution(row),
            "graph": {"nodes": nodes, "edges": edges},
            "execution_artifact_ids": execution_artifacts,
            "latest_live_diagnosis": diagnosis,
            "replay_capabilities": {"judge_only": unavailable, "pipeline_what_if": unavailable},
        }, artifacts

    def health(self, tenant, filters):
        selected = self.selected(tenant, filters)

        diagnoses = {item["execution_id"]: self.diagnosis(row) for row, item in selected}

        def counts(items):
            analyzed = [
                diagnoses[item["execution_id"]]
                for item in items
                if item["flagged"] and item["diagnosis_state"] == "completed"
            ]
            abstained = sum(run["verdict"]["status"] == "abstained" for run in analyzed)
            return {
                "observed_traces": len(items),
                "runtime_failed_traces": sum(item["status"] == "failed" for item in items),
                "flagged_traces": sum(item["flagged"] for item in items),
                "analyzed_flagged_traces": len(analyzed),
                "abstained_traces": abstained,
                "abstention_rate": abstained / len(analyzed) if analyzed else None,
            }

        def latency(rows):
            samples = defaultdict(list)
            for row in rows:
                detail, _ = self.detail(row)
                for node in detail["graph"]["nodes"]:
                    if node["duration_ms"] is not None:
                        samples[node["stage"]].append(node["duration_ms"])
            return sorted(
                [
                    {
                        "stage": stage,
                        "sample_count": len(values),
                        "avg_ms": sum(values) / len(values),
                    }
                    for stage, values in samples.items()
                ],
                key=lambda item: -item["avg_ms"],
            )

        with self.store.connection() as db:
            apps = db.execute(
                "SELECT app_id, service_name FROM apps WHERE tenant=? ORDER BY app_id", (tenant,)
            ).fetchall()
        return {
            "window": {key: filters[key] for key in ("from", "to")},
            "coverage": "unknown",
            "totals": counts([item for _, item in selected]),
            "apps": [
                {
                    "app_id": app["app_id"],
                    "service_name": app["service_name"],
                    "counts": counts(
                        [item for _, item in selected if item["app_id"] == app["app_id"]]
                    ),
                    "latency_bottlenecks": latency(
                        [row for row, item in selected if item["app_id"] == app["app_id"]]
                    ),
                }
                for app in apps
                if "app_id" not in filters or filters["app_id"] == app["app_id"]
            ],
            "latency_bottlenecks": latency([row for row, _ in selected]),
            "primary_category_counts": dict.fromkeys([f"FP{i}" for i in range(1, 8)], 0),
        }

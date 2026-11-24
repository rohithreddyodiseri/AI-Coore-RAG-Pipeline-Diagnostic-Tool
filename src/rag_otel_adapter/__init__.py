"""OTLP spans to normalized RAG execution records (ticket C3).

Person 1 calls ``normalize`` from the trace API (ticket A4) after the receiver
has authenticated and stored a trace. It turns OTLP/JSON spans into records
that match docs/telemetry.schema.json, which DiagnosticEngine.diagnose accepts.

Mapping rules come from "Evidence shapes and receiver mapping" in Person 1's
docs/diagnostic-tool.md:

- each ``rag.query`` span with ``rag.telemetry.version = "1.0"`` is one execution
- each span tagged ``rag.stage`` belongs to its nearest execution ancestor
- run_id is ``trace_id:execution_span_id``
- malformed or ambiguous evidence is dropped with a note in ``inspection_errors``
- a stage with no span is missing evidence; only declared stages become ``not_used``

Standard library only, so it adds nothing to Person 1's dependencies.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Iterable
from typing import Any

__all__ = ["normalize"]

CONTRACT_VERSION = "1.0"
EXECUTION_SPAN = "rag.query"
STAGES = {
    "corpus", "parsing", "chunking", "document_embedding", "indexing", "query_rewrite",
    "query_embedding", "retrieval", "reranking", "context_assembly", "generation",
}  # fmt: skip
OTLP_STATUS_ERROR = 2


def normalize(requests: dict[str, Any] | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map OTLP/JSON export requests to one execution record per ``rag.query`` span.

    ``requests`` is one ExportTraceServiceRequest in OTLP/JSON form (with
    ``resourceSpans``) or a list of them, for example every batch stored for one
    trace. Spans repeated across batches are counted once. Executions with an
    unsupported contract version are skipped, so they stay available only for
    trace inspection.
    """
    if isinstance(requests, dict):
        requests = [requests]
    spans = _flatten(requests)

    records = []
    for execution in sorted(spans.values(), key=lambda s: s["start"]):
        if execution["name"] != EXECUTION_SPAN:
            continue
        if execution["attributes"].get("rag.telemetry.version") != CONTRACT_VERSION:
            continue
        records.append(_execution_record(execution, spans))
    return records


# ----------------------------------------------------------------- reading OTLP/JSON


def _flatten(requests: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    spans: dict[tuple[str, str], dict[str, Any]] = {}
    for request in requests:
        for resource_spans in request.get("resourceSpans", []):
            resource = _attributes((resource_spans.get("resource") or {}).get("attributes"))
            for scope_spans in resource_spans.get("scopeSpans", []):
                for raw in scope_spans.get("spans", []):
                    trace_id = _hex_id(raw.get("traceId"), 16)
                    span_id = _hex_id(raw.get("spanId"), 8)
                    if not trace_id or not span_id:
                        continue
                    spans.setdefault(
                        (trace_id, span_id),  # deduplicate by span ID
                        {
                            "trace_id": trace_id,
                            "span_id": span_id,
                            "parent_id": _hex_id(raw.get("parentSpanId"), 8),
                            "name": raw.get("name", ""),
                            "start": int(raw.get("startTimeUnixNano") or 0),
                            "end": int(raw.get("endTimeUnixNano") or 0),
                            "status_code": int((raw.get("status") or {}).get("code") or 0),
                            "attributes": _attributes(raw.get("attributes")),
                            "service_name": resource.get("service.name"),
                        },
                    )
    return spans


def _hex_id(value: Any, size: int) -> str | None:
    """OTLP/JSON IDs are hex; protobuf-to-JSON tools sometimes emit base64."""
    if not isinstance(value, str) or not value:
        return None
    text = value.lower()
    if len(text) == size * 2 and all(c in "0123456789abcdef" for c in text):
        return None if set(text) == {"0"} else text
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw.hex() if len(raw) == size and any(raw) else None


def _attributes(items: Any) -> dict[str, Any]:
    return {
        item["key"]: _any_value(item.get("value") or {}) for item in items or [] if "key" in item
    }


def _any_value(value: dict[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "arrayValue" in value:
        return [_any_value(v) for v in (value["arrayValue"] or {}).get("values", [])]
    return None


# ----------------------------------------------------------------- building records


def _execution_record(
    execution: dict[str, Any], spans: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    attrs = execution["attributes"]
    notes: list[str] = []

    status = attrs.get("rag.execution.status")
    if status not in ("completed", "failed"):
        status = "failed" if execution["status_code"] == OTLP_STATUS_ERROR else "completed"
        notes.append(f"rag.execution.status was missing or invalid; used {status}.")

    service_name = execution["service_name"]
    if not isinstance(service_name, str) or not service_name.strip():
        service_name = "unknown_service"
        notes.append("The resource has no service.name.")

    flagged = status == "failed" or attrs.get("rag.failure.flagged") is True
    reason = attrs.get("rag.failure.reason")
    record: dict[str, Any] = {
        "schema_version": CONTRACT_VERSION,
        "run_id": f"{execution['trace_id']}:{execution['span_id']}",
        "otel": {
            "contract_version": CONTRACT_VERSION,
            "trace_id": execution["trace_id"],
            "execution_span_id": execution["span_id"],
            "service_name": service_name,
        },
        "execution_status": status,
        "failure_signal": {
            "flagged": flagged,
            "reason": reason if isinstance(reason, str) and reason.strip() else None,
        },
        "stages": _stages(execution, spans, notes),
    }

    query = attrs.get("rag.query.text")
    if isinstance(query, str) and query.strip():
        record["query"] = query
    for attribute, field in (
        ("rag.answer.text", "final_answer"),
        ("rag.generation.instructions", "generation_instructions"),
    ):
        if isinstance(attrs.get(attribute), str):
            record[field] = attrs[attribute]
    if notes:
        record["inspection_errors"] = notes
    return record


def _stages(
    execution: dict[str, Any], spans: dict[tuple[str, str], dict[str, Any]], notes: list[str]
) -> dict[str, dict[str, Any]]:
    trace_spans = {
        s["span_id"]: s for s in spans.values() if s["trace_id"] == execution["trace_id"]
    }

    tagged: dict[str, list[dict[str, Any]]] = {}
    for span in trace_spans.values():
        stage = span["attributes"].get("rag.stage")
        if stage is None:
            continue
        owner = _execution_ancestor(span, trace_spans)
        if owner is None:
            notes.append(
                f"Span {span['span_id']} ({stage}) has no rag.query ancestor in this trace; "
                "it was not used as evidence."
            )
            continue
        if owner != execution["span_id"]:
            continue
        if stage not in STAGES:
            notes.append(f"Span {span['span_id']} has unknown stage {stage!r}; it was ignored.")
            continue
        tagged.setdefault(stage, []).append(span)

    stages: dict[str, dict[str, Any]] = {}
    for stage, matches in sorted(tagged.items()):
        if len(matches) > 1:
            ids = ", ".join(sorted(s["span_id"] for s in matches))
            notes.append(f"Stage {stage} is tagged on several spans ({ids}); it was not used.")
            continue
        stages[stage] = _stage_record(stage, matches[0], notes)

    not_used = execution["attributes"].get("rag.stages.not_used") or []
    for stage in not_used if isinstance(not_used, list) else []:
        if stage in stages:
            notes.append(f"Stage {stage} is declared not used but has a span; the span was kept.")
        elif stage in STAGES:
            stages[stage] = {"status": "not_used"}
    return stages


def _execution_ancestor(span: dict[str, Any], trace_spans: dict[str, dict[str, Any]]) -> str | None:
    seen = set()
    parent_id = span["parent_id"]
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        parent = trace_spans.get(parent_id)
        if parent is None:
            return None  # parent not received: parentage is unknown
        if parent["name"] == EXECUTION_SPAN:
            return parent_id
        parent_id = parent["parent_id"]
    return None


def _stage_record(stage: str, span: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    attrs = span["attributes"]
    status = attrs.get("rag.stage.status")
    if status not in ("completed", "failed"):
        status = "failed" if span["status_code"] == OTLP_STATUS_ERROR else "unknown"
        notes.append(f"Stage {stage} has no valid rag.stage.status; used {status}.")

    record: dict[str, Any] = {"status": status, "span_id": span["span_id"]}
    if span["start"] and span["end"] >= span["start"]:
        record["duration_ms"] = (span["end"] - span["start"]) / 1_000_000
    k = attrs.get("rag.stage.k")
    if isinstance(k, int) and not isinstance(k, bool) and k >= 1:
        record["k"] = k

    for attribute, field in (
        ("rag.stage.input_json", "input_data"),
        ("rag.stage.output_json", "output_data"),
    ):
        if attribute not in attrs:
            continue
        try:
            value = json.loads(attrs[attribute])
        except (TypeError, ValueError):
            notes.append(f"Stage {stage} {attribute} is not valid JSON; it was discarded.")
            continue
        problem = _shape_problem(stage, field, value)
        if problem:
            notes.append(f"Stage {stage} {attribute} {problem}; it was discarded.")
            continue
        record[field] = value

    if isinstance(attrs.get("error.type"), str):
        record["error"] = {"type": attrs["error.type"]}
    return record


def _shape_problem(stage: str, field: str, value: Any) -> str | None:
    """Checks the evidence shapes in the producer contract. None means valid."""
    if field == "output_data":
        if stage in ("retrieval", "reranking"):
            if not isinstance(value, list) or not all(_is_ranked_chunk(c) for c in value):
                return "is not a list of {chunk_id, text, rank} objects"
        elif stage == "context_assembly":
            if not (
                isinstance(value, str)
                or (isinstance(value, dict) and isinstance(value.get("text"), str))
            ):
                return "is not a string or {text, chunk_ids} object"
        elif stage == "generation":
            if not isinstance(value, str):
                return "is not a string"
        elif stage == "query_embedding":
            if not isinstance(value, list) or not all(
                isinstance(v, list) and all(_is_number(x) for x in v) for v in value
            ):
                return "is not a list of vectors"
    elif stage == "generation" and not isinstance(value, dict):
        return "is not an object"
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_ranked_chunk(chunk: Any) -> bool:
    if not isinstance(chunk, dict):
        return False
    rank = chunk.get("rank")
    return (
        isinstance(chunk.get("chunk_id"), str)
        and bool(chunk["chunk_id"])
        and isinstance(chunk.get("text"), str)
        and isinstance(rank, int)
        and not isinstance(rank, bool)
        and rank >= 1
        and ("doc_id" not in chunk or (isinstance(chunk["doc_id"], str) and bool(chunk["doc_id"])))
        and ("score" not in chunk or _is_number(chunk["score"]))
    )

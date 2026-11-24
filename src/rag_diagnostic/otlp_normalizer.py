"""Map a completed OpenTelemetry RAG execution to the diagnostic envelope."""

from __future__ import annotations

import json
from collections import defaultdict

STAGES = frozenset({"query_embedding", "retrieval", "reranking", "context_assembly", "generation"})
FINAL_STATES = ("completed", "failed")


def attribute_values(message):
    values = {}
    for entry in message.attributes:
        value = entry.value
        kind = value.WhichOneof("value")
        if kind in {"string_value", "bool_value", "int_value", "double_value"}:
            values[entry.key] = getattr(value, kind)
        elif kind == "array_value" and all(
            item.WhichOneof("value") == "string_value" for item in value.array_value.values
        ):
            values[entry.key] = [item.string_value for item in value.array_value.values]
    return values


def execution_span(span, attrs):
    return span.name == "rag.query" and "rag.telemetry.version" in attrs


def supported_execution(span, attrs):
    return (
        execution_span(span, attrs)
        and attrs["rag.telemetry.version"] == "1.0"
        and attrs.get("rag.execution.status") in FINAL_STATES
    )


def _stage_owner(span, spans, roots):
    parent = span.parent_span_id
    seen = {span.span_id}
    while parent and parent not in seen:
        if parent in roots:
            return parent
        if parent not in spans:
            break
        seen.add(parent)
        parent = spans[parent].parent_span_id
    return None


def _reject_nonfinite(value):
    raise ValueError("non-finite JSON value")


def _add_evidence(attrs, stage, envelope, name, validator, notes):
    for attribute, field in (
        ("rag.stage.input_json", "input_data"),
        ("rag.stage.output_json", "output_data"),
    ):
        if attribute not in attrs:
            continue
        try:
            decoded = json.loads(attrs[attribute], parse_constant=_reject_nonfinite)
            # Reject floating point overflows such as 1e1000 too.
            json.dumps(decoded, allow_nan=False)
            stage[field] = decoded
            if not validator.is_valid(envelope):
                raise ValueError("wrong evidence shape")
        except (TypeError, ValueError, RecursionError):
            stage.pop(field, None)
            notes.append(f"{name}: invalid {field} was discarded.")


def normalize_execution(root_id, spans, service_name, validator):
    by_id = {span.span_id: span for span in spans}
    attrs = {span_id: attribute_values(span) for span_id, span in by_id.items()}
    roots = {span_id for span_id, span in by_id.items() if execution_span(span, attrs[span_id])}
    root = by_id[root_id]
    root_attrs = attrs[root_id]
    trace_id, root_hex = root.trace_id.hex(), root.span_id.hex()
    status = root_attrs["rag.execution.status"]
    stages = {}
    notes = []
    envelope = {
        "schema_version": "1.0",
        "run_id": f"{trace_id}:{root_hex}",
        "execution_status": status,
        "otel": {
            "contract_version": "1.0",
            "trace_id": trace_id,
            "execution_span_id": root_hex,
            "service_name": service_name,
        },
        "pipeline": {"name": service_name},
        "failure_signal": {
            "flagged": status == "failed" or root_attrs.get("rag.failure.flagged") is True,
        },
        "stages": stages,
    }
    for attribute, field in (
        ("rag.query.text", "query"),
        ("rag.answer.text", "final_answer"),
        ("rag.generation.instructions", "generation_instructions"),
    ):
        value = root_attrs.get(attribute)
        if isinstance(value, str) and (field != "query" or value.strip()):
            envelope[field] = value
    reason = root_attrs.get("rag.failure.reason")
    if isinstance(reason, str):
        envelope["failure_signal"]["reason"] = reason

    tagged = defaultdict(list)
    for span_id, span in by_id.items():
        name = attrs[span_id].get("rag.stage")
        if not isinstance(name, str) or name not in STAGES or span_id in roots:
            continue
        owner = _stage_owner(span, by_id, roots)
        if owner == root_id:
            tagged[name].append(span)
        elif owner is None and len(roots) == 1:
            notes.append(f"{name}: span parentage could not be resolved.")

    for name in sorted(tagged):
        spans_for_stage = tagged[name]
        if len(spans_for_stage) != 1:
            notes.append(f"{name}: duplicate stage tags were excluded.")
            continue
        span = spans_for_stage[0]
        fields = attrs[span.span_id]
        stage_status = fields.get("rag.stage.status")
        if stage_status not in FINAL_STATES:
            notes.append(f"{name}: missing or invalid stage status.")
            continue
        stage = {
            "status": stage_status,
            "span_id": span.span_id.hex(),
            "duration_ms": (span.end_time_unix_nano - span.start_time_unix_nano) / 1_000_000,
        }
        stages[name] = stage
        cutoff = fields.get("rag.stage.k")
        if name in {"retrieval", "reranking"} and type(cutoff) is int and cutoff > 0:
            stage["k"] = cutoff
        error = fields.get("error.type")
        if stage_status == "failed" and isinstance(error, str):
            stage["error"] = {"type": error}
        _add_evidence(fields, stage, envelope, name, validator, notes)

    bypassed = root_attrs.get("rag.stages.not_used")
    if isinstance(bypassed, list):
        for name in sorted(set(bypassed) & STAGES):
            if name not in tagged:
                stages[name] = {"status": "not_used"}
            else:
                notes.append(f"{name}: observed stage conflicts with bypass declaration.")
    if notes:
        envelope["inspection_errors"] = sorted(set(notes))
    validator.validate(envelope)
    return envelope

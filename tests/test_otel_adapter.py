"""C3: OTLP spans from the reference app normalize into docs/telemetry.schema.json."""

import base64
import copy
import json
from pathlib import Path

import pytest

from gemini_client import GeminiAPIError
from rag_otel_adapter import normalize
from rag_pipeline.telemetry import otlp_json

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "docs" / "telemetry.schema.json").read_text())
FIXTURES = ROOT / "fixtures" / "step0"


def assert_valid(record):
    jsonschema.Draft202012Validator(SCHEMA).validate(record)


def export(spans):
    return otlp_json(spans.get_finished_spans())


def test_healthy_query_normalizes_to_valid_record(make_pipeline, spans, monkeypatch):
    monkeypatch.setenv("RAG_TRACE_CONTENT", "1")
    pipeline = make_pipeline()
    pipeline.ask("How can retrieval improve?")

    [record] = normalize(export(spans))
    assert_valid(record)
    assert record["otel"]["trace_id"] == pipeline.last_trace_id
    assert record["run_id"] == f"{pipeline.last_trace_id}:{record['otel']['execution_span_id']}"
    assert record["otel"]["service_name"] == "reference-rag"
    assert record["execution_status"] == "completed"
    assert record["failure_signal"] == {"flagged": False, "reason": None}
    assert record["query"] == "How can retrieval improve?"
    assert record["final_answer"] == "Use hybrid search and reranking."
    assert set(record["stages"]) == {
        "query_embedding", "retrieval", "reranking", "context_assembly", "generation"
    }  # fmt: skip
    assert record["stages"]["reranking"] == {"status": "not_used"}
    retrieval = record["stages"]["retrieval"]
    assert retrieval["k"] == 5 and retrieval["status"] == "completed"
    assert [c["chunk_id"] for c in retrieval["output_data"]] == ["chunk-a", "chunk-b"]
    assert record["stages"]["generation"]["output_data"] == "Use hybrid search and reranking."
    assert "inspection_errors" not in record


def test_generation_failure_is_flagged_and_keeps_error_type(make_pipeline, spans, monkeypatch):
    monkeypatch.setenv("RAG_TRACE_CONTENT", "1")
    with pytest.raises(GeminiAPIError):
        make_pipeline(error=ConnectionError("down")).ask("How can retrieval improve?")

    [record] = normalize(export(spans))
    assert_valid(record)
    assert record["execution_status"] == "failed"
    assert record["failure_signal"]["flagged"] is True
    generation = record["stages"]["generation"]
    assert generation["status"] == "failed"
    assert generation["error"] == {"type": "GeminiAPIError"}
    assert "output_data" not in generation
    assert "final_answer" not in record


def test_metadata_only_trace_has_no_query(make_pipeline, spans, monkeypatch):
    monkeypatch.delenv("RAG_TRACE_CONTENT", raising=False)
    make_pipeline().ask("Private?", flag_reason="Looks wrong")

    [record] = normalize(export(spans))
    assert_valid(record)
    # Without a query, Person 1 reports insufficient evidence instead of calling the engine.
    assert "query" not in record
    assert record["failure_signal"] == {"flagged": True, "reason": "Looks wrong"}
    assert all("output_data" not in stage for stage in record["stages"].values())


def test_spans_split_across_batches_and_repeated_are_counted_once(make_pipeline, spans):
    make_pipeline().ask("How can retrieval improve?")
    finished = spans.get_finished_spans()
    first, second = otlp_json(finished[:2]), otlp_json(finished[2:])

    records = normalize([second, first, copy.deepcopy(first)])
    assert len(records) == 1
    assert len(records[0]["stages"]) == 5
    assert "inspection_errors" not in records[0]


def test_base64_ids_are_accepted(make_pipeline, spans):
    make_pipeline().ask("How can retrieval improve?")
    request = export(spans)
    for span in request["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        for key in ("traceId", "spanId", "parentSpanId"):
            if span.get(key):
                span[key] = base64.b64encode(bytes.fromhex(span[key])).decode()
    [record] = normalize(request)
    assert_valid(record)
    assert len(record["stages"]) == 5


# ---- hand-built traces for the evidence rules in the producer contract

TRACE = "aabbccddeeff00112233445566778899"
ROOT_SPAN = "1122334455667788"


def attr(key, value):
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, list):
        return {
            "key": key,
            "value": {"arrayValue": {"values": [{"stringValue": v} for v in value]}},
        }
    return {"key": key, "value": {"stringValue": value}}


def span(span_id, name, parent=None, **attributes):
    return {
        "traceId": TRACE,
        "spanId": span_id,
        **({"parentSpanId": parent} if parent else {}),
        "name": name,
        "startTimeUnixNano": "1000000000",
        "endTimeUnixNano": "1014000000",
        "attributes": [attr(k.replace("__", "."), v) for k, v in attributes.items()],
    }


def request(*spans_):
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [attr("service.name", "reference-rag")]},
                "scopeSpans": [{"spans": list(spans_)}],
            }
        ]
    }


def root(**extra):
    base = {
        "rag__telemetry__version": "1.0",
        "rag__execution__status": "completed",
        "rag__query__text": "How are documents ranked?",
    }
    return span(ROOT_SPAN, "rag.query", **{**base, **extra})


def test_contract_example_empty_retrieval():
    """The worked example in docs/diagnostic-tool.md produces the documented record."""
    record = normalize(
        request(
            root(
                rag__failure__flagged=True,
                rag__failure__reason="Empty retrieval",
                rag__stages__not_used=["reranking"],
            ),
            span(
                "2233445566778899",
                "retrieve",
                ROOT_SPAN,
                rag__stage="retrieval",
                rag__stage__status="completed",
                rag__stage__k=5,
                rag__stage__output_json="[]",
            ),  # fmt: skip
        )
    )
    assert record == [
        {
            "schema_version": "1.0",
            "run_id": f"{TRACE}:{ROOT_SPAN}",
            "otel": {
                "contract_version": "1.0",
                "trace_id": TRACE,
                "execution_span_id": ROOT_SPAN,
                "service_name": "reference-rag",
            },
            "query": "How are documents ranked?",
            "execution_status": "completed",
            "failure_signal": {"flagged": True, "reason": "Empty retrieval"},
            "stages": {
                "retrieval": {
                    "status": "completed",
                    "span_id": "2233445566778899",
                    "duration_ms": 14,
                    "k": 5,
                    "output_data": [],
                },
                "reranking": {"status": "not_used"},
            },
        }
    ]
    assert_valid(record[0])


def test_malformed_evidence_is_discarded_but_stage_metadata_kept():
    [record] = normalize(
        request(
            root(),
            span(
                "00000000000000a1",
                "retrieve",
                ROOT_SPAN,
                rag__stage="retrieval",
                rag__stage__status="completed",
                rag__stage__output_json="[{broken",
            ),
            span(
                "00000000000000a2",
                "context",
                ROOT_SPAN,
                rag__stage="context_assembly",
                rag__stage__status="completed",
                rag__stage__output_json='{"chunk_ids": []}',
            ),
        )  # fmt: skip
    )
    assert_valid(record)
    assert record["stages"]["retrieval"]["status"] == "completed"
    assert "output_data" not in record["stages"]["retrieval"]
    assert "output_data" not in record["stages"]["context_assembly"]
    assert len(record["inspection_errors"]) == 2


def test_duplicate_stage_tags_and_orphans_are_not_evidence():
    [record] = normalize(
        request(
            root(),
            span(
                "00000000000000b1",
                "gen-1",
                ROOT_SPAN,
                rag__stage="generation",
                rag__stage__status="completed",
            ),
            span(
                "00000000000000b2",
                "gen-2",
                ROOT_SPAN,
                rag__stage="generation",
                rag__stage__status="completed",
            ),
            span(
                "00000000000000b3",
                "lost",
                "00000000000000ff",
                rag__stage="retrieval",
                rag__stage__status="completed",
            ),
        )  # fmt: skip
    )
    assert_valid(record)
    assert record["stages"] == {}
    assert any("several spans" in note for note in record["inspection_errors"])
    assert any("no rag.query ancestor" in note for note in record["inspection_errors"])


def test_nested_stage_span_belongs_to_nearest_execution():
    [record] = normalize(
        request(
            root(),
            span("00000000000000c1", "http-call", ROOT_SPAN),
            span(
                "00000000000000c2",
                "llm",
                "00000000000000c1",
                rag__stage="generation",
                rag__stage__status="completed",
                rag__stage__output_json='"ok"',
            ),
        )  # fmt: skip
    )
    assert record["stages"]["generation"]["output_data"] == "ok"


def test_unsupported_contract_version_is_skipped():
    assert normalize(request(root(rag__telemetry__version="2.0"))) == []


# ---- step-0 fixtures stay in sync with the adapter


@pytest.mark.parametrize("name", ["healthy", "failing"])
def test_step0_fixture_normalizes_to_expected_payload(name):
    otlp = [json.loads(line) for line in (FIXTURES / f"{name}.otlp.jsonl").read_text().splitlines()]
    expected = json.loads((FIXTURES / f"{name}.expected.json").read_text())
    assert normalize(otlp) == expected
    for record in expected:
        assert_valid(record)

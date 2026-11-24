"""C2: the pipeline emits spans that follow the producer contract v1."""

import json

import pytest


def by_name(spans):
    return {span.name: span for span in spans.get_finished_spans()}


def test_query_produces_execution_span_with_stage_children(make_pipeline, spans):
    pipeline = make_pipeline()
    pipeline.ask("How can retrieval improve?")

    finished = by_name(spans)
    root = finished["rag.query"]
    assert root.attributes["rag.telemetry.version"] == "1.0"
    assert root.attributes["rag.execution.status"] == "completed"
    assert list(root.attributes["rag.stages.not_used"]) == ["reranking"]
    assert root.resource.attributes["service.name"] == "reference-rag"
    assert pipeline.last_trace_id == format(root.context.trace_id, "032x")

    for stage in ("query_embedding", "retrieval", "context_assembly", "generation"):
        span = finished[f"rag.{stage}"]
        assert span.attributes["rag.stage"] == stage
        assert span.attributes["rag.stage.status"] == "completed"
        assert span.parent.span_id == root.context.span_id
        assert span.context.trace_id == root.context.trace_id
    assert finished["rag.retrieval"].attributes["rag.stage.k"] == 5


def test_content_is_not_exported_unless_enabled(make_pipeline, spans, monkeypatch):
    monkeypatch.delenv("RAG_TRACE_CONTENT", raising=False)
    make_pipeline().ask("Private question?")

    for span in spans.get_finished_spans():
        assert not any(
            key in span.attributes
            for key in (
                "rag.query.text",
                "rag.answer.text",
                "rag.stage.input_json",
                "rag.stage.output_json",
            )
        )


def test_content_capture_records_evidence_shapes(make_pipeline, spans, monkeypatch):
    monkeypatch.setenv("RAG_TRACE_CONTENT", "1")
    make_pipeline().ask("How can retrieval improve?")
    finished = by_name(spans)

    root = finished["rag.query"].attributes
    assert root["rag.query.text"] == "How can retrieval improve?"
    assert root["rag.answer.text"] == "Use hybrid search and reranking."
    assert "only the provided context" in root["rag.generation.instructions"]

    retrieved = json.loads(finished["rag.retrieval"].attributes["rag.stage.output_json"])
    assert [c["rank"] for c in retrieved] == [1, 2]
    assert retrieved[0] == {
        "chunk_id": "chunk-a",
        "doc_id": "paper-1",
        "text": "Hybrid search combines keyword and vector retrieval.",
        "rank": 1,
        "score": pytest.approx(0.88),
    }
    context = json.loads(finished["rag.context_assembly"].attributes["rag.stage.output_json"])
    assert context["chunk_ids"] == ["chunk-a", "chunk-b"]
    assert context["text"].startswith("[1] Hybrid search")
    generation = finished["rag.generation"].attributes
    assert json.loads(generation["rag.stage.output_json"]) == "Use hybrid search and reranking."
    assert set(json.loads(generation["rag.stage.input_json"])) == {
        "prompt",
        "system_instruction",
        "model",
    }


def test_flag_reason_marks_the_query_for_diagnosis(make_pipeline, spans):
    make_pipeline().ask("How can retrieval improve?", flag_reason="Answer missed details")
    root = by_name(spans)["rag.query"].attributes
    assert root["rag.failure.flagged"] is True
    assert root["rag.failure.reason"] == "Answer missed details"


def test_generation_error_marks_stage_and_execution_failed(make_pipeline, spans):
    class AuthenticationError(Exception):
        pass

    pipeline = make_pipeline(error=AuthenticationError("Incorrect API key provided: sk-secret"))
    with pytest.raises(AuthenticationError):
        pipeline.ask("How can retrieval improve?")

    finished = by_name(spans)
    generation = finished["rag.generation"]
    assert generation.attributes["rag.stage.status"] == "failed"
    assert generation.attributes["error.type"] == "AuthenticationError"
    assert generation.status.status_code.name == "ERROR"
    root = finished["rag.query"]
    assert root.attributes["rag.execution.status"] == "failed"
    assert root.status.status_code.name == "ERROR"
    # The exception message (which can hold secrets) is never exported.
    for span in spans.get_finished_spans():
        assert not span.events
        assert "sk-secret" not in json.dumps(dict(span.attributes))


def test_empty_index_fails_at_retrieval(make_pipeline, spans):
    with pytest.raises(RuntimeError, match="index is empty"):
        make_pipeline(chunks=[]).ask("Anything?")
    finished = by_name(spans)
    assert finished["rag.retrieval"].attributes["rag.stage.status"] == "failed"
    assert "rag.generation" not in finished

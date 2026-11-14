import json

import pytest
from conftest import FakeClient

from rag_pipeline.corpus import Document
from rag_pipeline.pipeline import RAGPipeline


class FakeCollection:
    def count(self):
        return 1

    def query(self, **kwargs):
        assert kwargs["query_embeddings"] == [[0.1, 0.2]]
        return {
            "ids": [["paper:0"]],
            "documents": [["The diagnostic answer is 42."]],
            "distances": [[0.1]],
            "metadatas": [[{"title": "Diagnostics", "source": "paper.json"}]],
        }


def test_chunk_id_is_stable():
    document = Document("text", {"source": "paper.json", "section": 1})

    first = RAGPipeline._chunk_id(document, 0, "chunk text")
    second = RAGPipeline._chunk_id(document, 0, "chunk text")

    assert first == second
    assert len(first) == 64


def test_ask_logs_each_pipeline_stage(tmp_path, monkeypatch):
    from rag_pipeline.config import Settings

    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        chroma_dir=tmp_path / "chroma",
        collection="test",
        embedding_model="test-embedder",
        llm_model="test-model",
        chunk_size=100,
        chunk_overlap=10,
        batch_size=2,
        top_k=1,
        diagnostics_dir=tmp_path / "diagnostics",
    )
    client = FakeClient("  It is 42.  ")
    pipeline = RAGPipeline(settings, client=client)
    monkeypatch.setattr(pipeline, "_open_collection", lambda create=False: FakeCollection())
    monkeypatch.setattr(pipeline, "_embed", lambda texts: [[0.1, 0.2]])

    answer, sources = pipeline.ask("What is the diagnostic answer?")

    assert answer == "It is 42."
    assert client.calls == 1
    assert client.requests[0]["model"] == "test-model"
    assert "The diagnostic answer is 42." in client.requests[0]["contents"]
    assert sources[0].metadata["title"] == "Diagnostics"
    assert pipeline.last_trace_path is not None
    events = [
        json.loads(line)
        for line in pipeline.last_trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["stage"] for event in events] == [
        "query",
        "query_embedding",
        "retrieval",
        "prompt_construction",
        "generation",
        "query",
        "execution",
    ]
    assert events[1]["inputs"]["texts"] == ["What is the diagnostic answer?"]
    assert events[1]["outputs"] == [[0.1, 0.2]]
    assert events[2]["outputs"][0]["text"] == "The diagnostic answer is 42."
    assert events[3]["outputs"]["input"].startswith("Question:")
    assert events[4]["outputs"]["output_text"] == "It is 42."
    assert events[5]["outputs"]["answer"] == "It is 42."


def test_ask_logs_failed_stage(tmp_path, monkeypatch):
    from rag_pipeline.config import Settings

    settings = Settings(
        corpus_dir=tmp_path / "corpus",
        chroma_dir=tmp_path / "chroma",
        collection="test",
        embedding_model="test-embedder",
        llm_model="test-model",
        chunk_size=100,
        chunk_overlap=10,
        batch_size=2,
        top_k=1,
        diagnostics_dir=tmp_path / "diagnostics",
    )
    pipeline = RAGPipeline(settings)
    monkeypatch.setattr(pipeline, "_open_collection", lambda create=False: FakeCollection())
    monkeypatch.setattr(
        pipeline, "_embed", lambda texts: (_ for _ in ()).throw(RuntimeError("embed failed"))
    )

    try:
        pipeline.ask("Will this fail?")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the embedding stage to fail")

    events = [
        json.loads(line)
        for line in pipeline.last_trace_path.read_text(encoding="utf-8").splitlines()
    ]
    embedding_event = next(event for event in events if event["stage"] == "query_embedding")
    assert embedding_event["status"] == "failed"
    assert embedding_event["error"] == {"type": "RuntimeError", "message": "embed failed"}
    assert events[-3]["stage"] == "retrieval"
    assert events[-3]["status"] == "failed"
    assert events[-2]["stage"] == "query"
    assert events[-1]["stage"] == "execution"
    assert events[-1]["outputs"]["failure_signal"]["flagged"] is True


def test_search_uses_requested_k_and_embeds_only_once(tmp_path, monkeypatch):
    from dataclasses import replace

    from rag_pipeline.config import Settings
    from rag_pipeline.diagnostics import PipelineTrace

    class Collection(FakeCollection):
        calls = []

        def count(self):
            return 20

        def query(self, **kwargs):
            self.calls.append(kwargs)
            return super().query(**kwargs)

    settings = replace(Settings.from_env(), top_k=1, diagnostics_dir=tmp_path)
    pipeline, collection, embeddings = RAGPipeline(settings), Collection(), []
    monkeypatch.setattr(pipeline, "_open_collection", lambda create=False: collection)

    def embed(texts):
        embeddings.append(texts)
        return [[0.1, 0.2]]

    monkeypatch.setattr(pipeline, "_embed", embed)
    trace = PipelineTrace(tmp_path)
    results = pipeline.search("query", 10, trace=trace)
    assert embeddings == [["query"]]
    assert len(collection.calls) == 1
    assert collection.calls[0]["n_results"] == 10
    assert collection.calls[0]["include"] == ["documents", "metadatas", "distances"]
    assert results[0].chunk_id == "paper:0"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(0.9)
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert [event["stage"] for event in events] == ["query_embedding", "retrieval"]
    assert events[1]["inputs"]["query_embeddings"] == events[0]["outputs"]
    assert events[1]["inputs"]["n_results"] == 10


def test_execution_trace_matches_gemini_request_and_resets_logging(tmp_path, monkeypatch):
    from dataclasses import replace

    from rag_pipeline.config import Settings

    settings = replace(Settings.from_env(), diagnostics_dir=tmp_path)
    client, collection, embeddings = (
        FakeClient("The diagnostic answer is 42."),
        FakeCollection(),
        [],
    )
    pipeline = RAGPipeline(settings, client=client)
    monkeypatch.setattr(pipeline, "_open_collection", lambda create=False: collection)

    def embed(texts):
        embeddings.append(texts)
        return [[0.1, 0.2]]

    monkeypatch.setattr(pipeline, "_embed", embed)
    trace = pipeline.run_with_trace("Question?")
    request = client.requests[0]
    assert client.calls == 1
    assert embeddings == [["Question?"]]
    assert trace["stages"]["generation"]["input_data"]["prompt"] == request["contents"]
    assert trace["generation_instructions"] == request["config"]["system_instruction"]
    assert (
        trace["stages"]["generation"]["input_data"]["temperature"]
        == request["config"]["temperature"]
    )
    assert trace["final_answer"] == trace["stages"]["generation"]["output_data"]
    original_events = pipeline.last_trace_path.read_text()
    pipeline.search("Inspection probe", 10)
    assert pipeline.last_trace_path.read_text() == original_events


def test_failed_gemini_generation_is_logged_without_credentials(tmp_path, monkeypatch):
    from dataclasses import replace

    from gemini_client import GeminiAPIError
    from rag_pipeline.config import Settings

    def failure(request):
        raise RuntimeError("provider message contains private-test-key")

    pipeline = RAGPipeline(
        replace(Settings.from_env(), diagnostics_dir=tmp_path), client=FakeClient(failure)
    )
    monkeypatch.setattr(pipeline, "_open_collection", lambda create=False: FakeCollection())
    monkeypatch.setattr(pipeline, "_embed", lambda texts: [[0.1, 0.2]])
    with pytest.raises(GeminiAPIError):
        pipeline.ask("Question?")
    contents = pipeline.last_trace_path.read_text()
    assert "private-test-key" not in contents
    events = [json.loads(line) for line in contents.splitlines()]
    assert [event["stage"] for event in events[-3:]] == ["generation", "query", "execution"]
    assert all(event["status"] == "failed" for event in events[-2:])
    assert pipeline._active_trace.get() is None

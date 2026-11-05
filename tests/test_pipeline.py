import json
from types import SimpleNamespace

from rag_pipeline.corpus import Document
from rag_pipeline.pipeline import RAGPipeline


class FakeCollection:
    def count(self):
        return 1

    def query(self, **kwargs):
        assert kwargs["query_embeddings"] == [[0.1, 0.2]]
        return {
            "documents": [["The diagnostic answer is 42."]],
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
    pipeline = RAGPipeline(settings)
    monkeypatch.setattr(pipeline, "_open_collection", lambda create=False: FakeCollection())
    monkeypatch.setattr(pipeline, "_embed", lambda texts: [[0.1, 0.2]])

    response = SimpleNamespace(id="response-1", output_text="  It is 42.  ", usage={"tokens": 7})

    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["model"] == "test-model"
            assert "The diagnostic answer is 42." in kwargs["input"]
            return response

    monkeypatch.setattr(
        "openai.OpenAI", lambda: SimpleNamespace(responses=FakeResponses())
    )

    answer, sources = pipeline.ask("What is the diagnostic answer?")

    assert answer == "It is 42."
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
    ]
    assert events[1]["inputs"]["texts"] == ["What is the diagnostic answer?"]
    assert events[1]["outputs"] == [[0.1, 0.2]]
    assert events[2]["outputs"][0]["text"] == "The diagnostic answer is 42."
    assert events[3]["outputs"]["input"].startswith("Question:")
    assert events[4]["outputs"]["output_text"] == "  It is 42.  "
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
    assert events[-2]["stage"] == "query_embedding"
    assert events[-2]["status"] == "failed"
    assert events[-2]["error"] == {"type": "RuntimeError", "message": "embed failed"}
    assert events[-1]["stage"] == "query"
    assert events[-1]["status"] == "failed"

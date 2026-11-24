"""Shared fakes so pipeline tests run without models, Chroma, or an OpenAI key."""

import sys
import types

import pytest
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_pipeline.config import Settings
from rag_pipeline.pipeline import RAGPipeline

# A tracer provider can be set once per process, so every test shares this one.
SPANS = InMemorySpanExporter()
_provider = TracerProvider(resource=Resource.create({"service.name": "reference-rag"}))
_provider.add_span_processor(SimpleSpanProcessor(SPANS))
trace.set_tracer_provider(_provider)

CHUNKS = [
    ("chunk-a", "Hybrid search combines keyword and vector retrieval.", "paper-1", 0.12),
    ("chunk-b", "Reranking reorders candidates with a cross-encoder.", "paper-2", 0.25),
]


class FakeVectors(list):
    def tolist(self):
        return list(self)


class FakeEmbedder:
    def encode(self, texts, **_):
        return FakeVectors([[0.1, 0.2, 0.3] for _ in texts])


class FakeCollection:
    def __init__(self, chunks):
        self.chunks = chunks

    def count(self):
        return len(self.chunks)

    def query(self, query_embeddings, n_results, include):
        rows = self.chunks[:n_results]
        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[r[1] for r in rows]],
            "metadatas": [[{"doc_id": r[2], "title": f"Paper {r[2]}", "section": 0} for r in rows]],
            "distances": [[r[3] for r in rows]],
        }


def fake_openai(answer=None, error=None):
    class Responses:
        def create(self, **_):
            if error:
                raise error
            return types.SimpleNamespace(output_text=answer)

    class OpenAI:
        def __init__(self, *_, **__):
            self.responses = Responses()

    return types.SimpleNamespace(OpenAI=OpenAI)


@pytest.fixture
def spans():
    SPANS.clear()
    return SPANS


@pytest.fixture
def make_pipeline(monkeypatch, tmp_path):
    """Builds a RAGPipeline with fake retrieval and a fake OpenAI client."""

    def build(chunks=CHUNKS, answer="Use hybrid search and reranking.", error=None):
        monkeypatch.setitem(sys.modules, "openai", fake_openai(answer, error))
        settings = Settings.from_env()
        pipeline = RAGPipeline(settings.__class__(**{**settings.__dict__, "chroma_dir": tmp_path}))
        pipeline._embedder = FakeEmbedder()
        pipeline._collection = FakeCollection(list(chunks))
        return pipeline

    return build

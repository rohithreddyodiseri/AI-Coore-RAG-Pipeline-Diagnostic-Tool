"""Shared fakes so pipeline tests run without models, Chroma, or an OpenAI key."""

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


class FakeGeminiClient:
    """Stands in for google.genai.Client inside Person 1's GeminiInferenceEngine."""

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.requests = answer, error, []
        self.models = self

    def generate_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error
        stop = types.SimpleNamespace(finish_reason="STOP")
        return types.SimpleNamespace(
            text=self.answer, candidates=[stop], response_id="r1", model_version=model
        )


@pytest.fixture
def spans():
    SPANS.clear()
    return SPANS


@pytest.fixture
def make_pipeline(tmp_path):
    """Builds a RAGPipeline with fake retrieval and a fake Gemini client."""

    def build(chunks=CHUNKS, answer="Use hybrid search and reranking.", error=None):
        settings = Settings.from_env()
        pipeline = RAGPipeline(settings.__class__(**{**settings.__dict__, "chroma_dir": tmp_path}))
        pipeline._embedder = FakeEmbedder()
        pipeline._collection = FakeCollection(list(chunks))
        pipeline._inference._client = FakeGeminiClient(answer, error)
        pipeline._inference.min_request_interval_seconds = 0  # no pacing in tests
        return pipeline

    return build

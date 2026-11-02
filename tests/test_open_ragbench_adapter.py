"""Tests the adapter against the real rag_pipeline code and a real ChromaDB index.

A simple word-count embedder stands in for the embedding model, and a
stand-in replaces the OpenAI call, so no downloads or API key are needed.
These tests are skipped if rag_pipeline or chromadb isn't installed.
"""

import hashlib
import json
import re
from dataclasses import replace

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("langchain_text_splitters")
rag_config = pytest.importorskip("rag_pipeline.config")
rag_pipeline_mod = pytest.importorskip("rag_pipeline.pipeline")
rag_corpus = pytest.importorskip("rag_pipeline.corpus")

from rag_diagnostic.adapters.open_ragbench import OpenRAGBenchAdapter  # noqa: E402
from rag_diagnostic.cases import DiagnosticCase  # noqa: E402
from rag_diagnostic.config import DiagnosticConfig  # noqa: E402
from rag_diagnostic.diagnose import Diagnoser  # noqa: E402
from rag_diagnostic.judge import ReferenceRecallJudge  # noqa: E402
from rag_diagnostic.locate import tokens  # noqa: E402

DIM = 256


class WordCountPipeline(rag_pipeline_mod.RAGPipeline):
    def _embed(self, texts):
        vectors = []
        for text in texts:
            v = [0.0] * DIM
            for t in tokens(text):
                v[int(hashlib.md5(t.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class ExtractiveAdapter(OpenRAGBenchAdapter):
    def generate(self, query, context):
        wanted = set(tokens(query)) - {"what", "the", "of", "is", "how", "in"}
        sentences = re.split(r"(?<=[.!?])\s+|\n\n", context.text)
        best = max(sentences, key=lambda s: len(wanted & set(tokens(s))), default="")
        return best or "I do not know."


PAPERS = [
    {
        "id": "2401.00001",
        "title": "Sparse Probing of Retrieval Models",
        "abstract": "We study retrieval models with sparse probes.",
        "sections": [
            {
                "text": "Our probe reaches an accuracy of 87 percent on the held out split. "
                "Training took four hours on one GPU.",
                "tables": {"t1": "| model | acc |\n| probe | 87 |"},
                "images": {"i1": "base64-junk"},
            },
        ],
    },
    {
        "id": "2401.00002",
        "title": "Chunking Strategies Compared",
        "abstract": "We compare fixed and semantic chunking.",
        "sections": [{"text": "Semantic chunking improved recall by 6 points over fixed windows."}],
    },
]

CASE = DiagnosticCase(
    case_id="orb-1",
    query="What accuracy does the sparse probe reach on the held out split?",
    reference_answer="87 percent",
    gold=(),
)


@pytest.fixture
def corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for paper in PAPERS:
        (corpus_dir / f"{paper['id']}.json").write_text(json.dumps(paper))
    return corpus_dir


def make_adapter(tmp_path, corpus, **overrides):
    settings = replace(
        rag_config.Settings.from_env(),
        corpus_dir=corpus,
        chroma_dir=tmp_path / "chroma",
        collection="test",
        **overrides,
    )
    pipeline = WordCountPipeline(settings)
    pipeline.index(rag_corpus.OpenRAGBenchLoader(corpus).load(), reset=True)
    return ExtractiveAdapter(settings, pipeline=pipeline)


def with_gold(evidence):
    from rag_diagnostic.cases import GoldLocation

    return replace(CASE, gold=(GoldLocation("2401.00001", evidence),))


def test_adapter_matches_the_pipeline_loader(tmp_path, corpus):
    adapter = make_adapter(tmp_path, corpus)
    loader_text = "\n\n".join(
        d.text
        for d in rag_corpus.OpenRAGBenchLoader(corpus).load()
        if d.metadata["doc_id"] == "2401.00001"
    )
    assert adapter.parsed_text("2401.00001") == loader_text
    assert "base64-junk" not in adapter.raw_text("2401.00001")


def test_adapter_chunk_ids_match_the_index(tmp_path, corpus):
    adapter = make_adapter(tmp_path, corpus)
    ids = [c.chunk_id for c in adapter.chunks("2401.00001")]
    assert ids and adapter.indexed_ids(ids) == set(ids)


def test_clean_index_answers_correctly(tmp_path, corpus):
    adapter = make_adapter(tmp_path, corpus)
    evidence = "Our probe reaches an accuracy of 87 percent on the held out split."
    diagnoser = Diagnoser(DiagnosticConfig(), ReferenceRecallJudge())
    assert diagnoser.diagnose(with_gold(evidence), adapter).status == "no_failure"


def test_small_chunks_that_still_hold_the_answer_are_not_a_failure(tmp_path, corpus):
    # The chunks split the sentence, but "87 percent" still fits in one, so the answer is right.
    adapter = make_adapter(tmp_path, corpus, chunk_size=30, chunk_overlap=0)
    evidence = "Our probe reaches an accuracy of 87 percent on the held out split."
    diagnoser = Diagnoser(DiagnosticConfig(), ReferenceRecallJudge())
    assert diagnoser.diagnose(with_gold(evidence), adapter).status == "no_failure"


def test_tiny_chunks_are_diagnosed_as_chunking(tmp_path, corpus):
    adapter = make_adapter(tmp_path, corpus, chunk_size=15, chunk_overlap=0)
    evidence = "Our probe reaches an accuracy of 87 percent on the held out split."
    case = replace(with_gold(evidence), reference_answer="accuracy of 87 percent")
    diagnoser = Diagnoser(DiagnosticConfig(), ReferenceRecallJudge())
    verdict = diagnoser.diagnose(case, adapter)
    assert verdict.status == "diagnosed", verdict.render()
    assert verdict.stages == ("chunking",), verdict.render()

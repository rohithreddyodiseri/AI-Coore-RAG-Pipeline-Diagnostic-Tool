"""Tests the adapter against the real rag_pipeline code and a real ChromaDB index.

A simple word-count embedder stands in for the embedding model, and a
stand-in replaces the Gemini call, so no downloads or API key are needed.
These tests are skipped if rag_pipeline or chromadb isn't installed.
"""

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("langchain_text_splitters")
rag_config = pytest.importorskip("rag_pipeline.config")
rag_pipeline_mod = pytest.importorskip("rag_pipeline.pipeline")
rag_corpus = pytest.importorskip("rag_pipeline.corpus")

from rag_diagnostic.adapters.open_ragbench import OpenRAGBenchAdapter  # noqa: E402
from rag_diagnostic.config import DiagnosticConfig  # noqa: E402
from rag_diagnostic.diagnose import Diagnoser  # noqa: E402
from rag_diagnostic.judge import LLMJudge  # noqa: E402


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


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


class ExtractivePipeline(WordCountPipeline):
    def generate(self, query, context):
        wanted = set(tokens(query)) - {"what", "the", "of", "is", "how", "in"}
        sentences = re.split(r"(?<=[.!?])\s+|\n\n", context)
        return max(
            sentences, key=lambda text: len(wanted & set(tokens(text))), default="I do not know."
        )


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

QUERY = "What accuracy does the sparse probe reach on the held out split?"


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
    pipeline = ExtractivePipeline(settings)
    pipeline.index(rag_corpus.OpenRAGBenchLoader(corpus).load(), reset=True)
    return OpenRAGBenchAdapter(settings, pipeline=pipeline)


def diagnoser(tmp_path, reply):
    from conftest import FakeClient

    config = DiagnosticConfig(
        runs_dir=str(tmp_path / "runs"), judge_cache=str(tmp_path / "judge.json")
    )
    return Diagnoser(config, LLMJudge(config.judge_model, config.judge_cache, FakeClient(reply)))


def test_adapter_collects_actual_corpus(tmp_path, corpus):
    adapter = make_adapter(tmp_path, corpus)
    snapshot = adapter.corpus_snapshot()
    assert snapshot["complete"] is True
    assert len(snapshot["documents"]) == 2
    assert "base64-junk" not in snapshot["documents"][0]["text"]
    assert snapshot["snapshot_id"]


def test_real_pipeline_artifacts_reach_llm_without_gold(tmp_path, corpus):
    from conftest import cite, report, sample_trace

    observed = []

    def reply(request):
        artifacts = json.loads(request["contents"])["artifacts"]
        observed.append(artifacts)
        result = report(sample_trace(), clear=True)
        for row in result["evaluations"]:
            row["evidence"] = [
                cite(artifacts, "query"),
                cite(artifacts, "context.text"),
                cite(artifacts, "generation.answer"),
            ]
        return result

    adapter = make_adapter(tmp_path, corpus)
    verdict = diagnoser(tmp_path, reply).diagnose(QUERY, adapter)
    assert verdict.status == "no_failure", verdict.render()
    assert "generation.instructions" in observed[0]
    assert "deep.manifest" in observed[0]
    assert "87 percent" in observed[0]["context.text"]["text"]


def test_snapshot_changes_when_source_changes(tmp_path, corpus):
    adapter = make_adapter(tmp_path, corpus)
    before = adapter.corpus_snapshot()["snapshot_id"]
    path = corpus / "2401.00001.json"
    data = json.loads(path.read_text())
    data["abstract"] += " Additional source content."
    path.write_text(json.dumps(data))
    assert adapter.corpus_snapshot()["snapshot_id"] != before


def test_ask_and_diagnostic_capture_execute_same_real_pipeline(tmp_path, corpus):
    adapter = make_adapter(tmp_path, corpus)
    calls = []
    original = adapter.pipeline.generate

    def generate(query, context):
        calls.append((query, context))
        return original(query, context)

    adapter.pipeline.generate = generate
    answer, sources = adapter.pipeline.ask(QUERY)
    trace = adapter.run(QUERY)
    assert trace["final_answer"] == answer
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert trace["stages"]["retrieval"]["output_data"][0]["chunk_id"] == sources[0].chunk_id
    assert trace["stages"]["generation"]["input_data"]["prompt"].endswith(calls[1][1])
    assert trace["stages"]["reranking"]["status"] == "not_used"


def test_direct_query_cli_executes_target_once_and_writes_report(
    tmp_path, corpus, monkeypatch, capsys
):
    from conftest import FakeClient, cite, report, sample_trace

    from rag_diagnostic import cli

    adapter = make_adapter(tmp_path, corpus)
    generated = []
    generate = adapter.pipeline.generate

    def capture(query, context):
        generated.append(query)
        return generate(query, context)

    adapter.pipeline.generate = capture

    def reply(request):
        artifacts = json.loads(request["contents"])["artifacts"]
        result = report(sample_trace(), clear=True)
        for row in result["evaluations"]:
            row["evidence"] = [
                cite(artifacts, "query"),
                cite(artifacts, "context.text"),
                cite(artifacts, "generation.answer"),
            ]
        return result

    config = DiagnosticConfig(
        runs_dir=str(tmp_path / "runs"), judge_cache=str(tmp_path / "judge.json")
    )
    judge = LLMJudge(config.judge_model, config.judge_cache, FakeClient(reply))
    connected = Diagnoser(config, judge)
    factories = []

    def factory(spec):
        factories.append(spec)
        return lambda settings: adapter

    monkeypatch.setattr(cli, "_load_factory", factory)
    monkeypatch.setattr(cli, "Diagnoser", lambda settings: connected)
    output, captured = tmp_path / "report.json", tmp_path / "trace.json"
    cli.main([QUERY, "--report-out", str(output), "--trace-out", str(captured), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert factories == [cli.DEFAULT_ADAPTER]
    assert generated == [QUERY]
    assert data["query"] == QUERY
    assert data["run_id"] and "case_id" not in data
    assert json.loads(output.read_text())["answer"] == data["answer"]
    trace = json.loads(captured.read_text())
    assert len(trace["corpus"]["documents"]) == 2
    assert trace["corpus"]["complete"] is True
    assert trace["stages"]["retrieval"]["input_data"]["query"] == QUERY
    assert trace["stages"]["generation"]["output_data"] == data["answer"]
    assert (Path(config.runs_dir) / "diagnoses.jsonl").exists()

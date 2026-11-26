"""Exercise real trace emission and ingestion with a mocked semantic judge."""

import hashlib
import json
from dataclasses import replace

import pytest
from conftest import FakeClient, cite, report, sample_trace

from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnostic_engine import DiagnosticEngine, trace_artifacts
from rag_diagnostic.ingestion import TelemetryIngestor
from rag_diagnostic.judge import LLMJudge
from rag_diagnostic.telemetry import IncompleteExecution, flag_execution, load_execution
from rag_pipeline.config import Settings
from rag_pipeline.corpus import OpenRAGBenchLoader
from rag_pipeline.diagnostics import PipelineTrace
from rag_pipeline.pipeline import RAGPipeline


class MemoryCollection:
    def __init__(self):
        self.rows = {}

    def upsert(self, ids, documents, metadatas, embeddings):
        for values in zip(ids, documents, metadatas, embeddings, strict=True):
            self.rows[values[0]] = values[1:]

    def count(self):
        return len(self.rows)

    def get(self, where=None, include=None, limit=None, offset=0, ids=None):
        rows = list(self.rows.items())
        if ids is not None:
            rows = [row for row in rows if row[0] in ids]
        if where:
            rows = [
                row
                for row in rows
                if all(row[1][1].get(key) == value for key, value in where.items())
            ]
        rows = rows[offset : offset + limit if limit is not None else None]
        return {
            "ids": [key for key, _ in rows],
            "documents": [value[0] for _, value in rows],
            "metadatas": [value[1] for _, value in rows],
            "embeddings": [value[2] for _, value in rows],
        }

    def query(self, n_results, **kwargs):
        rows = list(self.rows.items())[:n_results]
        return {
            "ids": [[key for key, value in rows]],
            "documents": [[value[0] for key, value in rows]],
            "metadatas": [[value[1] for key, value in rows]],
            "distances": [[0.1] * len(rows)],
        }


@pytest.fixture
def indexed_pipeline(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "paper.json").write_text(
        json.dumps(
            {
                "id": "paper",
                "title": "Construction report",
                "abstract": "Ada built the tower in 1902.",
                "sections": [{"text": "The tower is 30 meters tall."}],
            }
        )
    )
    settings = replace(
        Settings.from_env(),
        corpus_dir=corpus,
        chroma_dir=tmp_path / "chroma",
        diagnostics_dir=tmp_path / "telemetry",
        chunk_size=120,
        chunk_overlap=0,
    )
    pipeline, collection = (
        RAGPipeline(settings, client=FakeClient("Ada built the tower.")),
        MemoryCollection(),
    )
    monkeypatch.setattr(pipeline, "_open_collection", lambda create=True: collection)
    monkeypatch.setattr(pipeline, "_embed", lambda texts: [[1.0, 0.0] for text in texts])
    pipeline.index(OpenRAGBenchLoader(corpus).load())
    return pipeline


def semantic_judge(request):
    artifacts = json.loads(request["contents"])["artifacts"]
    result = report(sample_trace())
    result["failure_category"], result["primary_fault_stage"] = "FP5", "generation"
    result["evaluations"][4] = {
        "category": "FP5",
        "status": "fault",
        "stage": "generation",
        "confidence": 0.93,
        "rationale": "The query requests JSON; generation emitted a prose sentence.",
        "evidence": [cite(artifacts, "query"), cite(artifacts, "generation.answer")],
        "contributing_stages": [],
    }
    return result


def ingestor(tmp_path, reply=semantic_judge):
    config = DiagnosticConfig(
        runs_dir=str(tmp_path / "reports"),
        judge_cache=str(tmp_path / "judge.json"),
        assessment_mode="legacy_fp",
    )
    client = FakeClient(reply)
    return TelemetryIngestor(
        DiagnosticEngine(
            config,
            LLMJudge(config.judge_model, config.judge_cache, client, assessment_mode="legacy_fp"),
        )
    ), client


def test_emitted_run_ingests_index_and_corpus_without_reexecution(indexed_pipeline, tmp_path):
    pipeline = indexed_pipeline
    pipeline.ask("Who built the tower? Return JSON.", failure_reason="Unexpected response format")
    native = load_execution(pipeline.last_trace_path)
    assert native["run_id"]
    assert native["schema_version"] == "1.0"
    assert native["failure_signal"]["flagged"] is True
    assert native["corpus"]["complete"] is True
    for stage in (
        "parsing",
        "chunking",
        "document_embedding",
        "indexing",
        "query_embedding",
        "retrieval",
        "context_assembly",
        "generation",
    ):
        assert "output_data" in native["stages"][stage]
    assert native["stages"]["reranking"]["status"] == "not_used"
    assert native["stages"]["retrieval"]["output_data"][0]["score"] == pytest.approx(0.9)
    assert "vector" in native["stages"]["document_embedding"]["output_data"][0]
    # Neither changed source files nor an unavailable target may affect a captured run.
    (pipeline.settings.corpus_dir / "paper.json").write_text('{"id":"changed","abstract":"new"}')
    pipeline.execute = lambda query: pytest.fail("ingestion must not execute a pipeline")
    consumer, client = ingestor(tmp_path)
    verdict, destination = consumer.ingest(pipeline.last_trace_path)
    assert verdict.failure_category == "FP5"
    assert verdict.run_id == native["run_id"]
    assert destination.exists()
    assert (
        "Ada built the tower in 1902."
        in verdict.observations["trace"]["corpus"]["documents"][0]["text"]
    )
    artifacts = json.loads(client.requests[0]["contents"])["artifacts"]
    assert "chunking.chunk.0" in artifacts
    assert "document_embedding.manifest" in artifacts
    assert "query_embedding.manifest" in artifacts


def test_watcher_flags_after_query_and_deduplicates_across_restarts(indexed_pipeline, tmp_path):
    pipeline = indexed_pipeline
    pipeline.ask("Who built the tower? Return JSON.")
    consumer, client = ingestor(tmp_path)
    assert consumer.poll(pipeline.settings.diagnostics_dir) == []
    pipeline.flag_last_run("Engineer marked this response as failing")
    first = consumer.poll(pipeline.settings.diagnostics_dir)
    assert len(first) == 1
    assert client.calls == 1
    assert consumer.poll(pipeline.settings.diagnostics_dir) == []
    restarted, restarted_client = ingestor(tmp_path)
    assert restarted.poll(pipeline.settings.diagnostics_dir) == []
    assert restarted_client.calls == 0


def test_partial_stream_is_retried_before_publishing(indexed_pipeline, tmp_path):
    store = indexed_pipeline.settings.diagnostics_dir
    active = PipelineTrace(store)
    active.record("query", {"query": "Pending query"}, status="started")
    consumer, _ = ingestor(tmp_path)
    assert consumer.poll(store) == []
    with active.path.open("a") as stream:
        stream.write('{"run_id":')
    assert consumer.poll(store) == []
    with pytest.raises(IncompleteExecution):
        load_execution(active.path)


@pytest.mark.parametrize("damage", ["missing", "checksum", "escape"])
def test_bad_index_artifacts_never_assert_corpus_completeness(indexed_pipeline, damage):
    pipeline = indexed_pipeline
    trace = pipeline.run_with_trace("Question?")
    reference = trace["pipeline"]["index_snapshot"]
    root = pipeline.settings.diagnostics_dir
    manifest_path = root / reference["path"]
    manifest = json.loads(manifest_path.read_text())
    corpus_ref = manifest["artifacts"]["corpus"]
    corpus_file = root / corpus_ref["path"]
    if damage == "missing":
        corpus_file.unlink()
    elif damage == "checksum":
        corpus_file.write_text('{"documents":[],"complete":true}')
    else:
        corpus_ref["path"] = "../outside.json"
        manifest_path.write_text(json.dumps(manifest))
        # Simulate a supplied manifest whose contents are checksummed but path is unsafe.
        reference["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        original = [json.loads(line) for line in pipeline.last_trace_path.read_text().splitlines()]
        original[-1]["outputs"]["pipeline"]["index_snapshot"] = reference
        pipeline.last_trace_path.write_text("".join(json.dumps(row) + "\n" for row in original))
    loaded = load_execution(pipeline.last_trace_path)
    assert not loaded.get("corpus", {}).get("complete")
    assert loaded["inspection_errors"]


def test_external_json_trace_requires_no_pipeline_adapter(tmp_path):
    trace = sample_trace()
    trace.update(run_id="external-execution", schema_version="1.0")
    path = tmp_path / "external.json"
    path.write_text(json.dumps(trace))
    flag_execution(path, "Format requirement failed")
    consumer, client = ingestor(tmp_path)
    results = consumer.poll(tmp_path)
    assert results[0][0].run_id == "external-execution"
    assert results[0][0].failure_category == "FP5"
    assert client.calls == 1


def test_multi_execution_stream_requires_selection(tmp_path):
    trace = sample_trace()
    path = tmp_path / "stream.jsonl"
    rows = []
    for run_id in ("first", "second"):
        rows.append(
            {
                "run_id": run_id,
                "stage": "execution",
                "status": "completed",
                "outputs": {
                    **trace,
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "failure_signal": {"flagged": True},
                },
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="run-id"):
        load_execution(path)
    assert load_execution(path, run_id="second")["run_id"] == "second"
    consumer, client = ingestor(tmp_path)
    assert len(consumer.poll(tmp_path)) == 2
    assert client.calls == 1  # Equal observed artifacts reuse the semantic judge response.


def test_pre_envelope_stage_logs_are_ingested(indexed_pipeline):
    pipeline = indexed_pipeline
    pipeline.ask("Question?")
    rows = [json.loads(line) for line in pipeline.last_trace_path.read_text().splitlines()]
    rows = [row for row in rows if row["stage"] != "execution"]
    path = pipeline.last_trace_path
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = load_execution(path)
    assert loaded["final_answer"] == "Ada built the tower."
    assert "Ada built" in loaded["stages"]["context_assembly"]["output_data"]["text"]


def test_failed_execution_is_published_and_preserves_missing_outputs(indexed_pipeline, monkeypatch):
    pipeline = indexed_pipeline
    monkeypatch.setattr(
        pipeline,
        "_embed",
        lambda texts: (_ for _ in ()).throw(RuntimeError("embedding unavailable")),
    )
    with pytest.raises(RuntimeError):
        pipeline.ask("Question?")
    loaded = load_execution(pipeline.last_trace_path)
    assert loaded["failure_signal"]["flagged"] is True
    assert loaded["stages"]["query_embedding"]["status"] == "failed"
    assert "output_data" not in loaded["stages"]["query_embedding"]
    assert "final_answer" not in loaded
    assert "generation.answer" not in trace_artifacts(loaded)


def test_ingest_cli_and_flag_cli_use_saved_execution(
    indexed_pipeline, tmp_path, monkeypatch, capsys
):
    from rag_diagnostic import cli

    pipeline = indexed_pipeline
    pipeline.ask("Who built the tower? Return JSON.")
    consumer, _ = ingestor(tmp_path)
    cli.main(["flag", "--trace", str(pipeline.last_trace_path), "--reason", "Missing JSON"])
    assert json.loads(capsys.readouterr().out)["flagged"]
    monkeypatch.setattr(cli, "DiagnosticEngine", lambda config: consumer.engine)
    cli.main(["ingest", "--trace", str(pipeline.last_trace_path), "--json"])
    assert json.loads(capsys.readouterr().out)["failure_category"] == "FP5"


def test_watch_once_cli_consumes_flagged_run(indexed_pipeline, tmp_path, monkeypatch, capsys):
    from rag_diagnostic import cli

    indexed_pipeline.ask("Who built the tower? Return JSON.", failure_reason="Wrong format")
    consumer, client = ingestor(tmp_path)
    monkeypatch.setattr(cli, "DiagnosticEngine", lambda config: consumer.engine)
    cli.main(
        ["watch", "--trace-dir", str(indexed_pipeline.settings.diagnostics_dir), "--once", "--json"]
    )
    assert json.loads(capsys.readouterr().out)["failure_category"] == "FP5"
    assert client.calls == 1
    cli.main(
        ["watch", "--trace-dir", str(indexed_pipeline.settings.diagnostics_dir), "--once", "--json"]
    )
    assert capsys.readouterr().out == ""
    assert client.calls == 1


def test_unknown_execution_schema_is_rejected(tmp_path):
    trace = {**sample_trace(), "schema_version": "2.0", "run_id": "external"}
    path = tmp_path / "external.json"
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="schema_version"):
        load_execution(path)


def test_non_boolean_flag_does_not_trigger_processing(tmp_path):
    trace = {**sample_trace(), "run_id": "external", "failure_signal": {"flagged": "false"}}
    path = tmp_path / "external.json"
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="boolean"):
        load_execution(path)


def test_in_progress_native_execution_is_not_diagnosed(tmp_path):
    trace = {
        **sample_trace(),
        "run_id": "external",
        "execution_status": "running",
        "failure_signal": {"flagged": True},
    }
    path = tmp_path / "external.json"
    path.write_text(json.dumps(trace))
    consumer, client = ingestor(tmp_path)
    assert consumer.poll(tmp_path) == []
    assert client.calls == 0


def test_index_build_is_preserved_across_pipeline_instances(
    indexed_pipeline, tmp_path, monkeypatch
):
    original = indexed_pipeline
    restarted = RAGPipeline(original.settings, client=FakeClient("Ada built the tower."))
    monkeypatch.setattr(restarted, "_open_collection", original._open_collection)
    monkeypatch.setattr(restarted, "_embed", original._embed)
    restarted.ask("Question?")
    payload = load_execution(restarted.last_trace_path)
    assert payload["index_manifest"]["chunks"] > 0
    assert payload["stages"]["chunking"]["recorded_at"] == "index_build"
    assert payload["corpus"]["complete"]


def test_judge_views_share_a_bounded_text_budget(make_engine):
    payload = sample_trace()
    payload["corpus"] = {
        "complete": True,
        "snapshot_id": "full",
        "scope": "all sources",
        "documents": [{"doc_id": "d", "text": "source " * 1000}],
    }
    payload["stages"]["parsing"] = {
        "output_data": [{"text": "parsed " * 1000, "metadata": {"doc_id": "d"}}]
    }
    payload["stages"]["chunking"] = {
        "output_data": [{"chunk_id": "d:0", "doc_id": "d", "text": "chunk " * 1000}]
    }
    engine, client = make_engine(report(sample_trace()), corpus_budget_chars=300)
    verdict = engine.diagnose(payload)
    artifacts = json.loads(client.requests[0]["contents"])["artifacts"]
    source_text = [
        value["text"]
        for key, value in artifacts.items()
        if key.startswith(("corpus.document.", "parsing.document.", "chunking.chunk."))
    ]
    assert sum(map(len, source_text)) <= 300
    assert not artifacts["corpus.manifest"]["complete"]
    assert not artifacts["chunking.manifest"]["complete"]
    assert len(verdict.observations["trace"]["corpus"]["documents"][0]["text"]) > 300


def test_flagging_a_stream_without_final_newline_remains_valid(indexed_pipeline):
    indexed_pipeline.ask("Question?")
    path = indexed_pipeline.last_trace_path
    path.write_text(path.read_text().rstrip("\n"))
    flag_execution(path, "Engineer review requested")
    assert load_execution(path)["failure_signal"]["flagged"]


def test_query_records_index_lineage_before_generation(indexed_pipeline, monkeypatch):
    pipeline = indexed_pipeline
    order = []
    original_metadata, original_generate = pipeline._telemetry_metadata, pipeline.generate

    def metadata():
        order.append("snapshot")
        return original_metadata()

    def generate(query, context):
        order.append("generate")
        return original_generate(query, context)

    monkeypatch.setattr(pipeline, "_telemetry_metadata", metadata)
    monkeypatch.setattr(pipeline, "generate", generate)
    pipeline.ask("Question?")
    assert order == ["snapshot", "generate"]

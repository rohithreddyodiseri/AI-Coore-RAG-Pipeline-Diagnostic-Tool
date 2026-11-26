"""Verify the default stage judge uses captured producer telemetry end to end."""

import json

from conftest import FakeClient
from test_telemetry_integration import indexed_pipeline  # noqa: F401

from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnostic_engine import DiagnosticEngine
from rag_diagnostic.ingestion import TelemetryIngestor
from rag_diagnostic.judge import LLMJudge
from rag_diagnostic.mechanisms import MECHANISM_STAGE
from rag_diagnostic.stage_diagnosis import unknown
from rag_diagnostic.telemetry import load_execution


def test_emission_to_stage_report(indexed_pipeline, tmp_path):  # noqa: F811
    pipeline = indexed_pipeline
    pipeline.ask("Return only valid JSON.", failure_reason="format review")
    trace = load_execution(pipeline.last_trace_path)
    chunks = trace["stages"]["chunking"]["output_data"]
    assert all(c["start_offset"] >= 0 and c["end_offset"] > c["start_offset"] for c in chunks)
    assert trace["stages"]["indexing"]["output_data"]["complete"] is True
    assert set(trace["stages"]["indexing"]["output_data"]["indexed_ids"]) == {
        c["chunk_id"] for c in chunks
    }
    actual = trace["stages"]["generation"]["input_data"]
    assert actual["prompt"] == pipeline._inference._client.requests[0]["contents"]
    assert actual["delivered_context"] in actual["prompt"]
    assert trace["stages"]["selection"]["output_data"]["complete"] is True

    def judge_response(request):
        payload = json.loads(request["contents"])
        assert "taxonomy" in payload and "FP1" not in request["config"]["system_instruction"]
        assert request["config"]["temperature"] == 0.1
        rows = [unknown(m, "insufficient evidence") for m in MECHANISM_STAGE]
        row = next(r for r in rows if r["mechanism"] == "OUTPUT_FORMAT_VIOLATION")
        row.update(
            status="FAULT",
            designation="primary",
            confidence=0.95,
            evidence=[{"artifact_id": key, "span_id": 0} for key in ("query", "generation.answer")],
            unresolved_questions=[],
        )
        return json.dumps({"evaluations": rows})

    config = DiagnosticConfig(
        runs_dir=str(tmp_path / "runs"), judge_cache=str(tmp_path / "cache.json")
    )
    client = FakeClient(judge_response)
    engine = DiagnosticEngine(config, LLMJudge(config.judge_model, config.judge_cache, client))
    ingestor = TelemetryIngestor(engine)
    report, destination = ingestor.ingest(pipeline.last_trace_path)
    assert report.root_cause["mechanism"] == "OUTPUT_FORMAT_VIOLATION"
    assert destination.exists()
    assert pipeline._inference._client.calls == 1
    ingestor.ingest(pipeline.last_trace_path)
    assert client.calls == 1

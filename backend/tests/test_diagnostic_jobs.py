"""Run captured OTLP executions through the real validator with a mocked judge transport."""

import json
import threading
import time

from conftest import FakeClient
from test_otlp_server import (
    ADMIN,
    OTHER,
    VIEWER,
    app_token,
    call,
    export,
    make_server,
    register,
    span,
    span_request,
    validate_ui,
)

from rag_diagnostic.diagnostic_engine import DiagnosticEngine
from rag_diagnostic.judge import LLMJudge
from rag_diagnostic.mechanisms import MECHANISM_STAGE
from rag_diagnostic.stage_diagnosis import unknown


def capture(server, *, flagged=True, content=True, answer="Not JSON"):
    package = json.loads(register(server)[2])
    attrs = {"rag.telemetry.version": "1.0", "rag.execution.status": "completed"}
    if content:
        attrs.update({"rag.query.text": "Return only valid JSON.", "rag.answer.text": answer})
    root = span(1, name="rag.query", parent=0, attrs=attrs)
    root.attributes.add(key="rag.failure.flagged").value.bool_value = flagged
    assert export(server, span_request("reference-rag", [root]), app_token(package))[0] == 200
    assert server.store.publish_one()
    row = server.ui.records("first")[0]
    return server.ui.execution(row)["execution_id"]


def mocked_factory(clients):
    def factory(config):
        client = FakeClient({"evaluations": [unknown(m, "Not observed") for m in MECHANISM_STAGE]})
        clients.append(client)
        return DiagnosticEngine(config, LLMJudge(config.judge_model, config.judge_cache, client))

    return factory


def test_flagged_execution_is_backfilled_diagnosed_and_durable(tmp_path):
    clients = []
    with make_server(tmp_path, assembly_window=0) as server:
        execution_id = capture(server)
        server.diagnostics.engine_factory = mocked_factory(clients)
        server.diagnostics.enqueue_flagged()
        assert (
            server.ui.execution(server.ui.find("first", execution_id))["diagnosis_state"]
            == "queued"
        )
        assert server.diagnostics.run_one()
        assert not server.diagnostics.run_one()
        detail, artifacts = server.ui.detail(server.ui.find("first", execution_id))
        validate_ui("TraceDetail", detail)
        run = detail["latest_live_diagnosis"]
        assert detail["execution"]["diagnosis_state"] == "completed"
        assert run["verdict"]["mechanism"] == "OUTPUT_FORMAT_VIOLATION"
        assert run["verdict"]["primary_stage"] == "generation"
        assert run["verdict"]["category"] is None
        assert run["verdict"]["confidence"] == 1.0
        for finding in run["verdict"]["evidence_trail"]:
            for citation in finding["citations"]:
                artifact = artifacts[citation["artifact_id"]]
                assert citation["quote"] in artifact["content"]
                validate_ui("Artifact", artifact)
        health = server.ui.health(
            "first", {"from": "1970-01-01T00:00:00Z", "to": "1970-01-02T00:00:00Z"}
        )
        validate_ui("Dashboard", health)
        assert health["totals"]["analyzed_flagged_traces"] == 1
        assert health["totals"]["abstention_rate"] == 0.0
        assert sum(client.calls for client in clients) == 1
    with make_server(tmp_path, assembly_window=0) as restarted:
        restarted.diagnostics.enqueue_flagged()
        assert not restarted.diagnostics.run_one()
        path = f"/api/v1/traces/{execution_id}"
        assert call(restarted, "GET", path, OTHER)[0] == 404
        saved = json.loads(call(restarted, "GET", path, VIEWER)[2])
        assert saved["latest_live_diagnosis"] == run
        validate_ui("TraceDetail", saved)


def test_manual_requests_are_scoped_idempotent_and_retryable(tmp_path):
    clients = []
    with make_server(tmp_path, assembly_window=0) as server:
        execution_id = capture(server, flagged=False)
        server.diagnostics.enqueue_flagged()
        assert not server.diagnostics.run_one()
        path = f"/api/v1/traces/{execution_id}/diagnosis"
        assert call(server, "POST", path, VIEWER)[0] == 403
        assert call(server, "POST", path, OTHER)[0] == 404
        assert call(server, "POST", path)[0] == 401
        status, _, body = call(server, "POST", path, ADMIN)
        assert status == 202
        queued = json.loads(body)
        validate_ui("DiagnosticRun", queued)
        assert json.loads(call(server, "POST", path, ADMIN)[2]) == queued

        def fail(config):
            raise ValueError("Set GEMINI_API_KEY in your environment or .env file")

        server.diagnostics.engine_factory = fail
        assert server.diagnostics.run_one()
        failed = server.ui.diagnosis(server.ui.find("first", execution_id))
        validate_ui("DiagnosticRun", failed)
        assert failed["state"] == "failed" and "GEMINI_API_KEY" in failed["error"]["message"]
        server.diagnostics.enqueue_flagged()
        assert not server.diagnostics.run_one()  # Never retry provider calls in a tight loop.
        server.diagnostics.engine_factory = mocked_factory(clients)
        retried = json.loads(call(server, "POST", path, ADMIN)[2])
        assert retried["diagnostic_run_id"] != failed["diagnostic_run_id"]
        assert not server.diagnostics.run_one("not-a-job")
        assert server.diagnostics.run_one(retried["diagnostic_run_id"])
        done = json.loads(call(server, "POST", path, ADMIN)[2])
        assert done["state"] == "completed"
        assert not server.diagnostics.run_one()
        assert clients[0].calls == 1


def test_missing_content_reports_insufficient_evidence_without_calling_judge(tmp_path):
    with make_server(tmp_path, assembly_window=0) as server:
        execution_id = capture(server, content=False)
        server.diagnostics.enqueue_flagged()
        server.diagnostics.engine_factory = lambda config: (_ for _ in ()).throw(AssertionError())
        assert server.diagnostics.run_one()
        detail, _ = server.ui.detail(server.ui.find("first", execution_id))
        validate_ui("TraceDetail", detail)
        assert detail["execution"]["diagnosis_state"] == "insufficient_evidence"
        assert "RAG_TRACE_CONTENT=1" in detail["latest_live_diagnosis"]["error"]["message"]


def test_abstention_is_completed_and_counted(tmp_path):
    with make_server(tmp_path, assembly_window=0) as server:
        execution_id = capture(server, answer='{"answer":"supported"}')
        server.diagnostics.engine_factory = mocked_factory([])
        server.diagnostics.enqueue_flagged()
        assert server.diagnostics.run_one()
        detail, _ = server.ui.detail(server.ui.find("first", execution_id))
        validate_ui("TraceDetail", detail)
        assert detail["latest_live_diagnosis"]["state"] == "completed"
        assert detail["latest_live_diagnosis"]["verdict"]["status"] == "abstained"
        health = server.ui.health(
            "first", {"from": "1970-01-01T00:00:00Z", "to": "1970-01-02T00:00:00Z"}
        )
        assert health["totals"]["abstention_rate"] == 1.0


def test_live_claim_prevents_duplicate_calls_and_expired_claim_recovers(tmp_path):
    with make_server(tmp_path, assembly_window=0) as server:
        execution_id = capture(server)
        server.diagnostics.enqueue_flagged()
        claimed = server.diagnostics._claim()
        assert claimed and not server.diagnostics._claim()
        with server.store.connection() as db:
            db.execute("UPDATE diagnostic_jobs SET lease_until=?", (time.time() - 1,))
        clients = []
        server.diagnostics.engine_factory = mocked_factory(clients)
        assert server.diagnostics.run_one()
        assert clients[0].calls == 1
        assert (
            server.ui.execution(server.ui.find("first", execution_id))["diagnosis_state"]
            == "completed"
        )


def test_queued_and_running_status_visible_during_provider_call(tmp_path):
    with make_server(tmp_path, assembly_window=0) as server:
        execution_id = capture(server)
        server.diagnostics.enqueue_flagged()
        entered, finish = threading.Event(), threading.Event()
        factory = mocked_factory([])

        def waiting_factory(config):
            entered.set()
            assert finish.wait(timeout=10)
            return factory(config)

        server.diagnostics.engine_factory = waiting_factory
        worker = threading.Thread(target=server.diagnostics.run_one)
        worker.start()
        try:
            assert entered.wait(timeout=10)
            detail, _ = server.ui.detail(server.ui.find("first", execution_id))
            validate_ui("TraceDetail", detail)
            assert detail["execution"]["diagnosis_state"] == "running"
            assert not server.diagnostics.run_one()
        finally:
            finish.set()
            worker.join(timeout=10)
        assert not worker.is_alive()


def test_server_background_worker_schedules_existing_flags(tmp_path):
    with make_server(tmp_path, assembly_window=0) as server:
        execution_id = capture(server)
        server.diagnostics.engine_factory = mocked_factory([])
        worker = threading.Thread(target=server._diagnose)
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                state = server.ui.execution(server.ui.find("first", execution_id))[
                    "diagnosis_state"
                ]
                if state == "completed":
                    break
                server.stop_worker.wait(0.02)
            assert state == "completed"
        finally:
            server.stop_worker.set()
            worker.join(timeout=10)
        assert not worker.is_alive()

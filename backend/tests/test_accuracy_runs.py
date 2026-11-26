"""Exercise the HTTP benchmark workflow with the real runner and an offline judge."""

import json

from test_otlp_server import ADMIN, OTHER, VIEWER, call, make_server

from rag_diagnostic import accuracy_runs
from rag_diagnostic.judge import JudgeResponse
from rag_diagnostic.mechanism_benchmark import run_mechanism_suite
from rag_diagnostic.mechanisms import MECHANISM_STAGE
from rag_diagnostic.stage_diagnosis import unknown

PATH = "/api/v1/accuracy-runs"


class RecordedJudge:
    def describe(self):
        return {"provider": "test", "model": "offline-judge"}

    def evaluate_trace(self, payload):
        assert "expected_root" not in payload
        rows = [unknown(label, "No supported conclusion") for label in MECHANISM_STAGE]
        return JudgeResponse(json.dumps({"evaluations": rows}), "offline")


def test_admin_run_persists_real_scores_and_is_tenant_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "unused-offline")
    monkeypatch.setattr(
        accuracy_runs,
        "run_mechanism_suite",
        lambda output, config: run_mechanism_suite(output, config, judge=RecordedJudge()),
    )
    with make_server(tmp_path) as server:
        assert call(server, "GET", PATH)[0] == 401
        assert call(server, "POST", PATH, VIEWER)[0] == 403

        status, _, raw = call(server, "POST", PATH, ADMIN)
        queued = json.loads(raw)
        assert status == 202 and queued["state"] == "queued"
        assert json.loads(call(server, "POST", PATH, ADMIN)[2]) == queued

        assert server.accuracy.run_one()
        assert not server.accuracy.run_one()
        path = f"{PATH}/{queued['run_id']}"
        status, headers, raw = call(server, "GET", path, VIEWER)
        assert status == 200 and headers["Cache-Control"] == "no-store"
        run = json.loads(raw)
        assert run["state"] == "completed"
        cases = run["report"]["cases"]
        attempted = len(cases)
        assert attempted == run["finished_cases"] == run["total_cases"]
        assert run["summary"]["suite_accuracy"] == round(
            sum(case["correct"] for case in cases) / attempted, 4
        )
        metrics = run["report"]["mechanism_metrics"]
        assert set(metrics) == set(MECHANISM_STAGE)
        assert metrics["OUTPUT_FORMAT_VIOLATION"]["support"] == 2
        assert metrics["OUTPUT_FORMAT_VIOLATION"]["precision"] == 1
        assert metrics["EMBEDDING_MISMATCH"]["precision"] is None
        assert metrics["EMBEDDING_MISMATCH"]["recall"] == 0
        assert any(case["outcome"] == "abstained" for case in cases)
        assert any(case["outcome"] == "correct" and case["status"] == "abstained" for case in cases)
        assert call(server, "GET", path, OTHER)[0] == 404
        assert json.loads(call(server, "GET", PATH, OTHER)[2])["items"] == []
        assert "config" not in run["report"]

    with make_server(tmp_path) as restarted:
        history = json.loads(call(restarted, "GET", PATH, VIEWER)[2])
        assert not history["can_run"]
        assert history["items"][0]["summary"] == run["summary"]
        assert json.loads(call(restarted, "GET", path, ADMIN)[2]) == run
        assert json.loads(call(restarted, "POST", PATH, ADMIN)[2])["run_id"] != queued["run_id"]


def test_errors_keep_ground_truth_and_remain_visible(tmp_path, monkeypatch):
    class FailingJudge(RecordedJudge):
        def evaluate_trace(self, payload):
            raise RuntimeError("private error details")

    monkeypatch.setenv("GEMINI_API_KEY", "unused-offline")
    monkeypatch.setattr(
        accuracy_runs,
        "run_mechanism_suite",
        lambda output, config: run_mechanism_suite(output, config, judge=FailingJudge()),
    )
    with make_server(tmp_path) as server:
        queued = server.accuracy.start("first")
        assert server.accuracy.run_one()
        run = server.accuracy.detail("first", queued["run_id"])
        assert run["state"] == "completed_with_errors"
        assert run["summary"]["errors"] == run["total_cases"]
        assert run["summary"]["suite_accuracy"] == 0
        assert run["report"]["mechanism_metrics"]["EMBEDDING_MISMATCH"]["recall"] == 0
        assert any(case["expected_faults"] for case in run["report"]["cases"])
        assert all(case["outcome"] == "error" for case in run["report"]["cases"])
        assert "private error details" not in json.dumps(run)

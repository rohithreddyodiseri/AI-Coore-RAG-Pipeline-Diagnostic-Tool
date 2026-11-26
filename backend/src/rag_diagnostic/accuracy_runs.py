"""Schedule the fault-test-runner's technical suite and publish tenant-scoped results."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone

from rag_diagnostic.mechanism_benchmark import run_mechanism_suite
from rag_diagnostic.mechanism_cases import build_cases

LOG = logging.getLogger(__name__)
SUMMARY_FIELDS = (
    "suite_accuracy",
    "primary_accuracy",
    "stage_accuracy",
    "mechanism_precision",
    "mechanism_recall",
    "attempted",
    "errors",
)
CASE_FIELDS = (
    "run_id",
    "kind",
    "description",
    "expected_root",
    "predicted_root",
    "expected_faults",
    "predicted_faults",
    "status",
    "correct",
    "root_correct",
    "stage_correct",
    "target_correct",
    "exact_fault_set_correct",
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def case_outcome(case):
    if "error" in case:
        return "error"
    if case["correct"]:
        return "correct"
    if case["status"] == "abstained":
        return "abstained"
    return "incorrect"


class AccuracyRuns:
    def __init__(self, store, config):
        self.store = store
        self.config = config
        self.directory = store.directory / "accuracy"
        self.case_count = len(build_cases())

    def _output(self, row):
        return self.directory / row["tenant"] / row["run_id"]

    def start(self, tenant):
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute(
                "SELECT * FROM accuracy_runs WHERE tenant=? AND state IN ('queued', 'running')",
                (tenant,),
            ).fetchone()
            if active is None:
                run_id = uuid.uuid4().hex
                db.execute(
                    """INSERT INTO accuracy_runs (run_id, tenant, state, created_at)
                       VALUES (?, ?, 'queued', ?)""",
                    (run_id, tenant, timestamp()),
                )
                active = db.execute(
                    "SELECT * FROM accuracy_runs WHERE run_id=?", (run_id,)
                ).fetchone()
        return self._metadata(active)

    def _metadata(self, row):
        fields = (
            "run_id",
            "state",
            "created_at",
            "started_at",
            "finished_at",
            "error_message",
        )
        summary = json.loads(row["summary"]) if row["summary"] else None
        progress = (
            summary["attempted"]
            if summary
            else sum(1 for _ in (self._output(row) / "results").glob("*.json"))
        )
        return {key: row[key] for key in fields} | {
            "summary": summary,
            "total_cases": summary["attempted"] if summary else self.case_count,
            "finished_cases": progress,
        }

    def list(self, tenant):
        with self.store.connection() as db:
            rows = db.execute(
                "SELECT * FROM accuracy_runs WHERE tenant=? ORDER BY created_at DESC",
                (tenant,),
            ).fetchall()
        return {"items": [self._metadata(row) for row in rows], "suite_cases": self.case_count}

    def detail(self, tenant, run_id):
        with self.store.connection() as db:
            row = db.execute(
                "SELECT * FROM accuracy_runs WHERE tenant=? AND run_id=?", (tenant, run_id)
            ).fetchone()
        if row is None:
            return None
        result = self._metadata(row)
        if row["summary"]:
            report = json.loads((self._output(row) / "report.json").read_text())
            result["report"] = {
                "mechanism_metrics": report["mechanism_metrics"],
                "cases": [
                    {key: case[key] for key in CASE_FIELDS if key in case}
                    | {"outcome": case_outcome(case)}
                    for case in report["cases"]
                ],
            }
        return result

    def _finish(self, row, report=None, error=None):
        summary = None
        state = "failed"
        if report is not None:
            summary = {key: report[key] for key in SUMMARY_FIELDS}
            summary.update(
                model=report["judge"].get("model"), taxonomy_version=report["taxonomy_version"]
            )
            state = report["status"]
        with self.store.connection() as db:
            db.execute(
                """UPDATE accuracy_runs SET state=?, finished_at=?, summary=?, error_message=?
                   WHERE run_id=?""",
                (
                    state,
                    timestamp(),
                    json.dumps(summary) if summary else None,
                    error,
                    row["run_id"],
                ),
            )

    def run_one(self):
        # The OS releases the lock on exit, so only a successor can recover interrupted work.
        self.directory.mkdir(mode=0o700, exist_ok=True)
        # Use a separate database so the worker lock never blocks registry or telemetry reads.
        with closing(sqlite3.connect(self.directory / "worker.sqlite3", timeout=0)) as lock:
            try:
                lock.execute("BEGIN EXCLUSIVE")
            except sqlite3.OperationalError as error:
                if str(error) == "database is locked":
                    return False
                raise
            return self._run_locked()

    def _run_locked(self):
        with self.store.connection() as db:
            interrupted = db.execute("SELECT * FROM accuracy_runs WHERE state='running'").fetchall()
            row = db.execute(
                "SELECT * FROM accuracy_runs WHERE state='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
        for previous in interrupted:
            path = self._output(previous) / "report.json"
            if path.exists():
                self._finish(previous, json.loads(path.read_text()))
            else:
                self._finish(
                    previous,
                    error="The server stopped before the suite finished. Start a new run.",
                )
        if row is None:
            return False
        with self.store.connection() as db:
            db.execute(
                "UPDATE accuracy_runs SET state='running', started_at=? WHERE run_id=?",
                (timestamp(), row["run_id"]),
            )
        if self.config.judge != "llm" or not (
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        ):
            self._finish(
                row,
                error="Set a Gemini API key on the diagnostic server and run the suite again.",
            )
            return True
        output = self._output(row)
        try:
            # Each run uses fresh judge responses so comparisons measure separate executions.
            config = replace(self.config, judge_cache=str(output.resolve() / "judge_cache.json"))
            report = run_mechanism_suite(output, config)
            self._finish(row, report)
        except Exception as error:
            LOG.warning("Accuracy suite failed (%s)", type(error).__name__)
            self._finish(
                row,
                error="The fault suite could not finish. Check the server logs and run it again.",
            )
        return True

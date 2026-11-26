"""Durable, leased diagnostic jobs for captured executions; never re-run a client pipeline."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone

from gemini_client import GeminiAPIError
from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnostic_engine import DiagnosticEngine

LOG = logging.getLogger(__name__)
LEASE_SECONDS = 60


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_name(stage):
    return {
        "normalization": "parsing",
        "embedding": "document_embedding",
        "selection": "retrieval",
        "context": "context_assembly",
    }.get(stage, stage)


def present_report(report):
    """Expose the validated technical mechanism without relabeling it as an FP category."""
    root = report["root_cause"]
    observations = report["observations"]
    judge = observations["judge"]
    run = report["run"]
    revision = run["code_revision"]
    faults = [row for row in report["evaluations"] if row["status"] == "FAULT"]
    return {
        "input_fingerprint": observations["input_fingerprint"],
        "audit": {
            "tool_version": run["tool_version"],
            "git_sha": None if revision["commit"] == "unknown" else revision["commit"],
            "dirty_worktree": revision["uncommitted_changes"],
            "engine_source_sha256": observations["input"]["source_sha256"],
            "config_fingerprint": run["config_fingerprint"],
            "judge": {
                "provider": judge["provider"],
                "model": judge["model"],
                "model_version": observations.get("judge_response", {}).get("actual_model"),
                "temperature": judge["temperature"],
                "prompt_version": str(judge["prompt_version"]),
                "prompt_sha256": judge["prompt_sha256"],
                "output_schema_sha256": judge["schema_sha256"],
            },
        },
        "verdict": {
            "status": report["status"],
            "primary_stage": stage_name(root["technical_stage"]) if root else None,
            "category": None,
            "mechanism": root["mechanism"] if root else None,
            "confidence": root["confidence"] if root else None,
            "contributing_stages": list(
                dict.fromkeys(
                    stage_name(row["technical_stage"]) for row in report["contributing_failures"]
                )
            ),
            "unresolved_categories": [],
            "unresolved_mechanisms": report["unresolved_issues"],
            "evaluations": report["evaluations"],
            "summary": root["rationale"]
            if root
            else (
                "The captured evidence does not support a root-cause attribution. "
                "Review the unresolved mechanism checks."
                if report["status"] == "abstained"
                else "No failure found in the observed evidence."
            ),
            "evidence_trail": [
                {
                    "stage": stage_name(row["technical_stage"]),
                    "category": None,
                    "mechanism": row["mechanism"],
                    "finding": row["rationale"],
                    "citations": row["evidence"],
                }
                for row in faults
            ],
            "recommendations": [],
        },
    }


class DiagnosticWorker:
    def __init__(self, store, model, config=None, engine_factory=DiagnosticEngine):
        self.store, self.model = store, model
        self.config = config or DiagnosticConfig()
        if self.config.assessment_mode != "stage":
            raise ValueError("The diagnostic server requires assessment_mode=stage")
        self.engine_factory = engine_factory

    def request(self, row, *, retry=False):
        identity = (row["app_id"], row["trace_id"], row["span_id"])
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if retry:
                db.execute(
                    """DELETE FROM diagnostic_jobs WHERE app_id=? AND trace_id=? AND span_id=?
                       AND state IN ('failed', 'insufficient_evidence')""",
                    identity,
                )
            db.execute(
                """INSERT OR IGNORE INTO diagnostic_jobs
                   (app_id, trace_id, span_id, job_id, state, created_at)
                   VALUES (?, ?, ?, ?, 'queued', ?)""",
                (*identity, uuid.uuid4().hex, now()),
            )
        return self.model.diagnosis(
            self.model.find(row["tenant"], self.model.execution(row)["execution_id"])
        )

    def enqueue_flagged(self):
        """Backfill already-ingested flagged executions as well as new arrivals."""
        with self.store.connection() as db:
            tenants = [row[0] for row in db.execute("SELECT DISTINCT tenant FROM apps")]
        for tenant in tenants:
            for row in self.model.records(tenant):
                if (
                    row["normalized_at"]
                    and not row["job_id"]
                    and self.model.execution(row)["flagged"]
                ):
                    self.request(row)

    def _claim(self, job_id=None):
        owner = uuid.uuid4().hex
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """UPDATE diagnostic_jobs SET state='queued', lease_owner=NULL, lease_until=NULL
                   WHERE state='running' AND lease_until < ?""",
                (time.time(),),
            )
            row = db.execute(
                """SELECT d.*, a.tenant FROM diagnostic_jobs d JOIN apps a USING(app_id)
                   JOIN executions e ON e.app_id=d.app_id AND e.trace_id=d.trace_id
                                    AND e.span_id=d.span_id
                   WHERE state='queued' AND e.due_at<=? AND (? IS NULL OR d.job_id=?)
                   ORDER BY d.created_at LIMIT 1""",
                (time.time(), job_id, job_id),
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE diagnostic_jobs SET state='running', lease_owner=?, lease_until=? "
                    "WHERE job_id=?",
                    (owner, time.time() + LEASE_SECONDS, row["job_id"]),
                )
                return dict(row), owner
        return None

    def _heartbeat(self, job_id, owner, stop):
        while not stop.wait(LEASE_SECONDS / 3):
            try:
                with self.store.connection() as db:
                    db.execute(
                        "UPDATE diagnostic_jobs SET lease_until=? WHERE job_id=? AND lease_owner=?",
                        (time.time() + LEASE_SECONDS, job_id, owner),
                    )
            except Exception:
                LOG.warning("Could not renew diagnostic job lease")

    def run_one(self, job_id=None):
        claimed = self._claim(job_id)
        if not claimed:
            return False
        job, owner = claimed
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat, args=(job["job_id"], owner, stop), daemon=True
        )
        heartbeat.start()
        state, result, code, message = "completed", None, None, None
        try:
            execution_id = f"{job['app_id']}_{job['trace_id'].hex()}_{job['span_id'].hex()}"
            row = self.model.find(job["tenant"], execution_id)
            envelope = self.model.envelope(row)
            if not envelope.get("query", "").strip():
                state, code = "insufficient_evidence", "content_not_captured"
                message = (
                    "Query content was not captured. Set RAG_TRACE_CONTENT=1 on the RAG service, "
                    "restart it, and send another flagged query."
                )
            elif self.config.judge == "unavailable":
                state, code, message = "failed", "judge_unavailable", "Configure an LLM judge."
            else:
                directory = self.store.directory / job["tenant"] / job["app_id"] / "diagnostics"
                config = replace(
                    self.config,
                    runs_dir=str(directory),
                    judge_cache=str(directory / "judge_cache.json"),
                )
                report = self.engine_factory(config).diagnose(envelope).to_dict()
                # Validate presentation before committing a completed state.
                present_report(report)
                result = json.dumps(report, allow_nan=False)
        except Exception as error:
            state, code = "failed", "diagnostic_error"
            if isinstance(error, GeminiAPIError):
                message = (
                    f"The judge provider request failed (code {error.code or 'unknown'}). "
                    "Retry diagnosis."
                )
            elif isinstance(error, ValueError) and str(error).startswith("Set GEMINI_API_KEY"):
                message = (
                    "Set GEMINI_API_KEY in the diagnostic backend environment and retry diagnosis."
                )
            else:
                message = (
                    "The diagnostic engine could not process this execution. "
                    "Check backend configuration and retry."
                )
            LOG.warning("Diagnostic job failed (%s)", type(error).__name__)
        finally:
            stop.set()
            heartbeat.join()
        with self.store.connection() as db:
            db.execute(
                """UPDATE diagnostic_jobs SET state=?, result=?, error_code=?, error_message=?,
                   finished_at=?, lease_owner=NULL, lease_until=NULL
                   WHERE job_id=? AND lease_owner=?""",
                (state, result, code, message, now(), job["job_id"], owner),
            )
        return True

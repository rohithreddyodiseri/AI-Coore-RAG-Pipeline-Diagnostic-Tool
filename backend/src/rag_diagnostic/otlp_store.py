"""Durable app registry, OTLP span intake, and normalized execution spool."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import tempfile
import time
import uuid
from collections import Counter
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from opentelemetry.proto.trace.v1.trace_pb2 import Span

from rag_diagnostic.otlp_normalizer import (
    attribute_values,
    execution_span,
    normalize_execution,
    supported_execution,
)

MAX_TRACE_SPANS = 4096
MAX_TRACE_BYTES = 16 * 1024 * 1024

_DDL = """
CREATE TABLE IF NOT EXISTS apps (
    app_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, service_name TEXT NOT NULL,
    framework TEXT NOT NULL, environments TEXT NOT NULL, created_at TEXT NOT NULL,
    last_trace_at TEXT, key_id TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE,
    UNIQUE(tenant, service_name)
);
CREATE INDEX IF NOT EXISTS apps_by_tenant_id ON apps(tenant, app_id);
CREATE TABLE IF NOT EXISTS spans (
    app_id TEXT NOT NULL REFERENCES apps(app_id), trace_id BLOB NOT NULL,
    span_id BLOB NOT NULL, span BLOB NOT NULL, resource BLOB NOT NULL, scope BLOB NOT NULL,
    PRIMARY KEY(app_id, trace_id, span_id)
);
CREATE TABLE IF NOT EXISTS executions (
    app_id TEXT NOT NULL, trace_id BLOB NOT NULL, span_id BLOB NOT NULL, due_at REAL NOT NULL,
    normalized_at TEXT, PRIMARY KEY(app_id, trace_id, span_id),
    FOREIGN KEY(app_id, trace_id, span_id) REFERENCES spans(app_id, trace_id, span_id)
);
CREATE INDEX IF NOT EXISTS pending_executions ON executions(due_at) WHERE normalized_at IS NULL;
CREATE TABLE IF NOT EXISTS diagnostic_jobs (
    app_id TEXT NOT NULL, trace_id BLOB NOT NULL, span_id BLOB NOT NULL,
    job_id TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
    created_at TEXT NOT NULL, finished_at TEXT,
    result TEXT, error_code TEXT, error_message TEXT,
    lease_owner TEXT, lease_until REAL,
    PRIMARY KEY(app_id, trace_id, span_id),
    FOREIGN KEY(app_id, trace_id, span_id) REFERENCES executions(app_id, trace_id, span_id)
);
CREATE INDEX IF NOT EXISTS diagnostic_jobs_by_state ON diagnostic_jobs(state, created_at);
CREATE TABLE IF NOT EXISTS accuracy_runs (
    run_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, state TEXT NOT NULL,
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
    summary TEXT, error_message TEXT
);
CREATE INDEX IF NOT EXISTS accuracy_runs_by_tenant ON accuracy_runs(tenant, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS active_accuracy_run_by_tenant ON accuracy_runs(tenant)
    WHERE state IN ('queued', 'running');
"""


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def token_digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Store:
    def __init__(self, directory, validator, assembly_window):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.database = self.directory / "ingestion.sqlite3"
        fd = os.open(self.database, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        self.validator = validator
        self.assembly_window = assembly_window
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_DDL)

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.database, timeout=5)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            yield db

    @staticmethod
    def _app(row):
        result = {
            field: row[field]
            for field in ("app_id", "service_name", "framework", "created_at", "last_trace_at")
        }
        result["environments"] = json.loads(row["environments"])
        return result

    def register(self, tenant, registration):
        token = f"diag_{secrets.token_urlsafe(32)}"
        app_id, key_id = uuid.uuid4().hex, uuid.uuid4().hex
        with self.connection() as db:
            db.execute(
                """INSERT INTO apps
                   (app_id, tenant, service_name, framework, environments,
                    created_at, key_id, key_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    app_id,
                    tenant,
                    registration["service_name"],
                    registration["framework"],
                    json.dumps(registration["environments"]),
                    _timestamp(),
                    key_id,
                    token_digest(token),
                ),
            )
            row = db.execute("SELECT * FROM apps WHERE app_id=?", (app_id,)).fetchone()
        return self._app(row), key_id, token

    def rotate(self, tenant, app_id):
        token, key_id = f"diag_{secrets.token_urlsafe(32)}", uuid.uuid4().hex
        with self.connection() as db:
            result = db.execute(
                "UPDATE apps SET key_id=?, key_hash=? WHERE tenant=? AND app_id=?",
                (key_id, token_digest(token), tenant, app_id),
            )
            if not result.rowcount:
                return None
            row = db.execute("SELECT * FROM apps WHERE app_id=?", (app_id,)).fetchone()
        return self._app(row), key_id, token

    def app(self, tenant, app_id):
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM apps WHERE tenant=? AND app_id=?", (tenant, app_id)
            ).fetchone()
        return self._app(row) if row else None

    def apps(self, tenant, cursor, limit):
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM apps WHERE tenant=? AND app_id>? ORDER BY app_id LIMIT ?",
                (tenant, cursor, limit + 1),
            ).fetchall()
        items = [self._app(row) for row in rows[:limit]]
        return {
            "items": items,
            "next_cursor": items[-1]["app_id"] if len(rows) > limit else None,
        }

    def authenticates_ingestion(self, token):
        with self.connection() as db:
            return (
                db.execute("SELECT 1 FROM apps WHERE key_hash=?", (token_digest(token),)).fetchone()
                is not None
            )

    def accept(self, token, request):
        """Commit accepted spans and execution deadlines before acknowledging the SDK."""
        accepted, rejected = 0, Counter()
        connected = False
        budgets = {}
        with self.connection() as db:
            # Serialize the token lookup with rotation and span insertion.
            db.execute("BEGIN IMMEDIATE")
            app = db.execute(
                "SELECT * FROM apps WHERE key_hash=?", (token_digest(token),)
            ).fetchone()
            if app is None:
                return None
            for resource_spans in request.resource_spans:
                resource = resource_spans.resource
                service = attribute_values(resource).get("service.name")
                matches = service == app["service_name"]
                resource_bytes = resource.SerializeToString()
                for scoped in resource_spans.scope_spans:
                    scope_bytes = scoped.scope.SerializeToString()
                    for span in scoped.spans:
                        attrs = attribute_values(span)
                        if execution_span(span, attrs) and not matches:
                            rejected["execution service.name does not match the app"] += 1
                        elif not _valid_span(span):
                            rejected["invalid span ID or timestamp"] += 1
                        elif not self._store_span(
                            db, app["app_id"], span, resource_bytes, scope_bytes, attrs, budgets
                        ):
                            rejected["trace storage limit exceeded"] += 1
                        else:
                            accepted += 1
                            connected |= matches
            if connected:
                db.execute(
                    "UPDATE apps SET last_trace_at=? WHERE app_id=?",
                    (_timestamp(), app["app_id"]),
                )
        return accepted, rejected

    def _store_span(self, db, app_id, span, resource, scope, attrs, budgets):
        identity = (app_id, span.trace_id, span.span_id)
        if db.execute(
            "SELECT 1 FROM spans WHERE app_id=? AND trace_id=? AND span_id=?", identity
        ).fetchone():
            return True
        if span.trace_id not in budgets:
            budgets[span.trace_id] = tuple(
                db.execute(
                    """SELECT count(*),
                              coalesce(sum(length(span)+length(resource)+length(scope)), 0)
                       FROM spans WHERE app_id=? AND trace_id=?""",
                    identity[:2],
                ).fetchone()
            )
        count, size = budgets[span.trace_id]
        encoded = span.SerializeToString()
        size += len(encoded) + len(resource) + len(scope)
        if count >= MAX_TRACE_SPANS or size > MAX_TRACE_BYTES:
            return False
        db.execute(
            "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
            (*identity, encoded, resource, scope),
        )
        budgets[span.trace_id] = (count + 1, size)
        if supported_execution(span, attrs):
            db.execute(
                "INSERT INTO executions (app_id, trace_id, span_id, due_at) VALUES (?, ?, ?, ?)",
                (*identity, time.time() + self.assembly_window),
            )
        return True

    def publish_one(self):
        """Atomically write one due envelope; retry the same path after a process crash."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """SELECT executions.*, apps.tenant, apps.service_name
                   FROM executions JOIN apps USING(app_id)
                   WHERE normalized_at IS NULL AND due_at<=?
                   ORDER BY due_at LIMIT 1""",
                (time.time(),),
            ).fetchone()
            if row is None:
                return None
            identity = (row["app_id"], row["trace_id"])
            spans = [
                Span.FromString(record["span"])
                for record in db.execute(
                    "SELECT span FROM spans WHERE app_id=? AND trace_id=?", identity
                )
            ]
            envelope = normalize_execution(
                row["span_id"], spans, row["service_name"], self.validator
            )
            # The legacy watcher assumes a query. Keep metadata-only executions separate.
            collection = "executions" if "query" in envelope else "metadata-only"
            destination = self.directory / row["tenant"] / row["app_id"] / collection
            destination.mkdir(mode=0o700, parents=True, exist_ok=True)
            file = destination / f"{row['trace_id'].hex()}-{row['span_id'].hex()}.json"
            _write_json(file, envelope)
            db.execute(
                """UPDATE executions SET normalized_at=?
                   WHERE app_id=? AND trace_id=? AND span_id=?""",
                (_timestamp(), *identity, row["span_id"]),
            )
            return file


def _valid_span(span):
    return (
        len(span.trace_id) == 16
        and any(span.trace_id)
        and len(span.span_id) == 8
        and any(span.span_id)
        and (
            not span.parent_span_id or (len(span.parent_span_id) == 8 and any(span.parent_span_id))
        )
        and 0 < span.start_time_unix_nano <= span.end_time_unix_nano
    )


def _write_json(path, data):
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(data, handle, ensure_ascii=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

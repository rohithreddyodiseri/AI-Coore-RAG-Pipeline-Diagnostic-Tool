"""File-store consumer for flagged executions; no target pipeline is imported or run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag_diagnostic.diagnostic_engine import DiagnosticEngine
from rag_diagnostic.telemetry import (
    IncompleteExecution,
    execution_ids,
    hydrate_index_artifacts,
    load_execution,
)


class TelemetryIngestor:
    def __init__(self, engine: DiagnosticEngine):
        self.engine = engine
        self.directory = Path(engine.config.runs_dir)
        self.ledger_path = self.directory / "ingestion.json"

    def ingest(self, path, *, run_id=None, artifact_root=None):
        payload = load_execution(Path(path), run_id, artifact_root)
        verdict = self.engine.diagnose(payload)
        report_name = hashlib.sha256(verdict.run_id.encode()).hexdigest()[:24] + ".json"
        destination = self.directory / "reports" / report_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(verdict.to_dict(), indent=2), encoding="utf-8")
        return verdict, destination

    def poll(self, trace_directory, *, artifact_root=None):
        """Consume terminal flagged runs once per payload, tool revision and judge settings.

        In-progress files are retried. Errors from terminal records/provider calls
        propagate so a supervisor can retry without marking an execution processed.
        """
        processed = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}
        source = hashlib.sha256()
        for path in sorted(Path(__file__).parent.rglob("*.py")):
            source.update(path.read_bytes())
        identity = [self.engine.config.to_dict(), self.engine.judge.describe(), source.hexdigest()]
        results = []
        for path in sorted(Path(trace_directory).iterdir()):
            if not path.is_file() or path.suffix not in (".json", ".jsonl"):
                continue
            try:
                ids = execution_ids(path)
            except IncompleteExecution:
                continue
            for selected in ids:
                try:
                    payload = load_execution(
                        path, run_id=selected, artifact_root=artifact_root, resolve_artifacts=False
                    )
                except IncompleteExecution:
                    continue
                if not payload.get("failure_signal", {}).get("flagged"):
                    continue
                payload = hydrate_index_artifacts(
                    payload, Path(artifact_root) if artifact_root else path.parent
                )
                run_id = payload.get("run_id")
                if not isinstance(run_id, str) or not run_id.strip():
                    raise ValueError("flagged executions require a run_id")
                key = hashlib.sha256(
                    json.dumps([identity, payload], sort_keys=True).encode()
                ).hexdigest()
                if key in processed:
                    continue
                verdict = self.engine.diagnose(payload)
                report_name = hashlib.sha256(verdict.run_id.encode()).hexdigest()[:24] + ".json"
                destination = self.directory / "reports" / report_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(verdict.to_dict(), indent=2))
                processed[key] = {"run_id": run_id, "source": str(path), "report": str(destination)}
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = self.ledger_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(processed, sort_keys=True))
                temporary.replace(self.ledger_path)
                results.append((verdict, destination))
        return results

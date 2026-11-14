"""Decoupled ingestion of structured executions and append-only stage event logs.

This module has no dependency on the reference pipeline or its adapter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class IncompleteExecution(ValueError):
    """The writer has not published a terminal execution yet."""


def _legacy_payload(events, run_id):
    query_events = [event for event in events if event.get("stage") == "query"]
    if not query_events or query_events[-1].get("status") not in ("completed", "failed"):
        raise IncompleteExecution("execution is still being written")
    query = query_events[0].get("inputs", {}).get("query")
    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "query": query,
        "stages": {},
        "execution_status": query_events[-1]["status"],
        "pipeline": {"name": "captured_stage_events", **query_events[0].get("inputs", {})},
    }
    stages = payload["stages"]
    for event in events:
        name, inputs = event.get("stage"), event.get("inputs", {})
        if name not in (
            "parsing",
            "chunking",
            "document_embedding",
            "indexing",
            "query_embedding",
            "retrieval",
            "reranking",
            "context_assembly",
            "prompt_construction",
            "generation",
        ):
            continue
        stage = {"status": event.get("status", "unknown"), "input_data": inputs}
        if "error" in event:
            stage["error"] = event["error"]
        if name == "prompt_construction":
            name = "context_assembly"
            prompt = event.get("outputs", {})
            prefix = f"Question:\n{query}\n\nContext:\n"
            if "context" in prompt:
                stage["output_data"] = {"text": prompt["context"]}
            elif isinstance(prompt.get("input"), str) and prompt["input"].startswith(prefix):
                stage["output_data"] = {"text": prompt["input"][len(prefix) :]}
            if "instructions" in prompt:
                payload["generation_instructions"] = prompt["instructions"]
        elif name == "generation":
            stage["input_data"] = {
                "prompt": inputs.get("prompt", inputs.get("input", "")),
                "system_instruction": inputs.get("instructions"),
                "model": inputs.get("model"),
            }
            if "outputs" in event:
                output = event["outputs"]
                text = output.get("output_text") if isinstance(output, dict) else output
                if isinstance(text, str):
                    stage["output_data"] = text
                    payload["final_answer"] = text
        elif "outputs" in event:
            stage["output_data"] = event["outputs"]
        if name == "retrieval":
            stage["k"] = inputs.get("k", inputs.get("n_results", payload["pipeline"].get("top_k")))
        stages[name] = stage
    return payload


def decode_events(events: list[dict], run_id: str | None = None) -> dict[str, Any]:
    if not events or any(not isinstance(event, dict) for event in events):
        raise ValueError("telemetry requires JSON event objects")
    ids = {event.get("run_id") for event in events}
    if run_id is None:
        if len(ids) != 1 or None in ids:
            raise ValueError("select --run-id when a log contains multiple executions")
        run_id = next(iter(ids))
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("telemetry run_id must be a nonempty string")
    selected = [event for event in events if event.get("run_id") == run_id]
    if not selected:
        raise ValueError(f"execution {run_id!r} not found")
    published = [event for event in selected if event.get("stage") == "execution"]
    payload = dict(published[-1]["outputs"]) if published else _legacy_payload(selected, run_id)
    if payload.get("run_id") != run_id:
        raise ValueError("published payload and event run_id do not match")
    if payload.get("schema_version", "1.0") != "1.0":
        raise ValueError("unsupported execution schema_version")
    flags = [event for event in selected if event.get("stage") == "failure_flag"]
    failed = payload.get("execution_status") == "failed"
    if flags:
        payload["failure_signal"] = {
            "flagged": True,
            "reason": flags[-1].get("inputs", {}).get("reason"),
        }
    elif failed:
        payload["failure_signal"] = {"flagged": True, "reason": "pipeline execution failed"}
    return payload


def _read_artifact(reference, root):
    relative = reference.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("artifact paths must be relative to the authorized artifact root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes the authorized artifact root")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != reference.get("sha256"):
        raise ValueError(f"artifact checksum mismatch: {relative}")
    if reference.get("format") == "jsonl":
        return [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    return json.loads(content)


def hydrate_index_artifacts(payload, root):
    reference = payload.get("pipeline", {}).get("index_snapshot")
    if not reference:
        if "corpus" not in payload:
            payload.setdefault("inspection_errors", []).append(
                "No captured index/corpus snapshot was supplied."
            )
        return payload
    try:
        if reference.get("status") != "completed":
            raise ValueError("the recorded index build was not completed")
        manifest = _read_artifact(reference, root)
        if (
            manifest.get("run_id") != reference.get("run_id")
            or manifest.get("status") != "completed"
        ):
            raise ValueError("index manifest identity/status mismatch")
        expected_collection = payload.get("pipeline", {}).get("settings", {}).get("collection")
        if expected_collection is not None and manifest.get("collection") != expected_collection:
            raise ValueError("index manifest belongs to a different collection")
        captured = {name: _read_artifact(ref, root) for name, ref in manifest["artifacts"].items()}
        payload["index_manifest"] = manifest
        stages = payload.setdefault("stages", {})
        for name in ("parsing", "chunking", "document_embedding"):
            stages[name] = {
                "status": "completed",
                "recorded_at": "index_build",
                "index_run_id": manifest["run_id"],
                "output_data": captured[name],
            }
        stages["indexing"] = {
            "status": "completed",
            "recorded_at": "index_build",
            "index_run_id": manifest["run_id"],
            "output_data": {
                "collection": manifest["collection"],
                "chunks": manifest["chunks"],
                "embedding_model": manifest["embedding_model"],
            },
        }
        payload.setdefault("corpus", captured["corpus"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        payload.setdefault("inspection_errors", []).append(f"Index artifacts unavailable: {error}")
        # Never claim corpus completeness after failed artifact verification.
        if isinstance(payload.get("corpus"), dict):
            payload["corpus"] = {**payload["corpus"], "complete": False}
    return payload


def load_execution(
    path: Path,
    run_id: str | None = None,
    artifact_root: Path | None = None,
    *,
    resolve_artifacts: bool = True,
) -> dict:
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    try:
        document = json.loads(content)
    except json.JSONDecodeError:
        # An incomplete last line is retried by the watcher, never treated as a missing stage.
        try:
            events = [json.loads(line) for line in content.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise IncompleteExecution(f"incomplete or malformed JSONL record: {error}") from error
        payload = decode_events(events, run_id)
    else:
        if not isinstance(document, dict):
            raise ValueError("execution must be a JSON object or JSONL event stream")
        if "query" in document:
            payload = document
            if run_id is not None and payload.get("run_id") != run_id:
                raise ValueError("execution run_id does not match selection")
        else:
            payload = decode_events([document], run_id)
    if payload.get("schema_version", "1.0") != "1.0":
        raise ValueError("unsupported execution schema_version")
    status = payload.get("execution_status")
    if status in ("started", "running", "in_progress"):
        raise IncompleteExecution("execution is still running")
    if status not in (None, "completed", "failed"):
        raise ValueError("invalid execution_status")
    signal = payload.get("failure_signal", {})
    if not isinstance(signal, dict) or type(signal.get("flagged", False)) is not bool:
        raise ValueError("failure_signal.flagged must be boolean")
    if status == "failed":
        payload["failure_signal"] = {
            "flagged": True,
            "reason": signal.get("reason", "execution failed"),
        }
    return (
        hydrate_index_artifacts(payload, Path(artifact_root) if artifact_root else path.parent)
        if resolve_artifacts
        else payload
    )


def execution_ids(path: Path) -> list[str | None]:
    """List record identities in a store file, without hydrating build artifacts."""
    content = path.read_text(encoding="utf-8")
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        try:
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise IncompleteExecution("store file is still being written") from error
        ids = {row.get("run_id") for row in rows}
        if None in ids:
            raise ValueError("event records must have a run_id") from None
        return sorted(ids)
    if isinstance(value, dict) and "query" in value:
        return [value.get("run_id")]
    if isinstance(value, dict):
        return [value.get("run_id")] if "stage" in value else []
    raise ValueError("telemetry file must contain execution objects")


def flag_execution(path: Path, reason: str, run_id: str | None = None):
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a nonempty failure reason is required")
    content = path.read_text(encoding="utf-8")
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict) and "query" in value:
        if run_id and run_id != value.get("run_id"):
            raise ValueError("execution run_id does not match selection")
        if not value.get("run_id"):
            raise ValueError("flagged executions require a run_id")
        value["failure_signal"] = {"flagged": True, "reason": reason}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value))
        temporary.replace(path)
        return value["run_id"]
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    payload = decode_events(rows, run_id)
    with path.open("a", encoding="utf-8") as stream:
        if content and not content.endswith("\n"):
            stream.write("\n")
        stream.write(
            json.dumps(
                {
                    "run_id": payload["run_id"],
                    "stage": "failure_flag",
                    "status": "completed",
                    "inputs": {"reason": reason},
                }
            )
            + "\n"
        )
    return payload["run_id"]

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)


class PipelineTrace:
    """Append-only diagnostic trace for a single pipeline run."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(timezone.utc)
        self.run_id = uuid.uuid4().hex
        self.events: list[dict[str, Any]] = []
        filename = f"{started_at:%Y%m%dT%H%M%S.%fZ}-{self.run_id}.jsonl"
        self.path = directory / filename

    def record(
        self,
        stage: str,
        inputs: Any,
        *,
        outputs: Any | None = None,
        status: str = "completed",
        duration_ms: float | None = None,
        error: BaseException | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "timestamp": _timestamp(),
            "run_id": self.run_id,
            "stage": stage,
            "status": status,
            "inputs": inputs,
        }
        if outputs is not None:
            event["outputs"] = outputs
        if duration_ms is not None:
            event["duration_ms"] = round(duration_ms, 3)
        if error is not None:
            event["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        with self.path.open("a", encoding="utf-8") as trace_file:
            json.dump(event, trace_file, ensure_ascii=False, default=_json_default)
            trace_file.write("\n")
        self.events.append(event)

    def capture(
        self,
        stage: str,
        inputs: Any,
        operation: Callable[[], T],
        output: Callable[[T], Any] | None = None,
    ) -> T:
        started = time.perf_counter()
        try:
            result = operation()
            serialized_output = output(result) if output else result
        except Exception as error:
            self.record(
                stage,
                inputs,
                status="failed",
                duration_ms=(time.perf_counter() - started) * 1000,
                error=error,
            )
            raise
        self.record(
            stage,
            inputs,
            outputs=serialized_output,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return result

    def execution_payload(self, query: str, pipeline: dict[str, Any]) -> dict[str, Any]:
        stages: dict[str, Any] = {"reranking": {"status": "not_used"}}
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "query": query,
            "pipeline": pipeline,
            "stages": stages,
            "reranker_used": False,
        }
        for event in self.events:
            name = event["stage"]
            if name not in ("query_embedding", "retrieval", "prompt_construction", "generation"):
                continue
            stage = {"status": event["status"], "input_data": event["inputs"]}
            if "error" in event:
                stage["error"] = event["error"]
            if name == "prompt_construction":
                name = "context_assembly"
                if "outputs" in event:
                    output = event["outputs"]
                    stage["output_data"] = {
                        "text": output["context"],
                        "chunk_ids": [
                            row["chunk_id"] for row in event["inputs"]["retrieved_results"]
                        ],
                    }
                    payload["generation_instructions"] = output["instructions"]
            elif name == "generation":
                stage["input_data"] = {
                    "prompt": event["inputs"]["input"],
                    "system_instruction": event["inputs"]["instructions"],
                    "model": event["inputs"]["model"],
                    "temperature": event["inputs"]["temperature"],
                }
                if "outputs" in event:
                    stage["output_data"] = event["outputs"]["output_text"]
                    payload["final_answer"] = stage["output_data"]
            elif "outputs" in event:
                stage["output_data"] = event["outputs"]
                if name == "retrieval":
                    stage["k"] = event["inputs"]["k"]
            stages[name] = stage
        return payload

    def publish(self, payload: dict[str, Any], *, failed: bool = False) -> None:
        """Publish a complete execution envelope to the standard JSONL stream."""
        self.record(
            "execution",
            {},
            outputs={
                "schema_version": "1.0",
                "run_id": self.run_id,
                **payload,
                "execution_status": "failed" if failed else "completed",
                "failure_signal": payload.get(
                    "failure_signal",
                    {"flagged": failed, "reason": "pipeline execution failed" if failed else None},
                ),
            },
            status="failed" if failed else "completed",
        )

    @staticmethod
    def flag(path: Path, reason: str) -> None:
        """Flag an already captured run for asynchronous diagnostic ingestion."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a nonempty failure reason is required")
        events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ids = {event.get("run_id") for event in events}
        if len(ids) != 1 or None in ids or not any(e.get("stage") == "execution" for e in events):
            raise ValueError("flagging requires one published execution")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "timestamp": _timestamp(),
                        "run_id": ids.pop(),
                        "stage": "failure_flag",
                        "status": "completed",
                        "inputs": {"reason": reason},
                    }
                )
                + "\n"
            )

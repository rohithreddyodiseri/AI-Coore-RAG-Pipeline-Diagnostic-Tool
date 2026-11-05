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

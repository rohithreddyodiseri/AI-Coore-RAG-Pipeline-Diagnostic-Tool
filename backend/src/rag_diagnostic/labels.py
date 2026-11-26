"""Optional offline labels for captured executions; never supplied to the judge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_diagnostic.taxonomy import BY_CODE, STAGES


@dataclass(frozen=True)
class ExecutionLabel:
    """Reviewed attribution for a recorded execution, used only for scoring."""

    run_id: str
    stages: frozenset[str]
    fault_type: str
    categories: frozenset[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionLabel:
        unknown_stages = set(data.get("stages") or []) - set(STAGES)
        unknown_categories = set(data.get("failure_categories") or []) - set(BY_CODE)
        if unknown_stages or unknown_categories:
            raise ValueError("label contains unknown seven-category stages or categories")
        return cls(
            run_id=str(data["run_id"]),
            stages=frozenset(data.get("stages") or []),
            fault_type=str(data.get("fault_type") or "none"),
            categories=frozenset(data["failure_categories"])
            if "failure_categories" in data
            else None,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def load_labels(path: Path) -> dict[str, ExecutionLabel]:
    labels = [ExecutionLabel.from_dict(row) for row in _read_jsonl(path)]
    return {label.run_id: label for label in labels}

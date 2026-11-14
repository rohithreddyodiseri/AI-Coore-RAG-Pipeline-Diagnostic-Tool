"""Query-only cases and evaluation labels, kept in separate files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_diagnostic.taxonomy import BY_CODE, STAGES


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    query: str
    pipeline: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticCase:
        forbidden = {
            "reference_answer",
            "gold",
            "stages",
            "fault_type",
            "failure_categories",
        } & data.keys()
        if forbidden:
            raise ValueError(f"case contains forbidden oracle/label fields: {sorted(forbidden)}")
        unknown = data.keys() - {"case_id", "query", "pipeline"}
        if unknown:
            raise ValueError(f"unknown case fields: {sorted(unknown)}")
        if not isinstance(data["query"], str) or not data["query"].strip():
            raise ValueError("query must be a nonempty string")
        return cls(str(data["case_id"]), data["query"], dict(data.get("pipeline") or {}))


@dataclass(frozen=True)
class FaultLabel:
    """The true cause of a case, used only for scoring."""

    case_id: str
    stages: frozenset[str]
    fault_type: str
    categories: frozenset[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaultLabel:
        unknown_stages = set(data.get("stages") or []) - set(STAGES)
        unknown_categories = set(data.get("failure_categories") or []) - set(BY_CODE)
        if unknown_stages or unknown_categories:
            raise ValueError("label contains unknown seven-category stages or categories")
        return cls(
            case_id=str(data["case_id"]),
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


def load_cases(path: Path) -> list[DiagnosticCase]:
    return [DiagnosticCase.from_dict(row) for row in _read_jsonl(path)]


def load_labels(path: Path) -> dict[str, FaultLabel]:
    labels = [FaultLabel.from_dict(row) for row in _read_jsonl(path)]
    return {label.case_id: label for label in labels}

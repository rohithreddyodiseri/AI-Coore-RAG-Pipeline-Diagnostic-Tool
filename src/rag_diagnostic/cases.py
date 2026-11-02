"""Reads the cases the tool diagnoses and the labels used to score it.

Cases and labels are kept in separate files. The tool only ever sees the
cases, so it can't peek at the right answer while it's diagnosing.

Case file (one JSON object per line):

    {"case_id": "c001",
     "query": "Who built the Velra lighthouse?",
     "reference_answer": "Mara Quill",
     "gold": [{"doc_id": "harbor",
               "evidence": "The Velra lighthouse was built in 1874 by the engineer Mara Quill."}],
     "pipeline": {"chunk_size": 5}}

    "evidence" is the shortest passage that supports the answer. A whole
    section is too long, since it would always look split across chunks.
    "pipeline" holds the settings for running this case. It shouldn't name
    the fault.

Label file (one JSON object per line):

    {"case_id": "c001", "stages": ["chunking"], "fault_type": "chunk_split_evidence"}

    "stages" can list more than one stage. A case with no fault uses an empty list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldLocation:
    """Where the answer lives: the document and the passage that supports it."""

    doc_id: str
    evidence: str


@dataclass(frozen=True)
class DiagnosticCase:
    """One question for the tool to diagnose."""

    case_id: str
    query: str
    reference_answer: str
    gold: tuple[GoldLocation, ...]
    pipeline: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticCase:
        # Refuse a case that carries its own label, so the answer can't leak in.
        if "stages" in data or "fault_type" in data:
            raise ValueError(
                f"case {data.get('case_id')} contains label fields; keep labels in the label file"
            )
        return cls(
            case_id=str(data["case_id"]),
            query=str(data["query"]),
            reference_answer=str(data["reference_answer"]),
            gold=tuple(GoldLocation(str(g["doc_id"]), str(g["evidence"])) for g in data["gold"]),
            pipeline=dict(data.get("pipeline") or {}),
        )


@dataclass(frozen=True)
class FaultLabel:
    """The true cause of a case, used only for scoring."""

    case_id: str
    stages: frozenset[str]
    fault_type: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaultLabel:
        return cls(
            case_id=str(data["case_id"]),
            stages=frozenset(data.get("stages") or []),
            fault_type=str(data.get("fault_type") or "none"),
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

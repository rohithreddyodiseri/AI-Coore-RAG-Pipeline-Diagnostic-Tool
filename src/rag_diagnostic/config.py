"""Settings for the diagnostic tool itself, not the pipeline it's checking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiagnosticConfig:
    # How much of a passage has to be found for it to count as present.
    match_threshold: float = 0.9
    # How far down the results to look for the answer chunk's real rank.
    deep_k: int = 100
    # Below this confidence, the tool abstains instead of giving a verdict.
    abstain_below: float = 0.6
    # Which judge to use: "reference_recall" (no model) or "llm".
    judge: str = "reference_recall"
    judge_threshold: float = 0.8
    judge_model: str = "gpt-4o-mini"
    judge_cache: str = "runs/judge_cache.json"
    # Share of cases held back for the final score, and the seed for picking them.
    held_out_fraction: float = 0.3
    split_seed: str = "fall-2026"
    # Where run records are saved.
    runs_dir: str = "runs"

    @classmethod
    def from_file(cls, path: Path | None) -> DiagnosticConfig:
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """A short ID for these exact settings."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

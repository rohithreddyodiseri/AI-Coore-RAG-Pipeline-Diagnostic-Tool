"""Settings for the diagnostic tool itself, not the pipeline it's checking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from gemini_client import DEFAULT_MODEL


@dataclass(frozen=True)
class DiagnosticConfig:
    deep_k: int = 20
    abstain_below: float = 0.7
    judge: str = "llm"
    judge_model: str = DEFAULT_MODEL
    judge_max_output_tokens: int = 6000
    max_judge_input_chars: int = 100000
    inspect_corpus: bool = True
    corpus_budget_chars: int = 40000
    judge_cache: str = "runs/judge_cache.json"
    held_out_fraction: float = 0.3
    split_seed: str = "fall-2026"
    runs_dir: str = "runs"

    def __post_init__(self):
        for key in ("abstain_below", "held_out_fraction"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not 0 < value <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        for key in (
            "deep_k",
            "judge_max_output_tokens",
            "max_judge_input_chars",
            "corpus_budget_chars",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.inspect_corpus) is not bool:
            raise ValueError("inspect_corpus must be boolean")

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

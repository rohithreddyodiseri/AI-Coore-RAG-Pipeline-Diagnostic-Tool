"""Saves a record of every run, with the code version and settings that produced it."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_diagnostic import __version__


def code_revision(path: Path | None = None) -> dict[str, Any]:
    """The current git commit, and whether there are uncommitted changes."""
    cwd = str(path or Path(__file__).resolve().parent)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "uncommitted_changes": None}


@dataclass(frozen=True)
class RunRecord:
    tool_version: str
    code_revision: dict[str, Any]
    config_fingerprint: str
    config: dict[str, Any]
    pipeline: dict[str, Any]
    started_at: str

    @classmethod
    def capture(cls, config, adapter) -> RunRecord:
        return cls(
            tool_version=__version__,
            code_revision=code_revision(),
            config_fingerprint=config.fingerprint(),
            config=config.to_dict(),
            pipeline=adapter.describe(),
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")

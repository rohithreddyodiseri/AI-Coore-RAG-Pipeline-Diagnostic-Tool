from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    text: str
    metadata: dict[str, str | int]


class OpenRAGBenchLoader:
    """Read paper JSON files from the Open RAGBench corpus directory."""

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _section_text(section: dict[str, Any]) -> str:
        parts = [str(section.get("text") or "").strip()]
        tables = section.get("tables") or {}
        if isinstance(tables, dict):
            parts.extend(str(table).strip() for table in tables.values() if table)
        return "\n\n".join(part for part in parts if part)

    def load(self, limit_files: int | None = None) -> Iterator[Document]:
        if not self.corpus_dir.is_dir():
            raise FileNotFoundError(
                f"Corpus directory not found: {self.corpus_dir}. "
                "Run `python download-dataset.py` first or pass --corpus-dir."
            )

        paths = sorted(self.corpus_dir.glob("*.json"))
        if limit_files is not None:
            if limit_files <= 0:
                raise ValueError("limit-files must be greater than zero")
            paths = paths[:limit_files]

        for path in paths:
            paper = self._read_json(path)
            doc_id = str(paper.get("id") or path.stem)
            title = str(paper.get("title") or path.stem)
            base_metadata: dict[str, str | int] = {
                "doc_id": doc_id,
                "title": title,
                "source": str(path),
            }

            abstract = str(paper.get("abstract") or "").strip()
            if abstract:
                yield Document(
                    text=f"{title}\n\nAbstract\n{abstract}",
                    metadata={**base_metadata, "section": -1},
                )

            for section_number, section in enumerate(paper.get("sections") or []):
                if not isinstance(section, dict):
                    continue
                text = self._section_text(section)
                if text:
                    yield Document(
                        text=f"{title}\n\n{text}",
                        metadata={**base_metadata, "section": section_number},
                    )

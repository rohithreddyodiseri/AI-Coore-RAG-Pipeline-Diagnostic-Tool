from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: Path) -> Path:
    value = Path(os.getenv(name, str(default))).expanduser()
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def load_environment() -> None:
    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file, override=False)


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    chroma_dir: Path
    collection: str
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    top_k: int

    @classmethod
    def from_env(cls) -> Settings:
        load_environment()
        return cls(
            corpus_dir=_path_from_env(
                "RAG_CORPUS_DIR", PROJECT_ROOT / "data/open_ragbench/pdf/arxiv/corpus"
            ),
            chroma_dir=_path_from_env("RAG_CHROMA_DIR", PROJECT_ROOT / "data/chroma"),
            collection=os.getenv("RAG_COLLECTION", "open_ragbench"),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm_model=os.getenv("RAG_LLM_MODEL", "gpt-4o-mini"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
            batch_size=int(os.getenv("RAG_BATCH_SIZE", "64")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
        )

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk size must be greater than zero")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        if self.batch_size <= 0 or self.top_k <= 0:
            raise ValueError("batch size and top-k must be greater than zero")

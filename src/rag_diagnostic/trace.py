"""Data types for what each pipeline stage produces.

Every adapter returns these types, so the diagnosis code works the same way for any pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# The stages in pipeline order. "corpus" isn't a real stage. It's the verdict
# when the answer was never in the documents to begin with.
STAGES: tuple[str, ...] = (
    "corpus",
    "parsing",
    "chunking",
    "indexing",
    "retrieval",
    "reranking",
    "context",
    "generation",
)


def stage_order(stage: str) -> int:
    return STAGES.index(stage)


@dataclass(frozen=True)
class Chunk:
    """A piece of a document after chunking."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk returned by retrieval or reranking, with its score and rank (starting at 1)."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int


@dataclass(frozen=True)
class Context:
    """The text the model actually receives."""

    chunk_ids: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class QueryTrace:
    """Everything the pipeline produced for one question."""

    query: str
    candidates: tuple[ScoredChunk, ...]  # what retrieval returned
    ranked: tuple[ScoredChunk, ...]  # after reranking; same as candidates if there's no reranker
    reranker_used: bool
    context: Context
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rerank_positions(items: list[ScoredChunk] | tuple[ScoredChunk, ...]) -> tuple[ScoredChunk, ...]:
    """Renumber the ranks 1, 2, 3... in the order given."""
    return tuple(
        ScoredChunk(item.chunk_id, item.doc_id, item.text, item.score, position)
        for position, item in enumerate(items, start=1)
    )

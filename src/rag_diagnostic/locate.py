"""Finds where the answer passage ended up.

Text is compared word by word, ignoring case and punctuation, so small
formatting changes from parsing or chunking don't count as losing the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from rag_diagnostic.trace import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

# Ignore matches shorter than two words, so a chunk that happens to share
# one common word with the passage doesn't count as holding part of it.
MIN_BLOCK = 2


def tokens(text: str) -> list[str]:
    """Split text into lowercase words and numbers."""
    return _TOKEN.findall(text.lower())


def matched_positions(haystack: str, evidence: str) -> set[int]:
    """Which words of the passage show up, in order, in the other text."""
    needle = tokens(evidence)
    if not needle:
        return set()
    matcher = SequenceMatcher(None, tokens(haystack), needle, autojunk=False)
    positions: set[int] = set()
    for block in matcher.get_matching_blocks():
        if block.size >= MIN_BLOCK or (block.size and len(needle) < MIN_BLOCK):
            positions.update(range(block.b, block.b + block.size))
    return positions


def coverage(haystack: str | None, evidence: str) -> float:
    """How much of the passage is in the text, from 0.0 to 1.0."""
    needle = tokens(evidence)
    if not haystack or not needle:
        return 0.0
    return len(matched_positions(haystack, evidence)) / len(needle)


@dataclass(frozen=True)
class ChunkMatch:
    """How the passage is spread across a document's chunks."""

    best_chunk_id: str | None
    best_coverage: float  # the most of the passage any one chunk holds
    combined_coverage: float  # how much all the chunks hold together
    contributing_ids: tuple[str, ...]  # the chunks that hold some of it

    def whole_in_one(self, threshold: float) -> bool:
        return self.best_coverage >= threshold

    def split(self, threshold: float) -> bool:
        return not self.whole_in_one(threshold) and self.combined_coverage >= threshold


def locate_in_chunks(chunks: list[Chunk], evidence: str) -> ChunkMatch:
    """Check whether the passage sits in one chunk, is split across several, or is missing."""
    needle_length = len(tokens(evidence))
    if not chunks or needle_length == 0:
        return ChunkMatch(None, 0.0, 0.0, ())

    union: set[int] = set()
    best_id: str | None = None
    best = 0.0
    contributing: list[str] = []
    for chunk in chunks:
        positions = matched_positions(chunk.text, evidence)
        if not positions:
            continue
        contributing.append(chunk.chunk_id)
        union |= positions
        share = len(positions) / needle_length
        if share > best:
            best, best_id = share, chunk.chunk_id
    return ChunkMatch(best_id, best, len(union) / needle_length, tuple(contributing))

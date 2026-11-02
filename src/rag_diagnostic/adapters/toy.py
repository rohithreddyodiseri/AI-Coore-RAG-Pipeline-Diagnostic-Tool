"""A small practice pipeline with a switch to break each stage.

It needs no models or internet, and it gives the same result every time. The
tests use it to make sure the tool blames the right stage. It's also the
shortest example of how to write an adapter.

Switches (set in a case's "pipeline" settings):
  parse_drop          phrases removed during parsing
  chunk_size          words per chunk (default 80); small values split sentences
  chunk_overlap       words shared between neighboring chunks (default 10)
  index_drop_docs     documents left out of the index
  retrieval           "bow" (normal) or "reverse" (worst match first)
  reranker            none (default), "keep" (no change) or "invert" (flips the order)
  context_char_limit  cut the context off at this many characters
  generator           "extractive" (normal), "refuse" (always says it doesn't know)
                      or "last_chunk" (only reads the last chunk)
  raw_text_available  set to false to hide the original documents
  retrieval_k, context_k
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from rag_diagnostic.adapter import PipelineAdapter
from rag_diagnostic.locate import tokens
from rag_diagnostic.trace import Chunk, Context, ScoredChunk, rerank_positions

TOY_DOCS: dict[str, str] = {
    "harbor": (
        "Port Velra is a fictional harbor town. The Velra lighthouse was built in 1874 by the "
        "engineer Mara Quill. The lighthouse beam reaches 22 nautical miles. Fishing boats leave "
        "the harbor before dawn."
    ),
    "observatory": (
        "The Kessler observatory sits on Mount Arden. Its main telescope has a mirror 2.4 meters "
        "wide. Astronomers there discovered the comet Halvorsen in 1996."
    ),
    "orchard": (
        "The Delmont orchard grows twelve apple varieties. Its oldest tree was planted in 1902. "
        "The orchard ships fruit to three counties each autumn."
    ),
    "coast": (
        "Lighthouses along the northern coast were built in the late 1800s. Many towers used oil "
        "lamps before electric beams arrived. Keepers lived in small cottages beside the towers."
    ),
    "telescopes": (
        "Large telescopes use mirrors rather than lenses. A wider mirror gathers more light from "
        "faint objects. Observatory sites are chosen for dry air and dark skies."
    ),
    "apples": (
        "Apple trees can live for more than a century. Orchard owners prune trees each winter to "
        "improve the harvest."
    ),
}

STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or the there this to "
    "was were what when where which who why with".split()
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n\n")


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


class ToyPipeline(PipelineAdapter):
    def __init__(self, settings: dict[str, Any] | None = None, docs: dict[str, str] | None = None):
        self.settings = dict(settings or {})
        self.docs = dict(TOY_DOCS if docs is None else docs)
        s = self.settings
        self.retrieval_k = int(s.get("retrieval_k", 5))
        self.context_k = int(s.get("context_k", 2))
        self.provides_raw_text = bool(s.get("raw_text_available", True))
        self.chunk_size = int(s.get("chunk_size", 80))
        self.chunk_overlap = int(s.get("chunk_overlap", 10))
        dropped = set(s.get("index_drop_docs", []))
        self._index = {
            c.chunk_id: c
            for doc_id in sorted(self.docs)
            if doc_id not in dropped
            for c in self.chunks(doc_id)
        }

    # ---- Documents and the index ----------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {"adapter": "toy", "documents": sorted(self.docs), **self.settings}

    def raw_text(self, doc_id: str) -> str | None:
        return self.docs.get(doc_id) if self.provides_raw_text else None

    def parsed_text(self, doc_id: str) -> str | None:
        text = self.docs.get(doc_id)
        if text is None:
            return None
        for phrase in self.settings.get("parse_drop", []):
            text = text.replace(phrase, "")
        return text

    def chunks(self, doc_id: str) -> list[Chunk]:
        text = self.parsed_text(doc_id)
        if text is None:
            return []
        words = text.split()
        step = max(1, self.chunk_size - self.chunk_overlap)
        out = []
        for number, start in enumerate(range(0, len(words), step)):
            piece = " ".join(words[start : start + self.chunk_size])
            out.append(Chunk(f"{doc_id}:{number}", doc_id, piece, {"chunk": number}))
            if start + self.chunk_size >= len(words):
                break
        return out

    def indexed_ids(self, chunk_ids: list[str]) -> set[str]:
        return {i for i in chunk_ids if i in self._index}

    # ---- Stages that run for each question -------------------------------------

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        q = Counter(content_tokens(query))
        scored = [(_cosine(q, Counter(content_tokens(c.text))), c) for c in self._index.values()]
        if self.settings.get("retrieval", "bow") == "reverse":
            scored.sort(key=lambda pair: (pair[0], pair[1].chunk_id))
        else:
            scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return [
            ScoredChunk(c.chunk_id, c.doc_id, c.text, round(score, 6), rank)
            for rank, (score, c) in enumerate(scored[:k], start=1)
        ]

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk] | None:
        mode = self.settings.get("reranker")
        if mode is None:
            return None
        if mode == "invert":
            return list(rerank_positions(list(reversed(candidates))))
        return list(rerank_positions(candidates))

    def assemble_context(self, query: str, ranked: list[ScoredChunk]) -> Context:
        top = list(ranked)[: self.context_k]
        text = "\n\n".join(f"[{n}] {c.text}" for n, c in enumerate(top, start=1))
        limit = self.settings.get("context_char_limit")
        if limit:
            text = text[: int(limit)]
        return Context(tuple(c.chunk_id for c in top), text)

    def generate(self, query: str, context: Context) -> str:
        mode = self.settings.get("generator", "extractive")
        if mode == "refuse":
            return "I do not know."
        text = context.text
        if mode == "last_chunk":
            text = re.split(r"\n\n", text)[-1]
        wanted = set(content_tokens(query))
        best, best_score = "", 0
        for sentence in _SENTENCE.split(text):
            sentence = re.sub(r"^\[\d+\]\s*", "", sentence.strip())
            score = len(wanted & set(content_tokens(sentence)))
            if score > best_score:
                best, best_score = sentence, score
        return best or "I do not know."


def build(pipeline_settings: dict[str, Any]) -> ToyPipeline:
    """Used by the command line: --adapter rag_diagnostic.adapters.toy:build"""
    return ToyPipeline(pipeline_settings)

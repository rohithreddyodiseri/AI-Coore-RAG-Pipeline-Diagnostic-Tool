"""The named failure modes, with the stage, what it looks like, and how to fix it.

This is a starting list. Add new modes as testing turns up failures it doesn't cover.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureMode:
    code: str
    stage: str
    name: str
    signature: str
    remedy: str


_MODES = [
    FailureMode(
        "answer_not_in_corpus",
        "corpus",
        "Answer not in the documents",
        "The evidence is in neither the raw nor the parsed documents.",
        "Add the missing source document, or treat the question as out of scope.",
    ),
    FailureMode(
        "parse_dropped_evidence",
        "parsing",
        "Lost during parsing",
        "The evidence is in the raw document but missing from the parsed text.",
        "Fix the parser for this format (tables, multi-column layout, headers, encoding).",
    ),
    FailureMode(
        "chunk_split_evidence",
        "chunking",
        "Split across chunks",
        "No single chunk holds the evidence, but adjacent chunks hold it together.",
        "Increase chunk overlap or split on sentence or section boundaries.",
    ),
    FailureMode(
        "chunk_dropped_evidence",
        "chunking",
        "Dropped during chunking",
        "The parsed text holds the evidence but no chunk does.",
        "Check the splitter for text it discards (very short sections, size filters).",
    ),
    FailureMode(
        "index_missing_chunk",
        "indexing",
        "Missing from index",
        "The chunk holding the evidence exists but is not in the index.",
        "Rebuild the index and check for failed or skipped batches.",
    ),
    FailureMode(
        "retrieval_low_rank",
        "retrieval",
        "Ranked too low by retrieval",
        "The evidence chunk is indexed but ranks below the retrieval cutoff.",
        "Try another embedding model, hybrid (keyword + vector) search, query rewriting, "
        "or a larger top-k.",
    ),
    FailureMode(
        "rerank_buried",
        "reranking",
        "Buried by reranker",
        "The evidence chunk was retrieved but the reranker pushed it below the context cutoff.",
        "Try another reranker model, or keep the top retrieval results regardless of rerank.",
    ),
    FailureMode(
        "context_truncated",
        "context",
        "Cut from context",
        "The evidence chunk was ranked into the context but its text did not reach the model.",
        "Raise the context length limit or put higher-ranked chunks first.",
    ),
    FailureMode(
        "generation_wrong_with_evidence",
        "generation",
        "Wrong answer despite evidence",
        "The model answers wrongly even when given only the evidence passage.",
        "Change the prompt or the generation model.",
    ),
    FailureMode(
        "generation_distracted",
        "generation",
        "Distracted by other context",
        "The model answers correctly from the evidence alone but not from the full context.",
        "Send fewer chunks, or tell the model to prefer the most specific passage.",
    ),
]

MODES: dict[str, FailureMode] = {mode.code: mode for mode in _MODES}


def mode(code: str) -> FailureMode:
    """Look up a failure mode by its code."""
    return MODES[code]

"""Stage checks that follow the answer passage through the pipeline.

Each check looks at one stage and asks whether the passage made it through.
The checks run in order and share what they learn along the way.

A failed check only makes a stage a suspect. The fix-and-rerun step in
diagnose.py is what confirms it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_diagnostic.adapter import PipelineAdapter
from rag_diagnostic.cases import DiagnosticCase
from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.locate import coverage, locate_in_chunks
from rag_diagnostic.trace import Chunk, QueryTrace

OK, FAULT, UNKNOWN, NOT_APPLICABLE = "ok", "fault", "unknown", "n/a"


@dataclass(frozen=True)
class Finding:
    """The result of one check."""

    stage: str
    status: str  # ok, fault, unknown or n/a
    summary: str
    code: str | None = None  # the failure mode, when the status is fault
    evidence: dict[str, Any] = field(default_factory=dict)  # the numbers behind the result


@dataclass
class Investigation:
    """What the checks share as they go."""

    case: DiagnosticCase
    adapter: PipelineAdapter
    config: DiagnosticConfig
    trace: QueryTrace
    # The checks fill these in, one entry per answer location.
    parsed_ok: list[bool] = field(default_factory=list)
    gold_chunks: list[list[Chunk]] = field(default_factory=list)
    indexed: list[set[str]] = field(default_factory=list)

    @property
    def threshold(self) -> float:
        return self.config.match_threshold

    def gold_ids(self) -> set[str]:
        return {chunk.chunk_id for chunks in self.gold_chunks for chunk in chunks}

    def indexed_gold_ids(self) -> set[str]:
        return set().union(*self.indexed) if self.indexed else set()

    def context_cutoff(self) -> int:
        """The rank a chunk needs to make it into the context."""
        if self.trace.reranker_used:
            return self.adapter.retrieval_k
        return min(self.adapter.retrieval_k, self.adapter.context_k)


def _worst(findings: list[Finding], stage: str, ok_summary: str) -> Finding:
    """Report the worst result when there's more than one answer location."""
    for status in (FAULT, UNKNOWN):
        for finding in findings:
            if finding.status == status:
                return finding
    if findings and all(f.status == NOT_APPLICABLE for f in findings):
        return findings[0]
    return Finding(stage, OK, ok_summary)


def check_parsing(inv: Investigation) -> Finding:
    """Is the passage in the parsed text? If not, was it ever in the document?"""
    results = []
    for gold in inv.case.gold:
        parsed = inv.adapter.parsed_text(gold.doc_id)
        parsed_share = coverage(parsed, gold.evidence)
        if parsed_share >= inv.threshold:
            inv.parsed_ok.append(True)
            results.append(Finding("parsing", OK, "evidence present in parsed text"))
            continue

        inv.parsed_ok.append(False)
        raw = inv.adapter.raw_text(gold.doc_id)
        raw_share = coverage(raw, gold.evidence)
        detail = {
            "doc_id": gold.doc_id,
            "parsed_coverage": round(parsed_share, 3),
            "raw_coverage": round(raw_share, 3),
            "document_parsed": parsed is not None,
        }
        if raw is not None and raw_share >= inv.threshold:
            results.append(
                Finding(
                    "parsing",
                    FAULT,
                    f"evidence is in the raw document but only {parsed_share:.0%} of it "
                    "survived parsing",
                    "parse_dropped_evidence",
                    detail,
                )
            )
        elif raw is not None or inv.adapter.provides_raw_text:
            results.append(
                Finding(
                    "corpus",
                    FAULT,
                    "evidence is not in the raw or parsed documents",
                    "answer_not_in_corpus",
                    detail,
                )
            )
        else:
            results.append(
                Finding(
                    "parsing",
                    UNKNOWN,
                    "evidence missing from parsed text and raw text is unavailable, so "
                    "parsing loss and absence from the documents cannot be told apart",
                    None,
                    detail,
                )
            )
    return _worst(results, "parsing", "evidence present in parsed text")


def check_chunking(inv: Investigation) -> Finding:
    """Does one chunk hold the whole passage, or did chunking split it up?"""
    results = []
    for gold, parsed_ok in zip(inv.case.gold, inv.parsed_ok, strict=True):
        if not parsed_ok:
            inv.gold_chunks.append([])
            results.append(Finding("chunking", NOT_APPLICABLE, "evidence lost before chunking"))
            continue
        chunks = inv.adapter.chunks(gold.doc_id)
        match = locate_in_chunks(chunks, gold.evidence)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        detail = {
            "doc_id": gold.doc_id,
            "chunks_in_document": len(chunks),
            "best_single_chunk": match.best_chunk_id,
            "best_single_coverage": round(match.best_coverage, 3),
            "combined_coverage": round(match.combined_coverage, 3),
            "contributing_chunks": list(match.contributing_ids),
        }
        if match.whole_in_one(inv.threshold):
            holders = [c for c in chunks if coverage(c.text, gold.evidence) >= inv.threshold]
            inv.gold_chunks.append(holders)
            results.append(Finding("chunking", OK, "evidence is whole inside one chunk"))
        elif match.split(inv.threshold):
            inv.gold_chunks.append([by_id[i] for i in match.contributing_ids])
            results.append(
                Finding(
                    "chunking",
                    FAULT,
                    f"evidence is split across {len(match.contributing_ids)} chunks; the best "
                    f"single chunk holds {match.best_coverage:.0%} of it",
                    "chunk_split_evidence",
                    detail,
                )
            )
        else:
            inv.gold_chunks.append([by_id[i] for i in match.contributing_ids])
            results.append(
                Finding(
                    "chunking",
                    FAULT,
                    f"parsed text holds the evidence but the chunks hold only "
                    f"{match.combined_coverage:.0%} of it",
                    "chunk_dropped_evidence",
                    detail,
                )
            )
    return _worst(results, "chunking", "evidence is whole inside one chunk")


def check_indexing(inv: Investigation) -> Finding:
    """Are the chunks that hold the passage in the index?"""
    results = []
    for gold, chunks in zip(inv.case.gold, inv.gold_chunks, strict=True):
        ids = [chunk.chunk_id for chunk in chunks]
        if not ids:
            inv.indexed.append(set())
            results.append(Finding("indexing", NOT_APPLICABLE, "no chunk holds the evidence"))
            continue
        present = inv.adapter.indexed_ids(ids)
        inv.indexed.append(present)
        missing = sorted(set(ids) - present)
        detail = {"doc_id": gold.doc_id, "gold_chunks": ids, "missing_from_index": missing}
        if not present:
            results.append(
                Finding(
                    "indexing",
                    FAULT,
                    "none of the chunks holding the evidence are in the index",
                    "index_missing_chunk",
                    detail,
                )
            )
        else:
            results.append(Finding("indexing", OK, "evidence chunks are indexed", None, detail))
    return _worst(results, "indexing", "evidence chunks are indexed")


def check_retrieval(inv: Investigation) -> Finding:
    """Did retrieval rank the answer chunk high enough? If not, where did it land?"""
    targets = inv.indexed_gold_ids()
    if not targets:
        return Finding("retrieval", NOT_APPLICABLE, "no indexed chunk holds the evidence")

    cutoff = inv.context_cutoff()
    in_run = {c.chunk_id: c for c in inv.trace.candidates}
    best_run_rank = min((in_run[i].rank for i in targets if i in in_run), default=None)

    deep_k = max(inv.config.deep_k, inv.adapter.retrieval_k)
    deep = inv.adapter.retrieve(inv.case.query, deep_k)
    deep_hits = [c for c in deep if c.chunk_id in targets]
    detail: dict[str, Any] = {
        "cutoff_rank": cutoff,
        "gold_rank": deep_hits[0].rank if deep_hits else f">{deep_k}",
        "gold_score": round(deep_hits[0].score, 4) if deep_hits else None,
        "score_at_cutoff": (
            round(inv.trace.candidates[cutoff - 1].score, 4)
            if len(inv.trace.candidates) >= cutoff
            else None
        ),
    }
    if best_run_rank is not None and best_run_rank <= cutoff:
        return Finding(
            "retrieval", OK, f"evidence chunk retrieved at rank {best_run_rank}", None, detail
        )
    return Finding(
        "retrieval",
        FAULT,
        f"evidence chunk ranks {detail['gold_rank']}, below the cutoff of {cutoff}",
        "retrieval_low_rank",
        detail,
    )


def check_reranking(inv: Investigation) -> Finding:
    """Did the reranker push the answer chunk out of the context?"""
    if not inv.trace.reranker_used:
        return Finding("reranking", NOT_APPLICABLE, "pipeline has no reranker")
    targets = inv.indexed_gold_ids()
    before = {c.chunk_id: c.rank for c in inv.trace.candidates}
    after = {c.chunk_id: c.rank for c in inv.trace.ranked}
    hits = [i for i in targets if i in before]
    if not hits:
        return Finding("reranking", NOT_APPLICABLE, "evidence chunk was not retrieved")
    best_before = min(before[i] for i in hits)
    best_after = min((after[i] for i in hits if i in after), default=None)
    detail = {
        "rank_before_rerank": best_before,
        "rank_after_rerank": best_after,
        "context_k": inv.adapter.context_k,
    }
    if best_after is not None and best_after <= inv.adapter.context_k:
        return Finding(
            "reranking", OK, f"evidence chunk at rank {best_after} after reranking", None, detail
        )
    return Finding(
        "reranking",
        FAULT,
        f"evidence chunk moved from rank {best_before} to {best_after or 'dropped'} after "
        f"reranking; only the top {inv.adapter.context_k} reach the context",
        "rerank_buried",
        detail,
    )


def check_context(inv: Investigation) -> Finding:
    """The answer chunk made the cut. Did its text actually reach the model?"""
    targets = inv.indexed_gold_ids()
    top = [c for c in inv.trace.ranked[: inv.adapter.context_k] if c.chunk_id in targets]
    if not top:
        return Finding("context", NOT_APPLICABLE, "evidence chunk was not ranked into the context")
    shares = {c.chunk_id: coverage(inv.trace.context.text, c.text) for c in top}
    detail = {"chunk_coverage_in_context": {k: round(v, 3) for k, v in shares.items()}}
    if max(shares.values()) >= inv.threshold:
        return Finding("context", OK, "evidence chunk text reached the model", None, detail)
    return Finding(
        "context",
        FAULT,
        f"evidence chunk was ranked into the context but only {max(shares.values()):.0%} of "
        "its text reached the model",
        "context_truncated",
        detail,
    )


# Generation has no check here. diagnose.py tests it by giving the model only the passage.
CHECKS = (
    check_parsing,
    check_chunking,
    check_indexing,
    check_retrieval,
    check_reranking,
    check_context,
)


def run_checks(inv: Investigation) -> list[Finding]:
    return [check(inv) for check in CHECKS]

"""Documented boundary between the attribution engine and any target RAG system.

Implement this interface by calling or instrumenting the target's real execution.
Do not reproduce its stage logic inside the adapter. See docs/diagnostic-tool.md.
"""

from abc import ABC, abstractmethod
from typing import Any

from rag_diagnostic.trace import QueryTrace, ScoredChunk


class PipelineAdapter(ABC):
    retrieval_k: int = 5
    context_k: int = 5

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Resolved non-secret settings, model/prompt identity and target code revision."""

    @abstractmethod
    def run(self, query: str) -> dict[str, Any] | QueryTrace:
        """Run the actual target once, returning exact stage inputs/outputs and response.

        May instead return a captured execution of this query. Omit unobserved
        stages and explicitly mark unused stages; never invent missing artifacts.
        """

    @abstractmethod
    def corpus_snapshot(self) -> dict[str, Any]:
        """Corpus documents, immutable snapshot_id, scope and truthful completeness.

        Return {documents: [...], snapshot_id: ..., scope: ..., complete: bool}.
        An inaccessible corpus must be explicit: documents=[], complete=False.
        """

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Optional deeper inspection search, kept separate from the original execution."""
        raise NotImplementedError("the target does not expose deeper retrieval")

    def generation_instructions(self) -> str | None:
        """Exact generation system instructions if not already in the trace."""
        return None

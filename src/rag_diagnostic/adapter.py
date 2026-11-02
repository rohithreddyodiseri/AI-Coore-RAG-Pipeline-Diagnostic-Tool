"""The interface a pipeline fills in so the tool can diagnose it.

To connect a new pipeline, subclass PipelineAdapter and fill in each method.
The tool uses these methods to see what the pipeline did, and to rerun the
later stages after fixing an earlier one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_diagnostic.trace import Chunk, Context, QueryTrace, ScoredChunk


class PipelineAdapter(ABC):
    # How many chunks retrieval returns.
    retrieval_k: int = 5
    # How many of the ranked chunks go into the context.
    context_k: int = 5
    # Set this to False if the adapter can't read the original documents.
    # Then a missing raw document means "unknown" instead of "doesn't exist".
    provides_raw_text: bool = True

    # ---- Documents and the index ----------------------------------------------

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """The pipeline's settings. These are saved with every run."""

    @abstractmethod
    def raw_text(self, doc_id: str) -> str | None:
        """The original document text before parsing, or None if it can't be found.

        Without raw text, the tool can't tell whether the answer was lost in
        parsing or was never there, so it abstains on those cases.
        """

    @abstractmethod
    def parsed_text(self, doc_id: str) -> str | None:
        """The document text after parsing, or None if it wasn't parsed."""

    @abstractmethod
    def chunks(self, doc_id: str) -> list[Chunk]:
        """The chunks made from one document, in order."""

    @abstractmethod
    def indexed_ids(self, chunk_ids: list[str]) -> set[str]:
        """Which of these chunks are in the index."""

    # ---- Stages that run for each question -------------------------------------

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """The top k chunks for the question, best first."""

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk] | None:
        """The candidates in a new order, or None if there's no reranker."""
        return None

    @abstractmethod
    def assemble_context(self, query: str, ranked: list[ScoredChunk]) -> Context:
        """Build the context the model will see."""

    @abstractmethod
    def generate(self, query: str, context: Context) -> str:
        """Answer the question from the context. Use temperature 0 so results repeat."""

    # ---- Full run ---------------------------------------------------------------

    def run(self, query: str) -> QueryTrace:
        """Run every stage in order and keep what each one produced.

        Override this if your pipeline runs differently, for example if the
        outputs come from saved logs.
        """
        candidates = self.retrieve(query, self.retrieval_k)
        reranked = self.rerank(query, candidates)
        ranked = reranked if reranked is not None else candidates
        context = self.assemble_context(query, ranked)
        answer = self.generate(query, context)
        return QueryTrace(
            query=query,
            candidates=tuple(candidates),
            ranked=tuple(ranked),
            reranker_used=reranked is not None,
            context=context,
            answer=answer,
        )

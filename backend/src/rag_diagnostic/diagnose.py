"""Capture a pipeline's observed trace and hand semantic attribution to the LLM."""

from __future__ import annotations

from dataclasses import asdict

from rag_diagnostic.adapter import PipelineAdapter
from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnostic_engine import DiagnosticEngine, Verdict  # noqa: F401
from rag_diagnostic.judge import Judge
from rag_diagnostic.runlog import RunRecord


class Diagnoser:
    def __init__(self, config: DiagnosticConfig, judge: Judge | None = None):
        self.config = config
        self.engine = DiagnosticEngine(config, judge)
        self.judge = self.engine.judge

    def diagnose(self, query: str, adapter: PipelineAdapter) -> Verdict:
        run_id = None
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        run = RunRecord.capture(self.config, adapter)
        try:
            observed = adapter.run(query)
            trace = observed.to_dict() if hasattr(observed, "to_dict") else dict(observed)
            if trace.get("query") != query:
                raise ValueError("adapter returned a trace for a different query")
            trace.setdefault("retrieval_k", adapter.retrieval_k)
            trace.setdefault("reranking_k", adapter.context_k)
            instructions = adapter.generation_instructions()
            if instructions is not None and "generation_instructions" not in trace:
                trace["generation_instructions"] = instructions
            # Deep search is a separate inspection probe, never the original top-K.
            try:
                trace["deep_retrieval"] = [
                    asdict(c) for c in adapter.retrieve(query, self.config.deep_k)
                ]
                trace["deep_k"] = self.config.deep_k
            except Exception as error:
                trace.setdefault("inspection_errors", []).append(
                    f"Deep search unavailable: {error}"
                )
            if self.config.inspect_corpus:
                try:
                    trace["corpus"] = adapter.corpus_snapshot()
                except Exception as error:
                    trace.setdefault("inspection_errors", []).append(
                        f"Corpus inspection unavailable: {error}"
                    )
        except Exception as error:
            self.engine.record_error({"query": query}, run, error, run_id)
            raise
        return self.engine.diagnose(trace, run_id, adapter.describe(), run)

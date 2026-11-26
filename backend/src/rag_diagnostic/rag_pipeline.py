"""Runtime fault hooks for the diagnostic tool's reference RAG pipeline."""

from __future__ import annotations

from dataclasses import replace

from rag_diagnostic.reference_pipeline import RAGPipeline, SearchResult


class BenchmarkRAGPipeline(RAGPipeline):
    def __init__(self, settings, plan=None, *, deep_k=20):
        super().__init__(settings)
        self.plan = plan or {}
        self.failure = self.plan.get("failure_type")
        self.deep_k = max(deep_k, settings.top_k + 1)
        self.original_candidates = []
        self.candidates = []
        self.original_response = None
        self.injection_applied = False
        self.injection_reason = "No runtime fault configured"
        self.generation_instructions = self.plan.get("instruction", self.generation_instructions)

    def _target(self, result: SearchResult) -> bool:
        target = self.plan["target_location"]
        return (
            result.metadata.get("doc_id") == target["doc_id"]
            and result.metadata.get("section") == target["section_index"]
        )

    def retrieve(self, query):
        self.original_candidates = self.search(query, self.deep_k)
        ranked = self.original_candidates
        k = self.settings.top_k
        if self.failure == "missed_top_ranked":
            targets = [row for row in ranked[:k] if self._target(row)]
            others = [row for row in ranked if not self._target(row)]
            if targets and len(others) >= k:
                ranked = others[:k] + [row for row in ranked if row not in others[:k]]
                self.injection_applied = True
                self.injection_reason = "Relevant naturally retrieved chunks demoted below top-k"
            else:
                self.injection_reason = "Target absent from natural top-k or too few alternatives"
        self.candidates = [replace(row, rank=i) for i, row in enumerate(ranked, 1)]
        return self.candidates[:k]

    def context_sources(self, results):
        if self.failure == "not_in_context":
            evidence = self.plan["evidence_text"]
            doc_id = self.plan["target_location"]["doc_id"]
            evidence_ids = {
                row.chunk_id
                for row in results
                if row.metadata.get("doc_id") == doc_id and self._contains(evidence, row.text)
            }
            if not evidence_ids:
                self.injection_reason = "Target evidence was not retrieved in natural top-k"
                return results
            retained = [
                row for row in results
                if not self._target(row) and row.chunk_id not in evidence_ids
            ]
            if any(self._contains(evidence, row.text) for row in retained):
                self.injection_reason = "Target evidence would remain in context after removal"
                return results
            self.injection_applied = True
            self.injection_reason = (
                "Removed naturally retrieved target evidence from generation context"
            )
            return retained
        return results

    @staticmethod
    def _contains(evidence, text):
        return " ".join(evidence.split()) in " ".join(text.split())

    def generate(self, query, context):
        self.original_response = super().generate(query, context)
        if "response" not in self.plan:
            return self.original_response
        if self.failure in {"not_extracted", "incomplete_answer"}:
            present = self._contains(self.plan["evidence_text"], context)
            if not present:
                self.injection_reason = "Required fact absent from natural generation context"
                return self.original_response
        self.injection_applied = True
        self.injection_reason = "Replaced response at generation output boundary"
        return self.plan["response"]

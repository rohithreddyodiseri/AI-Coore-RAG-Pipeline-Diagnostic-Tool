"""Seven-category LLM diagnosis of immutable trace artifacts, without gold answers.

Semantic decisions belong to the LLM. Python validates the response schema,
provenance, boundary observability, confidence, and summary consistency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.judge import Judge, build_judge
from rag_diagnostic.runlog import RunRecord, append_jsonl
from rag_diagnostic.taxonomy import BY_CODE, CATEGORIES, PAPER_URL, STAGES

FORBIDDEN = {
    "reference_answer",
    "gold",
    "expected_answer",
    "gold_answer",
    "fault_type",
    "failure_category",
    "failure_categories",
}


def _reject_oracles(value):
    if isinstance(value, dict):
        if FORBIDDEN & value.keys():
            raise ValueError("Trace inputs must not contain reference answers or fault labels")
        for nested in value.values():
            _reject_oracles(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_oracles(nested)


def _number(value):
    return type(value) in (int, float) and 0 <= value <= 1


def trace_artifacts(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize native QueryTrace dictionaries and the documented output_data format.

    Each citation names an artifact and a verbatim quote from that artifact's text.
    Missing stages stay missing; explicit empty outputs receive manifest artifacts.
    """
    _reject_oracles(trace)
    if (
        not isinstance(trace, dict)
        or not isinstance(trace.get("query"), str)
        or not trace["query"].strip()
    ):
        raise ValueError("trace.query must be a nonempty string")
    artifacts = {}

    def add(key, stage, kind, text, **metadata):
        if not isinstance(text, str):
            raise ValueError(f"{key} must contain text")
        artifacts[key] = {"stage": stage, "kind": kind, "text": text, **metadata}

    def manifest(key, stage, **data):
        add(key, stage, "manifest", json.dumps(data, sort_keys=True, allow_nan=False), **data)

    def chunks(prefix, stage, output, cutoff=None):
        if not isinstance(output, (list, tuple)):
            raise ValueError(f"{prefix} output_data must be an array")
        ids = []
        for i, item in enumerate(output):
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"{prefix} chunks require a text string")
            chunk_id = str(item.get("chunk_id", item.get("id", f"{prefix}:{i}")))
            rank = item.get("rank", i + 1)
            if type(rank) is not int or rank < 1:
                raise ValueError("chunk ranks must be positive integers")
            ids.append(chunk_id)
            add(
                f"{prefix}.chunk.{i}",
                stage,
                "chunk",
                item["text"],
                chunk_id=chunk_id,
                doc_id=str(item.get("doc_id", "")),
                rank=rank,
                score=item.get("score"),
            )
        if cutoff is not None and (type(cutoff) is not int or cutoff < 1):
            raise ValueError("pipeline cutoffs must be positive integers")
        manifest(f"{prefix}.manifest", stage, chunk_ids=ids, count=len(ids), cutoff=cutoff)

    add("query", "generation", "query", trace["query"])
    stages = trace.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("stages must be an object")
    for name, output_key in (("retrieval", "candidates"), ("reranking", "ranked")):
        stage = stages.get(name, {})
        if not isinstance(stage, dict):
            raise ValueError(f"stages.{name} must be an object")
        if "output_data" in stage:
            chunks(name, name, stage["output_data"], stage.get("k"))
        elif output_key in trace and (name != "reranking" or trace.get("reranker_used", True)):
            chunks(name, name, trace[output_key], trace.get(f"{name}_k"))
    context_stage = stages.get("context_assembly", {})
    if not isinstance(context_stage, dict):
        raise ValueError("stages.context_assembly must be an object")
    context = context_stage.get("output_data", trace.get("context"))
    if context is not None:
        text = context.get("text") if isinstance(context, dict) else context
        ids = context.get("chunk_ids") if isinstance(context, dict) else None
        add("context.text", "context_assembly", "context", text)
        manifest("context.manifest", "context_assembly", characters=len(text), chunk_ids=ids)
    answer_key = "final_answer" if "final_answer" in trace else "answer"
    if answer_key in trace:
        add("generation.answer", "generation", "answer", trace[answer_key])
        manifest("generation.manifest", "generation", characters=len(trace[answer_key]))
    instructions = trace.get("generation_instructions")
    if instructions is not None:
        add("generation.instructions", "generation", "instructions", instructions)
    if "deep_retrieval" in trace:
        chunks("deep", "retrieval", trace["deep_retrieval"], trace.get("deep_k"))
    if "corpus" in trace:
        corpus = trace["corpus"]
        if not isinstance(corpus, dict) or not isinstance(corpus.get("documents"), list):
            raise ValueError("corpus requires a documents array")
        complete = corpus.get("complete", False)
        if type(complete) is not bool:
            raise ValueError("corpus.complete must be boolean")
        if complete and any(
            not isinstance(corpus.get(key), str) or not corpus[key].strip()
            for key in ("snapshot_id", "scope")
        ):
            raise ValueError("a complete corpus requires snapshot_id and scope")
        doc_ids = []
        for i, doc in enumerate(corpus["documents"]):
            if (
                not isinstance(doc, dict)
                or not isinstance(doc.get("doc_id"), str)
                or not doc["doc_id"].strip()
            ):
                raise ValueError("corpus documents require a doc_id string")
            doc_ids.append(doc["doc_id"])
            add(f"corpus.document.{i}", "corpus", "document", doc.get("text"), doc_id=doc["doc_id"])
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("duplicate corpus document IDs")
        manifest(
            "corpus.manifest",
            "corpus",
            complete=complete,
            scope=corpus.get("scope"),
            snapshot_id=corpus.get("snapshot_id"),
            document_ids=doc_ids,
            total_documents=corpus.get("total_documents", len(doc_ids)),
            selection=corpus.get("selection", "all supplied documents"),
            truncated_document_ids=corpus.get("truncated_document_ids", []),
        )
    for name in ("retrieval", "reranking", "context_assembly", "generation"):
        stage = stages.get(name, {})
        if "input_data" in stage:
            add(
                f"{name}.input",
                name,
                "stage_input",
                json.dumps(stage["input_data"], sort_keys=True, allow_nan=False),
            )
        if stage.get("status") == "not_used":
            manifest(f"{name}.status", name, status="not_used")
    generation_input = stages.get("generation", {}).get("input_data", {})
    if isinstance(generation_input, dict) and "prompt" in generation_input:
        add("generation.prompt", "generation", "prompt", generation_input["prompt"])
    if trace.get("inspection_errors"):
        add(
            "inspection.errors",
            "retrieval",
            "inspection_note",
            json.dumps(trace["inspection_errors"], sort_keys=True),
        )
    return artifacts


@dataclass
class Verdict:
    run_id: str
    query: str
    status: str
    primary_fault_stage: str | None
    failure_category: str
    confidence: float
    contributing_stages: tuple[str, ...]
    stages: tuple[str, ...]
    failure_categories: tuple[str, ...]
    evidence: list[dict]
    evaluations: list[dict]
    unresolved_categories: tuple[str, ...]
    attribution_complete: bool
    answer: str
    explanation: list[str]
    suspects: tuple[str, ...] = ()
    observations: dict[str, Any] = field(default_factory=dict)
    run: RunRecord | None = None

    @property
    def case_id(self):
        """Compatibility for offline evaluation code; runtime reports use run_id."""
        return self.run_id

    @property
    def attributions(self):
        return [evaluation for evaluation in self.evaluations if evaluation["status"] == "fault"]

    def to_dict(self, include_run=True):
        result = json.loads(json.dumps(asdict(self), allow_nan=False))
        result["attributions"] = self.attributions
        if not include_run:
            result.pop("run")
        return result

    def render(self):
        lines = [
            f"Query: {self.query}",
            f"Run: {self.run_id}",
            f"Answer: {self.answer}",
            f"Verdict: {self.status}; {self.failure_category}; "
            f"stage: {self.primary_fault_stage or 'unknown'}; confidence: {self.confidence:.2f}",
            "Confidence is an uncalibrated evidence score.",
            *self.explanation,
        ]
        if self.contributing_stages:
            lines.append("Contributing stages: " + ", ".join(self.contributing_stages))
        for row in self.evaluations:
            lines.append(
                f"  {row['category']} {BY_CODE[row['category']].name}: "
                f"{row['status']} ({row['confidence']:.2f}) - {row['rationale']}"
            )
            for citation in row["evidence"]:
                lines.append(f"    [{citation['artifact_id']}] {citation['quote']}")
            for contributor in row["contributing_stages"]:
                lines.append(f"    Contributor {contributor['stage']}: {contributor['rationale']}")
                for citation in contributor["evidence"]:
                    lines.append(f"      [{citation['artifact_id']}] {citation['quote']}")
        if self.unresolved_categories:
            lines.append("Unresolved: " + ", ".join(self.unresolved_categories))
        return "\n".join(lines)


def _unknown(code, reason):
    return {
        "category": code,
        "status": "unknown",
        "stage": None,
        "confidence": 0.0,
        "rationale": reason,
        "evidence": [],
        "contributing_stages": [],
    }


def _citations(evidence, artifacts):
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("non-unknown evaluations require artifact citations")
    for citation in evidence:
        if not isinstance(citation, dict) or set(citation) != {"artifact_id", "quote"}:
            raise ValueError("invalid citation schema")
        artifact_id, quote = citation["artifact_id"], citation["quote"]
        if (
            not isinstance(artifact_id, str)
            or artifact_id not in artifacts
            or not isinstance(quote, str)
            or not quote.strip()
            or quote not in artifacts[artifact_id]["text"]
        ):
            raise ValueError("citation must reference a nonempty verbatim artifact quote")
    return {citation["artifact_id"] for citation in evidence}


def _boundaries(row, refs, artifacts):
    """Reject unsupported premises; do not decide semantic correctness with strings."""
    code, stage = row["category"], row["stage"]
    if stage not in BY_CODE[code].stages:
        raise ValueError("fault stage is incompatible with the category")

    def cited(prefix):
        return any(ref.startswith(prefix) for ref in refs)

    if code == "FP1":
        if (
            not artifacts.get("corpus.manifest", {}).get("complete")
            or "corpus.manifest" not in refs
        ):
            raise ValueError("FP1 requires a cited complete scoped corpus snapshot")
        if "query" not in refs:
            raise ValueError("FP1 requires a query citation")
    elif code == "FP2":
        if stage == "retrieval":
            if "retrieval.manifest" not in refs:
                raise ValueError("FP2 requires an observed retrieval cutoff/selection")
            cutoff = artifacts["retrieval.manifest"]["cutoff"]
            selected = artifacts["retrieval.manifest"]["chunk_ids"]
            broader = [
                artifacts[ref]
                for ref in refs
                if ref.startswith("deep.chunk.")
                and cutoff is not None
                and artifacts[ref]["rank"] > cutoff
                and artifacts[ref]["chunk_id"] not in selected
            ]
            if not broader:
                raise ValueError(
                    "FP2 requires an observed unselected lower-ranked source from deeper retrieval"
                )
        elif not (cited("retrieval.") and cited("reranking.")):
            raise ValueError("reranking fault requires both observed boundaries")
    elif code == "FP3":
        if not (cited("retrieval.chunk.") and cited("context.")):
            raise ValueError("FP3 requires retrieved content and the actual context boundary")
        if stage == "reranking" and not cited("reranking."):
            raise ValueError("reranking attribution requires observed reranker evidence")
    elif code in ("FP4", "FP7"):
        if code == "FP7" and "query" not in refs:
            raise ValueError("FP7 requires cited requested parts from the query")
        if "context.text" not in refs or not (
            cited("generation.answer") or cited("generation.manifest")
        ):
            raise ValueError("FP4/FP7 require context facts and the observed generated answer")
    elif code in ("FP5", "FP6"):
        if not (
            ("query" in refs or "generation.instructions" in refs)
            and (cited("generation.answer") or cited("generation.manifest"))
        ):
            raise ValueError("FP5/FP6 require request constraints and the generated answer")


def validate_assessment(raw, artifacts, threshold):
    """Strictly validate all seven rows, then downgrade unsupported claims to unknown."""
    if not isinstance(raw, dict) or set(raw) != {
        "failure_category",
        "primary_fault_stage",
        "confidence",
        "reason",
        "evaluations",
    }:
        raise ValueError("judge response has an invalid top-level schema")
    if (
        raw["failure_category"] not in (*BY_CODE, "NONE", "ABSTAIN")
        or raw["primary_fault_stage"] not in (*STAGES, None)
        or not _number(raw["confidence"])
        or not isinstance(raw["reason"], str)
        or not raw["reason"].strip()
    ):
        raise ValueError("invalid judge summary")
    rows = raw["evaluations"]
    if (
        not isinstance(rows, list)
        or len(rows) != 7
        or any(not isinstance(row, dict) for row in rows)
        or sorted(row.get("category", "") for row in rows) != sorted(BY_CODE)
    ):
        raise ValueError("judge must evaluate FP1-FP7 exactly once")
    validated = []
    required = {
        "category",
        "status",
        "stage",
        "confidence",
        "rationale",
        "evidence",
        "contributing_stages",
    }
    for row in sorted(rows, key=lambda row: row["category"]):
        try:
            if (
                set(row) != required
                or row["status"] not in ("fault", "clear", "unknown", "not_applicable")
                or row["stage"] not in (*STAGES, None)
                or not _number(row["confidence"])
                or not isinstance(row["rationale"], str)
                or not row["rationale"].strip()
                or not isinstance(row["contributing_stages"], list)
            ):
                raise ValueError("invalid category assessment schema")
            if row["status"] == "unknown":
                validated.append(_unknown(row["category"], row["rationale"]))
                continue
            refs = _citations(row["evidence"], artifacts)
            # Clear/not-applicable must not turn unobserved boundaries into successes.
            code = row["category"]
            if code in ("FP2", "FP3") and "retrieval.manifest" not in artifacts:
                raise ValueError("retrieval was not observed")
            if code in ("FP3", "FP4", "FP7") and "context.text" not in artifacts:
                raise ValueError("the actual generation context was not observed")
            if code in ("FP4", "FP7") and "generation.answer" not in artifacts:
                raise ValueError("the generated answer was not observed")
            if (
                code in ("FP5", "FP6")
                and row["status"] != "not_applicable"
                and "generation.answer" not in artifacts
            ):
                raise ValueError("the generated answer was not observed")

            if (
                code == "FP1"
                and row["status"] == "clear"
                and not any(
                    ref == "context.text" or ref.startswith("corpus.document.") for ref in refs
                )
            ):
                raise ValueError("clearing FP1 requires cited available source facts")
            if min(raw["confidence"], row["confidence"]) < threshold:
                raise ValueError("assessment is below the configured confidence threshold")
            if row["status"] == "fault":
                _boundaries(row, refs, artifacts)
            elif row["stage"] is not None or row["contributing_stages"]:
                raise ValueError("non-fault assessments cannot blame stages")
            for contributor in row["contributing_stages"]:
                if (
                    not isinstance(contributor, dict)
                    or set(contributor) != {"stage", "rationale", "evidence"}
                    or contributor["stage"] not in STAGES
                    or contributor["stage"] == row["stage"]
                    or not isinstance(contributor["rationale"], str)
                    or not contributor["rationale"].strip()
                ):
                    raise ValueError("invalid contributor attribution")
                contributor_refs = _citations(contributor["evidence"], artifacts)
                if not any(
                    artifacts[ref]["stage"] == contributor["stage"]
                    and artifacts[ref]["kind"] != "query"
                    for ref in contributor_refs
                ):
                    raise ValueError("contributor lacks evidence from its own stage")
            validated.append({**row, "confidence": min(raw["confidence"], row["confidence"])})
        except ValueError as error:
            validated.append(
                _unknown(row["category"], f"Abstained: {error}. {row.get('rationale', '')}")
            )
    return validated


class DiagnosticEngine:
    """Public entry point for saved traces from arbitrary RAG systems."""

    def __init__(self, config: DiagnosticConfig | None = None, judge: Judge | None = None):
        self.config = config or DiagnosticConfig()
        self.judge = judge or build_judge(self.config)

    def diagnose(
        self,
        trace: dict,
        case_id: str | None = None,
        pipeline: dict | None = None,
        run: RunRecord | None = None,
    ) -> Verdict:
        if pipeline is None:
            pipeline = (
                trace.get("pipeline", {"adapter": "captured_trace"})
                if isinstance(trace, dict)
                else {"adapter": "captured_trace"}
            )

        class Description:
            def describe(self):
                return pipeline

        run = run or RunRecord.capture(self.config, Description())
        try:
            if not isinstance(pipeline, dict):
                raise ValueError("pipeline metadata must be an object")
            _reject_oracles(pipeline)
            original_artifacts = trace_artifacts(trace)
            view = self._judge_view(trace)
            artifacts = original_artifacts if view is trace else trace_artifacts(view)
            source = hashlib.sha256()
            root = Path(__file__).parent.parent
            for path in sorted(root.rglob("*.py")):
                source.update(str(path.relative_to(root)).encode())
                source.update(path.read_bytes())
            payload = {
                "artifacts": artifacts,
                "pipeline": pipeline,
                "paper": PAPER_URL,
                "code_revision": run.code_revision,
                "source_sha256": source.hexdigest(),
                "config": self.config.to_dict(),
            }
            encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
            fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
            observations = {
                "input_fingerprint": fingerprint,
                "judge": self.judge.describe(),
                "input": payload,
                "trace": trace,
            }
            rows = [_unknown(c.code, "Judge has not established a verdict.") for c in CATEGORIES]
            raw, reason = None, ""
            if len(encoded) > self.config.max_judge_input_chars:
                reason = (
                    "Trace exceeds max_judge_input_chars; no artifacts were silently truncated."
                )
            else:
                response = self.judge.evaluate_trace(payload)
                observations["judge_response"] = asdict(response)
                try:
                    if response.completion_status != "completed":
                        raise ValueError("judge response was refused or incomplete")
                    raw = json.loads(response.raw_output)
                    rows = validate_assessment(raw, artifacts, self.config.abstain_below)
                    reason = raw["reason"]
                except (ValueError, TypeError, KeyError) as error:
                    reason = f"Abstained: invalid or insufficient judge response ({error})."
            faults = [row for row in rows if row["status"] == "fault"]
            unknown = tuple(row["category"] for row in rows if row["status"] == "unknown")
            primary = next(
                (
                    row
                    for row in faults
                    if raw
                    and row["category"] == raw["failure_category"]
                    and row["stage"] == raw["primary_fault_stage"]
                ),
                None,
            )
            if faults and primary is None:
                reason += (
                    " Primary summary normalized to the first supported category; "
                    "the LLM's primary selection did not survive evidence validation."
                )
            primary = primary or (faults[0] if faults else None)
            stages = {row["stage"] for row in faults}
            stages.update(c["stage"] for row in faults for c in row["contributing_stages"])
            ordered_stages = tuple(stage for stage in STAGES if stage in stages)
            status = "diagnosed" if faults else "abstained" if unknown else "no_failure"
            evidence = [
                {
                    "category": row["category"],
                    "stage": row["stage"],
                    "rationale": row["rationale"],
                    **citation,
                }
                for row in faults
                for citation in row["evidence"]
            ]
            verdict = Verdict(
                run_id=case_id or str(trace.get("run_id", trace.get("case_id", fingerprint[:12]))),
                query=trace["query"],
                status=status,
                primary_fault_stage=primary["stage"] if primary else None,
                failure_category=primary["category"]
                if primary
                else "ABSTAIN"
                if unknown
                else "NONE",
                confidence=min((row["confidence"] for row in faults), default=0.0)
                if unknown or faults
                else min(row["confidence"] for row in rows),
                contributing_stages=tuple(
                    stage for stage in ordered_stages if not primary or stage != primary["stage"]
                ),
                stages=ordered_stages,
                failure_categories=tuple(row["category"] for row in faults),
                evidence=evidence,
                evaluations=rows,
                unresolved_categories=unknown,
                attribution_complete=not unknown,
                answer=trace.get("final_answer", trace.get("answer", "")),
                explanation=[reason],
                observations=observations,
                run=run,
            )
            append_jsonl(Path(self.config.runs_dir) / "diagnoses.jsonl", verdict.to_dict())
            return verdict
        except Exception as error:
            self.record_error(trace, run, error, case_id)
            raise

    def _judge_view(self, trace):
        """Bound only corpus text sent to the LLM; preserve full original trace in the report.

        Source selection is inspection scoping, never a correctness or fault decision.
        Any omitted or shortened corpus content makes completeness false.
        """
        corpus = trace.get("corpus") if isinstance(trace, dict) else None
        if not isinstance(corpus, dict) or not isinstance(corpus.get("documents"), list):
            return trace
        docs = corpus["documents"]
        if sum(len(doc.get("text", "")) for doc in docs) <= self.config.corpus_budget_chars:
            return trace
        stages = trace.get("stages", {})
        retrieved = stages.get("retrieval", {}).get("output_data", trace.get("candidates", []))
        priority = {doc.get("doc_id") for doc in [*retrieved, *trace.get("deep_retrieval", [])]}
        remaining, selected, truncated = self.config.corpus_budget_chars, [], []
        for doc in sorted(
            docs, key=lambda d: (d.get("doc_id") not in priority, d.get("doc_id", ""))
        ):
            if remaining <= 0:
                break
            text = doc["text"][:remaining]
            selected.append({**doc, "text": text})
            if len(text) < len(doc["text"]):
                truncated.append(doc["doc_id"])
            remaining -= len(text)
        return {
            **trace,
            "corpus": {
                **corpus,
                "documents": selected,
                "complete": False,
                "total_documents": len(docs),
                "truncated_document_ids": truncated,
                "selection": "Bounded corpus view: retrieval/deep-search document IDs first, "
                "then stable document order; full corpus retained in recorded trace.",
            },
        }

    def record_error(self, trace, run, error, case_id=None):
        append_jsonl(
            Path(self.config.runs_dir) / "diagnoses.jsonl",
            {
                "run_id": case_id,
                "status": "error",
                "trace": trace,
                "error": str(error),
                "run": run.to_dict(),
            },
        )

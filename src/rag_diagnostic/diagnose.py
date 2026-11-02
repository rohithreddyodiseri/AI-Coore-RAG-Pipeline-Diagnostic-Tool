"""Takes a wrong answer and works out which stage caused it.

1. Run the pipeline and judge the answer. If it's right, there's nothing to do.
2. Run the stage checks to find where the answer passage was lost.
3. Give the model only the passage. If it still gets it wrong, generation is at fault.
4. Fix the first stage that failed its check, and rerun everything after it.
   If that gets the passage to the model, the stage is confirmed.
5. Set the confidence from how the fix went, and abstain if it's too low.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rag_diagnostic.adapter import PipelineAdapter
from rag_diagnostic.cases import DiagnosticCase
from rag_diagnostic.checks import FAULT, UNKNOWN, Finding, Investigation, run_checks
from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.judge import Judge, JudgeResult
from rag_diagnostic.locate import coverage
from rag_diagnostic.runlog import RunRecord
from rag_diagnostic.taxonomy import mode
from rag_diagnostic.trace import Chunk, Context, ScoredChunk, rerank_positions, stage_order

# Starting confidence values. Tune these against labeled cases so that a
# verdict reported at 0.9 is right about 90% of the time.
CONF_FIX_CORRECTED_ANSWER = 0.9
CONF_FIX_WITH_BROKEN_GENERATION = 0.8
CONF_FIX_WITH_DISTRACTED_GENERATION = 0.7
CONF_FIX_DID_NOT_HELP = 0.4
CONF_GENERATION_FAILS_ON_EVIDENCE = 0.85
CONF_GENERATION_DISTRACTED = 0.6
CONF_NOT_IN_DOCUMENTS = 0.9
PENALTY_UNKNOWN_FINDING = 0.2


@dataclass(frozen=True)
class FixResult:
    """What happened when the pipeline was rerun with one stage fixed."""

    stage: str
    context_recovered: bool  # did the passage reach the model?
    answer: str
    answer_correct: bool


@dataclass(frozen=True)
class Attribution:
    """One stage blamed for the failure, and why."""

    stage: str
    code: str
    confidence: float
    reason: str


@dataclass
class Verdict:
    """The tool's answer for one case."""

    case_id: str
    query: str
    status: str  # no_failure, diagnosed or abstained
    stages: tuple[str, ...]  # the stages at fault; empty if it abstained
    suspects: tuple[str, ...]  # the best guess, shown even when it abstains
    confidence: float
    attributions: list[Attribution]
    findings: list[Finding]
    fix: FixResult | None
    generation_probe: dict[str, Any] | None
    answer: str
    explanation: list[str]
    run: RunRecord | None = field(default=None)

    def to_dict(self, include_run: bool = True) -> dict[str, Any]:
        """The full verdict as a dictionary, for saving to JSON."""
        data = asdict(self)
        data["failure_modes"] = [
            {
                "stage": a.stage,
                "code": a.code,
                "name": mode(a.code).name,
                "remedy": mode(a.code).remedy,
            }
            for a in self.attributions
        ]
        if not include_run:
            data.pop("run")
        return data

    def render(self) -> str:
        """The verdict as readable text."""
        lines = [f"Case {self.case_id}: {self.query}", f"Answer: {self.answer}"]
        if self.status == "no_failure":
            lines.append("Verdict: the pipeline answered correctly; nothing to diagnose.")
        elif self.status == "abstained":
            guess = ", ".join(self.suspects) or "none"
            lines.append(
                f"Verdict: ABSTAINED (confidence {self.confidence:.2f}). "
                f"The evidence does not support a verdict. Leading suspects: {guess}."
            )
        else:
            lines.append(f"Verdict: {' + '.join(self.stages)} (confidence {self.confidence:.2f})")
        if self.attributions:
            lines.append("Failure modes:")
            for a in self.attributions:
                m = mode(a.code)
                lines.append(f"  - {m.name} [{a.stage}, {a.confidence:.2f}]: {a.reason}")
                lines.append(f"    Remedy: {m.remedy}")
        lines.append("Evidence:")
        lines.extend(f"  - {line}" for line in self.explanation)
        return "\n".join(lines)


class Diagnoser:
    def __init__(self, config: DiagnosticConfig, judge: Judge):
        self.config = config
        self.judge = judge

    # ---- Main entry point ------------------------------------------------------

    def diagnose(self, case: DiagnosticCase, adapter: PipelineAdapter) -> Verdict:
        """Diagnose one case and return the verdict."""
        run = RunRecord.capture(self.config, adapter)
        trace = adapter.run(case.query)
        first = self._judge(case, trace.answer)
        if first.correct:
            return Verdict(
                case_id=case.case_id,
                query=case.query,
                status="no_failure",
                stages=(),
                suspects=(),
                confidence=1.0,
                attributions=[],
                findings=[],
                fix=None,
                generation_probe=None,
                answer=trace.answer,
                explanation=[f"judge: {first.rationale}"],
                run=run,
            )

        inv = Investigation(case, adapter, self.config, trace)
        findings = run_checks(inv)
        explanation = [f"judge marked the answer wrong: {first.rationale}"]
        explanation += [f"{f.stage}: {f.status} - {f.summary}" for f in findings]
        has_unknown = any(f.status == UNKNOWN for f in findings)
        faults = [f for f in findings if f.status == FAULT]

        # If the answer was never in the documents, there's nothing to fix.
        missing = [f for f in faults if f.stage == "corpus"]
        if missing:
            attributions = [
                Attribution(
                    "corpus", "answer_not_in_corpus", CONF_NOT_IN_DOCUMENTS, missing[0].summary
                )
            ]
            return self._finish(
                case,
                trace.answer,
                attributions,
                findings,
                None,
                None,
                explanation,
                has_unknown,
                run,
            )

        probe = self._generation_probe(inv)
        explanation.append(
            "generation test (model given only the passage): "
            + ("answered correctly" if probe["correct"] else f"still wrong: {probe['answer']!r}")
        )

        suspects = sorted({f.stage for f in faults}, key=stage_order)
        attributions: list[Attribution] = []
        fix: FixResult | None = None

        if suspects:
            # Fix the first stage that failed its check and see what changes.
            stage = suspects[0]
            code = next(f.code for f in faults if f.stage == stage)
            fix = self.fix_and_rerun(inv, stage)
            explanation.append(
                f"fixed {stage} and reran: the passage "
                f"{'reached' if fix.context_recovered else 'still did not reach'} the model, "
                f"and the answer was {'correct' if fix.answer_correct else 'still wrong'}"
            )
            if len(suspects) > 1:
                explanation.append(
                    f"later stages also failed their checks ({', '.join(suspects[1:])}); "
                    "only the first one is tested"
                )

            if not fix.context_recovered:
                attributions.append(
                    Attribution(
                        stage,
                        code,
                        CONF_FIX_DID_NOT_HELP,
                        "failed its check, but fixing it didn't get the passage to the model, "
                        "so another stage may also be at fault",
                    )
                )
            elif fix.answer_correct:
                attributions.append(
                    Attribution(
                        stage,
                        code,
                        CONF_FIX_CORRECTED_ANSWER,
                        "fixing this stage got the passage to the model and corrected the answer",
                    )
                )
            else:
                confidence = (
                    CONF_FIX_WITH_BROKEN_GENERATION
                    if not probe["correct"]
                    else CONF_FIX_WITH_DISTRACTED_GENERATION
                )
                attributions.append(
                    Attribution(
                        stage,
                        code,
                        confidence,
                        "fixing this stage got the passage to the model, but the answer "
                        "was still wrong",
                    )
                )
                attributions.append(self._generation_attribution(probe))
        else:
            # Every check passed, so the passage should have reached the model.
            evidence_in_context = all(
                coverage(trace.context.text, g.evidence) >= self.config.match_threshold
                for g in case.gold
            )
            if evidence_in_context:
                explanation.append("the passage was in the context the model received")
                attributions.append(self._generation_attribution(probe))
            elif has_unknown:
                explanation.append(
                    "the passage is not in the context, and an unclear check keeps the tool "
                    "from naming the stage that lost it"
                )
            else:
                explanation.append(
                    "no check failed, yet the passage is not in the context; "
                    "check the case's answer location"
                )

        return self._finish(
            case,
            trace.answer,
            attributions,
            findings,
            fix,
            probe,
            explanation,
            has_unknown,
            run,
        )

    # ---- Helpers ----------------------------------------------------------------

    def _judge(self, case: DiagnosticCase, answer: str) -> JudgeResult:
        return self.judge.judge(case.query, answer, case.reference_answer)

    def _generation_probe(self, inv: Investigation) -> dict[str, Any]:
        """Give the model only the answer passage and see if it gets it right."""
        oracle = Context(("oracle",), "\n\n".join(g.evidence for g in inv.case.gold))
        answer = inv.adapter.generate(inv.case.query, oracle)
        result = self._judge(inv.case, answer)
        return {"answer": answer, "correct": result.correct, "judge": result.rationale}

    def _generation_attribution(self, probe: dict[str, Any]) -> Attribution:
        if not probe["correct"]:
            return Attribution(
                "generation",
                "generation_wrong_with_evidence",
                CONF_GENERATION_FAILS_ON_EVIDENCE,
                "the model answers wrongly even when given only the passage",
            )
        return Attribution(
            "generation",
            "generation_distracted",
            CONF_GENERATION_DISTRACTED,
            "the model answers correctly from the passage alone, but not from the full context",
        )

    def _oracle_chunks(self, inv: Investigation) -> list[Chunk]:
        """A perfect chunk for each answer passage."""
        return [Chunk(f"oracle:{n}", g.doc_id, g.evidence) for n, g in enumerate(inv.case.gold)]

    def fix_and_rerun(self, inv: Investigation, stage: str) -> FixResult:
        """Rerun the pipeline with one stage fixed and everything else left alone.

        How each stage is fixed:
          parsing, chunking   -> a perfect chunk with the passage is put first
          indexing, retrieval -> the real answer chunk is put first
          reranking           -> the reranker is skipped
          context             -> the top chunks go in whole, with nothing cut
        """
        adapter, query = inv.adapter, inv.case.query
        candidates = list(inv.trace.candidates)

        inject: list[Chunk] = []
        if stage in ("parsing", "chunking"):
            inject = self._oracle_chunks(inv)
        elif stage in ("indexing", "retrieval"):
            inject = [c for chunks in inv.gold_chunks for c in chunks] or self._oracle_chunks(inv)
        if inject:
            top = candidates[0].score if candidates else 1.0
            injected_ids = {c.chunk_id for c in inject}
            boosted = [ScoredChunk(c.chunk_id, c.doc_id, c.text, top, 0) for c in inject]
            rest = [c for c in candidates if c.chunk_id not in injected_ids]
            candidates = list(rerank_positions((boosted + rest)[: adapter.retrieval_k]))

        if inv.trace.reranker_used and stage != "reranking":
            ranked = adapter.rerank(query, candidates) or candidates
        else:
            ranked = candidates

        if stage == "context":
            top_chunks = ranked[: adapter.context_k]
            context = Context(
                tuple(c.chunk_id for c in top_chunks), "\n\n".join(c.text for c in top_chunks)
            )
        else:
            context = adapter.assemble_context(query, list(ranked))

        recovered = all(
            coverage(context.text, g.evidence) >= self.config.match_threshold for g in inv.case.gold
        )
        answer = adapter.generate(query, context)
        return FixResult(stage, recovered, answer, self._judge(inv.case, answer).correct)

    def _finish(
        self, case, answer, attributions, findings, fix, probe, explanation, has_unknown, run
    ) -> Verdict:
        """Set the confidence, decide whether to abstain, and build the verdict."""
        suspect_set = {a.stage for a in attributions}
        if not suspect_set:
            # Nothing was blamed, so name the stages whose checks were unclear.
            for f in findings:
                if f.status == UNKNOWN:
                    suspect_set.add(f.stage)
                    if f.stage == "parsing":
                        suspect_set.add("corpus")
        suspects = tuple(sorted(suspect_set, key=stage_order))
        confidence = min((a.confidence for a in attributions), default=0.0)
        if has_unknown:
            confidence = max(0.0, confidence - PENALTY_UNKNOWN_FINDING)
            explanation.append("confidence lowered: at least one check was unclear")
        confidence = round(confidence, 2)
        diagnosed = bool(attributions) and confidence >= self.config.abstain_below
        return Verdict(
            case_id=case.case_id,
            query=case.query,
            status="diagnosed" if diagnosed else "abstained",
            stages=suspects if diagnosed else (),
            suspects=suspects,
            confidence=confidence,
            attributions=attributions,
            findings=findings,
            fix=fix,
            generation_probe=probe,
            answer=answer,
            explanation=explanation,
            run=run,
        )

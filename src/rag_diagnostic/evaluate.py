"""Scores the tool against labeled cases.

A verdict is correct when it names exactly the labeled stages. The report
shows three numbers: accuracy on the cases it answered, how often it
abstained, and accuracy with abstentions counted as wrong. You need all
three, since a tool can look accurate just by abstaining a lot.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rag_diagnostic.adapter import PipelineAdapter
from rag_diagnostic.cases import DiagnosticCase, FaultLabel
from rag_diagnostic.diagnose import Diagnoser, Verdict

AdapterFactory = Callable[[dict[str, Any]], PipelineAdapter]


def split_cases(
    cases: list[DiagnosticCase], held_out_fraction: float, seed: str
) -> tuple[list[DiagnosticCase], list[DiagnosticCase]]:
    """Split the cases into a tuning set and a held-out set.

    Each case always lands on the same side, even as new cases are added,
    so held-out cases never end up in tuning.
    """
    tuning, held_out = [], []
    for case in cases:
        digest = hashlib.sha256(f"{seed}\0{case.case_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        (held_out if bucket < held_out_fraction else tuning).append(case)
    return tuning, held_out


def _label_name(stages) -> str:
    return "+".join(sorted(stages)) or "none"


@dataclass
class CaseResult:
    case_id: str
    fault_type: str
    true_stages: str
    predicted_stages: str
    suspects: str
    status: str
    confidence: float
    correct: bool


def _predicted(verdict: Verdict) -> frozenset[str]:
    return frozenset() if verdict.status == "no_failure" else frozenset(verdict.stages)


def evaluate(
    cases: list[DiagnosticCase],
    labels: dict[str, FaultLabel],
    adapter_factory: AdapterFactory,
    diagnoser: Diagnoser,
    on_verdict: Callable[[Verdict], None] | None = None,
) -> dict[str, Any]:
    results: list[CaseResult] = []
    for case in cases:
        label = labels.get(case.case_id)
        if label is None:
            raise KeyError(f"no label for case {case.case_id}")
        verdict = diagnoser.diagnose(case, adapter_factory(case.pipeline))
        if on_verdict:
            on_verdict(verdict)
        predicted = _predicted(verdict)
        results.append(
            CaseResult(
                case_id=case.case_id,
                fault_type=label.fault_type,
                true_stages=_label_name(label.stages),
                predicted_stages="abstained"
                if verdict.status == "abstained"
                else _label_name(predicted),
                suspects=_label_name(verdict.suspects),
                status=verdict.status,
                confidence=verdict.confidence,
                correct=verdict.status != "abstained" and predicted == label.stages,
            )
        )
    return summarize(results)


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    answered = [r for r in results if r.status != "abstained"]
    correct = sum(r.correct for r in results)

    per_type: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        grouped[r.fault_type].append(r)
    for fault_type, rows in sorted(grouped.items()):
        rows_answered = [r for r in rows if r.status != "abstained"]
        per_type[fault_type] = {
            "cases": len(rows),
            "abstained": len(rows) - len(rows_answered),
            "accuracy_when_answered": _ratio(
                sum(r.correct for r in rows_answered), len(rows_answered)
            ),
            "accuracy_overall": _ratio(sum(r.correct for r in rows), len(rows)),
        }

    confusion = Counter((r.true_stages, r.predicted_stages) for r in results)
    return {
        "cases": total,
        "answered": len(answered),
        "abstained": total - len(answered),
        "abstention_rate": _ratio(total - len(answered), total),
        "accuracy_when_answered": _ratio(sum(r.correct for r in answered), len(answered)),
        "accuracy_overall": _ratio(correct, total),
        "per_fault_type": per_type,
        "confusion": [
            {"true": t, "predicted": p, "count": n} for (t, p), n in sorted(confusion.items())
        ],
        "wrong_cases": [vars(r) for r in results if not r.correct],
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def render_report(report: dict[str, Any]) -> str:
    def pct(value):
        return "n/a" if value is None else f"{value:.1%}"

    lines = [
        f"Cases: {report['cases']}   answered: {report['answered']}   "
        f"abstained: {report['abstained']} ({pct(report['abstention_rate'])})",
        f"Accuracy when answered: {pct(report['accuracy_when_answered'])}",
        f"Accuracy with abstentions counted wrong: {pct(report['accuracy_overall'])}",
        "",
        "| Fault type | Cases | Abstained | Accuracy (answered) | Accuracy (overall) |",
        "|---|---|---|---|---|",
    ]
    for name, row in report["per_fault_type"].items():
        lines.append(
            f"| {name} | {row['cases']} | {row['abstained']} | "
            f"{pct(row['accuracy_when_answered'])} | {pct(row['accuracy_overall'])} |"
        )
    lines += ["", "| True stages | Predicted | Count |", "|---|---|---|"]
    lines += [f"| {c['true']} | {c['predicted']} | {c['count']} |" for c in report["confusion"]]
    if report["wrong_cases"]:
        lines += ["", "Cases the tool got wrong or abstained on:"]
        lines += [
            f"  - {r['case_id']} ({r['fault_type']}): true {r['true_stages']}, "
            f"predicted {r['predicted_stages']}, suspects {r['suspects']}, "
            f"confidence {r['confidence']:.2f}"
            for r in report["wrong_cases"]
        ]
    return "\n".join(lines)

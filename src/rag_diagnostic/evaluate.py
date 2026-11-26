"""Scores the tool against labeled executions.

A verdict is correct when it names exactly the labeled stages. The report
shows three numbers: accuracy on the executions it answered, how often it
abstained, and accuracy with abstentions counted as wrong. You need all
three, since a tool can look accurate just by abstaining a lot.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from rag_diagnostic.diagnostic_engine import Verdict
from rag_diagnostic.labels import ExecutionLabel


def split_executions(
    executions: list[dict], held_out_fraction: float, seed: str
) -> tuple[list[dict], list[dict]]:
    """Split the executions into a tuning set and a held-out set.

    Each execution always lands on the same side, even as new executions are added,
    so held-out executions never end up in tuning.
    """
    tuning, held_out = [], []
    for execution in executions:
        digest = hashlib.sha256(f"{seed}\0{execution['run_id']}".encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        (held_out if bucket < held_out_fraction else tuning).append(execution)
    return tuning, held_out


def _label_name(stages) -> str:
    return "+".join(sorted(stages)) or "none"


@dataclass
class ExecutionResult:
    run_id: str
    fault_type: str
    true_stages: str
    predicted_stages: str
    suspects: str
    status: str
    confidence: float
    correct: bool
    true_categories: str | None = None
    predicted_categories: str | None = None


def _predicted(verdict: Verdict) -> frozenset[str]:
    return frozenset() if verdict.status == "no_failure" else frozenset(verdict.stages)


def score_verdict(verdict: Verdict, label: ExecutionLabel) -> ExecutionResult:
    predicted = _predicted(verdict)
    return ExecutionResult(
        run_id=verdict.run_id,
        fault_type=label.fault_type,
        true_stages=_label_name(label.stages),
        predicted_stages="abstained" if verdict.status == "abstained" else _label_name(predicted),
        suspects=_label_name(verdict.suspects),
        status=verdict.status,
        confidence=verdict.confidence,
        correct=verdict.status != "abstained"
        and predicted == label.stages
        and (label.categories is None or frozenset(verdict.failure_categories) == label.categories),
        true_categories=_label_name(label.categories) if label.categories is not None else None,
        predicted_categories=_label_name(verdict.failure_categories),
    )


def evaluate_traces(traces, labels, engine):
    """Evaluate captured traces; labels never enter the engine or judge payload."""
    results = []
    seen = set()
    for trace in traces:
        run_id = trace.get("run_id")
        if not isinstance(run_id, str) or run_id in seen:
            raise ValueError("evaluation traces require unique string run_id values")
        seen.add(run_id)
        if run_id not in labels:
            raise KeyError(f"no label for execution {run_id}")
        verdict = engine.diagnose(trace, run_id)
        results.append(score_verdict(verdict, labels[run_id]))
    return summarize(results)


def summarize(results: list[ExecutionResult]) -> dict[str, Any]:
    total = len(results)
    answered = [r for r in results if r.status in ("diagnosed", "no_failure")]
    abstained = sum(r.status == "abstained" for r in results)
    correct = sum(r.correct for r in results)

    per_type: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[ExecutionResult]] = defaultdict(list)
    for r in results:
        grouped[r.fault_type].append(r)
    for fault_type, rows in sorted(grouped.items()):
        rows_answered = [r for r in rows if r.status in ("diagnosed", "no_failure")]
        per_type[fault_type] = {
            "executions": len(rows),
            "abstained": sum(r.status == "abstained" for r in rows),
            "errors": sum(r.status.endswith("_error") for r in rows),
            "accuracy_when_answered": _ratio(
                sum(r.correct for r in rows_answered), len(rows_answered)
            ),
            "accuracy_overall": _ratio(sum(r.correct for r in rows), len(rows)),
        }

    confusion = Counter((r.true_stages, r.predicted_stages) for r in results)
    return {
        "executions": total,
        "answered": len(answered),
        "abstained": abstained,
        "errors": sum(r.status.endswith("_error") for r in results),
        "abstention_rate": _ratio(abstained, total),
        "accuracy_when_answered": _ratio(sum(r.correct for r in answered), len(answered)),
        "accuracy_overall": _ratio(correct, total),
        "per_fault_type": per_type,
        "confusion": [
            {"true": t, "predicted": p, "count": n} for (t, p), n in sorted(confusion.items())
        ],
        "incorrect_runs": [vars(r) for r in results if not r.correct],
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def render_report(report: dict[str, Any]) -> str:
    def pct(value):
        return "n/a" if value is None else f"{value:.1%}"

    lines = [
        f"Executions: {report['executions']}   answered: {report['answered']}   "
        f"abstained: {report['abstained']} ({pct(report['abstention_rate'])})",
        f"Accuracy when answered: {pct(report['accuracy_when_answered'])}",
        f"Accuracy with abstentions counted wrong: {pct(report['accuracy_overall'])}",
        "",
        "| Fault type | Executions | Abstained | Accuracy (answered) | Accuracy (overall) |",
        "|---|---|---|---|---|",
    ]
    for name, row in report["per_fault_type"].items():
        lines.append(
            f"| {name} | {row['executions']} | {row['abstained']} | "
            f"{pct(row['accuracy_when_answered'])} | {pct(row['accuracy_overall'])} |"
        )
    lines += ["", "| True stages | Predicted | Count |", "|---|---|---|"]
    lines += [f"| {c['true']} | {c['predicted']} | {c['count']} |" for c in report["confusion"]]
    if report["incorrect_runs"]:
        lines += ["", "Executions the tool got wrong or abstained on:"]
        lines += [
            f"  - {r['run_id']} ({r['fault_type']}): true {r['true_stages']}, "
            f"predicted {r['predicted_stages']}, suspects {r['suspects']}, "
            f"confidence {r['confidence']:.2f}"
            for r in report["incorrect_runs"]
        ]
    return "\n".join(lines)

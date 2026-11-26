"""Run captured boundary interventions through the production diagnostic engine."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from rag_diagnostic.diagnostic_engine import DiagnosticEngine
from rag_diagnostic.mechanism_cases import build_cases
from rag_diagnostic.mechanisms import VERSION


def _write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def score(case, verdict):
    """Compare sealed expectations only AFTER diagnosis; unknowns are not successes on faults."""
    data = verdict.to_dict()
    root = data["root_cause"]
    predicted_root = {k: root[k] for k in ("mechanism", "technical_stage")} if root else None
    faults = {r["mechanism"] for r in data["evaluations"] if r["status"] == "FAULT"}
    expected = set(case["expected_faults"])
    statuses = {r["mechanism"]: r["status"] for r in data["evaluations"]}
    target_correct = all(statuses.get(m) == s for m, s in case["target_statuses"].items())
    return dict(
        run_id=data["run_id"],
        kind=case["kind"],
        description=case["description"],
        expected_root=case["expected_root"],
        predicted_root=predicted_root,
        status=data["status"],
        confidence=data["confidence"],
        root_correct=predicted_root == case["expected_root"],
        stage_correct=(predicted_root or {}).get("technical_stage")
        == (case["expected_root"] or {}).get("technical_stage"),
        target_correct=target_correct,
        expected_faults=sorted(expected),
        predicted_faults=sorted(faults),
        true_positives=len(expected & faults),
        false_positives=len(faults - expected),
        false_negatives=len(expected - faults),
        exact_fault_set_correct=faults == expected,
        correct=predicted_root == case["expected_root"] and faults == expected and target_correct,
        target_statuses={m: statuses.get(m) for m in case["target_statuses"]},
    )


def summarize(rows):
    def ratio(n, d):
        return round(n / d, 4) if d else None

    faults = [r for r in rows if r["kind"] != "control"]
    answered = [r for r in faults if r["status"] == "diagnosed"]
    valid = [r for r in rows if "error" not in r]
    tp = sum(r["true_positives"] for r in valid)
    fp = sum(r["false_positives"] for r in valid)
    fn = sum(r["false_negatives"] for r in rows)
    controls = [r for r in rows if r["kind"] == "control"]
    return dict(
        attempted=len(rows),
        errors=sum("error" in r for r in rows),
        suite_accuracy=ratio(sum(r["correct"] for r in rows), len(rows)),
        primary_accuracy=ratio(sum(r["root_correct"] for r in faults), len(faults)),
        stage_accuracy=ratio(sum(r["stage_correct"] for r in faults), len(faults)),
        accuracy_when_diagnosed=ratio(sum(r["root_correct"] for r in answered), len(answered)),
        fault_abstention_rate=ratio(sum(r["status"] == "abstained" for r in faults), len(faults)),
        control_accuracy=ratio(sum(r["correct"] for r in controls), len(controls)),
        mechanism_precision=ratio(tp, tp + fp),
        mechanism_recall=ratio(tp, tp + fn),
        confusion=[
            dict(
                expected=(r["expected_root"] or {}).get("mechanism", "NONE"),
                predicted=(r.get("predicted_root") or {}).get("mechanism", r["status"]),
            )
            for r in rows
        ],
        per_mechanism={
            r["expected_root"]["mechanism"]: dict(
                root_correct=r["root_correct"],
                target_correct=r["target_correct"],
                status=r["status"],
            )
            for r in rows
            if r["kind"] == "fault"
        },
    )


def run_mechanism_suite(
    output_dir,
    config,
    *,
    judge=None,
    mechanisms=None,
    prepare_only=False,
    resume=False,
    replay_dir=None,
):
    if config.assessment_mode != "stage":
        raise ValueError("Technical mechanism suite requires assessment_mode='stage'")
    output = Path(output_dir).resolve()
    if replay_dir:
        source = Path(replay_dir).resolve()
        expectations = json.loads((source / "ground_truth.json").read_text())
        cases = [
            dict(case, trace=json.loads((source / case["trace_path"]).read_text()))
            for case in expectations
        ]
    else:
        cases = build_cases()
    if mechanisms:
        cases = [
            c
            for c in cases
            if c["kind"] == "fault" and c["expected_root"]["mechanism"] in mechanisms
        ]
    if not cases:
        raise ValueError("No mechanism cases selected")
    output.mkdir(parents=True, exist_ok=resume)
    for subdir in ("traces", "verdicts", "results"):
        (output / subdir).mkdir(exist_ok=resume)
    labels = []
    for case in cases:
        relative = f"traces/{case['trace']['run_id']}.json"
        _write(output / relative, case["trace"])
        labels.append(
            {k: v for k, v in case.items() if k not in ("trace", "trace_path")}
            | {"trace_path": relative}
        )
    _write(output / "ground_truth.json", labels)
    if prepare_only:
        return dict(
            mode="technical_mechanisms", status="prepared", attempted=len(cases), cases=labels
        )
    engine = DiagnosticEngine(replace(config, runs_dir=str(output / "diagnostic")), judge=judge)
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    rows = []
    for case, label in zip(cases, labels, strict=True):
        identity = hashlib.sha256(
            json.dumps(
                [case, engine.config.to_dict(), engine.judge.describe(), source_hash.hexdigest()],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        checkpoint = output / "results" / f"{case['trace']['run_id']}.json"
        previous = json.loads(checkpoint.read_text()) if resume and checkpoint.exists() else {}
        if previous.get("identity") == identity and "error" not in previous:
            from rag_diagnostic.judge_logging import log_saved_judge_exchange

            saved = output / "verdicts" / f"{case['trace']['run_id']}.json"
            if saved.exists():
                log_saved_judge_exchange(json.loads(saved.read_text()))
            rows.append(previous)
            continue
        try:
            # Never pass case, description, target status, or expected root into diagnosis.
            verdict = engine.diagnose(case["trace"])
            _write(output / "verdicts" / f"{verdict.run_id}.json", verdict.to_dict())
            row = score(case, verdict)
        except Exception as error:
            row = dict(
                run_id=case["trace"]["run_id"],
                kind=case["kind"],
                description=case["description"],
                expected_root=case["expected_root"],
                status="diagnosis_error",
                predicted_root=None,
                confidence=0.0,
                correct=False,
                root_correct=False,
                stage_correct=False,
                target_correct=False,
                true_positives=0,
                false_positives=0,
                false_negatives=len(case["expected_faults"]),
                error=dict(type=type(error).__name__, message=str(error)),
            )
        row.update(identity=identity, trace_path=label["trace_path"])
        _write(checkpoint, row)
        rows.append(row)
    report = dict(
        schema_version="3.0",
        mode="technical_mechanisms",
        taxonomy_version=VERSION,
        status="completed_with_errors" if any("error" in r for r in rows) else "completed",
        evidence_scope="Controlled captured boundary interventions, not production-domain accuracy",
        judge=engine.judge.describe(),
        config=engine.config.to_dict(),
        source_sha256=source_hash.hexdigest(),
        cases=rows,
        **summarize(rows),
    )
    _write(output / "report.json", report)
    return report

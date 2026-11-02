"""Diagnose wrong RAG answers, or score the tool on labeled cases.

    rag-diagnose diagnose --cases cases.jsonl --adapter rag_diagnostic.adapters.toy:build
    rag-diagnose evaluate --cases cases.jsonl --labels labels.jsonl \\
        --adapter rag_diagnostic.adapters.toy:build --split held_out

--adapter points to a function, written as "module:function". It takes a
case's pipeline settings and returns an adapter.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from rag_diagnostic.cases import load_cases, load_labels
from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnose import Diagnoser
from rag_diagnostic.evaluate import evaluate, render_report, split_cases
from rag_diagnostic.judge import build_judge
from rag_diagnostic.runlog import append_jsonl


def _load_factory(spec: str):
    module_name, _, function = spec.partition(":")
    if not function:
        raise ValueError("--adapter must look like module.path:function")
    return getattr(importlib.import_module(module_name), function)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-diagnose", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    diagnose = commands.add_parser("diagnose", help="Diagnose failing queries")
    diagnose.add_argument("--cases", type=Path, required=True)
    diagnose.add_argument("--case-id", action="append", help="Only these cases (repeatable)")
    diagnose.add_argument("--adapter", required=True)
    diagnose.add_argument("--config", type=Path)
    diagnose.add_argument("--json", action="store_true", help="Print full JSON verdicts")

    evaluate_cmd = commands.add_parser("evaluate", help="Score the tool against labelled cases")
    evaluate_cmd.add_argument("--cases", type=Path, required=True)
    evaluate_cmd.add_argument("--labels", type=Path, required=True)
    evaluate_cmd.add_argument("--adapter", required=True)
    evaluate_cmd.add_argument("--config", type=Path)
    evaluate_cmd.add_argument("--split", choices=["held_out", "tuning", "all"], default="held_out")
    evaluate_cmd.add_argument("--report-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = DiagnosticConfig.from_file(args.config)
    diagnoser = Diagnoser(config, build_judge(config))
    factory = _load_factory(args.adapter)
    log_path = Path(config.runs_dir) / "diagnoses.jsonl"
    cases = load_cases(args.cases)

    def record(verdict):
        append_jsonl(log_path, verdict.to_dict())

    if args.command == "diagnose":
        if args.case_id:
            wanted = set(args.case_id)
            cases = [c for c in cases if c.case_id in wanted]
        for case in cases:
            verdict = diagnoser.diagnose(case, factory(case.pipeline))
            record(verdict)
            if args.json:
                print(json.dumps(verdict.to_dict(), indent=2, default=str))
            else:
                print(verdict.render(), end="\n\n")
        print(f"Run records appended to {log_path}", file=sys.stderr)
        return

    tuning, held_out = split_cases(cases, config.held_out_fraction, config.split_seed)
    chosen = {"held_out": held_out, "tuning": tuning, "all": cases}[args.split]
    if not chosen:
        raise SystemExit(f"the {args.split} split is empty; add cases or change held_out_fraction")
    report = evaluate(chosen, load_labels(args.labels), factory, diagnoser, on_verdict=record)
    report["split"] = args.split
    report["config_fingerprint"] = config.fingerprint()
    print(f"Split: {args.split} ({len(chosen)} of {len(cases)} cases)\n")
    print(render_report(report))
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

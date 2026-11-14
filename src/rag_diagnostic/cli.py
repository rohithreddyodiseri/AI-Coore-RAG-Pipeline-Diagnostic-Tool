"""Run a query against a target RAG pipeline and attribute failures from its trace."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnose import Diagnoser

DEFAULT_ADAPTER = "rag_diagnostic.adapters.open_ragbench:build"


def _load_factory(spec):
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("--adapter must look like module.path:function")
    return getattr(importlib.import_module(module), name)


def _parser():
    parser = argparse.ArgumentParser(prog="rag-diagnose", description=__doc__)
    parser.add_argument("query", help="Query to execute on the connected RAG pipeline")
    parser.add_argument(
        "--adapter", default=DEFAULT_ADAPTER, help="Target adapter factory (module:function)"
    )
    parser.add_argument("--pipeline-config", type=Path, help="JSON settings for the target adapter")
    parser.add_argument("--config", type=Path, help="Diagnostic settings JSON")
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--chroma-dir", type=Path)
    parser.add_argument("--collection")
    parser.add_argument("--embedding-model")
    parser.add_argument("--llm-model", help="Target pipeline generation model")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--judge-model", help="Diagnostic judge model")
    parser.add_argument("--report-out", type=Path, help="Structured JSON report destination")
    parser.add_argument(
        "--trace-out", type=Path, help="Also save the captured target execution trace"
    )
    parser.add_argument("--json", action="store_true", help="Print the structured report")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        config = DiagnosticConfig.from_file(args.config)
        if args.judge_model:
            config = replace(config, judge_model=args.judge_model)
        settings = json.loads(args.pipeline_config.read_text()) if args.pipeline_config else {}
        if not isinstance(settings, dict):
            raise ValueError("--pipeline-config must contain a JSON object")
        for name in (
            "corpus_dir",
            "chroma_dir",
            "collection",
            "embedding_model",
            "llm_model",
            "top_k",
        ):
            value = getattr(args, name)
            if value is not None:
                settings[name] = str(value) if isinstance(value, Path) else value
        adapter = _load_factory(args.adapter)(settings)
        verdict = Diagnoser(config).diagnose(args.query, adapter)
        data = verdict.to_dict()
        destination = (
            args.report_out or Path(config.runs_dir) / "reports" / f"{verdict.run_id}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if args.trace_out:
            args.trace_out.parent.mkdir(parents=True, exist_ok=True)
            args.trace_out.write_text(
                json.dumps(verdict.observations["trace"], indent=2), encoding="utf-8"
            )
        print(json.dumps(data, indent=2) if args.json else verdict.render())
        print(f"Diagnostic report: {destination}", file=sys.stderr)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        print(f"Diagnostic error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

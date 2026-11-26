"""Run a query against a target RAG pipeline and attribute failures from its trace."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from gemini_client import GeminiAPIError
from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnose import Diagnoser
from rag_diagnostic.diagnostic_engine import DiagnosticEngine
from rag_diagnostic.ingestion import TelemetryIngestor
from rag_diagnostic.telemetry import flag_execution

DEFAULT_ADAPTER = "rag_diagnostic.adapters.open_ragbench:build"
MAX_PROVIDER_RETRY_SECONDS = 60.0


def _load_factory(spec):
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("--adapter must look like module.path:function")
    return getattr(importlib.import_module(module), name)


def _parser():
    parser = argparse.ArgumentParser(
        prog="rag-diagnose",
        description=__doc__,
        epilog="Saved telemetry: rag-diagnose ingest --trace PATH; "
        "flagged-run consumer: rag-diagnose watch --trace-dir DIR; "
        "benchmark: rag-diagnose fault-test-runner --output-dir DIR",
    )
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


def _poll_with_provider_retry(ingestor, args):
    """Keep continuous watch alive while a retryable judge outage clears."""
    delay = max(5.0, args.poll_interval)
    while True:
        try:
            return ingestor.poll(args.trace_dir, artifact_root=args.artifact_root)
        except GeminiAPIError as error:
            if args.once or not error.retryable:
                raise
            print(
                f"Diagnostic judge temporarily unavailable ({error}); "
                f"retrying unacknowledged runs in {delay:g} seconds.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            delay = min(MAX_PROVIDER_RETRY_SECONDS, delay * 2)


def _ingestion_main(argv):
    parser = argparse.ArgumentParser(prog="rag-diagnose " + argv[0])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--artifact-root", type=Path, help="Authorized root for captured artifact references"
    )
    if argv[0] == "watch":
        parser.add_argument("--trace-dir", type=Path, required=True)
        parser.add_argument(
            "--once", action="store_true", help="Poll once, suitable for scheduled jobs"
        )
        parser.add_argument("--poll-interval", type=float, default=2)
    else:
        parser.add_argument("--trace", type=Path, required=True)
        parser.add_argument("--run-id", help="Select one execution from a multi-run event stream")
        parser.add_argument("--report-out", type=Path)
        if argv[0] == "flag":
            parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv[1:])
    try:
        if argv[0] == "flag":
            run_id = flag_execution(args.trace, args.reason, args.run_id)
            print(json.dumps({"run_id": run_id, "flagged": True}))
            return
        config = DiagnosticConfig.from_file(args.config)
        ingestor = TelemetryIngestor(DiagnosticEngine(config))
        if argv[0] == "ingest":
            verdict, destination = ingestor.ingest(
                args.trace, run_id=args.run_id, artifact_root=args.artifact_root
            )
            if args.report_out:
                args.report_out.parent.mkdir(parents=True, exist_ok=True)
                args.report_out.write_text(json.dumps(verdict.to_dict(), indent=2))
                destination = args.report_out
            print(json.dumps(verdict.to_dict(), indent=2) if args.json else verdict.render())
            print(f"Diagnostic report: {destination}", file=sys.stderr)
            return
        if not 0 < args.poll_interval <= 60:
            raise ValueError("--poll-interval must be between zero and 60 seconds")
        while True:
            for verdict, destination in _poll_with_provider_retry(ingestor, args):
                print(json.dumps(verdict.to_dict()) if args.json else verdict.render(), flush=True)
                print(f"Diagnostic report: {destination}", file=sys.stderr, flush=True)
            if args.once:
                return
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        print(f"Diagnostic ingestion error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "fault-test-runner":
        from rag_diagnostic.fault_test_runner import main as benchmark_main

        return benchmark_main(argv[1:])
    if argv and argv[0] in ("ingest", "watch", "flag"):
        return _ingestion_main(argv)
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

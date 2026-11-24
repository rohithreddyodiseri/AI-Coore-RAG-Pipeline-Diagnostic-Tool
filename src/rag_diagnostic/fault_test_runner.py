"""Execute and score the seven-category fault benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from rag_diagnostic.adapters.open_ragbench import OpenRAGBenchAdapter
from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnostic_engine import DiagnosticEngine
from rag_diagnostic.evaluate import ExecutionResult, render_report, score_verdict, summarize
from rag_diagnostic.labels import ExecutionLabel
from rag_diagnostic.rag_pipeline import BenchmarkRAGPipeline
from rag_diagnostic.reference_pipeline import RAGPipeline
from rag_diagnostic.taxonomy import BY_CODE
from rag_diagnostic.telemetry import load_execution
from rag_pipeline.config import Settings
from rag_pipeline.corpus import OpenRAGBenchLoader
from rag_pipeline.fault_injector import FaultInjectionEngine

FAULT_CODES = {
    "missing_content": "FP1",
    "missed_top_ranked": "FP2",
    "not_in_context": "FP3",
    "not_extracted": "FP4",
    "wrong_format": "FP5",
    "incorrect_specificity": "FP6",
    "incomplete_answer": "FP7",
}


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_local_json(path, label):
    flags = getattr(os.stat(path), "st_flags", 0)
    if flags & getattr(stat, "SF_DATALESS", 0):
        raise ValueError(
            f"{label} is an iCloud dataless file: {path}. "
            f"Download it first with: brctl download '{path}'"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_resume_registry(path):
    return _load_local_json(path, "Resume registry")


def _require_local_tree(path, label):
    dataless = [
        item
        for item in Path(path).rglob("*")
        if item.is_file()
        and getattr(os.stat(item), "st_flags", 0) & getattr(stat, "SF_DATALESS", 0)
    ]
    if dataless:
        examples = ", ".join(item.name for item in dataless[:3])
        raise ValueError(
            f"{label} contains {len(dataless)} iCloud dataless file(s) ({examples}). "
            f"Download the directory first with: brctl download '{Path(path)}'"
        )


def _progress(enabled):
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[detail]}"),
        TimeElapsedColumn(),
        console=Console(file=sys.stderr, force_terminal=enabled),
        disable=not enabled,
        refresh_per_second=10,
    )


def _index_with_progress(progress, pipeline, corpus_dir, description):
    paths = sorted(Path(corpus_dir).glob("*.json"))
    task = progress.add_task(description, total=len(paths), detail="0 chunks")
    previous_source = None

    def documents():
        nonlocal previous_source
        for document in OpenRAGBenchLoader(corpus_dir).load():
            source = document.metadata["source"]
            if previous_source is not None and source != previous_source:
                progress.advance(task)
            previous_source = source
            yield document
        if previous_source is not None:
            progress.advance(task)

    count = pipeline.index(
        documents(),
        progress=lambda chunks: progress.update(task, detail=f"{chunks:,} chunks"),
    )
    progress.update(task, completed=len(paths), detail=f"{count:,} chunks")
    return count


def _clone_index(progress, source_pipeline, target_pipeline, deleted_doc_id):
    source = source_pipeline._open_collection(create=False)
    target = target_pipeline._open_collection()
    total = source.count()
    deleted_count = len(
        source.get(where={"doc_id": deleted_doc_id}, include=["metadatas"])["ids"]
    )
    if deleted_count == 0:
        raise ValueError(
            f"FP1 target document {deleted_doc_id!r} was not found in the clean index"
        )
    expected = total - deleted_count
    if target.count() == expected:
        task = progress.add_task(
            "Reusing corrupted index", total=expected, completed=expected,
            detail=f"{expected:,} chunks",
        )
        progress.update(task, completed=expected)
        return expected
    if target.count():
        raise ValueError("Corrupted index is partial; use a fresh output directory")
    task = progress.add_task("Creating corrupted index", total=total, detail="0 chunks")
    copied = 0
    for offset in range(0, total, 1000):
        rows = source.get(
            limit=1000, offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        keep = [
            i for i, metadata in enumerate(rows["metadatas"])
            if metadata.get("doc_id") != deleted_doc_id
        ]
        if keep:
            embeddings = []
            for i in keep:
                vector = rows["embeddings"][i]
                embeddings.append(vector.tolist() if hasattr(vector, "tolist") else vector)
            target.upsert(
                ids=[rows["ids"][i] for i in keep],
                documents=[rows["documents"][i] for i in keep],
                metadatas=[rows["metadatas"][i] for i in keep],
                embeddings=embeddings,
            )
            copied += len(keep)
        progress.update(
            task, completed=min(offset + len(rows["ids"]), total),
            detail=f"{copied:,} chunks",
        )
    return copied


def run_suite(
    manifest, output_dir, settings, config, *, judge=None, show_progress=False, resume=False
):
    with _progress(show_progress) as progress:
        return _run_suite(manifest, output_dir, settings, config, judge, progress, resume)


def _run_suite(manifest, output_dir, settings, config, judge, progress, resume):
    output = Path(output_dir).expanduser().resolve()
    clean_corpus = Path(settings.corpus_dir).expanduser().resolve()
    injector = FaultInjectionEngine(clean_corpus, output, manifest, link_corpus=True)
    cases = injector.load_manifest()["test_cases"]
    types = [case.get("failure_type") for case in cases if isinstance(case, dict)]
    if len(cases) != len(FAULT_CODES) or sorted(map(str, types)) != sorted(FAULT_CODES):
        raise ValueError("Benchmark manifest must specify exactly one case for each of FP1–FP7")
    case_settings = {case["query_id"]: case for case in cases}
    for case in cases:
        cutoff = case.get("retrieval_top_k", settings.top_k)
        if type(cutoff) is not int or cutoff < 1:
            raise ValueError("retrieval_top_k must be a positive integer")
    settings = replace(
        settings,
        corpus_dir=output / "corpus",
        chroma_dir=output / "index",
        collection="fault_benchmark_corrupted",
        diagnostics_dir=output / "index_artifacts",
    )
    settings.validate()
    config = replace(config, runs_dir=str(output / "diagnostic"))
    engine = DiagnosticEngine(config, judge)
    prepare_task = progress.add_task("Preparing corrupted corpus", total=None, detail="")
    if resume:
        if not output.is_dir():
            raise ValueError("--resume requires an existing benchmark output directory")
        registry = _load_resume_registry(output / "ground_truth.json")
    else:
        registry = injector.run_injection_suite()
    progress.update(prepare_task, total=1, completed=1, detail="ready")
    run_metadata = {
        "schema_version": "2.0",
        "benchmark_id": uuid4().hex,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "full_corpus_reference_pipeline",
        "status": "running",
    }
    _write_json(output / "report.json", {**run_metadata, "cases": []})
    traces = output / "traces"
    traces.mkdir(exist_ok=True)
    legacy_index = output / "baseline_index"
    baseline_index = legacy_index if resume and legacy_index.is_dir() else output / "index"
    if resume:
        _require_local_tree(baseline_index, "Clean Chroma index")
    baseline_collection = (
        "fault_benchmark" if baseline_index == legacy_index else "fault_benchmark_clean"
    )
    baseline_settings = replace(
        settings,
        corpus_dir=clean_corpus,
        chroma_dir=baseline_index,
        collection=baseline_collection,
        diagnostics_dir=output / "baseline_artifacts",
    )
    baseline = RAGPipeline(baseline_settings)
    indexer = BenchmarkRAGPipeline(settings, deep_k=config.deep_k)
    try:
        try:
            baseline_count = baseline._open_collection(create=False).count()
        except Exception:
            baseline_count = 0
        if baseline_count:
            task = progress.add_task(
                "Reusing clean index", total=baseline_count, completed=baseline_count,
                detail=f"{baseline_count:,} chunks",
            )
            progress.update(task, completed=baseline_count)
        else:
            baseline_count = _index_with_progress(
                progress, baseline, clean_corpus, "Indexing clean corpus"
            )
        indexer._embedder = baseline._embedder
        deleted_target = next(
            entry["target_location"] for entry in registry["test_cases"].values()
            if entry["failure_type"] == "missing_content"
        )
        deleted = deleted_target["doc_id"]
        indexed = _clone_index(progress, baseline, indexer, deleted)
        snapshot_task = progress.add_task("Reading corpus snapshot", total=None, detail="")
        legacy_snapshots = list((output / "baseline_artifacts").glob("indexes/*/corpus.json"))
        if resume and legacy_snapshots:
            corpus = _load_local_json(legacy_snapshots[-1], "Baseline corpus snapshot")
            clean_files = {path.name for path in clean_corpus.glob("*.json")}
            faulted_files = {path.name for path in settings.corpus_dir.glob("*.json")}
            if clean_files - {deleted_target["file"]} != faulted_files:
                raise ValueError("Saved fault corpus files differ from the expected snapshot")
            corpus["documents"] = [
                document for document in corpus["documents"] if document["doc_id"] != deleted
            ]
            corpus["snapshot_id"] = hashlib.sha256(
                json.dumps(corpus["documents"], sort_keys=True).encode()
            ).hexdigest()
            corpus["scope"] = f"Source JSON documents in {(output / 'corpus').resolve()}"
        else:
            corpus = OpenRAGBenchAdapter(pipeline=indexer).corpus_snapshot()
        progress.update(snapshot_task, total=1, completed=1, detail="ready")
    except Exception as error:
        _write_json(
            output / "report.json",
            {
                **run_metadata,
                "status": "failed",
                "cases": [],
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        raise
    baseline_dir = output / "baseline_traces"
    baseline_dir.mkdir(exist_ok=True)
    print(
        f"Indexed {baseline_count} clean and {indexed} corrupted corpus chunks.",
        file=sys.stderr,
        flush=True,
    )

    results, rows = [], []
    query_task = progress.add_task(
        "Executing fault cases", total=len(registry["test_cases"]), detail=""
    )
    target_settings = replace(settings, diagnostics_dir=output / "query_events")
    for query_id, entry in registry["test_cases"].items():
        failure = entry["failure_type"]
        code = FAULT_CODES[failure]
        case = case_settings[query_id]
        cutoff = case.get("retrieval_top_k", settings.top_k)
        stage = entry["expected_failure_stage"]
        if stage not in BY_CODE[code].stages:
            raise ValueError(f"Invalid expected stage for {code}: {stage}")
        row = {
            "query_id": query_id,
            "expected_category": code,
            "expected_stage": stage,
            "retrieval_top_k": cutoff,
            "trace": f"traces/{code}.json",
            "injection_applied": False,
        }
        rows.append(row)
        pipeline = BenchmarkRAGPipeline(
            replace(target_settings, top_k=cutoff),
            entry["mutation"].get("runtime_plan"),
            deep_k=config.deep_k,
        )
        # The index and embedding model are shared; generation and hooks are per query.
        pipeline._embedder = indexer._embedder
        pipeline._collection = indexer._collection
        query = entry["query"]
        source_document = case.get("source_document")
        case_corpus = corpus
        if source_document is not None:
            if (
                failure != "missing_content"
                or source_document != entry["target_location"]["doc_id"]
            ):
                raise ValueError("source_document must identify the missing_content case's source")
            query = f"Use only source document {source_document}. {query}"
            documents = [doc for doc in corpus["documents"] if doc["doc_id"] == source_document]
            case_corpus = {
                "snapshot_id": corpus["snapshot_id"],
                "scope": f"All records for source document {source_document} in {corpus['scope']}",
                "complete": corpus["complete"],
                "documents": documents,
                "source_document": source_document,
            }
            row["corpus_scope"] = case_corpus["scope"]
        trace = None
        try:
            progress.update(query_task, description=f"{code} clean baseline", detail=failure)
            baseline.settings = replace(baseline_settings, top_k=cutoff)
            baseline.generation_instructions = pipeline.generation_instructions
            baseline_trace = baseline.run_with_trace(query)
            indexer._embedder = baseline._embedder
            pipeline._embedder = indexer._embedder
            _write_json(baseline_dir / f"{code}.json", baseline_trace)
            row["baseline_trace"] = f"baseline_traces/{code}.json"
            progress.update(query_task, description=f"{code} faulted pipeline", detail=failure)
            trace = pipeline.run_with_trace(query)
            if failure == "missing_content":
                trace["corpus"] = case_corpus
            trace["deep_retrieval"] = pipeline._serialize_results(pipeline.candidates)
            trace["deep_k"] = len(pipeline.candidates)
            _write_json(output / row["trace"], trace)
            row["injection_applied"] = (
                entry["injection_status"] == "applied" or pipeline.injection_applied
            )
            row["injection_reason"] = pipeline.injection_reason
            progress.update(query_task, description=f"{code} diagnosis", detail=failure)
            verdict = engine.diagnose(trace, trace["run_id"])
            label = ExecutionLabel(trace["run_id"], frozenset({stage}), failure, frozenset({code}))
            result = score_verdict(verdict, label)
            applied = entry["injection_status"] == "applied" or pipeline.injection_applied
            row["injection_applied"] = applied
            row["injection_reason"] = (
                "Corpus mutation applied before indexing"
                if entry["injection_status"] == "applied"
                else pipeline.injection_reason
            )
            if not applied:
                result.correct = False
            results.append(result)
            row.update(
                asdict(result),
                predicted_primary_category=verdict.failure_category,
                predicted_primary_stage=verdict.primary_fault_stage,
                primary_correct=applied
                and (verdict.failure_category, verdict.primary_fault_stage) == (code, stage),
                evidence=verdict.evidence,
                explanation=verdict.explanation,
                evaluations=verdict.evaluations,
                judge_corpus_complete=verdict.observations.get("input", {})
                .get("artifacts", {})
                .get("corpus.manifest", {})
                .get("complete"),
            )
        except Exception as error:
            row.update(
                error={"type": type(error).__name__, "message": str(error)},
                status="diagnosis_error" if trace is not None else "execution_error",
                correct=False,
                primary_correct=False,
            )
            results.append(
                ExecutionResult(
                    trace["run_id"] if trace else uuid4().hex,
                    failure,
                    stage,
                    row["status"],
                    "none",
                    row["status"],
                    0.0,
                    False,
                    code,
                    None,
                )
            )
            if trace is None:
                trace = {
                    "query": query,
                    "run_id": uuid4().hex,
                    "execution_status": "failed",
                }
                if pipeline.last_trace_path and pipeline.last_trace_path.is_file():
                    trace = load_execution(pipeline.last_trace_path, resolve_artifacts=False)
                _write_json(output / row["trace"], trace)
        row["original_retrieval_order"] = [
            result.chunk_id for result in pipeline.original_candidates
        ]
        row["original_response"] = pipeline.original_response
        print(f"{code}: {row['status']}", file=sys.stderr, flush=True)
        _write_json(output / "report.json", {**run_metadata, "cases": rows})
        progress.advance(query_task)

    report = summarize(results)
    report.update(
        **run_metadata,
        cases=rows,
        attempted=len(rows),
        errors=sum("error" in row for row in rows),
        suite_accuracy=round(sum(row["correct"] for row in rows) / len(rows), 4),
        primary_accuracy=round(sum(row["primary_correct"] for row in rows) / len(rows), 4),
        scope=corpus["scope"],
        reference_pipeline="rag_diagnostic.reference_pipeline.RAGPipeline",
        baseline_chunks=baseline_count,
        indexed_chunks=indexed,
        corpus_documents=len(corpus["documents"]),
        applied_cases=sum(row.get("injection_applied", False) for row in rows),
        corpus_snapshot_id=corpus["snapshot_id"],
        manifest_sha256=registry["manifest_sha256"],
        judge=engine.judge.describe(),
        diagnostic_config_id=config.fingerprint(),
    )
    report["status"] = "completed_with_errors" if report["errors"] else "completed"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["category_confusion"] = [
        {
            "actual": row["expected_category"],
            "predicted": row.get("predicted_categories") or row["status"],
            "count": 1,
        }
        for row in rows
    ]
    eligible = [row for row in rows if row.get("injection_applied")]
    report["injection_coverage"] = round(len(eligible) / len(rows), 4)
    report["diagnostic_accuracy"] = (
        round(sum(row["correct"] for row in eligible) / len(eligible), 4) if eligible else None
    )
    report_task = progress.add_task("Writing benchmark report", total=1, detail="")
    _write_json(output / "report.json", report)
    progress.update(report_task, completed=1, detail="complete")
    return report


def replay_suite(source, output_dir, config, *, show_progress=False):
    """Score the saved observations again with the current judge and validation."""
    with _progress(show_progress) as progress:
        return _replay_suite(source, output_dir, config, progress)


def _replay_suite(source, output_dir, config, progress):
    source = Path(source).resolve()
    output = Path(output_dir).resolve()
    previous = json.loads((source / "report.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    engine = DiagnosticEngine(replace(config, runs_dir=str(output / "diagnostic")))
    results, rows = [], []
    replay_task = progress.add_task("Replaying diagnoses", total=len(previous["cases"]), detail="")
    for original in previous["cases"]:
        code, stage = original["expected_category"], original["expected_stage"]
        trace = load_execution(source / original["trace"], resolve_artifacts=False)
        progress.update(replay_task, description=f"Replaying {code}", detail=stage)
        _write_json(output / "traces" / f"{code}.json", trace)
        row = {
            "query_id": original["query_id"],
            "expected_category": code,
            "expected_stage": stage,
            "trace": f"traces/{code}.json",
        }
        try:
            if trace.get("execution_status") == "failed":
                raise ValueError("Saved execution failed; run a new suite to execute this query")
            verdict = engine.diagnose(trace, trace["run_id"])
            label = ExecutionLabel(
                trace["run_id"],
                frozenset({stage}),
                next(name for name, value in FAULT_CODES.items() if value == code),
                frozenset({code}),
            )
            result = score_verdict(verdict, label)
            if original.get("injection_applied") is False:
                result.correct = False
            row.update(
                asdict(result),
                evaluations=verdict.evaluations,
                explanation=verdict.explanation,
                evidence=verdict.evidence,
                primary_correct=original.get("injection_applied", True)
                and (verdict.failure_category, verdict.primary_fault_stage) == (code, stage),
                injection_applied=original.get("injection_applied"),
            )
        except Exception as error:
            result = ExecutionResult(
                trace["run_id"],
                code,
                stage,
                "diagnosis_error",
                "none",
                "diagnosis_error",
                0.0,
                False,
                code,
                None,
            )
            row.update(
                error={"type": type(error).__name__, "message": str(error)},
                status="diagnosis_error",
                correct=False,
                primary_correct=False,
            )
        results.append(result)
        rows.append(row)
        print(f"{code}: {row['status']}", file=sys.stderr, flush=True)
        progress.advance(replay_task)
    report = summarize(results)
    report.update(
        schema_version="2.0",
        benchmark_id=uuid4().hex,
        mode=previous.get("mode", "legacy_saved_traces"),
        status="completed_with_errors" if any("error" in row for row in rows) else "completed",
        completed_at=datetime.now(timezone.utc).isoformat(),
        cases=rows,
        attempted=len(rows),
        errors=sum("error" in row for row in rows),
        suite_accuracy=round(sum(row["correct"] for row in rows) / len(rows), 4),
        primary_accuracy=round(sum(row["primary_correct"] for row in rows) / len(rows), 4),
        source_run=str(source),
        judge=engine.judge.describe(),
    )
    _write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fault-test-runner", description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/faults.example.yaml"))
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Diagnostic judge settings JSON")
    parser.add_argument("--top-k", type=int, help="Number of naturally retrieved chunks selected")
    parser.add_argument("--replay-dir", type=Path, help="Re-score saved traces without generation")
    parser.add_argument("--no-progress", action="store_true", help="Disable animated progress bars")
    parser.add_argument("--resume", action="store_true", help="Resume a failed benchmark run")
    args = parser.parse_args(argv)
    try:
        if args.replay_dir:
            report = replay_suite(
                args.replay_dir,
                args.output_dir,
                DiagnosticConfig.from_file(args.config),
                show_progress=not args.no_progress,
            )
        else:
            report = _run_new_suite(args)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        print(f"Benchmark error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(render_report(report))
    print(
        f"Suite accuracy: {report['suite_accuracy']:.1%} across {report['attempted']} cases "
        f"({report['errors']} execution or diagnosis errors)."
    )
    print(f"Primary category and stage accuracy: {report['primary_accuracy']:.1%}")
    print(f"Report: {args.output_dir / 'report.json'}")
    if report["errors"]:
        raise SystemExit(1)


def _run_new_suite(args):
    settings = Settings.from_env()
    if args.corpus_dir is not None:
        settings = replace(settings, corpus_dir=args.corpus_dir)
    if args.top_k is not None:
        settings = replace(settings, top_k=args.top_k)
    return run_suite(
        args.manifest,
        args.output_dir,
        settings,
        DiagnosticConfig.from_file(args.config),
        show_progress=not args.no_progress,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

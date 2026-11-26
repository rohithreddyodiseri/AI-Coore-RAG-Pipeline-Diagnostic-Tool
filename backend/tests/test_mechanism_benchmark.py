"""Fixture boundary integrity and independent benchmark scoring."""

import json
from types import SimpleNamespace

import pytest

from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.judge import JudgeResponse
from rag_diagnostic.mechanism_benchmark import run_mechanism_suite, score, summarize
from rag_diagnostic.mechanism_cases import build_cases
from rag_diagnostic.mechanisms import MECHANISM_STAGE
from rag_diagnostic.stage_diagnosis import reconstruct, unknown, validate

KEYS = {
    "MISSING_SOURCE_INFORMATION": ["query", "corpus.manifest"],
    "EXTRACTION_LOSS": ["boundary.parsing", "corpus.document.0"],
    "CHUNK_FRAGMENTATION": [
        "boundary.chunking",
        "boundary.parsing",
        "chunking.chunk.0",
        "chunking.chunk.1",
    ],
    "EMBEDDING_MISMATCH": ["boundary.document_embedding", "boundary.query_embedding"],
    "INDEXING_FAILURE": ["boundary.indexing", "boundary.chunking", "boundary.document_embedding"],
    "TOP_K_EXCLUSION": ["retrieval.manifest", "deep.chunk.1"],
    "RERANKING_DEGRADATION": ["retrieval.manifest", "retrieval.chunk.0", "reranking.manifest"],
    "SIMILARITY_CUTOFF_EXCLUSION": ["boundary.selection", "retrieval.chunk.1"],
    "TOKEN_BUDGET_EXCLUSION": ["boundary.selection", "retrieval.chunk.1"],
    "CONTEXT_TRUNCATION": ["boundary.context_assembly"],
    "CONTEXT_DELIVERY_LOSS": [
        "context.text",
        "generation.delivered_context",
        "boundary.generation",
    ],
    "INCORRECT_EVIDENCE_EXTRACTION": [
        "generation.delivered_context",
        "generation.answer",
        "boundary.generation",
    ],
    "PARTIAL_EVIDENCE_EXTRACTION": [
        "query",
        "generation.delivered_context",
        "generation.answer",
        "boundary.generation",
    ],
    "OUTPUT_FORMAT_VIOLATION": ["query", "generation.answer"],
    "SPECIFICITY_MISMATCH": ["query", "generation.answer"],
    "UNCLASSIFIED_OPERATIONAL_FAILURE": ["boundary.generation"],
}


def hypothesis(mechanisms):
    rows = [unknown(m, "insufficient evidence") for m in MECHANISM_STAGE]
    for row in rows:
        if row["mechanism"] in mechanisms:
            row.update(
                status="FAULT",
                confidence=0.95,
                designation="primary",
                unresolved_questions=[],
                evidence=[dict(artifact_id=k, span_id=0) for k in KEYS[row["mechanism"]]],
            )
    return dict(evaluations=rows)


@pytest.mark.parametrize(
    "case", [c for c in build_cases() if c["kind"] != "control"], ids=lambda c: c["description"]
)
def test_every_injection_has_valid_boundary_evidence(case):
    trace = case["trace"]
    artifacts, _ = reconstruct(trace)
    rows = validate(hypothesis(case["expected_faults"]), artifacts, trace, 0.7)
    for mechanism in case["expected_faults"]:
        row = next(r for r in rows if r["mechanism"] == mechanism)
        assert row["status"] == "FAULT", row
    root = next(r for r in rows if r["designation"] == "primary")
    assert root["mechanism"] == case["expected_root"]["mechanism"]
    assert root["technical_stage"] == case["expected_root"]["technical_stage"]


def test_coverage_reproducibility_and_labels_are_separate(tmp_path):
    assert build_cases() == build_cases()
    cases = build_cases()
    assert {c["expected_root"]["mechanism"] for c in cases if c["kind"] == "fault"} == set(
        MECHANISM_STAGE
    )
    result = run_mechanism_suite(tmp_path / "prepared", DiagnosticConfig(), prepare_only=True)
    assert result["status"] == "prepared"
    assert not (tmp_path / "prepared/diagnostic").exists()
    for c in cases:
        encoded = json.dumps(c["trace"])
        assert "expected_root" not in encoded and "expected_faults" not in encoded
        assert not any(m in c["trace"]["run_id"] for m in MECHANISM_STAGE)


def test_actual_engine_scoring_resume_and_replay(tmp_path):
    class Judge:
        calls = 0

        def describe(self):
            return {"provider": "test"}

        def evaluate_trace(self, payload):
            self.calls += 1
            assert "expected_root" not in payload and "target_statuses" not in payload
            # Stub provides a hypothesis; production validation and scoring still run.
            return JudgeResponse(json.dumps(hypothesis(["EMBEDDING_MISMATCH"])), "test")

    judge = Judge()
    config = DiagnosticConfig()
    destination = tmp_path / "run"
    report = run_mechanism_suite(
        destination, config, judge=judge, mechanisms=["EMBEDDING_MISMATCH"]
    )
    assert report["primary_accuracy"] == 1 and report["suite_accuracy"] == 1
    run_mechanism_suite(
        destination, config, judge=judge, mechanisms=["EMBEDDING_MISMATCH"], resume=True
    )
    assert judge.calls == 1
    replay = run_mechanism_suite(tmp_path / "replay", config, judge=judge, replay_dir=destination)
    assert replay["primary_accuracy"] == 1 and judge.calls == 2


def test_abstention_errors_and_false_contributors_are_penalized():
    case = next(c for c in build_cases() if c["expected_faults"] == ["EMBEDDING_MISMATCH"])

    def verdict(root, evaluations, status="diagnosed"):
        return SimpleNamespace(
            to_dict=lambda: dict(
                run_id="id", root_cause=root, evaluations=evaluations, status=status, confidence=0.9
            )
        )

    abstained = score(case, verdict(None, [], "abstained"))
    assert not abstained["correct"]
    assert summarize([abstained])["primary_accuracy"] == 0
    root = case["expected_root"]
    extra = score(
        case,
        verdict(
            root,
            [dict(mechanism=m, status="FAULT") for m in ["EMBEDDING_MISMATCH", "INDEXING_FAILURE"]],
        ),
    )
    assert extra["root_correct"] and not extra["correct"] and extra["false_positives"] == 1
    metrics = summarize([abstained, extra])["mechanism_metrics"]
    assert metrics["EMBEDDING_MISMATCH"]["precision"] == 1
    assert metrics["EMBEDDING_MISMATCH"]["recall"] == 0.5
    assert metrics["INDEXING_FAILURE"]["precision"] == 0
    assert metrics["INDEXING_FAILURE"]["recall"] is None

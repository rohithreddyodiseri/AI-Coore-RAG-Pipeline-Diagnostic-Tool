"""Technical attribution requires recorded, consistent boundary evidence."""

import json

import pytest

from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnostic_engine import DiagnosticEngine
from rag_diagnostic.judge import JudgeResponse
from rag_diagnostic.mechanisms import MECHANISM_STAGE, OUTPUT_SCHEMA
from rag_diagnostic.stage_diagnosis import reconstruct, unknown, validate


@pytest.fixture
def execution():
    return {
        "run_id": "technical-run",
        "query": "Who built it? Return JSON.",
        "final_answer": "Bob.",
        "corpus": {
            "complete": True,
            "scope": "captured source JSON",
            "snapshot_id": "v1",
            "documents": [{"doc_id": "d", "text": "Ada built it. In 1902."}],
        },
        "stages": {
            "parsing": {
                "complete": True,
                "output_data": [
                    {"text": "Ada built it. In 1902.", "metadata": {"doc_id": "d", "section": 0}}
                ],
            },
            "chunking": {
                "output_data": [
                    {
                        "chunk_id": "a",
                        "doc_id": "d",
                        "text": "Ada built it.",
                        "metadata": {"section": 0, "chunk": 0, "start_offset": 0, "end_offset": 13},
                    },
                    {
                        "chunk_id": "b",
                        "doc_id": "d",
                        "text": "In 1902.",
                        "metadata": {
                            "section": 0,
                            "chunk": 1,
                            "start_offset": 14,
                            "end_offset": 22,
                        },
                    },
                ]
            },
            "document_embedding": {
                "configuration": {"model": "one"},
                "output_data": [
                    {"chunk_id": "a", "vector": [1.0, 0.0]},
                    {"chunk_id": "b", "vector": [0.0, 1.0]},
                ],
            },
            "query_embedding": {"configuration": {"model": "two"}, "output_data": [[1.0, 0.0]]},
            "indexing": {"output_data": {"complete": True, "indexed_ids": ["a"]}},
            "retrieval": {
                "k": 1,
                "output_data": [
                    {
                        "chunk_id": "a",
                        "doc_id": "d",
                        "text": "Ada built it.",
                        "rank": 1,
                        "score": 0.6,
                    }
                ],
            },
            "reranking": {"status": "not_used"},
            "selection": {
                "output_data": {
                    "complete": True,
                    "decisions": [
                        {
                            "chunk_id": "a",
                            "selected": False,
                            "reason": "similarity_threshold",
                            "score": 0.6,
                            "threshold": 0.7,
                        }
                    ],
                }
            },
            "context_assembly": {
                "pre_truncation": "Ada built it.",
                "post_truncation": "Ada",
                "output_data": "Ada built it.",
            },
            "generation": {
                "input_data": {
                    "prompt": "Question: Who? Context: Ada built it.",
                    "delivered_context": "Ada built it.",
                }
            },
        },
        "deep_retrieval": [
            {"chunk_id": "b", "doc_id": "d", "text": "In 1902.", "rank": 2, "score": 0.5}
        ],
    }


def assessment(mechanism, keys, status="FAULT"):
    rows = [unknown(m, "Not enough evidence") for m in MECHANISM_STAGE]
    row = next(r for r in rows if r["mechanism"] == mechanism)
    row.update(
        status=status,
        designation="primary" if status == "FAULT" else None,
        confidence=0.95,
        rationale="The cited boundaries establish this technical mechanism.",
        evidence=[{"artifact_id": k, "span_id": 0} for k in keys],
        unresolved_questions=[],
    )
    return {"evaluations": rows}


FAULTS = [
    ("MISSING_SOURCE_INFORMATION", ["corpus.manifest", "query"]),
    (
        "CHUNK_FRAGMENTATION",
        ["boundary.chunking", "boundary.parsing", "chunking.chunk.0", "chunking.chunk.1"],
    ),
    ("EMBEDDING_MISMATCH", ["boundary.document_embedding", "boundary.query_embedding"]),
    ("INDEXING_FAILURE", ["boundary.indexing", "boundary.chunking", "boundary.document_embedding"]),
    ("TOP_K_EXCLUSION", ["retrieval.manifest", "deep.chunk.0"]),
    ("SIMILARITY_CUTOFF_EXCLUSION", ["boundary.selection", "retrieval.chunk.0"]),
    ("CONTEXT_TRUNCATION", ["boundary.context_assembly"]),
    (
        "INCORRECT_EVIDENCE_EXTRACTION",
        ["generation.delivered_context", "generation.answer", "boundary.generation"],
    ),
    (
        "PARTIAL_EVIDENCE_EXTRACTION",
        ["query", "generation.delivered_context", "generation.answer", "boundary.generation"],
    ),
    ("OUTPUT_FORMAT_VIOLATION", ["query", "generation.answer"]),
    ("SPECIFICITY_MISMATCH", ["query", "generation.answer"]),
]


@pytest.mark.parametrize("mechanism,keys", FAULTS)
def test_supported_mechanisms(execution, mechanism, keys):
    artifacts, _ = reconstruct(execution)
    rows = validate(assessment(mechanism, keys), artifacts, execution, 0.7)
    row = next(r for r in rows if r["mechanism"] == mechanism)
    assert row["status"] == "FAULT", row["rationale"]
    assert row["designation"] == "primary"


@pytest.mark.parametrize("mechanism", list(MECHANISM_STAGE))
def test_absent_telemetry_never_establishes_fault(mechanism):
    trace = {"query": "What happened?"}
    artifacts, _ = reconstruct(trace)
    row = next(
        r
        for r in validate(assessment(mechanism, ["query"]), artifacts, trace, 0.7)
        if r["mechanism"] == mechanism
    )
    assert row["status"] == "UNKNOWN"


def test_extraction_loss_differs_from_source_absence(execution):
    execution["stages"]["parsing"]["output_data"][0]["text"] = "Title only"
    artifacts, _ = reconstruct(execution)
    raw = assessment("EXTRACTION_LOSS", ["boundary.parsing", "corpus.document.0"])
    assert validate(raw, artifacts, execution, 0.7)[1]["status"] == "FAULT"
    execution["stages"]["parsing"]["complete"] = False
    artifacts, _ = reconstruct(execution)
    assert validate(raw, artifacts, execution, 0.7)[1]["status"] == "UNKNOWN"


def test_unused_reranker_cannot_be_clear(execution):
    artifacts, _ = reconstruct(execution)
    for status, expected in [
        ("CLEAR", "UNKNOWN"),
        ("FAULT", "UNKNOWN"),
        ("NOT_APPLICABLE", "NOT_APPLICABLE"),
    ]:
        rows = validate(
            assessment("RERANKING_DEGRADATION", ["boundary.reranking"], status),
            artifacts,
            execution,
            0.7,
        )
        assert rows[6]["status"] == expected


def test_rank_threshold_and_budget_guards(execution):
    decision = execution["stages"]["selection"]["output_data"]["decisions"][0]
    decision.update(reason="token_budget", token_count=20, used_tokens=90, budget=100)
    raw = assessment("TOKEN_BUDGET_EXCLUSION", ["boundary.selection", "retrieval.chunk.0"])
    artifacts, _ = reconstruct(execution)
    assert validate(raw, artifacts, execution, 0.7)[8]["status"] == "FAULT"
    decision["used_tokens"] = 0
    artifacts, _ = reconstruct(execution)
    assert validate(raw, artifacts, execution, 0.7)[8]["status"] == "UNKNOWN"


def test_context_delivery_loss_and_generation_dependency(execution):
    execution["stages"]["generation"]["input_data"].update(
        delivered_context="", prompt="Question: Who?"
    )
    artifacts, _ = reconstruct(execution)
    # Empty context is cited using exact request boundary, not an empty quote.
    raw = assessment(
        "INCORRECT_EVIDENCE_EXTRACTION",
        ["context.text", "generation.answer", "boundary.generation"],
    )
    assert validate(raw, artifacts, execution, 0.7)[11]["status"] == "UNKNOWN"
    execution["stages"]["generation"]["input_data"].update(
        delivered_context="Ada", prompt="Context: Ada"
    )
    artifacts, _ = reconstruct(execution)
    raw = assessment(
        "CONTEXT_DELIVERY_LOSS",
        ["context.text", "generation.delivered_context", "boundary.generation"],
    )
    assert validate(raw, artifacts, execution, 0.7)[10]["status"] == "FAULT"


def test_reranker_demotion_requires_candidate_identity(execution):
    execution["stages"]["reranking"] = {"output_data": []}
    artifacts, _ = reconstruct(execution)
    raw = assessment(
        "RERANKING_DEGRADATION", ["retrieval.manifest", "retrieval.chunk.0", "reranking.manifest"]
    )
    assert validate(raw, artifacts, execution, 0.7)[6]["status"] == "FAULT"


def test_invalid_citation_confidence_and_contradiction_abstain(execution):
    artifacts, _ = reconstruct(execution)
    for modification in (
        {"confidence": 0.69},
        {"evidence": [{"artifact_id": "invented", "span_id": 0}]},
        {"contradicting_evidence": [{"artifact_id": "query", "span_id": 0}]},
    ):
        raw = assessment("SPECIFICITY_MISMATCH", ["query", "generation.answer"])
        raw["evaluations"][14].update(modification)
        assert validate(raw, artifacts, execution, 0.7)[14]["status"] == "UNKNOWN"


@pytest.mark.parametrize(
    "answer,expected",
    [
        ('{"answer": "Ada"}', "CLEAR"),
        ('```json\n{"answer": "Ada"}\n```', "FAULT"),
        ('{"answer": "Ada", "year": 1902}', "FAULT"),
    ],
)
def test_exact_json_check_overrides_judge_hypothesis(execution, answer, expected):
    execution["query"] = 'Who built it?\nReturn only valid JSON with a single key "answer".'
    execution["final_answer"] = answer
    artifacts, _ = reconstruct(execution)
    raw = assessment("OUTPUT_FORMAT_VIOLATION", ["boundary.generation"], "CLEAR")

    row = validate(raw, artifacts, execution, 0.7)[13]

    assert row["status"] == expected
    assert row["confidence"] == 1.0
    assert {citation["artifact_id"] for citation in row["evidence"]} == {
        "query",
        "generation.answer",
    }


def test_python_selects_earliest_confirmed_root(execution, tmp_path):
    raw = assessment("OUTPUT_FORMAT_VIOLATION", ["query", "generation.answer"])
    embedding = assessment(
        "EMBEDDING_MISMATCH", ["boundary.document_embedding", "boundary.query_embedding"]
    )["evaluations"][3]
    embedding["designation"] = "contributing"
    raw["evaluations"][3] = embedding

    class Judge:
        def describe(self):
            return {"provider": "test"}

        def evaluate_trace(self, payload):
            return JudgeResponse(json.dumps(raw), "test")

    report = (
        DiagnosticEngine(DiagnosticConfig(runs_dir=str(tmp_path)), Judge())
        .diagnose(execution)
        .to_dict()
    )
    assert report["root_cause"]["mechanism"] == "EMBEDDING_MISMATCH"
    assert report["contributing_failures"][0]["mechanism"] == "OUTPUT_FORMAT_VIOLATION"
    assert report["schema_version"] == "2.0"
    assert "failure_category" not in report
    assert len(report["evaluations"]) == 16
    assert report["observations"]["trace"] == execution


def test_operational_exception_retains_actual_stage(execution):
    execution["stages"]["parsing"].update(status="failed", error={"type": "ParserError"})
    artifacts, _ = reconstruct(execution)
    raw = assessment("UNCLASSIFIED_OPERATIONAL_FAILURE", ["boundary.parsing"])
    row = validate(raw, artifacts, execution, 0.7)[-1]
    assert row["status"] == "FAULT"
    assert row["technical_stage"] == "parsing"


def test_strict_mechanism_schema(execution):
    artifacts, _ = reconstruct(execution)
    raw = assessment("SPECIFICITY_MISMATCH", ["query", "generation.answer"])
    raw["evaluations"].pop()
    with pytest.raises(ValueError, match="exactly once"):
        validate(raw, artifacts, execution, 0.7)
    assert OUTPUT_SCHEMA["additionalProperties"] is False
    assert DiagnosticConfig().assessment_mode == "stage"


@pytest.mark.parametrize("mutation", ["duplicate_id", "invalid_score"])
def test_ranking_identity_and_score_validation(execution, mutation):
    if mutation == "duplicate_id":
        execution["deep_retrieval"].append(dict(execution["deep_retrieval"][0]))
    else:
        execution["deep_retrieval"][0]["score"] = "high"
    artifacts, _ = reconstruct(execution)
    raw = assessment("TOP_K_EXCLUSION", ["retrieval.manifest", "deep.chunk.0"])
    assert validate(raw, artifacts, execution, 0.7)[5]["status"] == "UNKNOWN"

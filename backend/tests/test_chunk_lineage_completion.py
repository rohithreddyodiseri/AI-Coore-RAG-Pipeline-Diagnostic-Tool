"""Structural completion preserves semantic judge hypotheses and attribution gates."""

from copy import deepcopy

import pytest

from rag_diagnostic.mechanism_cases import build_cases
from rag_diagnostic.mechanisms import MECHANISM_STAGE
from rag_diagnostic.stage_diagnosis import reconstruct, unknown, validate


def trace_and_assessment(status="FAULT"):
    case = next(c for c in build_cases() if c["description"] == "CHUNK_FRAGMENTATION")
    rows = [unknown(m, "insufficient evidence") for m in MECHANISM_STAGE]
    row = next(r for r in rows if r["mechanism"] == "CHUNK_FRAGMENTATION")
    row.update(
        status=status,
        confidence=0.92,
        designation="primary" if status == "FAULT" else None,
        rationale="The contrast qualification is separated from the main statement.",
        evidence=[dict(artifact_id=f"chunking.chunk.{i}", span_id=0) for i in range(2)],
        unresolved_questions=[],
    )
    return deepcopy(case["trace"]), {"evaluations": rows}


def test_links_verified_lineage_without_changing_semantic_confidence():
    trace, raw = trace_and_assessment()
    original = deepcopy(raw)
    artifacts, _ = reconstruct(trace)
    row = validate(raw, artifacts, trace, 0.7)[2]
    assert row["status"] == "FAULT" and row["designation"] == "primary"
    assert row["confidence"] == 0.92
    assert {c["artifact_id"] for c in row["derived_evidence"]} == {
        "boundary.parsing",
        "boundary.chunking",
    }
    assert raw == original


@pytest.mark.parametrize(
    "broken", ["offset", "document", "section", "missing_boundary", "single_chunk"]
)
def test_broken_lineage_still_abstains(broken):
    trace, raw = trace_and_assessment()
    chunk = trace["stages"]["chunking"]["output_data"][1]
    if broken == "offset":
        chunk["metadata"]["start_offset"] += 1
    elif broken == "document":
        chunk["doc_id"] = "unobserved-doc"
    elif broken == "section":
        chunk["metadata"]["section"] = 99
    elif broken == "missing_boundary":
        trace["stages"].pop("parsing")
    else:
        raw["evaluations"][2]["evidence"].pop()
    artifacts, _ = reconstruct(trace)
    row = validate(raw, artifacts, trace, 0.7)[2]
    assert row["status"] == "UNKNOWN" and "derived_evidence" not in row


def test_structural_split_never_creates_a_semantic_fault():
    trace, raw = trace_and_assessment("UNKNOWN")
    artifacts, _ = reconstruct(trace)
    row = validate(raw, artifacts, trace, 0.7)[2]
    assert row["status"] == "UNKNOWN"

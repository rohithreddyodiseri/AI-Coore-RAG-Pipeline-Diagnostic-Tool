"""Controlled boundary interventions; labels are never part of execution telemetry."""

import json
from copy import deepcopy
from hashlib import sha256

from rag_diagnostic.mechanisms import MECHANISM_STAGE

METHOD = "The Aster study improved retrieval using hybrid search and a cross-encoder reranker."
LIMIT = "The method worked on observed domains but failed to generalize to unseen domains."
SOURCE = METHOD + " " + LIMIT
NOISE = "The archive stores monthly publication dates."
QUERY = "Which methods improved retrieval in Aster, and what generalization limitation remained?"


def _base():
    docs = [{"doc_id": "d1", "text": SOURCE}, {"doc_id": "d2", "text": NOISE}]
    chunks = [
        dict(
            chunk_id=f"c{i}",
            doc_id=d["doc_id"],
            text=d["text"],
            metadata=dict(section=0, chunk=0, start_offset=0, end_offset=len(d["text"])),
        )
        for i, d in enumerate(docs, 1)
    ]
    configuration = dict(
        model="encoder-a", revision="r1", dimension=2, preprocessing="identity", normalized=True
    )
    return dict(
        schema_version="1.0",
        query=QUERY,
        execution_status="completed",
        failure_signal={"flagged": True, "reason": "quality review"},
        corpus=dict(
            complete=True,
            snapshot_id="aster-v1",
            scope="Two captured Aster source records",
            documents=docs,
        ),
        generation_instructions=(
            "Use only the supplied context. State uncertainty if facts are absent."
        ),
        stages={
            "parsing": dict(
                status="completed",
                complete=True,
                output_data=[
                    dict(text=d["text"], metadata=dict(doc_id=d["doc_id"], section=0)) for d in docs
                ],
            ),
            "chunking": dict(status="completed", complete=True, output_data=chunks),
            "document_embedding": dict(
                status="completed",
                configuration=configuration,
                output_data=[dict(chunk_id=c["chunk_id"], vector=[1.0, 0.0]) for c in chunks],
            ),
            "query_embedding": dict(
                status="completed", configuration=deepcopy(configuration), output_data=[[1.0, 0.0]]
            ),
            "indexing": dict(
                status="completed", output_data=dict(complete=True, indexed_ids=["c1", "c2"])
            ),
            "reranking": dict(status="not_used"),
        },
    )


def _finish(trace, retrieved, selected=None, assembled=None, delivered=None, answer=None):
    stages = trace["stages"]
    stages["retrieval"] = dict(
        status="completed", k=len(retrieved) or 1, output_data=deepcopy(retrieved)
    )
    selected = retrieved if selected is None else selected
    stages["selection"] = dict(
        status="completed",
        output_data=dict(
            complete=True,
            decisions=[
                dict(
                    chunk_id=c["chunk_id"],
                    selected=c in selected,
                    score=c["score"],
                    reason="included" if c in selected else "unspecified",
                )
                for c in retrieved
            ],
        ),
    )
    text = "\n\n".join(c["text"] for c in selected) if assembled is None else assembled
    stages["context_assembly"] = dict(
        status="completed",
        pre_truncation=text,
        post_truncation=text,
        output_data=dict(text=text, chunk_ids=[c["chunk_id"] for c in selected]),
    )
    delivered = text if delivered is None else delivered
    stages["generation"] = dict(
        status="completed",
        input_data=dict(
            prompt=f"Question:\n{trace['query']}\n\nContext:\n{delivered}",
            delivered_context=delivered,
            system_instruction=trace["generation_instructions"],
            model="captured-generator",
        ),
    )
    trace["final_answer"] = SOURCE if answer is None else answer


def _candidate(chunk, rank=1, score=0.9):
    return dict(
        chunk_id=chunk["chunk_id"],
        doc_id=chunk["doc_id"],
        text=chunk["text"],
        rank=rank,
        score=score,
    )


def _case(mechanism):
    t = _base()
    s = t["stages"]
    target, noise = [
        _candidate(c, i, 0.9 if i == 1 else 0.3)
        for i, c in enumerate(s["chunking"]["output_data"], 1)
    ]
    retrieved, selected, assembled, delivered, answer = [target], None, None, None, SOURCE
    if mechanism == "MISSING_SOURCE_INFORMATION":
        t["corpus"]["documents"][0]["text"] = NOISE
        s["parsing"]["output_data"][0]["text"] = NOISE
        s["chunking"]["output_data"][0].update(text=NOISE)
        s["chunking"]["output_data"][0]["metadata"]["end_offset"] = len(NOISE)
        retrieved, answer = (
            [noise],
            "The supplied records do not specify the requested methods or limitation.",
        )
    elif mechanism == "EXTRACTION_LOSS":
        s["parsing"]["output_data"][0]["text"] = METHOD
        s["chunking"]["output_data"][0]["text"] = METHOD
        s["chunking"]["output_data"][0]["metadata"]["end_offset"] = len(METHOD)
        target["text"] = METHOD
        answer = METHOD + " The limitation is not available in context."
    elif mechanism == "CHUNK_FRAGMENTATION":
        split = SOURCE.index(" but ")
        pieces = [SOURCE[:split], SOURCE[split:]]
        chunks = []
        start = 0
        for i, text in enumerate(pieces):
            chunks.append(
                dict(
                    chunk_id=f"part{i}",
                    doc_id="d1",
                    text=text,
                    metadata=dict(
                        section=0, chunk=i, start_offset=start, end_offset=start + len(text)
                    ),
                )
            )
            start += len(text)
        s["chunking"]["output_data"] = chunks
        s["document_embedding"]["output_data"] = [
            dict(chunk_id=c["chunk_id"], vector=[1.0, 0.0]) for c in chunks
        ]
        s["indexing"]["output_data"]["indexed_ids"] = [c["chunk_id"] for c in chunks]
        retrieved = [_candidate(c, i + 1) for i, c in enumerate(chunks)]
        selected = retrieved[:1]
        answer = METHOD + " It worked on observed domains."
    elif mechanism == "EMBEDDING_MISMATCH":
        s["query_embedding"]["configuration"]["model"] = "encoder-b"
        retrieved, answer = [noise], "The retrieved archive record does not answer the question."
    elif mechanism == "INDEXING_FAILURE":
        s["indexing"]["output_data"]["indexed_ids"] = ["c2"]
        retrieved, answer = [noise], "No study evidence was retrieved."
    elif mechanism == "TOP_K_EXCLUSION":
        noise.update(rank=1, score=0.95)
        target.update(rank=2, score=0.85)
        t["deep_retrieval"] = [noise, target]
        t["deep_k"] = 2
        retrieved, answer = [noise], "No methods or limitations appear in context."
    elif mechanism == "RERANKING_DEGRADATION":
        retrieved, selected, answer = (
            [target, noise],
            [noise],
            "No methods or limitations appear in context.",
        )
        s["reranking"] = dict(
            status="completed", input_data=deepcopy(retrieved), output_data=[dict(noise, rank=1)]
        )
    elif mechanism in ("SIMILARITY_CUTOFF_EXCLUSION", "TOKEN_BUDGET_EXCLUSION"):
        target["score"] = 0.61
        noise.update(score=0.9, rank=1)
        target["rank"] = 2
        retrieved, selected, answer = (
            [noise, target],
            [noise],
            "No study evidence appears in context.",
        )
    elif mechanism == "CONTEXT_TRUNCATION":
        assembled, answer = METHOD, METHOD + " The limitation is not available in context."
    elif mechanism == "CONTEXT_DELIVERY_LOSS":
        delivered, answer = NOISE, "No study evidence appears in context."
    elif mechanism == "INCORRECT_EVIDENCE_EXTRACTION":
        answer = (
            "The study used keyword search alone and generalized perfectly to all unseen domains."
        )
    elif mechanism == "PARTIAL_EVIDENCE_EXTRACTION":
        answer = METHOD
    elif mechanism == "OUTPUT_FORMAT_VIOLATION":
        t["query"] += '\nReturn only valid JSON with a single key "answer".'
        answer = "```json\n" + json.dumps({"answer": SOURCE}) + "\n```"
    elif mechanism == "SPECIFICITY_MISMATCH":
        t["query"] = (
            "Name the two exact retrieval methods used in Aster; "
            "give their technical names, not generic descriptions."
        )
        answer = "It used some advanced techniques to find things more effectively."
    _finish(t, retrieved, selected, assembled, delivered, answer)
    if mechanism in ("SIMILARITY_CUTOFF_EXCLUSION", "TOKEN_BUDGET_EXCLUSION"):
        decision = next(
            d for d in s["selection"]["output_data"]["decisions"] if d["chunk_id"] == "c1"
        )
        if mechanism.startswith("SIMILARITY"):
            decision.update(reason="similarity_threshold", threshold=0.7)
        else:
            decision.update(reason="token_budget", token_count=60, used_tokens=70, budget=100)
    if mechanism == "CONTEXT_TRUNCATION":
        s["context_assembly"]["pre_truncation"] = SOURCE
    if mechanism == "UNCLASSIFIED_OPERATIONAL_FAILURE":
        s["generation"] = dict(
            status="failed",
            input_data=s["generation"]["input_data"],
            error=dict(type="TransportError", message="Generator connection closed"),
        )
        t.pop("final_answer")
        t["execution_status"] = "failed"
    return t


def build_cases():
    """Return reproducible observations and separate evaluation-only expectations."""
    cases = []
    for mechanism, stage in MECHANISM_STAGE.items():
        cases.append(
            dict(
                trace=_case(mechanism),
                kind="fault",
                description=mechanism,
                expected_root=dict(
                    mechanism=mechanism,
                    technical_stage="generation" if stage == "operational" else stage,
                ),
                expected_faults=[mechanism],
                target_statuses={mechanism: "FAULT"},
            )
        )
    multi = _case("EMBEDDING_MISMATCH")
    multi["query"] += '\nReturn only valid JSON with a single key "answer".'
    multi["stages"]["generation"]["input_data"]["prompt"] = (
        f"Question:\n{multi['query']}\nContext:\n{NOISE}"
    )
    cases.append(
        dict(
            trace=multi,
            kind="multi_fault",
            description="Embedding mismatch plus format violation",
            expected_root=dict(mechanism="EMBEDDING_MISMATCH", technical_stage="embedding"),
            expected_faults=["EMBEDDING_MISMATCH", "OUTPUT_FORMAT_VIOLATION"],
            target_statuses={"EMBEDDING_MISMATCH": "FAULT", "OUTPUT_FORMAT_VIOLATION": "FAULT"},
        )
    )
    compliant = _case("OUTPUT_FORMAT_VIOLATION")
    compliant["final_answer"] = json.dumps({"answer": SOURCE})
    cases.append(
        dict(
            trace=compliant,
            kind="control",
            description="Compliant JSON response",
            expected_root=None,
            expected_faults=[],
            target_statuses={"OUTPUT_FORMAT_VIOLATION": "CLEAR"},
        )
    )
    withheld = _case("TOP_K_EXCLUSION")
    withheld.pop("deep_retrieval")
    cases.append(
        dict(
            trace=withheld,
            kind="control",
            description="Unobserved deeper retrieval must abstain",
            expected_root=None,
            expected_faults=[],
            target_statuses={"TOP_K_EXCLUSION": "UNKNOWN"},
        )
    )
    for i, case in enumerate(cases):
        # Opaque IDs prevent ground-truth labels leaking through telemetry identifiers.
        identity = sha256(f"mechanism-suite-v1:{i}".encode()).hexdigest()[:24]
        case["trace"]["run_id"] = identity
    return cases

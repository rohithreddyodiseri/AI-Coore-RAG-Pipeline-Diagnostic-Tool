"""Seven-category LLM attribution, evidence guards, and recorded replay."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from conftest import FACT, FakeClient, RecordedTarget, cite, report, sample_trace

from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnose import Diagnoser
from rag_diagnostic.diagnostic_engine import DiagnosticEngine, trace_artifacts
from rag_diagnostic.judge import LLMJudge, UnavailableJudge


@pytest.mark.parametrize(
    "code,stage",
    [
        ("FP1", "corpus"),
        ("FP2", "retrieval"),
        ("FP3", "context_assembly"),
        ("FP4", "generation"),
        ("FP5", "generation"),
        ("FP6", "generation"),
        ("FP7", "generation"),
    ],
)
def test_all_seven_categories_are_attributed_by_llm(make_engine, trace, code, stage):
    if code == "FP1":
        trace["stages"]["retrieval"]["output_data"] = []
        trace["stages"]["reranking"]["output_data"] = []
        trace["stages"]["context_assembly"]["output_data"] = ""
        trace["corpus"] = {
            "documents": [],
            "complete": True,
            "snapshot_id": "empty-v1",
            "scope": "entire test knowledge base",
        }
        keys = ["query", "corpus.manifest"]
    elif code == "FP2":
        trace["stages"]["retrieval"]["output_data"] = []
        trace["deep_retrieval"] = [{"chunk_id": "d:0", "doc_id": "d", "text": FACT, "rank": 2}]
        keys = ["retrieval.manifest", "deep.chunk.0"]
    elif code == "FP3":
        trace["stages"]["context_assembly"]["output_data"] = ""
        keys = ["retrieval.chunk.0", "context.manifest"]
    elif code in ("FP4", "FP7"):
        keys = ["query", "context.text", "generation.answer"]
    else:
        keys = ["query", "generation.answer"]
    engine, client = make_engine(report(trace, [(code, stage, keys)]))
    verdict = engine.diagnose(trace)
    assert verdict.status == "diagnosed", verdict.render()
    assert verdict.failure_category == code
    assert verdict.primary_fault_stage == stage
    assert verdict.failure_categories == (code,)
    assert verdict.confidence == 0.93
    assert len(verdict.evaluations) == 7
    assert not verdict.attribution_complete
    assert len(verdict.unresolved_categories) == 6
    assert client.calls == 1
    assert client.requests[0]["config"]["response_mime_type"] == "application/json"
    assert "reference_answer" not in client.requests[0]["contents"]


def test_format_can_fail_without_corpus_or_answer_evidence(make_engine, trace):
    trace.pop("stages")
    engine, _ = make_engine(report(trace, [("FP5", "generation", ["query", "generation.answer"])]))
    verdict = engine.diagnose(trace)
    assert verdict.failure_category == "FP5"
    assert verdict.status == "diagnosed"


def test_multiple_categories_and_stages(make_engine, trace):
    trace["stages"]["retrieval"]["output_data"] = []
    trace["deep_retrieval"] = [
        {"chunk_id": "source:0", "doc_id": "source", "text": FACT, "rank": 2}
    ]
    reply = report(
        trace,
        [
            ("FP2", "retrieval", ["retrieval.manifest", "deep.chunk.0"]),
            ("FP5", "generation", ["query", "generation.answer"]),
        ],
    )
    engine, _ = make_engine(reply)
    verdict = engine.diagnose(trace)
    assert verdict.failure_categories == ("FP2", "FP5")
    assert verdict.stages == ("retrieval", "generation")
    assert verdict.contributing_stages == ("generation",)
    assert len(verdict.attributions) == 2


def test_reranking_contributor_has_separate_evidence(make_engine, trace):
    trace["stages"]["context_assembly"]["output_data"] = ""
    reply = report(trace, [("FP3", "context_assembly", ["retrieval.chunk.0", "context.manifest"])])
    row = reply["evaluations"][2]
    row["contributing_stages"] = [
        {
            "stage": "reranking",
            "rationale": "Selection worsened loss.",
            "evidence": [cite(trace_artifacts(trace), "reranking.manifest")],
        }
    ]
    engine, _ = make_engine(reply)
    verdict = engine.diagnose(trace)
    assert verdict.stages == ("reranking", "context_assembly")
    assert verdict.contributing_stages == ("reranking",)


def test_topk_absence_does_not_prove_missing_corpus_content(make_engine, trace):
    engine, _ = make_engine(report(trace, [("FP1", "corpus", ["query", "retrieval.chunk.0"])]))
    verdict = engine.diagnose(trace)
    assert verdict.status == "abstained"
    assert "complete scoped corpus" in verdict.evaluations[0]["rationale"]


def test_incomplete_corpus_snapshot_cannot_establish_fp1(make_engine, trace):
    trace["corpus"] = {
        "documents": [{"doc_id": "partial", "text": "Unrelated document."}],
        "complete": False,
    }
    engine, _ = make_engine(report(trace, [("FP1", "corpus", ["query", "corpus.manifest"])]))
    assert engine.diagnose(trace).status == "abstained"


def test_fp2_needs_evidence_outside_cutoff(make_engine, trace):
    trace["deep_retrieval"] = deepcopy(trace["stages"]["retrieval"]["output_data"])
    engine, _ = make_engine(
        report(trace, [("FP2", "retrieval", ["retrieval.manifest", "deep.chunk.0"])])
    )
    assert engine.diagnose(trace).status == "abstained"


@pytest.mark.parametrize("code", ["FP4", "FP7"])
def test_missing_context_cannot_blame_extraction(make_engine, trace, code):
    trace["stages"].pop("context_assembly")
    engine, _ = make_engine(report(trace, [(code, "generation", ["query", "generation.answer"])]))
    verdict = engine.diagnose(trace)
    assert verdict.status == "abstained"
    assert "context" in verdict.evaluations[int(code[-1]) - 1]["rationale"]


@pytest.mark.parametrize("mutation", ["quote", "artifact", "stage", "confidence", "contributor"])
def test_invalid_evidence_never_becomes_a_verdict(make_engine, trace, mutation):
    reply = report(trace, [("FP5", "generation", ["query", "generation.answer"])])
    row = reply["evaluations"][4]
    if mutation == "quote":
        row["evidence"][0]["quote"] = "Invented format instruction."
    elif mutation == "artifact":
        row["evidence"][0]["artifact_id"] = "not-present"
    elif mutation == "stage":
        row["stage"] = "retrieval"
    elif mutation == "confidence":
        row["confidence"] = 0.69
    else:
        row["contributing_stages"] = [
            {
                "stage": "retrieval",
                "rationale": "unsupported",
                "evidence": [cite(trace_artifacts(trace), "query")],
            }
        ]
    engine, _ = make_engine(reply)
    assert engine.diagnose(trace).status == "abstained"


def test_confident_none_requires_complete_matrix(make_engine, trace):
    engine, _ = make_engine(report(trace, clear=True))
    verdict = engine.diagnose(trace)
    assert verdict.status == "no_failure"
    assert verdict.failure_category == "NONE"
    assert verdict.attribution_complete
    assert verdict.stages == ()


def test_unobserved_context_cannot_be_reported_clear(make_engine, trace):
    reply = report(trace, clear=True)
    trace["stages"].pop("context_assembly")
    # Replace invalid context quotes with observed query quotes: still cannot claim a clear context.
    for row in reply["evaluations"]:
        row["evidence"] = [cite(trace_artifacts(trace), "query")]
    engine, _ = make_engine(reply)
    verdict = engine.diagnose(trace)
    assert verdict.status == "abstained"
    assert "FP7" in verdict.unresolved_categories


@pytest.mark.parametrize(
    "reply", ["not json", "{}", "[]", '{"confidence":true}', '{"failure_category":"FP8"}']
)
def test_malformed_reply_abstains_and_is_preserved(make_engine, trace, reply):
    engine, _ = make_engine(reply)
    verdict = engine.diagnose(trace)
    assert verdict.status == "abstained"
    assert verdict.observations["judge_response"]["raw_output"] == reply


def test_duplicate_or_omitted_categories_abstain(make_engine, trace):
    reply = report(trace)
    reply["evaluations"][6] = deepcopy(reply["evaluations"][0])
    engine, _ = make_engine(reply)
    assert engine.diagnose(trace).status == "abstained"


def test_overall_low_confidence_forces_abstention(make_engine, trace):
    reply = report(trace, [("FP5", "generation", ["query", "generation.answer"])])
    reply["confidence"] = 0.5
    engine, _ = make_engine(reply)
    assert engine.diagnose(trace).status == "abstained"


def test_same_trace_reuses_llm_result_and_logs_each_run(make_engine, trace):
    engine, client = make_engine(
        report(trace, [("FP7", "generation", ["query", "context.text", "generation.answer"])])
    )
    first, second = engine.diagnose(trace), engine.diagnose(trace)
    assert client.calls == 1
    assert first.to_dict(False) == second.to_dict(False)
    log_path = Path(engine.config.runs_dir) / "diagnoses.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["run"]["code_revision"]["commit"] for row in rows)
    assert rows[0]["observations"]["input"]["source_sha256"]
    reloaded = DiagnosticEngine(
        engine.config,
        LLMJudge(
            engine.config.judge_model, engine.config.judge_cache, FakeClient("must not be called")
        ),
    )
    assert reloaded.diagnose(trace).to_dict(False) == first.to_dict(False)


def test_changed_artifacts_invalidate_cache(make_engine, trace):
    engine, client = make_engine(report(trace))
    engine.diagnose(trace)
    trace["final_answer"] = "A different recorded output."
    engine.diagnose(trace)
    assert client.calls == 2


def test_input_budget_abstains_without_silent_truncation(make_engine, trace):
    engine, client = make_engine(report(trace), max_judge_input_chars=10)
    verdict = engine.diagnose(trace)
    assert verdict.status == "abstained"
    assert client.calls == 0
    assert "silently truncated" in verdict.explanation[0]


def test_incomplete_llm_response_abstains(make_engine, trace):
    engine, client = make_engine(report(trace, clear=True))
    client.status = "incomplete"
    assert engine.diagnose(trace).status == "abstained"


def test_pipeline_adapter_uses_trace_judge_without_corpus_or_gold(tmp_path):
    class TraceOnlyPipeline(RecordedTarget):
        def snapshot_id(self):
            raise AssertionError("Corpus snapshot not needed for format judgments")

        def document_ids(self):
            raise AssertionError("Corpus inspection was not configured")

    config = DiagnosticConfig(
        runs_dir=str(tmp_path / "runs"),
        judge_cache=str(tmp_path / "judge.json"),
        inspect_corpus=False,
    )

    def reply(request):
        artifacts = json.loads(request["contents"])["artifacts"]
        output = report(sample_trace())
        output["failure_category"], output["primary_fault_stage"] = "FP5", "generation"
        output["evaluations"][4] = {
            "category": "FP5",
            "status": "fault",
            "stage": "generation",
            "confidence": 0.93,
            "rationale": "JSON was explicitly requested but the response is prose.",
            "evidence": [cite(artifacts, "query"), cite(artifacts, "generation.answer")],
            "contributing_stages": [],
        }
        return output

    client = FakeClient(reply)
    diagnoser = Diagnoser(config, LLMJudge(config.judge_model, config.judge_cache, client))
    query = "Who built the tower? Return JSON."
    verdict = diagnoser.diagnose(query, TraceOnlyPipeline())
    assert verdict.failure_category == "FP5"
    assert "deep.manifest" in verdict.observations["input"]["artifacts"]


def test_unavailable_judge_abstains(tmp_path, trace):
    engine = DiagnosticEngine(DiagnosticConfig(runs_dir=str(tmp_path)), UnavailableJudge())
    assert engine.diagnose(trace).status == "abstained"


def test_pipeline_capture_failure_is_recorded(tmp_path):
    class Broken(RecordedTarget):
        def run(self, query):
            raise RuntimeError("pipeline unavailable")

    config = DiagnosticConfig(runs_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="pipeline unavailable"):
        Diagnoser(config, UnavailableJudge()).diagnose("Why?", Broken())
    record = json.loads((tmp_path / "diagnoses.jsonl").read_text())
    assert record["status"] == "error"
    assert record["run"]["code_revision"]


def test_api_failure_is_recorded_as_operational_error(make_engine, trace):
    def failure(request):
        raise RuntimeError("mock API unavailable")

    engine, _ = make_engine(failure)
    with pytest.raises(RuntimeError, match="Gemini request failed"):
        engine.diagnose(trace)
    record = json.loads((Path(engine.config.runs_dir) / "diagnoses.jsonl").read_text())
    assert record["status"] == "error"
    assert record["trace"] == trace


def test_real_adapter_corpus_inspection_declares_scope(tmp_path):
    config = DiagnosticConfig(
        inspect_corpus=True,
        runs_dir=str(tmp_path / "runs"),
        judge_cache=str(tmp_path / "judge.json"),
    )
    observed = []

    def reply(request):
        artifacts = json.loads(request["contents"])["artifacts"]
        observed.append(artifacts)
        result = report(sample_trace())
        result["failure_category"], result["primary_fault_stage"] = "FP1", "corpus"
        result["evaluations"][0] = {
            "category": "FP1",
            "status": "fault",
            "stage": "corpus",
            "confidence": 0.93,
            "rationale": "The scoped knowledge base has no ferry information.",
            "evidence": [cite(artifacts, "query"), cite(artifacts, "corpus.manifest")],
            "contributing_stages": [],
        }
        return result

    client = FakeClient(reply)
    diagnoser = Diagnoser(config, LLMJudge(config.judge_model, config.judge_cache, client))
    verdict = diagnoser.diagnose("When did the ferry first sail?", RecordedTarget())
    assert verdict.failure_category == "FP1"
    assert observed[0]["corpus.manifest"]["complete"] is True


def test_paraphrased_answer_does_not_need_string_match(make_engine, trace):
    trace["final_answer"] = '{"builder":"Ada","year":1902}'
    engine, _ = make_engine(report(trace, clear=True))
    assert engine.diagnose(trace).status == "no_failure"


def test_large_corpus_is_retained_but_partial_model_view_cannot_claim_fp1(make_engine, trace):
    trace["corpus"] = {
        "documents": [
            {"doc_id": "a", "text": "Unrelated text. " * 100},
            {"doc_id": "d", "text": FACT},
        ],
        "complete": True,
        "snapshot_id": "full-source-v1",
        "scope": "complete test corpus",
    }
    reply = report(trace, [("FP1", "corpus", ["query", "corpus.manifest"])])

    # The script intentionally claims complete coverage on a bounded view; guard must reject it.
    def assess(request):
        artifacts = json.loads(request["contents"])["artifacts"]
        reply["evaluations"][0]["evidence"] = [
            cite(artifacts, "query"),
            cite(artifacts, "corpus.manifest"),
        ]
        return reply

    engine, _ = make_engine(assess, corpus_budget_chars=len(FACT) + 10)
    verdict = engine.diagnose(trace)
    assert verdict.status == "abstained"
    original = verdict.observations["trace"]["corpus"]
    assert original["complete"] is True and len(original["documents"][0]["text"]) > 1000
    artifacts = verdict.observations["input"]["artifacts"]
    assert artifacts["corpus.manifest"]["complete"] is False
    assert artifacts["corpus.manifest"]["total_documents"] == 2
    assert artifacts["corpus.document.0"]["doc_id"] == "d"
    assert artifacts["corpus.manifest"]["truncated_document_ids"] == ["a"]


def test_stage_inputs_and_unused_stages_are_visible_to_judge(make_engine, trace):
    trace["stages"]["retrieval"]["input_data"] = {"query": trace["query"], "k": 1}
    trace["stages"]["reranking"] = {"status": "not_used"}
    trace["stages"]["generation"] = {"input_data": {"prompt": "Exact generation prompt."}}
    engine, _ = make_engine(report(trace))
    verdict = engine.diagnose(trace)
    artifacts = verdict.observations["input"]["artifacts"]
    assert artifacts["generation.prompt"]["text"] == "Exact generation prompt."
    assert "retrieval.input" in artifacts
    assert artifacts["reranking.status"]["status"] == "not_used"
    assert "reranking.manifest" not in artifacts

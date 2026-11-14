import json
from copy import deepcopy

import pytest
from conftest import FACT, FakeClient, report

from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnostic_engine import trace_artifacts
from rag_diagnostic.evaluate import evaluate_traces, render_report, split_executions
from rag_diagnostic.judge import LLMJudge, build_judge
from rag_diagnostic.labels import load_labels
from rag_diagnostic.trace import Context, QueryTrace, ScoredChunk


def test_default_is_llm_as_judge(tmp_path):
    config = DiagnosticConfig(judge_cache=str(tmp_path / "cache.json"))
    assert isinstance(build_judge(config), LLMJudge)


@pytest.mark.parametrize("key", ["reference_answer", "gold", "expected_answer", "fault_type"])
def test_nested_trace_oracles_are_rejected(trace, key):
    trace["metadata"] = {key: "leaked benchmark answer"}
    with pytest.raises(ValueError, match="reference answers"):
        trace_artifacts(trace)


def test_native_trace_and_standard_trace_preserve_observed_text(trace):
    chunk = ScoredChunk("d:0", "d", FACT, 0.9, 1)
    native = QueryTrace(
        trace["query"], (chunk,), (chunk,), True, Context(("d:0",), FACT), trace["final_answer"]
    )
    artifacts = trace_artifacts(native.to_dict())
    assert artifacts["context.text"]["text"] == FACT
    assert artifacts["retrieval.chunk.0"]["text"] == FACT
    assert artifacts["generation.answer"]["text"] == trace["final_answer"]


def test_missing_and_empty_are_distinct(trace):
    missing = deepcopy(trace)
    missing["stages"].pop("retrieval")
    empty = deepcopy(trace)
    empty["stages"]["retrieval"]["output_data"] = []
    assert "retrieval.manifest" not in trace_artifacts(missing)
    assert trace_artifacts(empty)["retrieval.manifest"]["count"] == 0


@pytest.mark.parametrize(
    "change", ["missing_scope", "missing_snapshot", "duplicate_ids", "not_bool"]
)
def test_complete_corpus_contract(trace, change):
    trace["corpus"] = {"documents": [], "complete": True, "scope": "test KB", "snapshot_id": "v1"}
    if change == "missing_scope":
        trace["corpus"].pop("scope")
    elif change == "missing_snapshot":
        trace["corpus"].pop("snapshot_id")
    elif change == "not_bool":
        trace["corpus"]["complete"] = "true"
    else:
        trace["corpus"]["documents"] = [{"doc_id": "d", "text": "text"}] * 2
    with pytest.raises(ValueError):
        trace_artifacts(trace)


def test_split_is_deterministic():
    executions = [{"run_id": str(i), "query": f"Query {i}"} for i in range(20)]
    first = split_executions(executions, 0.3, "seed")
    second = split_executions(list(reversed(executions)), 0.3, "seed")
    assert {row["run_id"] for row in first[1]} == {row["run_id"] for row in second[1]}


def test_evaluation_scores_category_and_stage(make_engine, trace):
    from rag_diagnostic.labels import ExecutionLabel

    engine, _ = make_engine(report(trace, [("FP5", "generation", ["query", "generation.answer"])]))
    trace["run_id"] = "format"
    labels = {
        "format": ExecutionLabel(
            "format", frozenset({"generation"}), "wrong_format", frozenset({"FP5"})
        )
    }
    scored = evaluate_traces([trace], labels, engine)
    assert scored["accuracy_overall"] == 1
    labels["format"] = ExecutionLabel(
        "format", frozenset({"generation"}), "wrong_label", frozenset({"FP7"})
    )
    scored = evaluate_traces([trace], labels, engine)
    assert scored["accuracy_overall"] == 0
    assert scored["incorrect_runs"][0]["true_categories"] == "FP7"
    assert "| Fault type |" in render_report(scored)


def test_label_categories_are_optional(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text('{"run_id":"a","stages":[],"failure_categories":[]}\n')
    assert load_labels(path)["a"].categories == frozenset()


def test_cache_identity_includes_model_prompt_and_schema(tmp_path, trace):
    cache = tmp_path / "judge.json"
    client = FakeClient(report(trace))
    first = LLMJudge("pinned-v1", cache, client)
    first.evaluate_trace({"trace": trace})
    first.evaluate_trace({"trace": trace})
    assert client.calls == 1
    second = LLMJudge("pinned-v2", cache, client)
    second.evaluate_trace({"trace": trace})
    assert client.calls == 2
    second.VERSION += 1
    second.evaluate_trace({"trace": trace})
    assert client.calls == 3


@pytest.mark.parametrize(
    "settings",
    [
        {"deep_k": 0},
        {"abstain_below": 1.1},
        {"deep_k": 1.2},
        {"inspect_corpus": "yes"},
        {"judge_max_output_tokens": 0},
        {"max_judge_input_chars": True},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        DiagnosticConfig(**settings)


def test_removed_matching_config_is_rejected(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"match_threshold":0.5}')
    with pytest.raises(ValueError):
        DiagnosticConfig.from_file(path)


def test_trace_evaluation_does_not_send_labels_to_judge(make_engine, trace):
    from rag_diagnostic.evaluate import evaluate_traces
    from rag_diagnostic.labels import ExecutionLabel

    trace["run_id"] = "case-1"
    engine, client = make_engine(
        report(trace, [("FP5", "generation", ["query", "generation.answer"])])
    )
    labels = {
        "case-1": ExecutionLabel(
            "case-1", frozenset({"generation"}), "SECRET_LABEL", frozenset({"FP5"})
        )
    }
    scored = evaluate_traces([trace], labels, engine)
    assert scored["accuracy_overall"] == 1
    assert "SECRET_LABEL" not in client.requests[0]["contents"]


def test_duplicate_trace_ids_are_rejected(make_engine, trace):
    from rag_diagnostic.evaluate import evaluate_traces
    from rag_diagnostic.labels import ExecutionLabel

    trace["run_id"] = "case-1"
    engine, _ = make_engine(report(trace))
    with pytest.raises(ValueError, match="unique"):
        evaluate_traces(
            [trace, trace], {"case-1": ExecutionLabel("case-1", frozenset(), "none")}, engine
        )


def test_unknown_category_labels_are_rejected():
    from rag_diagnostic.labels import ExecutionLabel

    with pytest.raises(ValueError):
        ExecutionLabel.from_dict(
            {"run_id": "a", "stages": ["generation"], "failure_categories": ["FP8"]}
        )


def test_original_pipeline_settings_are_recorded(make_engine, trace):
    trace["pipeline"] = {
        "name": "external",
        "revision": "external-commit",
        "settings": {"top_k": 1},
    }
    engine, client = make_engine(report(trace))
    verdict = engine.diagnose(trace)
    assert verdict.run.pipeline == trace["pipeline"]
    assert json.loads(client.requests[0]["contents"])["pipeline"] == trace["pipeline"]

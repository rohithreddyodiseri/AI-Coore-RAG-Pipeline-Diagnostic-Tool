import json
from pathlib import Path

import pytest

from rag_diagnostic.adapters.toy import build
from rag_diagnostic.cases import DiagnosticCase, load_cases, load_labels
from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnose import Diagnoser
from rag_diagnostic.evaluate import evaluate, render_report, split_cases
from rag_diagnostic.judge import LLMJudge, ReferenceRecallJudge
from rag_diagnostic.locate import coverage, locate_in_chunks
from rag_diagnostic.trace import Chunk

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_coverage_ignores_case_whitespace_and_punctuation():
    assert coverage("THE mirror is 2.4   meters, wide!", "the mirror is 2.4 meters wide") == 1.0
    assert coverage("nothing related here", "the mirror is 2.4 meters wide") == 0.0
    assert coverage(None, "anything") == 0.0


def test_locate_detects_split_evidence():
    chunks = [
        Chunk("a", "d", "Port Velra. The Velra lighthouse was built"),
        Chunk("b", "d", "in 1874 by the engineer Mara Quill."),
    ]
    match = locate_in_chunks(
        chunks, "The Velra lighthouse was built in 1874 by the engineer Mara Quill."
    )
    assert match.split(0.9)
    assert match.contributing_ids == ("a", "b")


def test_cases_refuse_label_fields():
    with pytest.raises(ValueError):
        DiagnosticCase.from_dict(
            {
                "case_id": "x",
                "query": "q",
                "reference_answer": "r",
                "gold": [],
                "stages": ["parsing"],
            }
        )


def test_split_is_deterministic_and_stable():
    cases = load_cases(EXAMPLES / "toy_cases.jsonl")
    first = split_cases(cases, 0.3, "seed")
    second = split_cases(list(reversed(cases)), 0.3, "seed")
    assert {c.case_id for c in first[1]} == {c.case_id for c in second[1]}
    assert len(first[0]) + len(first[1]) == len(cases)


def test_evaluation_reports_accuracy_and_abstentions():
    cases = load_cases(EXAMPLES / "toy_cases.jsonl")
    labels = load_labels(EXAMPLES / "toy_labels.jsonl")
    report = evaluate(cases, labels, build, Diagnoser(DiagnosticConfig(), ReferenceRecallJudge()))
    assert report["cases"] == 13
    assert report["abstained"] == 2
    assert report["accuracy_when_answered"] == 1.0
    assert {r["case_id"] for r in report["wrong_cases"]} == {"toy-11", "toy-13"}
    assert "| Fault type |" in render_report(report)


def test_reference_recall_judge():
    judge = ReferenceRecallJudge(0.8)
    assert judge.judge("q", "It was built by Mara Quill.", "Mara Quill").correct
    assert not judge.judge("q", "I do not know.", "Mara Quill").correct


class _FakeResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs["temperature"] == 0

        class Out:
            output_text = json.dumps({"correct": True, "reason": "same fact"})

        return Out()


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


def test_llm_judge_caches_verdicts(tmp_path):
    client = _FakeClient()
    judge = LLMJudge("some-model", tmp_path / "cache.json", client=client)
    assert judge.judge("q", "a", "r").correct
    assert judge.judge("q", "a", "r").correct
    assert client.responses.calls == 1
    reloaded = LLMJudge("some-model", tmp_path / "cache.json", client=_FakeClient())
    assert reloaded.judge("q", "a", "r").correct


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"match_threshhold": 0.5}))
    with pytest.raises(ValueError):
        DiagnosticConfig.from_file(path)

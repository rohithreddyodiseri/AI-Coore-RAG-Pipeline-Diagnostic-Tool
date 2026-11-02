"""Full runs on the practice pipeline. There's one broken stage per test, plus
tests for two broken stages, a missing answer, abstaining, and repeat runs."""

from pathlib import Path

import pytest

from rag_diagnostic.adapters.toy import ToyPipeline
from rag_diagnostic.cases import load_cases, load_labels
from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnose import Diagnoser
from rag_diagnostic.judge import ReferenceRecallJudge

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CASES = {c.case_id: c for c in load_cases(EXAMPLES / "toy_cases.jsonl")}
LABELS = load_labels(EXAMPLES / "toy_labels.jsonl")


@pytest.fixture
def diagnoser():
    return Diagnoser(DiagnosticConfig(), ReferenceRecallJudge())


def run(diagnoser, case_id):
    case = CASES[case_id]
    return diagnoser.diagnose(case, ToyPipeline(case.pipeline))


def test_clean_pipeline_reports_no_failure(diagnoser):
    verdict = run(diagnoser, "toy-01")
    assert verdict.status == "no_failure"
    assert verdict.stages == ()


@pytest.mark.parametrize(
    "case_id",
    ["toy-02", "toy-03", "toy-04", "toy-05", "toy-06", "toy-07", "toy-08", "toy-09", "toy-12"],
)
def test_single_stage_faults_are_attributed(diagnoser, case_id):
    verdict = run(diagnoser, case_id)
    label = LABELS[case_id]
    assert verdict.status == "diagnosed", verdict.render()
    assert set(verdict.stages) == label.stages, verdict.render()
    assert [a.code for a in verdict.attributions] == [label.fault_type]


def test_multi_stage_upstream_plus_generation(diagnoser):
    verdict = run(diagnoser, "toy-10")
    assert set(verdict.stages) == {"indexing", "generation"}
    assert verdict.generation_probe["correct"] is False


def test_abstains_when_fixing_one_stage_is_not_enough(diagnoser):
    # Retrieval and the reranker are both broken. Fixing retrieval alone doesn't get
    # the passage through, so the tool abstains instead of blaming retrieval alone.
    verdict = run(diagnoser, "toy-11")
    assert verdict.status == "abstained"
    assert verdict.suspects == ("retrieval",)
    assert verdict.fix.stage == "retrieval"
    assert verdict.fix.context_recovered is False


def test_abstains_when_it_cannot_tell_parsing_from_a_missing_document(diagnoser):
    verdict = run(diagnoser, "toy-13")
    assert verdict.status == "abstained"
    assert verdict.stages == ()
    assert set(verdict.suspects) == {"corpus", "parsing"}


def test_raising_the_bar_turns_weak_verdicts_into_abstentions():
    strict = Diagnoser(DiagnosticConfig(abstain_below=0.7), ReferenceRecallJudge())
    verdict = run(strict, "toy-09")  # generation_distracted starts at 0.6
    assert verdict.status == "abstained"
    assert verdict.suspects == ("generation",)


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_same_case_twice_gives_same_verdict(diagnoser, case_id):
    first = run(diagnoser, case_id).to_dict(include_run=False)
    second = run(diagnoser, case_id).to_dict(include_run=False)
    assert first == second


def test_verdict_carries_run_record(diagnoser):
    verdict = run(diagnoser, "toy-05")
    assert verdict.run.config_fingerprint == DiagnosticConfig().fingerprint()
    assert verdict.run.pipeline["retrieval"] == "reverse"
    assert "commit" in verdict.run.code_revision

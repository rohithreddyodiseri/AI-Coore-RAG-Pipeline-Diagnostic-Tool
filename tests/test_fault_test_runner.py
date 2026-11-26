import json
from dataclasses import replace

import yaml
from conftest import FakeClient, report
from test_telemetry_integration import MemoryCollection

from gemini_client import GeminiInferenceEngine
from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnostic_engine import trace_artifacts
from rag_diagnostic.fault_test_runner import FAULT_CODES, run_suite
from rag_diagnostic.judge import LLMJudge
from rag_diagnostic.reference_pipeline import RAGPipeline
from rag_pipeline.config import Settings


def test_scoped_absence_and_json_are_checked_from_observed_artifacts(make_engine, trace):
    trace["query"] = "Use only source document paper-9. What happened?"
    trace["corpus"] = {
        "complete": True,
        "scope": "All records for paper-9 in this dataset",
        "snapshot_id": "snapshot",
        "source_document": "paper-9",
        "documents": [],
    }
    trace["generation_instructions"] = 'Return only valid JSON with a single key "answer".'
    engine, _ = make_engine(report(trace, clear=True))
    verdict = engine.diagnose(trace)
    assert set(verdict.failure_categories) == {"FP1", "FP5"}
    assert "corpus.source_membership" in {item["artifact_id"] for item in verdict.evidence}

    trace["corpus"]["complete"] = False
    trace["final_answer"] = '{"answer":"Ada"}'
    engine, _ = make_engine(report(trace))
    verdict = engine.diagnose(trace)
    assert not verdict.failure_categories
    assert verdict.evaluations[4]["status"] == "clear"


def test_fp4_repairs_only_verbatim_evidence_once(make_engine, trace):
    claim = (("FP4", "generation", ["context.text", "generation.answer"]),)

    def reply(request):
        payload = json.loads(request["contents"])
        result = report(trace, claim)
        row = result["evaluations"][3]
        if "citation_repair" not in payload:
            row["evidence"][0]["quote"] = "Ada assembled the tower in 1902"
        else:
            assert payload["citation_repair"]["category"] == "FP4"
            row["evidence"][0]["quote"] = "Ada built the tower in 1902."
        return result

    engine, client = make_engine(reply)
    verdict = engine.diagnose(trace)
    assert client.calls == 2
    assert verdict.failure_category == "FP4"
    assert (
        verdict.evaluations[3]["evidence"][0]["quote"]
        in (trace_artifacts(trace)["context.text"]["text"])
    )


def test_suite_injects_all_boundaries_and_scores_without_oracle(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "dataset/corpus"
    corpus.mkdir(parents=True)
    cases, queries, qrels, answers = [], {}, {}, {}
    for index, fault in enumerate(FAULT_CODES, 1):
        query_id, doc_id = str(index), f"paper-{index}"
        facts = f"Supporting fact {index}. Another fact {index}."
        response = f"Replacement answer {index}."
        case = {"query_id": query_id, "failure_type": fault}
        if index > 1:
            case["injection_config"] = {"evidence_quote": facts}
            if index in (2, 4, 7):
                case["injection_config"]["mode"] = "runtime"
            if index >= 4:
                case["injection_config"]["response"] = response
            if index in (5, 6):
                case["injection_config"]["instruction"] = f"Constraint {index}."
        cases.append(case)
        queries[query_id] = {"query": f"Question {index}?"}
        qrels[query_id] = {"doc_id": doc_id, "section_id": 12}
        answers[query_id] = "PRIVATE SCORING ANSWER"
        (corpus / f"{doc_id}.json").write_text(
            json.dumps({"id": doc_id, "sections": [{"section_id": 12, "text": facts}]})
        )
    for name, mapping in (("queries", queries), ("qrels", qrels), ("answers", answers)):
        (corpus.parent / f"{name}.json").write_text(json.dumps(mapping))
    unrelated = {
        "id": "paper-8",
        "title": "Unrelated full document",
        "sections": [{"section_id": 50, "text": "A document outside all manifest targets."}],
        "authors": ["Preserved author"],
        "custom_metadata": {"retain": True},
    }
    (corpus / "paper-8.json").write_text(json.dumps(unrelated))
    manifest = tmp_path / "faults.yaml"
    manifest.write_text(yaml.safe_dump({"version": 1, "test_cases": cases}))

    collections = {}
    contexts = {}
    monkeypatch.setattr(
        RAGPipeline,
        "_open_collection",
        lambda self, **kw: collections.setdefault(
            (str(self.settings.chroma_dir), self.settings.collection), MemoryCollection()
        ),
    )
    monkeypatch.setattr(RAGPipeline, "_embed", lambda _, texts: [[1.0] for _ in texts])

    def generate(self, prompt, system_instruction, temperature):
        query = prompt.splitlines()[1]
        contexts[query] = (prompt, system_instruction)
        return "Original model answer."

    monkeypatch.setattr(GeminiInferenceEngine, "generate", generate)

    judges = {
        1: ("corpus", ["query", "corpus.manifest"]),
        2: ("retrieval", ["retrieval.manifest", "deep.chunk.5"]),
        3: ("context_assembly", ["retrieval.chunk.1", "context.text"]),
        4: ("generation", ["context.text", "generation.answer"]),
        5: ("generation", ["generation.instructions", "generation.answer"]),
    }

    def reply(request):
        assert "PRIVATE SCORING ANSWER" not in request["contents"]
        assert "expected_category" not in request["contents"]
        artifacts = json.loads(request["contents"])["artifacts"]
        query = artifacts["query"]["text"]
        index = int(query.split()[1].rstrip("?"))
        context = artifacts["context.text"]["text"]
        prompt, instruction = contexts[query]
        assert prompt.endswith(context)
        assert artifacts["generation.instructions"]["text"] == instruction
        if index == 1:
            assert artifacts["corpus.manifest"]["complete"]
            assert "paper-1" not in artifacts["corpus.manifest"]["document_ids"]
        if index == 2:
            assert artifacts["deep.chunk.5"]["doc_id"] == "paper-2"
            assert "Supporting fact 2." not in context
        if index == 3:
            assert artifacts["retrieval.chunk.1"]["doc_id"] == "paper-3"
            assert "Supporting fact 3." not in context
        if 4 <= index <= 6:
            assert f"Supporting fact {index}." in context
            assert artifacts["generation.answer"]["text"] == f"Replacement answer {index}."
        minimal_trace = {
            "query": query,
            "context": context,
            "final_answer": "Answer.",
        }
        verdict = report(minimal_trace)
        if index in judges:
            stage, keys = judges[index]
            verdict.update(failure_category=f"FP{index}", primary_fault_stage=stage)
            verdict["evaluations"][index - 1].update(
                status="fault",
                stage=stage,
                confidence=0.95,
                evidence=[{"artifact_id": key, "quote": artifacts[key]["text"]} for key in keys],
            )
        return verdict

    client = FakeClient(reply)
    judge = LLMJudge(
        "test-model", tmp_path / "cache.json", client=client, assessment_mode="legacy_fp"
    )
    result = run_suite(
        manifest,
        tmp_path / "benchmark",
        replace(Settings.from_env(), corpus_dir=corpus, top_k=5),
        DiagnosticConfig(assessment_mode="legacy_fp"),
        judge=judge,
        show_progress=True,
    )
    assert client.calls == 7
    assert result["errors"] == 0
    assert result["abstained"] == 2
    assert result["accuracy_when_answered"] == 1
    assert result["suite_accuracy"] == round(5 / 7, 4)
    assert result["injection_coverage"] == round(6 / 7, 4)
    fp7 = result["cases"][-1]
    assert not fp7["injection_applied"]
    assert "absent" in fp7["injection_reason"]
    assert len(list((tmp_path / "benchmark/baseline_traces").glob("*.json"))) == 7
    assert len(collections) == 2
    assert sorted(collection.count() for collection in collections.values()) == [7, 8]
    assert json.loads((tmp_path / "benchmark/corpus/paper-8.json").read_text()) == unrelated
    assert json.loads((tmp_path / "benchmark/corpus/paper-4.json").read_text()) == (
        json.loads((corpus / "paper-4.json").read_text())
    )
    traces = list((tmp_path / "benchmark/traces").glob("*.json"))
    assert len(traces) == 7
    fp3 = json.loads((tmp_path / "benchmark/traces/FP3.json").read_text())
    retrieved = fp3["stages"]["retrieval"]["output_data"]
    ids = fp3["stages"]["context_assembly"]["output_data"]["chunk_ids"]
    assert next(row["chunk_id"] for row in retrieved if row["doc_id"] == "paper-3") not in ids
    progress_output = capsys.readouterr().err
    assert "Indexing clean corpus" in progress_output
    assert "Creating corrupted index" in progress_output
    assert "FP7" in progress_output


def test_replay_keeps_failed_cases_in_denominator(tmp_path):
    from conftest import sample_trace

    from rag_diagnostic.fault_test_runner import replay_suite

    source = tmp_path / "original"
    source.mkdir()
    trace = {**sample_trace(), "run_id": "saved", "execution_status": "failed"}
    (source / "trace.json").write_text(json.dumps(trace))
    (source / "report.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "query_id": "q",
                        "expected_category": "FP3",
                        "expected_stage": "context_assembly",
                        "trace": "trace.json",
                    }
                ]
            }
        )
    )
    result = replay_suite(
        source,
        tmp_path / "replay",
        DiagnosticConfig(judge="unavailable", assessment_mode="legacy_fp"),
    )
    assert result["executions"] == result["attempted"] == result["errors"] == 1
    assert result["answered"] == result["abstained"] == 0
    assert result["suite_accuracy"] == 0

"""Shared Gemini transport contracts; API calls are mocked."""

from types import SimpleNamespace

import pytest
from conftest import FakeClient

import gemini_client
from gemini_client import DEFAULT_MODEL, GeminiAPIError, GeminiInferenceEngine
from rag_diagnostic.judge import OUTPUT_SCHEMA
from rag_pipeline.config import Settings
from rag_pipeline.pipeline import RAGPipeline, SearchResult


def test_generation_and_judging_share_client_with_distinct_configs():
    client = FakeClient(' {"ok": true} ')
    engine = GeminiInferenceEngine(client=client)
    assert engine.generate("question", "system prompt") == '{"ok": true}'
    result = engine.judge_trace_result("trace", "judge instructions", OUTPUT_SCHEMA, 1234)
    assert result.text == ' {"ok": true} '
    assert result.model_version == "mock-pinned-model"
    generation, judge = client.requests
    assert generation["model"] == DEFAULT_MODEL
    assert generation["config"] == {"temperature": 0.2, "system_instruction": "system prompt"}
    assert judge["config"]["temperature"] == 0.1
    assert judge["config"]["response_mime_type"] == "application/json"
    assert judge["config"]["response_json_schema"] == OUTPUT_SCHEMA
    assert judge["config"]["max_output_tokens"] == 1234
    assert judge["config"]["system_instruction"] == "judge instructions"
    assert result.completion_status == "completed"


def test_rag_pipeline_and_adapter_use_same_gemini_generation(tmp_path):
    from rag_diagnostic.adapters.open_ragbench import OpenRAGBenchAdapter

    client = FakeClient("The grounded answer.")
    from dataclasses import replace

    settings = replace(Settings.from_env(), diagnostics_dir=tmp_path)
    pipeline = RAGPipeline(settings, client=client)
    pipeline.retrieve = lambda query: [SearchResult("Source fact.", {"doc_id": "d"})]
    answer, results = pipeline.ask("Question?")
    adapter = OpenRAGBenchAdapter(settings, pipeline=pipeline)
    observed_answer = adapter.run("Question?")["final_answer"]
    assert answer == observed_answer == "The grounded answer."
    assert results[0].text == "Source fact."
    assert client.requests[0] == client.requests[1]


@pytest.mark.parametrize("status", ["incomplete", "blocked"])
def test_incomplete_generation_cannot_masquerade_as_success(status):
    client = FakeClient("partial text")
    client.status = status
    engine = GeminiInferenceEngine(client=client)
    assert engine.judge_trace_result("trace").completion_status == "incomplete"
    with pytest.raises(GeminiAPIError, match="incomplete"):
        engine.generate("question")


def test_empty_judge_reply_is_preserved_for_abstention():
    engine = GeminiInferenceEngine(client=FakeClient(""))
    assert engine.judge_trace("trace") == ""
    with pytest.raises(GeminiAPIError):
        engine.generate("question")


@pytest.mark.parametrize("code", [404, 429])
def test_provider_errors_exclude_credentials_and_do_not_switch_model(code, monkeypatch):
    monkeypatch.setattr(gemini_client.time, "sleep", lambda _: None)
    secret = "private-test-credential"

    def failure(request):
        error = RuntimeError("request URL contained key=" + secret)
        error.code = code
        raise error

    client = FakeClient(failure)
    engine = GeminiInferenceEngine("requested-model", client=client)
    with pytest.raises(GeminiAPIError) as caught:
        engine.generate("question")
    assert secret not in str(caught.value)
    assert str(code) in str(caught.value)
    assert client.calls == (3 if code == 429 else 1)
    assert client.requests[0]["model"] == "requested-model"


def test_transient_provider_error_retries_then_recovers(monkeypatch):
    sleeps = []
    monkeypatch.setattr(gemini_client.time, "sleep", sleeps.append)
    count = 0

    def reply(request):
        nonlocal count
        count += 1
        if count < 3:
            error = RuntimeError("unavailable")
            error.code = 503
            raise error
        return "Recovered."

    client = FakeClient(reply)
    assert GeminiInferenceEngine(client=client).generate("Question") == "Recovered."
    assert client.calls == 3
    assert sleeps == [2, 4]


def test_missing_key_has_actionable_error(monkeypatch):
    import rag_pipeline.config

    monkeypatch.setattr(rag_pipeline.config, "load_environment", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiInferenceEngine().generate("question")


def test_environment_key_priority_without_network(monkeypatch):
    import sys

    import rag_pipeline.config

    monkeypatch.setattr(rag_pipeline.config, "load_environment", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "other-test-key")
    calls = []
    fake_google = SimpleNamespace(
        genai=SimpleNamespace(Client=lambda **kwargs: calls.append(kwargs))
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)
    GeminiInferenceEngine()._get_client()
    assert calls == [{"api_key": "gemini-test-key", "vertexai": False}]


def test_pacing_is_shared_between_engine_instances(monkeypatch):
    current, sleeps = [100.0], []
    monkeypatch.setattr(gemini_client, "_LAST_REQUEST", None)
    monkeypatch.setattr(gemini_client.time, "monotonic", lambda: current[0])

    def sleep(delay):
        sleeps.append(delay)
        current[0] += delay

    monkeypatch.setattr(gemini_client.time, "sleep", sleep)
    first = GeminiInferenceEngine(client=FakeClient("first"), min_request_interval_seconds=2)
    second = GeminiInferenceEngine(client=FakeClient("second"), min_request_interval_seconds=2)
    first.generate("one")
    second.generate("two")
    assert sleeps == [2]


@pytest.mark.parametrize("interval", [-1, float("nan"), True, 61])
def test_invalid_pacing_is_rejected(interval):
    with pytest.raises(ValueError):
        GeminiInferenceEngine(min_request_interval_seconds=interval)


def test_model_override_is_shared_default(monkeypatch):
    monkeypatch.setenv("RAG_LLM_MODEL", "custom-gemini-model")
    assert Settings.from_env().llm_model == "custom-gemini-model"

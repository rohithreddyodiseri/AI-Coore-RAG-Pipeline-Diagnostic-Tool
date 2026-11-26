"""Logged judge I/O matches the SDK request and covers cached responses."""

import json

import pytest
from conftest import FakeClient

from rag_diagnostic.judge import LLMJudge
from rag_diagnostic.judge_logging import judge_io_logging, log_saved_judge_exchange


def events(stderr):
    decoder = json.JSONDecoder()
    return [
        decoder.raw_decode(block.split(" ===\n", 1)[1])[0]
        for block in stderr.split("=== GEMINI JUDGE ")[1:]
    ]


def test_input_matches_actual_spanned_request_and_output(tmp_path, capsys):
    client = FakeClient('{"evaluations": []}')
    judge = LLMJudge("test-model", tmp_path / "cache.json", client)
    payload = {
        "artifacts": {
            "context.text": {
                "stage": "generation",
                "kind": "context",
                "text": "context words " * 100,
            }
        }
    }
    with judge_io_logging():
        judge.evaluate_trace(payload)
    captured = capsys.readouterr()
    input_log, output_log = events(captured.err)
    assert captured.out == ""
    request = client.requests[0]
    assert input_log["contents"] == request["contents"]
    assert input_log["system_instruction"] == request["config"]["system_instruction"]
    assert input_log["response_json_schema"] == request["config"]["response_json_schema"]
    assert "spans" in json.loads(input_log["contents"])["artifacts"]["context.text"]
    assert output_log["raw_output"] == '{"evaluations": []}'


def test_default_silent_and_cache_hit_is_logged_without_api_call(tmp_path, capsys):
    client = FakeClient("raw-response")
    judge = LLMJudge("test-model", tmp_path / "cache.json", client)
    judge.evaluate_trace({})
    assert capsys.readouterr().err == ""
    with judge_io_logging():
        judge.evaluate_trace({})
    assert client.calls == 1
    assert all(e["source"] == "cache (no API call)" for e in events(capsys.readouterr().err))
    judge.evaluate_trace({})
    assert capsys.readouterr().err == ""


def test_saved_resume_exchange_and_validation_summary_are_logged(capsys):
    log = dict(
        run_id="opaque-id",
        status="diagnosed",
        root_cause={"mechanism": "TEST"},
        observations=dict(
            input={},
            judge={"model": "test-model"},
            judge_response=dict(
                raw_output="recorded reply",
                request_instructions="system",
                response_schema={},
                completion_status="completed",
            ),
        ),
    )
    with judge_io_logging():
        log_saved_judge_exchange(log)
    logs = events(capsys.readouterr().err)
    assert logs[0]["source"] == "saved_verdict (no API call)"
    assert logs[1]["raw_output"] == "recorded reply"
    assert logs[2]["root_cause"] == {"mechanism": "TEST"}


def test_provider_failure_log_never_prints_exception_payload(tmp_path, capsys):
    def failure(_request):
        raise RuntimeError("request headers contained credential=private-key")

    judge = LLMJudge("test-model", tmp_path / "cache.json", FakeClient(failure))
    with judge_io_logging(), pytest.raises(RuntimeError):
        judge.evaluate_trace({})
    text = capsys.readouterr().err
    assert "GEMINI JUDGE ERROR" in text
    assert "private-key" not in text


def test_cli_flag_logs_live_and_resumed_execution(tmp_path, monkeypatch, capsys):
    from gemini_client import GeminiInferenceEngine
    from rag_diagnostic.fault_test_runner import main
    from rag_diagnostic.mechanisms import MECHANISM_STAGE
    from rag_diagnostic.stage_diagnosis import unknown

    client = FakeClient(
        json.dumps({"evaluations": [unknown(m, "Insufficient evidence") for m in MECHANISM_STAGE]})
    )
    monkeypatch.setattr(GeminiInferenceEngine, "_get_client", lambda self: client)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"judge_cache": str(tmp_path / "judge-cache.json")}))
    args = [
        "--config",
        str(config),
        "--mechanism",
        "OUTPUT_FORMAT_VIOLATION",
        "--output-dir",
        str(tmp_path / "run"),
        "--log-llm",
    ]
    main(args)
    captured = capsys.readouterr()
    assert "Technical cases: 1" in captured.out
    assert "GEMINI JUDGE INPUT" not in captured.out
    assert events(captured.err)[0]["source"] == "Gemini API"
    assert "GEMINI JUDGE VALIDATED VERDICT" in captured.err
    main([*args, "--resume"])
    assert events(capsys.readouterr().err)[0]["source"] == "saved_verdict (no API call)"
    assert client.calls == 1

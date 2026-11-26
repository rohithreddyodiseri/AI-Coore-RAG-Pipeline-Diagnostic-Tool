"""Opt-in, credential-free judge request/response logging to stderr."""

import json
import sys
from contextlib import contextmanager
from contextvars import ContextVar

_ENABLED = ContextVar("judge_io_logging", default=False)


@contextmanager
def judge_io_logging(enabled=True):
    token = _ENABLED.set(enabled)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def log_judge_event(event, **fields):
    if _ENABLED.get():
        print(f"\n=== GEMINI JUDGE {event} ===", file=sys.stderr, flush=True)
        print(json.dumps(fields, indent=2, ensure_ascii=False), file=sys.stderr, flush=True)


def log_saved_judge_exchange(verdict):
    """Show recorded I/O when resume skips an already diagnosed execution."""
    if not _ENABLED.get():
        return
    from rag_diagnostic.judge import judge_prompt

    observations = verdict.get("observations", {})
    response = observations.get("judge_response")
    if not response:
        log_judge_event(
            "RESUME",
            run_id=verdict.get("run_id"),
            message="Saved verdict has no recorded judge exchange.",
        )
        return
    log_judge_event(
        "INPUT",
        source="saved_verdict (no API call)",
        run_id=verdict.get("run_id"),
        model=observations.get("judge", {}).get("model"),
        cache_key=response.get("cache_key"),
        system_instruction=response.get("request_instructions"),
        contents=judge_prompt(observations.get("input", {})),
        response_json_schema=response.get("response_schema"),
    )
    log_judge_event(
        "OUTPUT",
        source="saved_verdict (no API call)",
        run_id=verdict.get("run_id"),
        raw_output=response.get("raw_output"),
        completion_status=response.get("completion_status"),
        actual_model=response.get("actual_model"),
    )
    log_validated_verdict(verdict)


def log_validated_verdict(verdict):
    if _ENABLED.get():
        log_judge_event(
            "VALIDATED VERDICT",
            run_id=verdict.get("run_id"),
            status=verdict.get("status"),
            root_cause=verdict.get("root_cause"),
            evaluations=[
                {k: row.get(k) for k in ("mechanism", "status", "confidence", "rationale")}
                for row in verdict.get("evaluations", [])
            ],
        )

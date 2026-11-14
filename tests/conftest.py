"""Mocked LLM protocol fixtures; these do not measure semantic judge accuracy."""

import json
from copy import deepcopy

import pytest

from rag_diagnostic.config import DiagnosticConfig
from rag_diagnostic.diagnostic_engine import DiagnosticEngine, trace_artifacts
from rag_diagnostic.judge import LLMJudge
from rag_diagnostic.taxonomy import CATEGORIES

FACT = "Ada built the tower in 1902."
QUERY = "Who built the tower, and when? Return a JSON object with builder and year."


def sample_trace():
    chunks = [{"chunk_id": "d:0", "doc_id": "d", "text": FACT, "rank": 1, "score": 0.9}]
    return {
        "query": QUERY,
        "stages": {
            "retrieval": {"k": 1, "output_data": chunks},
            "reranking": {"k": 1, "output_data": deepcopy(chunks)},
            "context_assembly": {"output_data": {"text": FACT, "chunk_ids": ["d:0"]}},
        },
        "final_answer": "Ada built the tower.",
    }


def cite(artifacts, key, quote=None):
    return {"artifact_id": key, "quote": artifacts[key]["text"] if quote is None else quote}


def report(trace, faults=(), clear=False):
    artifacts = trace_artifacts(trace)
    rows = [
        {
            "category": c.code,
            "status": "clear" if clear else "unknown",
            "stage": None,
            "confidence": 0.93 if clear else 0,
            "rationale": "Supported." if clear else "Unobserved.",
            "evidence": [
                cite(artifacts, "query"),
                cite(artifacts, "context.text"),
                cite(artifacts, "generation.answer"),
            ]
            if clear
            else [],
            "contributing_stages": [],
        }
        for c in CATEGORIES
    ]
    for code, stage, keys in faults:
        rows[int(code[-1]) - 1] = {
            "category": code,
            "status": "fault",
            "stage": stage,
            "confidence": 0.93,
            "rationale": f"Observed {code} evidence at the cited boundaries.",
            "evidence": [cite(artifacts, key) for key in keys],
            "contributing_stages": [],
        }
    return {
        "failure_category": faults[0][0] if faults else "NONE" if clear else "ABSTAIN",
        "primary_fault_stage": faults[0][1] if faults else None,
        "confidence": 0.93,
        "reason": "Inspect the cited artifacts; unobserved checks are unknown.",
        "evaluations": rows,
    }


class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls, self.requests, self.models = reply, 0, [], self
        self.status = "completed"

    def generate_content(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        reply = self.reply(kwargs) if callable(self.reply) else self.reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return type(
            "Response",
            (),
            {
                "text": text,
                "response_id": "mock-response",
                "model_version": "mock-pinned-model",
                "candidates": [
                    type(
                        "Candidate",
                        (),
                        {"finish_reason": "STOP" if self.status == "completed" else "MAX_TOKENS"},
                    )()
                ],
            },
        )()


@pytest.fixture
def trace():
    return sample_trace()


@pytest.fixture
def make_engine(tmp_path):
    def make(reply, **overrides):
        config = DiagnosticConfig(
            runs_dir=str(tmp_path / "runs"), judge_cache=str(tmp_path / "judge.json"), **overrides
        )
        client = FakeClient(reply)
        judge = LLMJudge(
            config.judge_model, config.judge_cache, client, config.judge_max_output_tokens
        )
        return DiagnosticEngine(config, judge), client

    return make


@pytest.fixture(autouse=True)
def disable_live_rate_delays(monkeypatch):
    # Unit tests inject a mock Gemini client; no free-tier quota is consumed.
    monkeypatch.setenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "0")


class RecordedTarget:
    """Test double for an external system exposing recorded execution artifacts."""

    retrieval_k = context_k = 1

    def describe(self):
        return {"target": "recorded_test_system", "revision": "test-v1"}

    def run(self, query):
        from rag_diagnostic.trace import Context, QueryTrace, ScoredChunk

        chunk = ScoredChunk("d:0", "d", FACT, 0.9, 1)
        return QueryTrace(
            query, (chunk,), (chunk,), False, Context(("d:0",), FACT), "Ada built the tower."
        )

    def generation_instructions(self):
        return "Use the source context."

    def retrieve(self, query, k):
        from rag_diagnostic.trace import ScoredChunk

        return [ScoredChunk("d:0", "d", FACT, 0.9, 1)]

    def corpus_snapshot(self):
        return {
            "documents": [{"doc_id": "d", "text": FACT}],
            "complete": True,
            "snapshot_id": "test-corpus-v1",
            "scope": "all test source documents",
        }

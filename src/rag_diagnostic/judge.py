"""Decides whether an answer is right.

The judge only says whether the answer is wrong. Finding which stage caused
it is the job of the checks and the fix-and-rerun step.

- ReferenceRecallJudge: no model needed. The answer is right if it contains
  most of the reference answer's words. Works well for short facts and tests.
- LLMJudge: asks a model to compare the answer with the reference. It runs at
  temperature 0 and saves every result, so a rerun gives the same answer.
  Check it by hand on a sample before trusting it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_diagnostic.locate import tokens


@dataclass(frozen=True)
class JudgeResult:
    correct: bool
    score: float
    rationale: str


class Judge(Protocol):
    def judge(self, query: str, answer: str, reference: str) -> JudgeResult: ...


class ReferenceRecallJudge:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def judge(self, query: str, answer: str, reference: str) -> JudgeResult:
        wanted = tokens(reference)
        if not wanted:
            return JudgeResult(False, 0.0, "reference answer is empty")
        present = set(tokens(answer))
        recall = sum(token in present for token in wanted) / len(wanted)
        return JudgeResult(
            recall >= self.threshold,
            round(recall, 4),
            f"{recall:.0%} of reference words found in answer",
        )


_LLM_INSTRUCTIONS = (
    "You grade answers from a question-answering system. Compare the system answer to the "
    "reference answer. The system answer is correct if it states the same facts as the "
    "reference, even in different words. Extra correct detail is fine. A refusal, a different "
    'fact, or a missing key fact is incorrect. Reply with JSON only: {"correct": true or false, '
    '"reason": "<one sentence>"}'
)


class LLMJudge:
    def __init__(self, model: str, cache_path: Path, client=None):
        self.model = model
        self.cache_path = Path(cache_path)
        self._client = client
        self._cache: dict[str, dict] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\0{_LLM_INSTRUCTIONS}\0{prompt}".encode()).hexdigest()

    def judge(self, query: str, answer: str, reference: str) -> JudgeResult:
        prompt = f"Question:\n{query}\n\nReference answer:\n{reference}\n\nSystem answer:\n{answer}"
        key = self._key(prompt)
        # Only call the model the first time. After that, use the saved result.
        if key not in self._cache:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI()
            response = self._client.responses.create(
                model=self.model,
                instructions=_LLM_INSTRUCTIONS,
                input=prompt,
                temperature=0,
            )
            try:
                parsed = json.loads(response.output_text)
                self._cache[key] = {
                    "correct": bool(parsed["correct"]),
                    "reason": str(parsed.get("reason", "")),
                }
            except (json.JSONDecodeError, KeyError, TypeError):
                self._cache[key] = {
                    "correct": False,
                    "reason": f"unparseable judge output: {response.output_text[:200]}",
                }
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
        cached = self._cache[key]
        return JudgeResult(cached["correct"], 1.0 if cached["correct"] else 0.0, cached["reason"])


def build_judge(config) -> Judge:
    """Pick the judge named in the config."""
    if config.judge == "reference_recall":
        return ReferenceRecallJudge(config.judge_threshold)
    if config.judge == "llm":
        return LLMJudge(config.judge_model, Path(config.judge_cache))
    raise ValueError(f"unknown judge: {config.judge}")

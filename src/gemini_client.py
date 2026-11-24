"""Shared Google Gen AI inference for RAG generation and diagnostic judging."""

from __future__ import annotations

import math
import os
import stat
import threading
import time
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "gemini-3.5-flash-lite"
_PACING_LOCK = threading.Lock()
_LAST_REQUEST: float | None = None


def _load_environment() -> None:
    from dotenv import find_dotenv, load_dotenv

    env_file = find_dotenv(usecwd=True)
    if env_file:
        flags = getattr(os.stat(env_file), "st_flags", 0)
        if flags & getattr(stat, "SF_DATALESS", 0):
            return
        load_dotenv(env_file, override=False)


@dataclass(frozen=True)
class InferenceResult:
    text: str
    response_id: str | None
    model_version: str | None
    completion_status: str


class GeminiAPIError(RuntimeError):
    """Provider failure with credentials and response payloads excluded."""


class GeminiInferenceEngine:
    VERSION = 2

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        client=None,
        min_request_interval_seconds: float | None = None,
        request_timeout_seconds: float | None = None,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Gemini model_name must be a nonempty string")
        self.model_name, self._client = model_name, client
        if min_request_interval_seconds is None:
            _load_environment()
            min_request_interval_seconds = float(
                os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "2")
            )
        if (
            type(min_request_interval_seconds) not in (int, float)
            or not math.isfinite(min_request_interval_seconds)
            or not 0 <= min_request_interval_seconds <= 60
        ):
            raise ValueError("GEMINI_MIN_REQUEST_INTERVAL_SECONDS must be between 0 and 60")
        self.min_request_interval_seconds = min_request_interval_seconds
        if request_timeout_seconds is None:
            request_timeout_seconds = float(
                os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "120")
            )
        if (
            type(request_timeout_seconds) not in (int, float)
            or not math.isfinite(request_timeout_seconds)
            or not 1 <= request_timeout_seconds <= 600
        ):
            raise ValueError("GEMINI_REQUEST_TIMEOUT_SECONDS must be between 1 and 600")
        self.request_timeout_seconds = request_timeout_seconds

    def _get_client(self):
        if self._client is None:
            _load_environment()
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("Set GEMINI_API_KEY in your environment or .env file")
            from google import genai
            from google.genai import types

            # Keep retries in this engine so one request cannot multiply the SDK's
            # default five attempts by our three attempts. The explicit deadline also
            # prevents a stalled provider connection from hanging a benchmark case.
            self._client = genai.Client(
                api_key=key,
                vertexai=False,
                http_options=types.HttpOptions(
                    timeout=int(self.request_timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return self._client

    def _pace(self):
        global _LAST_REQUEST
        with _PACING_LOCK:
            now = time.monotonic()
            if _LAST_REQUEST is not None and self.min_request_interval_seconds:
                delay = self.min_request_interval_seconds - (now - _LAST_REQUEST)
                if delay > 0:
                    time.sleep(delay)
            _LAST_REQUEST = time.monotonic()

    def _request(self, prompt: str, config: dict[str, Any]) -> InferenceResult:
        client = self._get_client()
        try:
            for attempt in range(3):
                self._pace()
                try:
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                    break
                except Exception as error:
                    if (
                        getattr(error, "code", None) not in (429, 500, 502, 503, 504)
                        or attempt == 2
                    ):
                        raise
                    time.sleep(2 ** (attempt + 1))
        except Exception as error:
            code = getattr(error, "code", None)
            # API exception messages may include request URLs, headers, or credentials.
            detail = str(code) if isinstance(code, (str, int)) else type(error).__name__
            message = f"Gemini request failed for model {self.model_name!r} ({detail})."
            if code == 404:
                message += " Check model availability; configure RAG_LLM_MODEL and judge_model."
            elif code == 429:
                message += " Rate limit exceeded; increase GEMINI_MIN_REQUEST_INTERVAL_SECONDS."
            raise GeminiAPIError(message) from None
        candidates = getattr(response, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        reason = getattr(reason, "name", reason)
        status = "completed" if reason == "STOP" else "incomplete"
        return InferenceResult(
            text=response.text or "",
            response_id=getattr(response, "response_id", None),
            model_version=getattr(response, "model_version", None),
            completion_status=status,
        )

    def generate(
        self, prompt: str, system_instruction: str | None = None, temperature: float = 0.2
    ) -> str:
        result = self._request(
            prompt, {"system_instruction": system_instruction, "temperature": temperature}
        )
        if result.completion_status != "completed" or not result.text.strip():
            raise GeminiAPIError("Gemini generation was blocked, empty, or incomplete")
        return result.text.strip()

    def judge_trace(
        self,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: dict | None = None,
        max_output_tokens: int = 6000,
    ) -> str:
        return self.judge_trace_result(
            prompt, system_instruction, response_schema, max_output_tokens
        ).text

    def judge_trace_result(
        self,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: dict | None = None,
        max_output_tokens: int = 6000,
    ) -> InferenceResult:
        config = {
            "system_instruction": system_instruction,
            "temperature": 0.1,
            "response_mime_type": "application/json",
            "max_output_tokens": max_output_tokens,
        }
        if response_schema is not None:
            config["response_json_schema"] = response_schema
        return self._request(prompt, config)

"""Continuous watch survives temporary judge availability failures."""

from types import SimpleNamespace

import pytest

from gemini_client import GeminiAPIError
from rag_diagnostic import cli


class FlakyIngestor:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def poll(self, trace_dir, *, artifact_root=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise GeminiAPIError("temporarily unavailable", code=503)
        return []


def args(*, once=False):
    return SimpleNamespace(
        once=once,
        poll_interval=2,
        trace_dir="executions",
        artifact_root=None,
    )


def test_continuous_watch_retries_with_bounded_backoff(monkeypatch, capsys):
    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    ingestor = FlakyIngestor(3)

    assert cli._poll_with_provider_retry(ingestor, args()) == []

    assert ingestor.calls == 4
    assert sleeps == [5, 10, 20]
    error = capsys.readouterr().err
    assert error.count("retrying unacknowledged runs") == 3


def test_once_mode_propagates_provider_failure(monkeypatch):
    monkeypatch.setattr(cli.time, "sleep", lambda _delay: pytest.fail("must not sleep"))
    with pytest.raises(GeminiAPIError):
        cli._poll_with_provider_retry(FlakyIngestor(1), args(once=True))


def test_nonretryable_provider_failure_is_not_hidden(monkeypatch):
    class InvalidRequest:
        def poll(self, trace_dir, *, artifact_root=None):
            raise GeminiAPIError("invalid", code=400)

    monkeypatch.setattr(cli.time, "sleep", lambda _delay: pytest.fail("must not sleep"))
    with pytest.raises(GeminiAPIError):
        cli._poll_with_provider_retry(InvalidRequest(), args())

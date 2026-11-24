"""C1: the web UI serves the page and answers through the traced pipeline."""

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from rag_pipeline.web import make_handler


@pytest.fixture
def server(make_pipeline):
    started = []

    def start(**pipeline_options):
        pipeline = make_pipeline(**pipeline_options)
        httpd = HTTPServer(("127.0.0.1", 0), make_handler(pipeline))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        return f"http://127.0.0.1:{httpd.server_port}"

    yield start
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


def post(url, body):
    request = urllib.request.Request(
        f"{url}/api/ask",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_page_is_served(server):
    with urllib.request.urlopen(server()) as response:
        assert response.status == 200
        assert b"Reference RAG App" in response.read()


def test_ask_returns_answer_sources_and_trace(server, spans):
    status, data = post(server(), {"query": "How can retrieval improve?"})
    assert status == 200
    assert data["answer"] == "Use hybrid search and reranking."
    assert [s["title"] for s in data["sources"]] == ["Paper paper-1", "Paper paper-2"]
    assert data["sources"][0]["number"] == 1
    root = next(s for s in spans.get_finished_spans() if s.name == "rag.query")
    assert data["trace_id"] == format(root.context.trace_id, "032x")
    assert data["flagged"] is False


def test_flag_reason_reaches_the_trace(server, spans):
    status, data = post(server(), {"query": "How?", "flag_reason": "Missing details"})
    assert status == 200 and data["flagged"] is True
    root = next(s for s in spans.get_finished_spans() if s.name == "rag.query")
    assert root.attributes["rag.failure.reason"] == "Missing details"


def test_pipeline_error_is_shown_with_trace_id(server):
    status, data = post(server(error=ConnectionError("model down")), {"query": "How?"})
    assert status == 500
    assert data["error"] == "ConnectionError: model down"
    assert len(data["trace_id"]) == 32


def test_blank_question_is_rejected(server):
    assert post(server(), {"query": "  "})[0] == 400

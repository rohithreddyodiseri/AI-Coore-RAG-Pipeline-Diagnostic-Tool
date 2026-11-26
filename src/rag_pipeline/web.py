"""Web UI for the reference RAG app (ticket C1).

Standard library only. ``GET /`` serves the page and ``POST /api/ask`` runs
one query through the same ``RAGPipeline.ask`` the CLI uses, so every query
from the browser produces the same OpenTelemetry trace.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from typing import Any

MAX_BODY_BYTES = 64_000


def ask_payload(pipeline: Any, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Runs one query and returns (HTTP status, JSON response)."""
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return 400, {"error": "Enter a question."}
    query = query.strip()
    flag_reason = body.get("flag_reason")
    if flag_reason is not None and not isinstance(flag_reason, str):
        return 400, {"error": "Failure reason must be text."}
    flag_reason = (flag_reason or "").strip() or None

    try:
        answer, results = pipeline.ask(query, failure_reason=flag_reason)
    except Exception as error:
        return 500, {
            "error": f"{type(error).__name__}: {error}",
            "trace_id": pipeline.last_trace_id,
            "flagged": True,  # the receiver flags runtime failures automatically
        }

    sources = [
        {
            "number": number,
            "title": str(result.metadata.get("title") or "Untitled"),
            "doc_id": result.metadata.get("doc_id"),
            "section": result.metadata.get("section"),
            "score": result.score,
            "text": result.text,
        }
        for number, result in enumerate(results, start=1)
    ]
    return 200, {
        "answer": answer,
        "sources": sources,
        "trace_id": pipeline.last_trace_id,
        "flagged": flag_reason is not None,
    }


def make_handler(pipeline: Any) -> type[BaseHTTPRequestHandler]:
    page = files("rag_pipeline").joinpath("static/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, data: dict[str, Any]) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send_json(404, {"error": "Not found."})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/ask":
                self._send_json(404, {"error": "Not found."})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length < 0:
                    raise ValueError
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length."})
                return
            if length > MAX_BODY_BYTES:
                self._send_json(413, {"error": "Request is too large."})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                self._send_json(400, {"error": "Send a JSON object."})
                return
            self._send_json(*ask_payload(pipeline, body))

    return Handler


def serve(pipeline: Any, host: str = "127.0.0.1", port: int = 8000) -> None:
    # Single-threaded on purpose: one query at a time keeps the pipeline simple.
    server = HTTPServer((host, port), make_handler(pipeline))
    print(f"Reference RAG app running at http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    finally:
        server.server_close()

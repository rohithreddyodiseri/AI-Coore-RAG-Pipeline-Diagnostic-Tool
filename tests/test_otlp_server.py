"""Exercise the app registry and real SDK OTLP wire messages through the HTTP handler."""

from __future__ import annotations

import gzip
import io
import json
from http.client import HTTPResponse
from pathlib import Path
from urllib.parse import unquote

import pytest

pytest.importorskip("opentelemetry.proto")
pytest.importorskip("google.rpc.status_pb2")
pytest.importorskip("jsonschema")

from google.protobuf import json_format  # noqa: E402
from google.rpc.status_pb2 import Status  # noqa: E402
from jsonschema import Draft202012Validator, FormatChecker  # noqa: E402
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (  # noqa: E402
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.trace.v1.trace_pb2 import Span  # noqa: E402

from rag_diagnostic.server import JSON, OTLP_PATH, PROTOBUF, DiagnosticServer, Handler  # noqa: E402

CONTRACTS = Path(__file__).resolve().parents[1] / "docs"
ADMIN = "admin-" + 32 * "a"
VIEWER = "viewer-" + 32 * "v"
OTHER = "other-" + 32 * "b"


class Wire:
    def __init__(self, data):
        self.inbound = io.BytesIO(data)
        self.outbound = io.BytesIO()

    def makefile(self, *_):
        return self.inbound

    def sendall(self, data):
        self.outbound.write(data)

    def settimeout(self, _):
        pass


def call(server, method, path, token=None, payload=b"", content_type=JSON, **headers):
    lines = {
        "Content-Type": content_type,
        "Content-Length": str(len(payload)),
        **headers,
    }
    if token:
        lines["Authorization"] = f"Bearer {token}"
    head = "\r\n".join([f"{method} {path} HTTP/1.1", *(f"{k}: {v}" for k, v in lines.items())])
    wire = Wire((head + "\r\n\r\n").encode() + payload)
    Handler(wire, ("127.0.0.1", 12345), server)
    wire.inbound = io.BytesIO(wire.outbound.getvalue())
    response = HTTPResponse(wire)
    response.begin()
    return response.status, response.headers, response.read()


def make_server(data_dir, **settings):
    return DiagnosticServer(
        ("127.0.0.1", 0),
        data_dir=data_dir,
        contract_dir=CONTRACTS,
        public_url="http://127.0.0.1:4318",
        bind_and_activate=False,
        credentials={"first": {"admin": ADMIN, "viewer": VIEWER}, "second": {"admin": OTHER}},
        **settings,
    )


@pytest.fixture
def server(tmp_path):
    with make_server(tmp_path, assembly_window=0, max_body_bytes=8192) as instance:
        yield instance


def register(server, auth=ADMIN, name="reference-rag"):
    return call(
        server,
        "POST",
        "/api/v1/apps",
        auth,
        json.dumps(
            {
                "service_name": name,
                "framework": "custom",
                "environments": ["staging"],
            }
        ).encode(),
    )


def app_token(package):
    header = unquote(package["env"]["OTEL_EXPORTER_OTLP_HEADERS"])
    return header.split("Bearer ", 1)[1]


def span_request(service, spans):
    request = ExportTraceServiceRequest()
    resource = request.resource_spans.add()
    resource.resource.attributes.add(key="service.name").value.string_value = service
    resource.scope_spans.add().spans.extend(spans)
    return request


def span(index, *, name="step", parent=1, attrs=None, trace_id=b"t" * 16):
    result = Span(
        name=name,
        trace_id=trace_id,
        span_id=index.to_bytes(8, "big"),
        parent_span_id=parent.to_bytes(8, "big") if parent else b"",
        start_time_unix_nano=1_000_000,
        end_time_unix_nano=2_000_000,
    )
    for key, value in (attrs or {}).items():
        result.attributes.add(key=key).value.string_value = value
    return result


def export(server, request, token):
    return call(server, "POST", OTLP_PATH, token, request.SerializeToString(), PROTOBUF)


def test_registry_scopes_roles_credentials_and_pagination(server):
    status, headers, body = register(server)
    assert status == 201
    assert headers["Cache-Control"] == "no-store"
    package = json.loads(body)
    ui_schema = json.loads((CONTRACTS / "ui-api.schema.json").read_text())
    validator = Draft202012Validator(
        {**ui_schema, "$ref": "#/$defs/OnboardingPackage"}, format_checker=FormatChecker()
    )
    validator.validate(package)
    registered = package["app"]
    app_id, key = registered["app_id"], app_token(package)
    assert registered["last_trace_at"] is None
    assert key not in package["snippets"][0]["code"]
    assert register(server, VIEWER)[0] == 403
    assert register(server)[0] == 409
    assert register(server, OTHER)[0] == 201
    assert register(server, name="another")[0] == 201
    path = f"/api/v1/apps/{app_id}"
    assert call(server, "GET", path, OTHER)[0] == 404
    assert call(server, "GET", path, key)[0] == 401
    assert json.loads(call(server, "GET", path, VIEWER)[2]) == registered
    page = json.loads(call(server, "GET", "/api/v1/apps?limit=1", VIEWER)[2])
    assert len(page["items"]) == 1
    page2 = json.loads(
        call(server, "GET", f"/api/v1/apps?limit=1&cursor={page['next_cursor']}", VIEWER)[2]
    )
    assert len(page2["items"]) == 1 and page2["next_cursor"] is None
    assert page["items"] != page2["items"]

    rotate = f"{path}/ingestion-key/rotate"
    assert call(server, "POST", rotate, OTHER)[0] == 404
    assert call(server, "POST", rotate, VIEWER)[0] == 403
    status, headers, body = call(server, "POST", rotate, ADMIN)
    assert status == 200 and headers["Cache-Control"] == "no-store"
    assert app_token(json.loads(body)) != key
    assert export(server, ExportTraceServiceRequest(), key)[0] == 401
    assert export(server, ExportTraceServiceRequest(), app_token(json.loads(body)))[0] == 200


def test_sdk_split_batch_normalizes_and_survives_restart(tmp_path):
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    resource = pytest.importorskip("opentelemetry.sdk.resources")
    encoder = pytest.importorskip("opentelemetry.exporter.otlp.proto.common.trace_encoder")

    captured = []

    class Capture(sdk_trace.SpanProcessor):
        def on_end(self, span):
            captured.append(span)

    provider = sdk_trace.TracerProvider(
        resource=resource.Resource.create({"service.name": "reference-rag"})
    )
    provider.add_span_processor(Capture())
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span(
        "rag.query",
        attributes={
            "rag.telemetry.version": "1.0",
            "rag.execution.status": "completed",
            "rag.query.text": "Where is it?",
            "rag.failure.flagged": True,
            "rag.stages.not_used": ("reranking",),
        },
    ):
        with tracer.start_as_current_span(
            "retrieve",
            attributes={
                "rag.stage": "retrieval",
                "rag.stage.status": "completed",
                "rag.stage.k": 4,
                "rag.stage.output_json": "[]",
            },
        ):
            pass
    provider.shutdown()
    with make_server(tmp_path, assembly_window=60) as first:
        package = json.loads(register(first)[2])
        key = app_token(package)
        assert export(first, encoder.encode_spans([captured[1]]), key)[0] == 200
        assert first.store.publish_one() is None
        request = encoder.encode_spans([captured[0]])
        body = gzip.compress(request.SerializeToString())
        assert (
            call(
                first,
                "POST",
                OTLP_PATH,
                key,
                body,
                PROTOBUF,
                **{
                    "Content-Encoding": "gzip",
                },
            )[0]
            == 200
        )
        assert export(first, request, key)[0] == 200
    with make_server(tmp_path) as restarted:
        with restarted.store.connection() as db:
            assert db.execute("SELECT count(*) FROM spans").fetchone()[0] == 2
            db.execute("UPDATE executions SET due_at=0")
        path = restarted.store.publish_one()
        assert path is not None and path.parent.name == "executions"
        data = json.loads(path.read_text())
        assert data["run_id"] == (
            f"{captured[1].context.trace_id:032x}:{captured[1].context.span_id:016x}"
        )
        assert data["query"] == "Where is it?"
        assert data["failure_signal"]["flagged"] is True
        assert data["stages"]["retrieval"]["output_data"] == []
        assert data["stages"]["retrieval"]["k"] == 4
        assert data["stages"]["reranking"] == {"status": "not_used"}
        assert restarted.store.publish_one() is None
        assert restarted.store.app("first", package["app"]["app_id"])["last_trace_at"]


def test_otlp_partial_success_invalid_evidence_and_metadata_only(server):
    key = app_token(json.loads(register(server)[2]))
    attrs = {"rag.telemetry.version": "1.0", "rag.execution.status": "failed"}
    root = span(1, name="rag.query", parent=0, attrs=attrs)
    other_root = span(9, name="rag.query", parent=0, attrs=attrs)
    bad = span(
        2,
        attrs={
            "rag.stage": "retrieval",
            "rag.stage.status": "completed",
            "rag.stage.output_json": '[{"chunk_id":"bad","text":"","rank":0}]',
        },
    )
    for stage_id in (3, 4):
        extra = span(stage_id, attrs={"rag.stage": "generation", "rag.stage.status": "failed"})
        if stage_id == 3:
            duplicate = extra
    request = span_request("reference-rag", [root, bad, duplicate, extra])
    request.resource_spans.extend(span_request("wrong-app", [other_root]).resource_spans)
    status, _, body = export(server, request, key)
    assert status == 200
    response = ExportTraceServiceResponse.FromString(body)
    assert response.partial_success.rejected_spans == 1
    assert "service.name" in response.partial_success.error_message
    output = server.store.publish_one()
    assert output.parent.name == "metadata-only"
    record = json.loads(output.read_text())
    assert "query" not in record and record["failure_signal"]["flagged"]
    assert set(record["stages"]) == {"retrieval"}
    assert "output_data" not in record["stages"]["retrieval"]
    assert len(record["inspection_errors"]) == 2
    assert server.store.publish_one() is None


def test_otlp_json_wire_errors_and_body_limit(server):
    key = app_token(json.loads(register(server)[2]))
    body = json_format.MessageToJson(ExportTraceServiceRequest()).encode()
    status, headers, response = call(server, "POST", OTLP_PATH, key, body)
    assert status == 200 and headers["Content-Type"] == JSON and json.loads(response) == {}
    status, _, body = call(server, "POST", OTLP_PATH, key, b"\xff", PROTOBUF)
    assert status == 400 and Status.FromString(body).message == "Malformed OTLP trace request"
    oversized = gzip.compress(b"x" * (server.max_body_bytes + 1))
    status, _, _ = call(
        server,
        "POST",
        OTLP_PATH,
        key,
        oversized,
        PROTOBUF,
        **{
            "Content-Encoding": "gzip",
        },
    )
    assert status == 413

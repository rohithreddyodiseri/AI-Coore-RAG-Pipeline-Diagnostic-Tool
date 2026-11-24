"""Run the app registry and OTLP/HTTP receiver behind a TLS reverse proxy."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import math
import os
import re
import sqlite3
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from google.protobuf import json_format
from google.protobuf.message import DecodeError
from google.rpc.status_pb2 import Status
from jsonschema import Draft202012Validator, FormatChecker
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

from rag_diagnostic.otlp_store import Store, token_digest

LOG = logging.getLogger(__name__)
PROTOBUF = "application/x-protobuf"
JSON = "application/json"
OTLP_PATH = "/v1/traces"
CONTRACT_PATH = "/docs/span-contract/v1"
TENANT = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
APP_ID = re.compile(r"[0-9a-f]{32}")
SNIPPET = """from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider(resource=Resource.create())
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)"""


class RequestError(Exception):
    def __init__(self, status, code, message, fields=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.fields = fields


def registry_tokens(config):
    """Map bootstrap user tokens to an explicit tenant and role."""
    if not isinstance(config, dict) or not config:
        raise ValueError("configure at least one tenant in RAG_DIAGNOSTIC_REGISTRY_TOKENS")
    result = {}
    for tenant, roles in config.items():
        if not TENANT.fullmatch(tenant) or not isinstance(roles, dict) or "admin" not in roles:
            raise ValueError("registry tenants require a valid identifier and an admin token")
        for role, token in roles.items():
            if role not in {"admin", "viewer"} or not isinstance(token, str):
                raise ValueError("registry credentials must use the admin or viewer role")
            if not re.fullmatch(r"[A-Za-z0-9._~-]{32,256}", token):
                raise ValueError("registry bearer tokens must be URL-safe and 32–256 characters")
            digest = token_digest(token)
            if digest in result:
                raise ValueError("registry bearer tokens must be unique")
            result[digest] = tenant, role
    return result


def _public_endpoint(value):
    url = urlsplit(value)
    local = url.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        not url.hostname
        or url.username is not None
        or url.query
        or url.fragment
        or url.path not in ("", "/")
        or not (url.scheme == "https" or (url.scheme == "http" and local))
    ):
        raise ValueError("public URL must be an HTTPS base URL or an HTTP loopback base URL")
    _ = url.port  # Validate an explicitly configured port.
    return value.rstrip("/")


class DiagnosticServer(ThreadingHTTPServer):
    def __init__(
        self,
        address,
        *,
        data_dir,
        contract_dir,
        credentials,
        public_url,
        assembly_window=5,
        max_body_bytes=4 * 1024 * 1024,
        bind_and_activate=True,
    ):
        if not math.isfinite(assembly_window) or assembly_window < 0 or max_body_bytes <= 0:
            raise ValueError("Assembly window must be nonnegative and request size positive")
        self.public_url = _public_endpoint(public_url)
        self.credentials = registry_tokens(credentials)
        self.max_body_bytes = max_body_bytes
        contract_dir = Path(contract_dir)
        normalized = json.loads((contract_dir / "telemetry.schema.json").read_text())
        ui = json.loads((contract_dir / "ui-api.schema.json").read_text())
        Draft202012Validator.check_schema(normalized)
        Draft202012Validator.check_schema(ui)
        self.registration_validator = Draft202012Validator(
            {**ui, "$ref": "#/$defs/RegisterApp"}, format_checker=FormatChecker()
        )
        guide = (contract_dir / "diagnostic-tool.md").read_text()
        heading = "## OpenTelemetry producer contract v1"
        _, found, remainder = guide.partition(heading)
        if not found:
            raise ValueError("producer contract heading is missing from diagnostic-tool.md")
        section = heading + remainder.split("\n## ", 1)[0]
        self.producer_contract = section.replace(
            "(telemetry.schema.json)", f"({self.public_url}/docs/telemetry.schema.json)"
        ).encode("utf-8")
        self.telemetry_schema = json.dumps(normalized, indent=2).encode("utf-8")
        self.store = Store(data_dir, Draft202012Validator(normalized), assembly_window)
        self.stop_worker = threading.Event()
        super().__init__(address, Handler, bind_and_activate=bind_and_activate)

    def serve_forever(self, poll_interval=0.5):
        worker = threading.Thread(target=self._publish, name="normalize-otlp", daemon=True)
        worker.start()
        try:
            super().serve_forever(poll_interval=poll_interval)
        finally:
            self.stop_worker.set()
            worker.join()

    def _publish(self):
        while not self.stop_worker.is_set():
            try:
                if self.store.publish_one() is not None:
                    continue
            except (OSError, sqlite3.Error):
                LOG.warning("Could not publish a normalized trace; will retry")
            self.stop_worker.wait(0.25)

    def package(self, result):
        app, key_id, token = result
        return {
            "app": app,
            "key_id": key_id,
            "span_contract_version": "1.0",
            "span_contract_url": self.public_url + CONTRACT_PATH,
            "env": {
                "OTEL_EXPORTER_OTLP_ENDPOINT": self.public_url,
                "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
                "OTEL_EXPORTER_OTLP_HEADERS": f"Authorization=Bearer%20{token}",
                "OTEL_SERVICE_NAME": app["service_name"],
            },
            "snippets": [{"language": "python", "label": "Python OTLP traces", "code": SNIPPET}],
        }


class Handler(BaseHTTPRequestHandler):
    server: DiagnosticServer

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def do_GET(self):
        self._route()

    def do_POST(self):
        self._route()

    def log_message(self, format, *args):
        # Request lines may contain client data. Log the server lifecycle only.
        pass

    def _route(self):
        url = urlsplit(self.path)
        is_otlp = url.path == OTLP_PATH
        try:
            if is_otlp:
                if self.command != "POST":
                    raise RequestError(405, "method_not_allowed", "Use POST for OTLP traces")
                self._ingest()
            elif self.command == "GET" and url.path == CONTRACT_PATH:
                self._send(200, self.server.producer_contract, "text/plain; charset=utf-8")
            elif self.command == "GET" and url.path == "/docs/telemetry.schema.json":
                self._send(200, self.server.telemetry_schema, JSON)
            else:
                self._registry(url)
        except RequestError as error:
            self._error(error, is_otlp)
        except sqlite3.OperationalError:
            self._error(
                RequestError(503, "storage_unavailable", "Storage temporarily unavailable"),
                is_otlp,
            )

    def _token(self):
        headers = self.headers.get_all("Authorization", [])
        if len(headers) == 1:
            parts = headers[0].split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1]
        raise RequestError(401, "unauthorized", "A valid bearer token is required")

    def _registry(self, url):
        base = "/api/v1/apps"
        if url.path != base and not url.path.startswith(base + "/"):
            raise RequestError(404, "not_found", "Route not found")
        account = self.server.credentials.get(token_digest(self._token()))
        if not account:
            raise RequestError(401, "unauthorized", "A valid registry bearer token is required")
        tenant, role = account
        store = self.server.store
        if self.command == "POST" and role != "admin":
            raise RequestError(403, "forbidden", "App registry administration is required")
        if url.path == base and self.command == "GET":
            cursor, limit = _pagination(url.query)
            self._json(200, store.apps(tenant, cursor, limit))
        elif url.path == base and self.command == "POST":
            self._register(tenant)
        else:
            suffix = url.path.removeprefix(base + "/").split("/")
            app_id, action = suffix[0], suffix[1:]
            if APP_ID.fullmatch(app_id):
                if not action and self.command == "GET":
                    app = store.app(tenant, app_id)
                    if app:
                        self._json(200, app)
                        return
                elif action == ["ingestion-key", "rotate"] and self.command == "POST":
                    package = store.rotate(tenant, app_id)
                    if package:
                        self._json(200, self.server.package(package), no_store=True)
                        return
            raise RequestError(404, "not_found", "App or route not found")

    def _register(self, tenant):
        if self.headers.get_content_type() != JSON:
            raise RequestError(415, "unsupported_media_type", "Expected application/json")
        try:
            data = json.loads(self._body(64 * 1024))
        except (ValueError, RecursionError) as error:
            raise RequestError(400, "invalid_json", "Expected a JSON object") from error
        fields = {}
        for error in self.server.registration_validator.iter_errors(data):
            # Avoid echoing values from the user's request.
            if error.validator == "required":
                missing = [key for key in error.validator_value if key not in data]
                fields.update(dict.fromkeys(missing, "Required."))
            else:
                path = ".".join(map(str, error.path)) or "body"
                fields[path] = f"Invalid value ({error.validator})."
        if fields:
            raise RequestError(422, "invalid_registration", "Invalid app registration", fields)
        try:
            result = self.server.store.register(tenant, data)
        except sqlite3.IntegrityError as error:
            raise RequestError(
                409, "service_name_exists", "Service name already registered"
            ) from error
        self._json(201, self.server.package(result), no_store=True)

    def _body(self, limit):
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
            raise RequestError(411, "length_required", "Provide one Content-Length header")
        raw_length = lengths[0]
        if not re.fullmatch(r"[0-9]{1,18}", raw_length):
            raise RequestError(400, "invalid_length", "Invalid Content-Length")
        length = int(raw_length)
        if length > limit:
            raise RequestError(413, "body_too_large", "Request exceeds the size limit")
        body = self.rfile.read(length)
        if len(body) != length:
            raise RequestError(400, "incomplete_body", "Request body ended early")
        return body

    def _ingest(self):
        token = self._token()
        if not self.server.store.authenticates_ingestion(token):
            raise RequestError(401, "unauthorized", "A valid app ingestion token is required")
        content_type = self.headers.get_content_type()
        if content_type not in {PROTOBUF, JSON}:
            raise RequestError(415, "unsupported_media_type", "Expected OTLP protobuf or JSON")
        encoding = self.headers.get("Content-Encoding", "identity").lower()
        if encoding not in {"gzip", "identity"}:
            raise RequestError(415, "unsupported_encoding", "Expected gzip or identity encoding")
        limit = self.server.max_body_bytes
        body = self._body(limit)
        request = ExportTraceServiceRequest()
        try:
            if encoding == "gzip":
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as decompressor:
                    body = decompressor.read(limit + 1)
                if len(body) > limit:
                    raise RequestError(413, "body_too_large", "Decompressed request exceeds limit")
            if content_type == PROTOBUF:
                request.ParseFromString(body)
            else:
                json_format.Parse(body, request, ignore_unknown_fields=True)
        except (
            DecodeError,
            json_format.ParseError,
            ValueError,
            OSError,
            EOFError,
            RecursionError,
            zlib.error,
        ):
            raise RequestError(400, "invalid_otlp", "Malformed OTLP trace request") from None
        result = self.server.store.accept(token, request)
        if result is None:
            raise RequestError(401, "unauthorized", "A valid app ingestion token is required")
        accepted, rejected = result
        response = ExportTraceServiceResponse()
        if rejected:
            message = "; ".join(f"{cause} ({count})" for cause, count in sorted(rejected.items()))
            if not accepted:
                raise RequestError(400, "spans_rejected", message)
            response.partial_success.rejected_spans = sum(rejected.values())
            response.partial_success.error_message = message
        self._otlp(200, response)

    def _error(self, error, is_otlp):
        if is_otlp:
            self._otlp(error.status, Status(message=str(error)))
        else:
            body = {"code": error.code, "message": str(error)}
            if error.fields:
                body["field_errors"] = error.fields
            self._json(error.status, body)

    def _json(self, status, response, no_store=False):
        self._send(status, json.dumps(response).encode("utf-8"), JSON, no_store=no_store)

    def _otlp(self, status, response):
        if self.headers.get_content_type() == JSON:
            self._send(status, json_format.MessageToJson(response).encode("utf-8"), JSON)
        else:
            self._send(status, response.SerializeToString(), PROTOBUF)

    def _send(self, status, body, content_type, no_store=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        if status == 503:
            self.send_header("Retry-After", "1")
        self.end_headers()
        self.wfile.write(body)


def _pagination(query):
    fields = parse_qs(query, keep_blank_values=True)
    cursors = fields.get("cursor", [""])
    limits = fields.get("limit", ["50"])
    if len(cursors) != 1 or len(limits) != 1:
        raise RequestError(400, "invalid_pagination", "Supply a single cursor and limit")
    cursor, limit = cursors[0], limits[0]
    if (
        (cursor and not APP_ID.fullmatch(cursor))
        or not re.fullmatch(r"[0-9]{1,3}", limit)
        or not 1 <= int(limit) <= 100
    ):
        raise RequestError(400, "invalid_pagination", "Invalid cursor or limit")
    return cursor, int(limit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", default=4318, type=int)
    parser.add_argument("--public-url", required=True, help="Public OTLP/HTTP base URL")
    parser.add_argument("--data-dir", type=Path, default=Path("data/otlp"))
    parser.add_argument("--contract-dir", type=Path, default=Path("docs"))
    parser.add_argument("--assembly-window", type=float, default=5)
    parser.add_argument("--max-body-bytes", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        credentials = json.loads(os.environ.get("RAG_DIAGNOSTIC_REGISTRY_TOKENS", "{}"))
        server = DiagnosticServer(
            (args.listen, args.port),
            data_dir=args.data_dir,
            contract_dir=args.contract_dir,
            credentials=credentials,
            public_url=args.public_url,
            assembly_window=args.assembly_window,
            max_body_bytes=args.max_body_bytes,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    with server:
        LOG.info("Diagnostic server listening on %s:%s", *server.server_address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

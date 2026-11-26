"""OpenTelemetry spans for the reference RAG app (ticket C2).

Follows the "OpenTelemetry producer contract v1" in Person 1's
docs/diagnostic-tool.md (Fault_Diagnostic_Tool branch):

- one execution span named ``rag.query`` per user query
- one child span per observed stage, tagged with ``rag.stage``
- metadata is always sent; queries, prompts, passages and answers are sent
  only when RAG_TRACE_CONTENT=1, because the contract makes content opt-in

Export is configured from the standard OpenTelemetry environment variables at
startup. Without an endpoint or trace file, spans are not recorded at all.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

TELEMETRY_VERSION = "1.0"
DEFAULT_SERVICE_NAME = "reference-rag"

tracer = trace.get_tracer("rag_pipeline")


def capture_content() -> bool:
    return os.getenv("RAG_TRACE_CONTENT", "").strip().lower() in {"1", "true", "yes"}


def setup_tracing() -> Any:
    """Configure span export once at app startup. Returns the provider, or None.

    OTLP/HTTP export uses the standard variables from the onboarding package:
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT (or OTEL_EXPORTER_OTLP_ENDPOINT),
    OTEL_EXPORTER_OTLP_TRACES_HEADERS (or OTEL_EXPORTER_OTLP_HEADERS) and
    OTEL_SERVICE_NAME. RAG_TRACE_FILE also appends each trace as OTLP/JSON to a
    local file, which is how the shared step-0 fixtures are produced before
    Person 1's receiver exists.
    """
    from dotenv import load_dotenv
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    # Deployment environment variables take precedence over local settings.
    load_dotenv(Path(__file__).resolve().parents[2] / '.env.otel', override=False)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    trace_file = os.getenv("RAG_TRACE_FILE")
    if not endpoint and not trace_file:
        return None
    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        return None

    service_name = os.getenv("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    if trace_file:
        provider.add_span_processor(SimpleSpanProcessor(OtlpJsonFileExporter(Path(trace_file))))
    trace.set_tracer_provider(provider)
    return provider


def otlp_json(spans: Sequence[Any]) -> dict[str, Any]:
    """Encode finished SDK spans as an OTLP/JSON ExportTraceServiceRequest."""
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    data = MessageToDict(encode_spans(spans), use_integers_for_enums=True)
    # OTLP/JSON writes trace and span IDs as hex, not protobuf's default base64.
    for resource_spans in data.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                for key in ("traceId", "spanId", "parentSpanId"):
                    if span.get(key):
                        span[key] = base64.b64decode(span[key]).hex()
    return data


class OtlpJsonFileExporter:
    """Appends one OTLP/JSON request per line to a local file."""

    def __init__(self, path: Path):
        self.path = path

    def export(self, spans: Sequence[Any]) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(otlp_json(spans)) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _mark_failed(span: Span, error: BaseException) -> None:
    # Only the exception class is exported: messages can carry user data or keys.
    span.set_attribute("error.type", type(error).__name__)
    span.set_status(Status(StatusCode.ERROR))


class StageSpan:
    def __init__(self, span: Span):
        self.span = span

    def output(self, value: Any) -> None:
        if capture_content():
            self.span.set_attribute("rag.stage.output_json", _json(value))


class ExecutionSpan:
    def __init__(self, span: Span):
        self.span = span
        context = span.get_span_context()
        self.trace_id = trace.format_trace_id(context.trace_id) if context.is_valid else None

    def finish(self, answer: str, instructions: str) -> None:
        if capture_content():
            self.span.set_attribute("rag.answer.text", answer)
            self.span.set_attribute("rag.generation.instructions", instructions)


@contextmanager
def execution_span(query: str, flag_reason: str | None = None) -> Iterator[ExecutionSpan]:
    """The ``rag.query`` span around one user query."""
    with tracer.start_as_current_span(
        "rag.query", record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute("rag.telemetry.version", TELEMETRY_VERSION)
        span.set_attribute("rag.stages.not_used", ["reranking"])  # main has no reranker
        if capture_content() and query.strip():
            span.set_attribute("rag.query.text", query)
        if flag_reason:
            span.set_attribute("rag.failure.flagged", True)
            span.set_attribute("rag.failure.reason", flag_reason)
        try:
            yield ExecutionSpan(span)
        except BaseException as error:
            span.set_attribute("rag.execution.status", "failed")
            _mark_failed(span, error)
            raise
        span.set_attribute("rag.execution.status", "completed")


@contextmanager
def stage_span(stage: str, *, k: int | None = None, input_data: Any = None) -> Iterator[StageSpan]:
    """A child span for one logical stage, tagged with ``rag.stage``."""
    with tracer.start_as_current_span(
        f"rag.{stage}", record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute("rag.stage", stage)
        if k is not None:
            span.set_attribute("rag.stage.k", k)
        if input_data is not None and capture_content():
            span.set_attribute("rag.stage.input_json", _json(input_data))
        try:
            yield StageSpan(span)
        except BaseException as error:
            span.set_attribute("rag.stage.status", "failed")
            _mark_failed(span, error)
            raise
        span.set_attribute("rag.stage.status", "completed")

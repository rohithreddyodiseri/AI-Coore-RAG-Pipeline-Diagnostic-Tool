# Connect a RAG application through OTLP

## 1. Register the RAG application

Save the onboarding response immediately. The ingestion secret in
`OTEL_EXPORTER_OTLP_HEADERS` is returned when the app is registered or its key
is rotated. Ordinary app-detail responses do not return it.

Do not commit the onboarding response or any ingestion token to source control.
Place the generated values in the application's secret manager, deployment
secrets, or an ignored local `.env` file.

## 2. Configure the client application

Set the exact values returned in the onboarding package:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT='http://127.0.0.1:4318'
export OTEL_EXPORTER_OTLP_PROTOCOL='http/protobuf'
export OTEL_EXPORTER_OTLP_HEADERS='Authorization=Bearer%20diag_generated-secret'
export OTEL_SERVICE_NAME='reference-rag'
export OTEL_TRACES_EXPORTER='otlp'
export OTEL_RESOURCE_ATTRIBUTES='deployment.environment.name=staging'
```

`OTEL_EXPORTER_OTLP_ENDPOINT` is a base URL. With `http/protobuf`, the
OpenTelemetry exporter adds the `/v1/traces` signal path. If the application
uses `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` instead, set its value to the full
URL:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT='http://127.0.0.1:4318/v1/traces'
```

Use one endpoint variable style. Keep the service name exactly equal to the
registered `service_name`. Set
`deployment.environment.name` to one of the environments registered for the
app.

For a shared deployment, the endpoint should use HTTPS. The loopback HTTP URL
is intended for local development.

## 3. Install the Python OpenTelemetry packages

For a Python RAG application:

```bash
python -m pip install \
  'opentelemetry-api>=1.30,<2' \
  'opentelemetry-sdk>=1.30,<2' \
  'opentelemetry-exporter-otlp-proto-http>=1.30,<2'
```

Framework-specific or zero-code instrumentation can create HTTP, database,
model, and vector-store spans. The diagnostic contract still requires one
tagged RAG execution span and tagged logical stage spans.

## 4. Initialize tracing when the app starts

Initialize one tracer provider for the application process:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


resource = Resource.create()
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("reference-rag")
```

The exporter reads the `OTEL_*` environment variables. Initialize it once
during process startup. Call `provider.shutdown()` during a graceful process
shutdown so buffered spans can be exported.

## 5. Instrument one RAG execution

Create one `rag.query` span per user query. Create descendant spans for the
logical RAG stages that actually run.

```python
import json
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


@contextmanager
def rag_stage(span_name: str, stage_name: str, **attributes):
    with tracer.start_as_current_span(
        span_name,
        attributes={"rag.stage": stage_name, **attributes},
    ) as stage_span:
        try:
            yield stage_span
        except Exception as error:
            stage_span.set_attribute("rag.stage.status", "failed")
            stage_span.set_attribute("error.type", type(error).__name__)
            stage_span.record_exception(error)
            stage_span.set_status(Status(StatusCode.ERROR))
            raise
        else:
            stage_span.set_attribute("rag.stage.status", "completed")


def answer_query(query: str) -> str:
    with tracer.start_as_current_span(
        "rag.query",
        attributes={
            "rag.telemetry.version": "1.0",
            "rag.query.text": query,
            "rag.stages.not_used": ("reranking",),
        },
    ) as execution_span:
        try:
            with rag_stage(
                "retrieve",
                "retrieval",
                **{"rag.stage.k": 5},
            ) as retrieval_span:
                retrieved_chunks = retrieve(query, top_k=5)
                retrieval_span.set_attribute(
                    "rag.stage.output_json",
                    json.dumps(
                        [
                            {
                                "chunk_id": chunk.id,
                                "doc_id": chunk.document_id,
                                "text": chunk.text,
                                "rank": rank,
                                "score": chunk.score,
                            }
                            for rank, chunk in enumerate(retrieved_chunks, start=1)
                        ],
                        allow_nan=False,
                    ),
                )

            with rag_stage(
                "assemble-context",
                "context_assembly",
            ) as context_span:
                context = build_context(retrieved_chunks)
                context_span.set_attribute(
                    "rag.stage.output_json",
                    json.dumps(
                        {
                            "text": context,
                            "chunk_ids": [chunk.id for chunk in retrieved_chunks],
                        }
                    ),
                )

            with rag_stage(
                "generate",
                "generation",
            ) as generation_span:
                answer = generate_answer(query, context)
                generation_span.set_attribute(
                    "rag.stage.output_json",
                    json.dumps(answer),
                )

            execution_span.set_attribute("rag.answer.text", answer)
            execution_span.set_attribute("rag.execution.status", "completed")
            execution_span.set_attribute("rag.failure.flagged", False)
            return answer
        except Exception as error:
            execution_span.set_attribute("rag.execution.status", "failed")
            execution_span.set_attribute("rag.failure.flagged", True)
            execution_span.set_attribute(
                "rag.failure.reason", "RAG pipeline execution failed"
            )
            execution_span.record_exception(error)
            execution_span.set_status(Status(StatusCode.ERROR))
            raise
```

Replace `retrieve`, `build_context`, and `generate_answer` with the client's
existing functions. The surrounding spans are the required integration work.

Set `rag.execution.status` before the execution span ends. A failed stage
should set `rag.stage.status="failed"`, record the exception, and set the
stage span status to `ERROR`.

The supported logical stages are:

| Stage                | `rag.stage` value  |
| -------------------- | ------------------ |
| Query embedding      | `query_embedding`  |
| Retrieval            | `retrieval`        |
| Reranking            | `reranking`        |
| Context construction | `context_assembly` |
| Model generation     | `generation`       |

Declare a deliberately bypassed stage with the
`rag.stages.not_used` string-array attribute on the execution span. An
unobserved stage should otherwise remain absent.

## 6. Capture evidence intentionally

The diagnostic engine can reason only from evidence that was captured.
`rag.stage.input_json` and `rag.stage.output_json` must each be a string
containing one complete JSON value.

The important output shapes are:

- Retrieval and reranking use an ordered JSON array. Each item includes
  `chunk_id`, `text`, and a one-based `rank`. Include `doc_id` and
  numerical `score` when available.
- Context assembly uses the context string or an object containing `text`
  and ordered `chunk_ids`.
- Generation uses a JSON string containing the actual model output.
- Query embedding uses an array of vectors when vector evidence is enabled.

Capture query text, prompts, retrieved text, context, and answers only when the
client's data policy permits it. Apply redaction before setting span attributes.
Never include API keys, authorization headers, reference answers, or known
failure labels in telemetry.

An empty retrieval result is represented by the literal JSON string `"[]"`
as the value of `rag.stage.output_json`. If retrieval content was withheld,
omit the attribute. These cases have different diagnostic meanings.

## 7. Flag a trace for diagnosis

Runtime failures are flagged automatically when the execution status is
`failed`. A completed query can be submitted for quality diagnosis by setting
these attributes before its execution span ends:

```python
execution_span.set_attribute("rag.failure.flagged", True)
execution_span.set_attribute(
    "rag.failure.reason",
    "User reported that the answer omitted requested details",
)
```

User feedback often arrives after the original span has ended. The current
OTLP contract cannot modify an ended span. Until a separate feedback endpoint
is implemented, decide whether a known synchronous quality signal should flag
the execution before it ends.

## 8. Verify the connection

Run one instrumented query and allow the batch exporter a few seconds to send
the spans. During local tests, shut down the application gracefully or call:

```python
provider.force_flush()
```

List registered apps using the registry token:

```bash
curl -s http://127.0.0.1:4318/api/v1/apps \
  -H "Authorization: Bearer ${REGISTRY_TOKEN}" \
  | python -m json.tool
```

The registered app's `last_trace_at` should change from `null` to a UTC
timestamp after the receiver accepts a batch with the matching service name.

Normalized executions are written under:

```text
data/otlp/<tenant>/<app_id>/executions/
```

Executions without captured query text are stored under:

```text
data/otlp/<tenant>/<app_id>/metadata-only/
```

The diagnostic engine requires a nonblank query to judge a flagged execution.

## 9. Run the diagnostic consumer

For a query-bearing app, run the existing watcher against that app's execution
directory:

```bash
rag-diagnose watch \
  --trace-dir 'data/otlp/demo/<app_id>/executions' \
  --config '/path/to/app-diagnostic-config.json'
```

Use app-specific `runs_dir` and `judge_cache` values in the diagnostic
configuration. The watcher processes terminal executions whose
`failure_signal.flagged` value is true.

## 10. Rotate a compromised ingestion key

Use the app ID and registry administrator token:

```bash
curl -i -X POST \
  "http://127.0.0.1:4318/api/v1/apps/<app_id>/ingestion-key/rotate" \
  -H "Authorization: Bearer ${REGISTRY_TOKEN}"
```

Update the client's `OTEL_EXPORTER_OTLP_HEADERS` with the value in the new
onboarding package and restart the client. Rotation immediately revokes the
previous ingestion key.

## Troubleshooting

### The app remains disconnected

- Confirm the diagnostic server is still running.
- Confirm the client can reach the endpoint from its own host or container.
  `127.0.0.1` refers to the client container itself when the RAG app and
  receiver run in different containers.
- Confirm `OTEL_SERVICE_NAME` exactly matches the registered service name.
- Confirm the client uses the current ingestion key after a rotation.
- Call `provider.force_flush()` during local tests to send buffered spans.

### The receiver returns 401

The authorization header is absent, malformed, or contains an expired
ingestion key. Use the exact `OTEL_EXPORTER_OTLP_HEADERS` value returned in
the latest onboarding package.

### The receiver returns 400 or partial success

Check that trace IDs, span IDs, timestamps, and OTLP payload encoding are
valid. An execution span with a different `service.name` is rejected. Partial
success means some spans were accepted and some were rejected. OpenTelemetry
exporters do not retry spans reported through a partial-success response.

### A trace is metadata-only

Enable permitted query capture and set a nonblank `rag.query.text` attribute
on the `rag.query` span. A metadata-only trace remains useful for connection
and operational inspection, but the diagnostic judge cannot assess it.

### Evidence is missing from the diagnosis

Confirm that stage spans are descendants of the `rag.query` span, use a
supported `rag.stage` value, and set a valid `rag.stage.status`. Verify that
JSON evidence is complete and matches the documented shape. The receiver
discards malformed evidence and records an inspection note.

## Related contracts

- [OpenTelemetry producer contract](diagnostic-tool.md#opentelemetry-producer-contract-v1)
- [Normalized execution schema](telemetry.schema.json)
- [Diagnostic receiver setup](otlp-server.md)

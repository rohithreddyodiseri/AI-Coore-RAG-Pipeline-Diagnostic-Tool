# Diagnostic UI API v1

Person 1 implements these endpoints and maps stored data into
[ui-api.schema.json](ui-api.schema.json). Person 2 renders the API's response
objects and can start from the [example payloads](ui-api.examples.json).
Validate each body against the named definition at `#/$defs/<Name>`. The
[telemetry schema](telemetry.schema.json) governs ingestion separately.
The example file's `artifacts_by_id` map supplies individual `Artifact`
responses for evidence link mocks.

## Implemented telemetry server routes

The local Python server implements the registry routes, `GET /traces`,
`GET /traces/{execution_id}`, artifact reads, and `GET /analytics/health` below.
It also exposes `GET /traces/{execution_id}/spans`, returning OTLP-shaped JSON
with hexadecimal trace/span IDs for the span inspector. Reads use registry
admin or viewer credentials, enforce tenant scope, and return `Cache-Control:
no-store`. Execution IDs are opaque and cursors follow execution-ID ordering.

Health counts describe received supported RAG executions. Sampling coverage is
unknown. Flagged executions are queued after trace assembly, including a backfill
of existing flagged captures on restart. The judge runs separately from ingestion;
the OTLP acknowledgment does not wait for the provider. State progresses through
`queued` and `running` to `completed`, `failed`, or `insufficient_evidence`.
Completed work and its evidence are persisted per app and tenant; leases prevent
concurrent processing and recover interrupted jobs after 60 seconds without a
heartbeat. Failures are retried only by an explicit admin request.

`POST /traces/{execution_id}/diagnosis` requests diagnosis or retries failed work.
It requires registry admin credentials and returns 202 `DiagnosticRun`. Repeated
requests during queued/running/completed states return the existing run. It also
supports unflagged executions. `--no-auto-diagnose` disables automatic scheduling.
Missing query content produces `insufficient_evidence` without calling the judge.

`latest_live_diagnosis` includes the validated result, audit fingerprints, and
technical `verdict.mechanism`. Technical diagnoses have a null legacy FP category;
they are not mapped to FP1–FP7. `verdict.evaluations` and
`verdict.unresolved_mechanisms` expose individual mechanism checks and missing
evidence. Citation artifact IDs resolve through the existing artifact endpoint to
the exact captured judge evidence. Health analyzed/abstention counts use completed
live diagnoses; FP category counts remain zero for technical diagnoses. Replay
capabilities remain unavailable. Standalone diagnostic-run listing, benchmark,
and replay routes below describe the planned contract and are not implemented.
The live UI does not substitute examples for missing results.

## Routes

All routes are relative to `/api/v1` and use JSON. Scope every lookup to the
authenticated user's active tenant. Require tenant admin permission for
registration and key rotation, and sandbox permission for replay. Ingestion
credentials authorize telemetry export only. Use `Error` for non-success JSON
bodies. Use 404 for resources outside the tenant's scope.

| Screen | Method and path | Request | Success |
|---|---|---|---|
| Registry | `GET /apps` | `cursor`, `limit` | 200 `AppList` |
| Registry | `POST /apps` | `RegisterApp` | 201 `OnboardingPackage` |
| Connection status | `GET /apps/{app_id}` | — | 200 `App` |
| Replace ingestion key | `POST /apps/{app_id}/ingestion-key/rotate` | — | 200 `OnboardingPackage` |
| Health dashboard | `GET /analytics/health` | `from`, `to`, optional `app_id`, `environment` | 200 `Dashboard` |
| Benchmark reports | `GET /benchmarks` | `cursor`, `limit` | 200 `BenchmarkList` |
| Benchmark detail | `GET /benchmarks/{benchmark_id}` | — | 200 `BenchmarkReport` |
| Trace explorer | `GET /traces` | `cursor`, `limit`, optional `from`, `to`, `app_id`, `environment`, `status`, `flagged`, `diagnosis_state` | 200 `ExecutionList` |
| Request or retry diagnosis | `POST /traces/{execution_id}/diagnosis` | — | 202 `DiagnosticRun` |
| Trace and verdict | `GET /traces/{execution_id}` | — | 200 `TraceDetail` |
| Evidence detail | `GET /traces/{execution_id}/artifacts/{artifact_id}` | — | 200 `Artifact` |
| Audit log | `GET /diagnostic-runs` | `cursor`, `limit`, optional `execution_id`, `source`, `state` | 200 `RunList` |
| Run and audit detail | `GET /diagnostic-runs/{diagnostic_run_id}` | — | 200 `DiagnosticRun` |
| Start replay | `POST /traces/{execution_id}/replays` | `ReplayRequest` | 202 `ReplayJob` |
| Poll replay | `GET /replays/{replay_id}` | — | 200 `ReplayJob` |

Use opaque cursors and return `next_cursor: null` on the last page. Use UTC
timestamps and half-open ranges `[from, to)`. The dashboard requires both
window bounds and rejects reversed or empty ranges. Queryable status values
match their response enums. All IDs except OpenTelemetry IDs are opaque to the UI.

## Registry and onboarding

- Person 1 enforces unique `service_name` within a tenant and uses
  `deployment.environment.name` from the execution span's OpenTelemetry
  resource for environment filtering. A missing environment is `null` in
  trace responses. `last_trace_at: null` means the app is waiting for its
  first accepted trace.
- Return the secret-bearing `env` only on successful registration or rotation.
  Use `Cache-Control: no-store` for those responses. Rotation replaces and
  revokes the prior key. Person 2 keeps the package in memory for copy/paste
  and clears it when leaving the wizard. List and detail responses expose
  ordinary app metadata.
- The generic OTLP endpoint is an HTTPS **base URL**. With `http/protobuf`,
  exporters use its `/v1/traces` path. A snippet using the trace-specific
  endpoint variable instead must include the full `/v1/traces` path.
  Snippets configure the trace exporter and refer to environment variables for
  credentials. [OpenTelemetry exporter configuration](https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/)

## Dashboard, benchmarks, and audit

- Count distinct normalized executions in the selected window. In a trace
  containing several RAG runs, each execution counts once. Count the latest
  live verdict for a flagged execution, including abstentions and
  `no_failure`. Exclude replay runs, engine errors, and pre-judge
  `insufficient_evidence` from `analyzed_flagged_traces`. Calculate
  `abstention_rate = abstained_traces / analyzed_flagged_traces`, with
  `null` when the denominator is zero.
- Rank top failing apps by `flagged_traces` and display runtime failures
  separately. Return latency bottlenecks in descending mean duration using
  known, observed stage timings. Use `sample_count` to show how many
  observations contributed to each mean. Return zero for empty category
  buckets. Count only the primary FP1–FP7 category once per diagnosed trace.
  Render category names using the [engine taxonomy](../backend/src/rag_diagnostic/taxonomy.py).
  Display `sampled` or `unknown` coverage alongside KPIs when applicable.
- Use benchmark `actual_labels` as rows and `predicted_labels` as columns.
  Omitted cells are zero. Represent an exact multi-label set as a sorted
  `+`-joined string, an empty set as `NONE`, and a judge abstention as
  `ABSTAIN` in the prediction column. Include category matrices and scores
  only for cases with category ground truth. For per-label metrics, precision
  is TP / (TP + FP), recall is TP / (TP + FN), and support is the actual
  positive count. An abstention is a missed prediction for any true label.
  Score undefined rates as `null`. The current evaluator already produces
  an exact stage-set confusion matrix and accuracy. Person 1 needs case-level
  expected and predicted label sets to calculate the per-label metrics and a
  category matrix.
- Map audit fields from `run.code_revision`, `run.config_fingerprint`,
  `observations.input.source_sha256`, and `observations.judge`. Resolve the
  provider's exact model version when available and use `null` otherwise.
  Capture the effective settings at execution time. A replay gets its own
  audit row. Expose sanitized audit fields instead of serving raw engine logs.

## Graph and verdict

- Use the normalized run ID as the `execution_id` or map it to a stable
  backend ID. Return the OpenTelemetry trace ID separately. Render graph
  nodes by `node_id` and keep graph edges acyclic and local to the execution.
  Use `parent_span` for span hierarchy and `observed_dependency` for
  captured stage dependencies. Render `query_rewrite` when observed by an
  app. The current producer contract does not require it.
- Use node `status` to highlight runtime failures and the verdict to
  highlight attributed faults. Fetch node evidence on demand. All artifact
  references in the graph and verdict must resolve within the execution,
  including a metadata-only response when content is withheld. Apply the
  same content permissions to citation quotes and artifact contents.
  Use `execution_artifact_ids` for evidence such as the original query or
  final answer that belongs to the whole execution.
  An available `chunks` artifact with `content: []` is an observed empty result.
- Map the engine's `primary_fault_stage`, `failure_category`, and validated
  evidence into the verdict. Show confidence as an uncalibrated evidence
  score. Display abstention with no gauge. A completed job may have an
  abstained verdict. If the judge could not run because required evidence was
  missing, use job state `insufficient_evidence` with no verdict. Return an
  empty recommendations list until the backend has a recommendation source.

## Replay

- `judge_only` runs the diagnostic judge again against the same captured
  evidence. Its `result_execution_id` equals the source execution ID. Its
  separate `diagnostic_run_id` identifies the new verdict.
- `pipeline_what_if` accepts only the supported pipeline overrides advertised
  for this execution. It requires a sandbox runner with authorized inputs.
  For example, a higher `top_k` needs a deeper authorized candidate set or a
  sandbox retrieval run. A prompt change requires generation in the sandbox.
  Save the resulting execution as a new record before judging it. The original
  trace and live verdict remain available unchanged.
- Render only advertised controls. Use the prompt choices in capability
  `options`. When a mode is unavailable, show its reason and disable it.
  The backend validates permissions and overrides again at submission and
  returns a structured error for unsupported requests. Poll until
  `completed`, `failed`, or `insufficient_evidence`. A completed job links
  to its result execution and exact diagnostic run for comparison.

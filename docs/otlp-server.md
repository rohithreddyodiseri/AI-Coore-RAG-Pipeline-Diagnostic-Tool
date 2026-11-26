# Diagnostic OTLP server

The service implements the app registration, listing, detail, and key rotation
routes in the [UI API contract](ui-api.md). RAG producers send OTLP/HTTP
trace spans to `POST /v1/traces`. The receipt acknowledges a durable span
write. Normalization runs in the background after the assembly window.

Client application owners should follow
[Connect a RAG application through OTLP](client-otlp-onboarding.md).

## Start locally

From the repository root:

```bash
python -m pip install -e '.[server]'
export RAG_DIAGNOSTIC_REGISTRY_TOKENS="$(
  python -c 'import json, secrets; print(json.dumps({"demo": {"admin": secrets.token_urlsafe(32)}}))'
)"
rag-diagnostic-server --public-url http://127.0.0.1:4318
```

Set `RAG_DIAGNOSTIC_REGISTRY_TOKENS` from a secret store for shared deployments.
It is a JSON object keyed by tenant, with an `admin` token and optional
`viewer` token for each tenant. Admins can register apps and rotate keys.
Viewers can list and inspect apps. These bootstrap tokens are separate from
per-app ingestion keys. Production deployments should terminate HTTPS at a
reverse proxy, keep the listener private, and set `--public-url` to the public
HTTPS base URL. Use one receiver process per data directory.

In another terminal with the same registry environment, register an app:

```bash
REGISTRY_TOKEN="$(python -c 'import json, os; print(json.loads(os.environ["RAG_DIAGNOSTIC_REGISTRY_TOKENS"])["demo"]["admin"])')"
curl --fail-with-body http://127.0.0.1:4318/api/v1/apps \
  -H "Authorization: Bearer ${REGISTRY_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"service_name":"reference-rag","framework":"custom","environments":["staging"]}'
```

Copy the returned `env` values into the reference RAG server's environment
and follow the returned span contract. The package contains the ingestion key
once. App reads return connection metadata only. After the SDK exports an
accepted span carrying the registered service name, `last_trace_at` is set.
Key rotation revokes the previous ingestion key. The receiver acknowledges
retries without storing duplicate spans.

## Where executions go

- Raw spans, OTLP resources, instrumentation scopes, app records, and pending
  assembly deadlines persist in `data/otlp/ingestion.sqlite3` by default.
- Normalized query-bearing records go to
  `data/otlp/<tenant>/<app_id>/executions/`. The assembly window defaults to
  five seconds after the execution span arrives. Later spans remain in the
  raw store. The published snapshot stays stable.
- Metadata-only records go into a sibling `metadata-only/` directory.
  They can support trace inspection after read APIs are added. The current
  engine requires actual query text to judge a run.

To diagnose flagged, query-bearing executions with the existing file watcher,
run it for the specific app. Use a separate diagnostic configuration with
app-specific `runs_dir` and `judge_cache` paths:

```bash
rag-diagnose watch --trace-dir 'data/otlp/demo/<app_id>/executions' --config /path/to/app-config.json
```

The HTTP receiver does not invoke the judge in the request path. The trace,
dashboard, and replay read APIs are separate backend work. Use `--data-dir`,
`--contract-dir`, `--assembly-window`, and `--max-body-bytes` to configure the
receiver. The contract directory must contain the existing telemetry JSON
Schema, UI JSON Schema, and producer guide.

# Diagnostic frontend

Use Node.js 22.18+ and Python 3.10–3.13.

Next.js renders applications, health metrics, executions, captured artifacts,
and span timings from the Python diagnostic server. Browser requests go through
`src/app/api/`; `src/lib/backend.ts` adds the registry token server-side.

From the repository root:

```bash
python -m pip install -e '.[dev,server]'
npm --prefix frontend ci
python scripts/dev.py
```

Open http://127.0.0.1:3000. The launcher creates an ephemeral local admin token
and shares it between the two servers. If the default ports are occupied, use
`python scripts/dev.py --backend-port 14318 --frontend-port 13000`. Registry and telemetry persist in
`data/otlp/`. Stop both servers with Ctrl+C. No Gemini key is needed to inspect
telemetry or register an application.

## Running the servers separately

Set a URL-safe token of 32–256 characters on both servers. For example, generate
one with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`.

Backend, from the repository root:

```bash
export RAG_DIAGNOSTIC_REGISTRY_TOKENS='{"local":{"admin":"YOUR_GENERATED_TOKEN"}}'
rag-diagnostic-server --public-url http://127.0.0.1:4318
```

Frontend: copy `.env.example` to `.env.local` in this directory and replace
`RAG_REGISTRY_TOKEN` with that same token, then run:

```bash
npm run dev -- --hostname 127.0.0.1
```

`RAG_API_BASE_URL` defaults to `http://127.0.0.1:4318`. These environment
variables are server-only; never prefix the registry token with `NEXT_PUBLIC_`.
This is a local admin console: keep the frontend on loopback. A remote deployment
needs user authentication before exposing the server's registry credential.

## Receiving real telemetry

Use **Connect application**, save the returned exporter settings on your RAG
service, and instrument it according to the returned span contract. Enable
content capture on that service to inspect inputs and outputs. After sending a
query, refresh the dashboard or trace explorer. Registration alone does not
produce traces. See [client onboarding](../docs/client-otlp-onboarding.md).

Flagged executions automatically enter a durable diagnostic queue after the
trace assembly window. Previously received flagged traces are picked up on backend
restart too. Set `GEMINI_API_KEY` in the diagnostic backend's environment or root
`.env`; the worker uses the existing stage diagnostic engine and captured evidence.
A judge request can consume API quota. The original RAG query is never replayed.

The trace detail view polls queued/running diagnoses and displays validated
technical mechanisms, supporting citations, and all mechanism checks. Missing
query content produces `insufficient_evidence` with capture instructions; provider
errors produce `failed` with a retry action. Use **Run diagnosis** for an unflagged
trace, or **Retry diagnosis** after fixing an error. Completed diagnoses are
idempotent and are not repeated by refreshes or repeated requests.

`rag-diagnostic-server --no-auto-diagnose` disables automatic scheduling while
retaining explicit requests. Use `--diagnostic-config configs/diagnostic.json` to
choose stage judge settings. Benchmark examples and simulated states are not
displayed. Replay remains unavailable. The span inspector lists up to 100 executions;
individual trace pages always link to the requested execution's spans.

## Structure and checks

- `src/app/`: routes and thin server handlers.
- `src/features/traces/`: execution explorer and span inspector.
- `src/components/`: shared graph and status views.
- `src/hooks/`: cancellable API loading.
- `src/lib/`: API types, browser client, and server gateway.
- `../docs/ui-api.schema.json`: shared API contract; examples stay in `docs/`.

```bash
npm run lint
npm test
npx tsc --noEmit
npm run build
```

# Step-0 fixtures (Person 3 drafts)

The handoff asks all three people to agree on three fixtures for one failing
query:

1. an OTLP trace
2. the expected payload for the diagnostic engine
3. the API response the diagnostic UI renders

These files are Person 3's drafts of the first two. The third belongs to
Persons 1 and 2.

| File | Contents |
|---|---|
| `healthy.otlp.jsonl` | OTLP/JSON trace of a successful query (one export request per line) |
| `healthy.expected.json` | What `rag_otel_adapter.normalize` returns for it |
| `failing.otlp.jsonl` | The same query failing at generation because no `GEMINI_API_KEY` is set (error type `ValueError`) |
| `failing.expected.json` | The normalized, automatically flagged record |

They come from the instrumented pipeline and the real `RAG_TRACE_FILE`
exporter, but retrieval and the model were stubbed. Content capture was on.
Replace them with traces from a real run once the index is built and
`GEMINI_API_KEY` is in `.env`. For the failing query, give the app a fake key so
Gemini rejects the request.

PowerShell (Windows):

```powershell
$env:RAG_TRACE_CONTENT = "1"
$env:RAG_TRACE_FILE = "fixtures/step0/healthy.otlp.jsonl"
rag-pipeline ask "What methods are used to improve retrieval quality?"

$env:RAG_TRACE_FILE = "fixtures/step0/failing.otlp.jsonl"
$env:GEMINI_API_KEY = "not-a-real-key"
rag-pipeline ask "What methods are used to improve retrieval quality?"
Remove-Item Env:GEMINI_API_KEY, Env:RAG_TRACE_FILE

python -m rag_otel_adapter fixtures/step0/healthy.otlp.jsonl > fixtures/step0/healthy.expected.json
python -m rag_otel_adapter fixtures/step0/failing.otlp.jsonl > fixtures/step0/failing.expected.json
```

bash (macOS/Linux):

```bash
export RAG_TRACE_CONTENT=1
RAG_TRACE_FILE=fixtures/step0/healthy.otlp.jsonl rag-pipeline ask "What methods are used to improve retrieval quality?"
GEMINI_API_KEY=not-a-real-key RAG_TRACE_FILE=fixtures/step0/failing.otlp.jsonl rag-pipeline ask "What methods are used to improve retrieval quality?"
python -m rag_otel_adapter fixtures/step0/healthy.otlp.jsonl > fixtures/step0/healthy.expected.json
python -m rag_otel_adapter fixtures/step0/failing.otlp.jsonl > fixtures/step0/failing.expected.json
```

On Windows, PowerShell 5's `>` writes UTF-16, so use PowerShell 7 (`pwsh`, which
VS Code's terminal uses when it is installed) or add `| Out-File -Encoding utf8`.

Delete the old `.otlp.jsonl` file first, because the exporter appends to it.
`tests/test_otel_adapter.py` checks that each expected file matches the
adapter's output and validates against `docs/telemetry.schema.json`.

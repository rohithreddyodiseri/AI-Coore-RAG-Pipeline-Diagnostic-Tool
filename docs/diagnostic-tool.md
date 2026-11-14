# RAG Pipeline Diagnostic Tool

The tool connects to a target RAG pipeline and acts as an automated attribution
engine. It executes a query once, captures the pipeline's actual stage inputs and
outputs, ingests its source corpus, and uses a Gemini LLM judge to produce an
evidence-cited diagnostic report. It accepts neither a reference answer nor a
known answer location. When observations are insufficient, it abstains.

This project's default connection targets `rag_pipeline.pipeline.RAGPipeline`.
Both `rag-pipeline ask` and the diagnostic adapter call the same `execute()`
method, which uses the real retrieval, context assembly, and generation functions.
The adapter does not reconstruct those stages. This pipeline currently has no
reranker; the trace explicitly records reranking as unused.

## Run a query

From the project root:

1. Activate the virtual environment and install the project:

   ```bash
   source .venv/bin/activate
   python -m pip install -e '.[dev]'
   ```

2. Set `GEMINI_API_KEY` in the ignored `.env` file. Generation and judging default
   to `gemini-2.5-flash-lite`. `RAG_LLM_MODEL` changes generation; `judge_model` in
   `configs/diagnostic.json` changes the judge. No key belongs in source or reports.

3. If the corpus has not been downloaded, run:

   ```bash
   python download-dataset.py
   ```

4. Build the target pipeline's index if it does not already exist:

   ```bash
   rag-pipeline index
   ```

   Keep embedding settings consistent with the existing index. Use `--reset`
   only when deliberately rebuilding the selected collection.

5. Run the query through the diagnostic tool:

   ```bash
   rag-diagnose "What methods improve retrieval quality?" \
     --config configs/diagnostic.json \
     --report-out runs/report.json \
     --trace-out runs/trace.json
   ```

   This executes the connected pipeline, generates its response, captures its
   trace and corpus, invokes the judge, and prints the attribution. The JSON
   report is saved to `runs/report.json`; the original trace is saved separately
   when `--trace-out` is given. Add `--json` to print the structured report.
   Without `--report-out`, reports go to `runs/reports/<run_id>.json`.
   Every diagnostic attempt is also recorded in `runs/diagnoses.jsonl`.

The module invocation works without installing the command entry point:

```bash
PYTHONPATH=src .venv/bin/python -m rag_diagnostic \
  "What methods improve retrieval quality?" \
  --config configs/diagnostic.json --report-out runs/report.json
```

To select a different local corpus/index:

```bash
rag-diagnose "Your question" \
  --corpus-dir /path/to/corpus --chroma-dir /path/to/chroma \
  --collection my_collection --top-k 5 --report-out runs/report.json
```

Additional options include `--embedding-model`, `--llm-model`, and `--judge-model`.
Use `rag-diagnose --help` to inspect the interface. Corpus download, embedding
model download, and uncached Gemini calls need network access. The current index
is used as configured; diagnosis does not rebuild or mutate it.

## Captured execution and report

The original trace contains:

- Query and actual target settings/code identity.
- Retrieval input, cutoff, returned chunk IDs, source IDs, text, scores, and ranks.
- Reranking inputs/outputs when present, or an explicit unused-stage record.
- Context assembly inputs and the exact text passed to generation.
- Generation prompt, instructions, model, temperature, and final response.
- Source corpus documents, declared scope, and a content-derived snapshot ID.
- A separate deeper retrieval inspection probe, distinguished from the execution.

The report contains `run_id`, `query`, `answer`, `status`, `primary_fault_stage`,
`failure_category`, `confidence`, `evidence`, `contributing_stages`, and all seven
category assessments. `failure_categories`/`stages` include all supported findings;
`unresolved_categories` and `attribution_complete` make partial attribution explicit.
Each assessment cites stable artifact IDs and verbatim excerpts. The engineer can
check those citations against `observations.input.artifacts` and the full original
execution/corpus in `observations.trace`.

## Failure categories and abstention

The judge uses [Barnett et al.'s seven failure points, §5](https://arxiv.org/pdf/2401.05856):

| Category | Attribution | Required evidence |
|---|---|---|
| FP1 Missing Content | Corpus | A complete, explicitly scoped corpus inspection. Top-K absence is insufficient. |
| FP2 Missed Top Ranked | Retrieval/reranking | Relevant indexed information demonstrably below the observed selection cutoff. |
| FP3 Not in Context | Context assembly/reranking | Retrieved information lost before the actual generation context. |
| FP4 Not Extracted | Generation | Required facts present in context but missed by the response. |
| FP5 Wrong Format | Generation/prompt | An explicit output-format instruction violated by the response. |
| FP6 Incorrect Specificity | Generation | A mismatch between requested scope/detail and response specificity. |
| FP7 Incomplete Answer | Generation | Requested parts omitted despite available context. |

Schema validation, exact citation checks, observability rules, and confidence
thresholds constrain the LLM's findings. Unknown, contradictory, incomplete, or
low-confidence evidence causes a category to remain unresolved. Supported faults
can coexist with unresolved categories. `ABSTAIN` means no fault is supported;
`NONE` requires all checks to be confidently clear or inapplicable. Multiple
contributing stages and categories are retained rather than forced into one cause.
The confidence is an uncalibrated model evidence score. Observational attribution
is not a guarantee that changing one component will repair the answer.

## Corpus scope and model input limits

Corpus ingestion is enabled by default. The full supplied corpus is retained in
the recorded trace. `corpus_budget_chars` bounds the corpus text actually sent to
the judge; documents referenced by retrieval and deeper inspection are selected
first, then stable source order. When any content is omitted or shortened, the
judge receives an explicitly incomplete corpus view with selection metadata.
It cannot claim FP1 on that view. No selection or text overlap decides a verdict;
semantic attribution remains the LLM's job. Increase the budget only within the
judge model's input limits. A source enumeration failure is recorded as an
inspection error, not inferred to mean an empty knowledge base.

`max_judge_input_chars` caps the entire judge request. If the recorded stage
artifacts plus the corpus view exceed it, the tool abstains. Stage inputs and
outputs are not silently dropped. `deep_k` controls the separate retrieval probe.
`inspect_corpus: false` disables source inspection and limits corpus-level claims.

## Connect a pipeline this tool did not build

Implement `PipelineAdapter` in `src/rag_diagnostic/adapter.py` by calling the target
system's existing functions or reading its instrumented logs. The engine depends
only on this interface:

| Method | Contract |
|---|---|
| `describe()` | Resolved settings, target code/model/prompt revisions, and provider identity, excluding secrets. |
| `run(query)` | Execute the real target once and return its recorded trace, or a `QueryTrace` for an already captured execution. |
| `corpus_snapshot()` | Return source `documents`, immutable `snapshot_id`, a named `scope`, and truthful `complete` flag. |
| `retrieve(query, k)` | Optional deeper search. Raise `NotImplementedError` if unavailable; the missing probe remains explicit. |
| `generation_instructions()` | Optional actual instructions if not already in the trace. |

`run()` returns a dictionary with `query`, `final_answer`, and `stages`.
Each observed stage (`retrieval`, `reranking`, `context_assembly`, `generation`)
has `input_data` and `output_data`. Retrieval/reranking outputs are arrays of
objects containing `chunk_id`, `doc_id`, `text`, one-based `rank`, and `score`.
Their `k` records the relevant selection cutoff. Context output is the exact
string or `{chunk_ids, text}`. Generation input records the actual prompt,
instructions and model settings. Mark absent stages with `status: "not_used"`;
unknown outputs must not be invented. Native `QueryTrace` values are also accepted.

`corpus_snapshot()` returns `{documents: [{doc_id, text}], snapshot_id, scope,
complete}`. A sample, bounded search, or inaccessible corpus must have
`complete: false`. A complete external corpus is an engineer's interface guarantee;
the tool cannot independently establish that a remote system disclosed everything.
No reference answers or benchmark fault labels may enter the trace or adapter metadata.

Expose a factory `build(settings: dict) -> PipelineAdapter`. Then connect it with:

```bash
rag-diagnose "Your query" --adapter your_package.connection:build \
  --pipeline-config target.json --report-out runs/report.json
```

`target.json` contains the target's own settings. The default factory is
`rag_diagnostic.adapters.open_ragbench:build`, already wired to this project's
pipeline. In Python, call `Diagnoser(config).diagnose(query, adapter)`. A service
that already has a complete trace can also call `DiagnosticEngine(config).diagnose(trace,
pipeline=metadata)` without re-executing the query.

## Repeatability and validation

Run records contain the tool/target revisions, source hashes, settings, actual
trace, corpus view metadata, exact judge prompt/schema, response metadata, raw
judge output, and validated report. Identical captured artifacts, code, model and
settings reuse the cached assessment. A new live response is a new diagnostic
input, even when the query is unchanged. Low temperatures alone do not guarantee
remote-model repeatability. Retain the logs/cache and pin model identities.

The shared Gemini wrapper spaces uncached requests by
`GEMINI_MIN_REQUEST_INTERVAL_SECONDS` (default 2). Adjust for the account's actual
quota; 429/API errors are recorded as operational failures rather than blamed on a
RAG stage. Test locally with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest
```

Tests cover real index execution with mocked model responses, direct CLI
connection, trace fidelity, source ingestion, evidence validation, abstention and
cached replay. They do not establish semantic judge accuracy or confidence
calibration; validate those against domain-reviewed failures.

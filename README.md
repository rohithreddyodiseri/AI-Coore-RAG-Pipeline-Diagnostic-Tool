# Open RAGBench RAG Pipeline

## Diagnostic frontend and backend

The repository now separates `backend/` (Python packages and tests) and
`frontend/` (Next.js). Shared API contracts live in `docs/`; configurations,
fixtures, and runtime data remain at the repository root. Utility scripts live
in `scripts/`. Run Python commands from the repository root.

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,server]"
npm --prefix frontend ci
python scripts/dev.py
```

Open http://127.0.0.1:3000. This starts both servers with matching local registry
credentials. Connect an application and send OpenTelemetry traces to see health
metrics, execution details, artifacts, and span timings. Flagged executions are
automatically queued for the diagnostic judge after trace assembly. The UI
shows progress, validated mechanisms, cited evidence, and retryable errors. See [frontend setup](frontend/README.md) and
[backend layout](backend/README.md).

```text
backend/
  src/                  Python engine, OTLP server, and reference pipeline
  tests/                Python tests
frontend/
  src/app/              Next.js pages and API gateway
  src/features/         Trace explorer and span inspector
  src/components/       Shared UI components
  src/hooks/            API data loading
  src/lib/              API types and backend connection
configs/                Diagnostic configurations
fixtures/               Shared telemetry fixtures
docs/                   API contracts, architecture, and guides
scripts/                Development launcher and dataset download
data/, runs/            Local runtime data (ignored by Git)
pyproject.toml          Python packaging and test configuration
```

## Project Overview

This project is a command-line retrieval-augmented generation (RAG) application that answers questions using documents from Open RAGBench.

The application loads document text, splits it into smaller chunks, creates embeddings, and stores them in ChromaDB. When a user asks a question, it retrieves relevant chunks and sends them to `gemini-3.5-flash-lite` to generate an answer. The answer and retrieved source paper titles appear in the terminal.

This pipeline is the starting point for a larger project focused on diagnosing RAG failures.

## Requirements

- Python 3.11 is recommended. The project supports Python 3.10–3.13.
- Internet access for downloading the dataset and embedding model.
- A Gemini API key with API access for generating answers.
- A Hugging Face read token is optional for authenticated downloads.

## 1. Set Up the Environment

Open a terminal in the project directory and run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The final command installs the application and development tools, including pytest.

Activate the environment again when opening a new terminal:

```bash
source .venv/bin/activate
```

## 2. Configure API Access

Create a file named `.env` in the project root and add:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_API_KEY=your_gemini_api_key
```

The application loads this file when running the indexing and question-answering commands. A Gemini API key is required for answer generation, but not for local embedding or indexing.

## 3. Download the Dataset

From the project root, run:

```bash
python scripts/download-dataset.py
```

The script downloads `vectara/open_ragbench` into:

```text
data/open_ragbench
```

To change the dataset repository or download location:

```bash
python scripts/download-dataset.py --repo-id vectara/open_ragbench --output data/open_ragbench
```

For authenticated Hugging Face downloads, create a read token at:

https://huggingface.co/settings/tokens

Set it in the terminal before running the download script:

```bash
read -s -p "Hugging Face token: " HF_TOKEN
echo
export HF_TOKEN
python scripts/download-dataset.py
```

The download script reads `HF_TOKEN` from the terminal environment. It does not automatically load tokens from `.env`.

## 4. Build the Index

Run:

```bash
rag-pipeline index --reset
```

This loads documents, splits the text into chunks, creates embeddings, and saves them in a persistent ChromaDB collection.

The default corpus directory is:

```text
data/open_ragbench/pdf/arxiv/corpus
```

The default database directory is:

```text
data/chroma
```

The `--reset` option deletes the selected collection before rebuilding it.

For a smaller initial run, index only the first 20 JSON files:

```bash
rag-pipeline index --limit-files 20 --reset
```

This limits indexing, not the dataset download.

To use another compatible corpus directory:

```bash
rag-pipeline index --corpus-dir /path/to/corpus --reset
```

The loader reads JSON files directly inside the selected directory. It supports paper titles, abstracts, section text, and table content stored in the supported JSON structure. Images are skipped.

## 5. Ask a Question

After building the index, run:

```bash
rag-pipeline ask "What methods are used to improve retrieval quality?"
```

The answer and the source paper titles are printed in the terminal.

Every query also writes an append-only JSONL diagnostic trace under
`data/diagnostics/`. The trace records the inputs, outputs, duration, and status
of query embedding, retrieval, prompt construction, and model generation. The
CLI prints the exact trace path, including when a traced stage fails. Use
`--diagnostics-dir /path/to/logs` to choose another directory.

## Configuration

Default settings:

| Setting              | Default                                  |
| -------------------- | ---------------------------------------- |
| Embedding model      | `sentence-transformers/all-MiniLM-L6-v2` |
| Generation model     | `gemini-3.5-flash-lite`                  |
| Vector database      | Persistent ChromaDB                      |
| Similarity measure   | Cosine distance                          |
| Collection           | `open_ragbench`                          |
| Chunk size           | 800 characters                           |
| Chunk overlap        | 120 characters                           |
| Embedding batch size | 64                                       |
| Retrieved chunks     | 5                                        |

View available command-line options:

```bash
rag-pipeline index --help
rag-pipeline ask --help
```

For example, change the number of retrieved chunks:

```bash
rag-pipeline ask "What methods are used to improve retrieval quality?" --top-k 3
```

Settings can also be configured through environment variables or `.env`:

```text
RAG_CORPUS_DIR
RAG_CHROMA_DIR
RAG_COLLECTION
RAG_EMBEDDING_MODEL
RAG_LLM_MODEL
GEMINI_MIN_REQUEST_INTERVAL_SECONDS
RAG_CHUNK_SIZE
RAG_CHUNK_OVERLAP
RAG_BATCH_SIZE
RAG_TOP_K
RAG_DIAGNOSTICS_DIR
```

The indexer accepts an iterable of `Document` objects, so another dataset can be
used by writing a small loader without changing the retrieval or generation code.

## Fault injection engine

The engine copies the clean Open RAGBench paper JSON corpus and applies the
mutations in a YAML manifest. It needs no model calls, embeddings, or API keys.
Install the updated dependencies with `pip install -e .`, then run:

```bash
python -m rag_pipeline.fault_injector \
  --manifest manifests/faults.example.yaml \
  --output-dir data/fault_runs/example
```

The installed `fault-injector` command accepts the same arguments. Use
`--corpus-dir` to select another downloaded Open RAGBench `corpus` directory.
Its parent must contain the corresponding `queries.json`, `qrels.json`, and
`answers.json`. Paths on the command line are relative to the working directory.

The run produces:

```text
data/fault_runs/example/
  corpus/             # Corrupted paper JSON files; input for the indexer
  ground_truth.json   # Query-keyed registry; keep outside the index
  runtime_faults.json # Plans for faults applied after indexing
```

To index this corpus with the existing pipeline:

```bash
rag-pipeline index --corpus-dir data/fault_runs/example/corpus \
  --chroma-dir data/chroma_faults --collection fault_example --reset
```

The example manifest uses seven real query IDs from the downloaded dataset.
Each case specifies a `query_id`, `failure_type`, and optional `injection_config`.
The default query text and clean answer come from the dataset. FP7 supplies an
explicit two-part query. `qrels.json` supplies the
default document and section. Sections are selected by their `section_id`, which
can differ from their position in the sections array.

| Failure type        | Mutation                                      | Configuration options                          |
| ------------------- | --------------------------------------------- | ---------------------------------------------- |
| `missing_content`   | Delete the entire paper JSON file             | `target_file`                                  |
| `missed_top_ranked` | Append noise to the selected field            | `noise`, `repeat` (default 500)                |
| `not_extracted`     | Wrap the selected field in contradictory text | `prefix`, `suffix`                             |
| `incomplete_answer` | Truncate the selected field                   | `keep_fraction` (default 0.5), or `keep_chars` |

For text mutations, `target_file` optionally overrides the paper filename and
`section_id` overrides the relevant section. The default `field` is `text`;
`field: table` also requires `table_id`, while `field: abstract` targets the
paper abstract without a section ID. For example, a table mutation can use:

```yaml
version: 1
test_cases:
  - query_id: "YOUR_QUERY_ID"
    failure_type: incomplete_answer
    injection_config:
      section_id: 3
      field: table
      table_id: "YOUR_TABLE_KEY"
      keep_chars: 120
```

The registry records each query and clean answer, original relevance target,
failure category, intended failure stage, generated root cause, exact file and
JSON pointer, injection configuration, and original/corrupted file SHA-256 hashes.
Text edits include the removed and inserted text with zero-based, end-exclusive
Unicode character offsets in the original selected field. A deleted document
has a null corrupted hash. Section array positions are also recorded to match
the existing loader's section metadata.

These labels describe intended downstream failures. The registry proves what
was changed; noise or truncation does not guarantee a retrieval or generation
failure, or removal of every occurrence of an answer. No diagnostic evaluation
is performed by this engine.

The clean corpus stays untouched. Unknown options, missing queries/files/sections,
duplicate query IDs, and multiple cases targeting the same paper are rejected.
All cases are validated before output is staged. Each run requires a new output
directory; regenerate into a different directory with the same manifest and
clean input to get identical corpus content and registry bytes. To test another
fault on the same document, create a separate manifest/run.

python -m pytest

````

The tests cover:

- Loading abstracts, section text, and tables while excluding image data.
- Limiting the number of loaded corpus files.
- Generating stable chunk identifiers.
- The spans each query emits, including failures, flags and content capture (`backend/tests/test_telemetry.py`).
- The web UI endpoint (`backend/tests/test_web.py`).
- The OTLP adapter's output against `docs/telemetry.schema.json`, including the step-0 fixtures (`backend/tests/test_otel_adapter.py`).

The pipeline tests use fake retrieval and a fake model, so they need no index or API key. These tests do not measure retrieval accuracy or validate live model answers.

## Release notes

### Initial RAG Pipeline Submission

This release implements the basic command-line RAG workflow:

- Downloading the Open RAGBench dataset.
- Loading abstracts, section text, and supported table content from JSON files.
- Splitting text with `RecursiveCharacterTextSplitter`.
- Creating local embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Storing embeddings, text, and metadata in ChromaDB.
- Retrieving relevant chunks using vector similarity.
- Generating answers through the Gemini API.
- Displaying retrieved source paper titles.
- Configuring settings through command-line options and environment variables.
- Providing basic automated tests.

The implementation indexes the dataset’s JSON representations of papers. It does not parse raw PDFs or process images.

The base RAG pipeline does not include hybrid retrieval or reranking. The separate diagnostic package provides adapters, fault fixtures, stage reports, and evaluation. Displayed source titles identify retrieved papers; the code does not verify that individual answer statements are supported by those sources.


## Additional Branches

All code is in the `main` branch and its `origin/main` tracking reference.

## Diagnostic telemetry integration

Every query emits structured execution telemetry through `PipelineTrace`, including
retrieval scores, exact context, generation inputs/output, and an immutable reference
to captured index-build artifacts. The diagnostic tool consumes these records without
rerunning the query or requiring a reference answer.

```bash
rag-pipeline ask "What methods improve retrieval quality?"
rag-diagnose flag --trace /path/to/execution.jsonl --reason "Incomplete response"
rag-diagnose ingest --trace /path/to/execution.jsonl --json
```

To automatically consume flagged records:

```bash
rag-diagnose watch --trace-dir data/diagnostics --config configs/diagnostic.json --json
```

Index builds capture parsing, chunking, embedding and corpus artifacts. Logs from
older indexes remain usable with explicit limits on missing upstream evidence.
The consumer uses the Gemini judge for FP1–FP7 attribution and preserves the original
run ID, evidence, confidence and unresolved checks in its JSON report.

See [the telemetry contract and operating guide](docs/diagnostic-tool.md).

## Fault benchmark

The default stage-based benchmark covers all 15 technical mechanisms, the
operational fallback, a multi-fault execution, and two controls:

```bash
rag-diagnose fault-test-runner --config configs/mechanism-benchmark.json \
  --output-dir data/fault_runs/mechanisms
```

Use `--prepare-only` to inspect the 19 captured executions before inference,
`--mechanism OUTPUT_FORMAT_VIOLATION` for a quick targeted run, and `--resume`
to retry failed diagnoses. See [mechanism cases and scoring](docs/mechanism-benchmark.md).

### Legacy FP1–FP7 benchmark

The full-corpus workflow below requires `assessment_mode: "legacy_fp"` in its
diagnostic configuration.

With Open RAGBench downloaded and Gemini credentials available, run the suite.

```bash
fault-test-runner --manifest manifests/faults.example.yaml \
  --corpus-dir data/open_ragbench/pdf/arxiv/corpus \
  --config configs/open-ragbench-diagnostic.json \
  --output-dir data/fault_runs/diagnostic_benchmark
```

The terminal displays live progress for corpus preparation, clean indexing,
creation of the corrupted index, each baseline/fault/diagnosis case, and report writing.
Indexing bars show source-file completion and the current indexed chunk count.
Use `--no-progress` when redirecting output to a log or running without a terminal.

Use a fresh output directory for each run. The same command is available via
`rag-diagnose fault-test-runner` or `python -m rag_diagnostic.fault_test_runner`.

The benchmark uses the full downloaded corpus. It indexes the clean corpus,
applies the manifest's corpus mutations to a hard-linked full copy, then builds the
corrupted Chroma collection from the clean collection's stored embeddings while
omitting FP1's deleted document. Runtime faults do not require re-embedding unchanged
documents. The independent reference pipeline implementation lives in
`backend/src/rag_diagnostic/reference_pipeline.py` and executes text loading, chunking,
sentence-transformer embeddings, Chroma vector search, prompt construction and
Gemini generation. Shared configuration, corpus loading and telemetry formats
keep its inputs and observations compatible with the main pipeline.

All seven queries execute against the clean baseline and the corrupted pipeline.
Retrieval uses the configured top-k and deeper inspection cutoff. Target evidence
is selected by the actual search. The runner records a runtime fault as unapplied
when its required evidence fails to reach the relevant boundary. FP4/FP7 use a
whitespace-normalized evidence quote to check the exact context precondition.

| Case | Injection boundary |
| --- | --- |
| FP1 | Delete the source paper from the full copied corpus before indexing |
| FP2 | Move the supporting indexed chunk below the retrieval cutoff |
| FP3 | Remove the supporting retrieved chunk before assembling the prompt |
| FP4 | Preserve the supporting context and replace the answer with one that misses the fact |
| FP5 | Send an explicit JSON instruction and replace the answer with plain text |
| FP6 | Send an explicit detail requirement and replace the answer with a vague response |
| FP7 | Keep evidence for both query parts and replace the answer with a partial response |

The run saves clean baseline traces under `baseline_traces/`, faulted diagnostic
inputs under `traces/`, and persistent query events and index artifacts. Each
directory contains one trace per category. The judge receives the observed artifacts. The
separate `report.json` compares predictions with the registry and includes
verdicts, evidence, errors, and the pre-injection retrieval order and model
answer. Suite accuracy requires the exact stage and category sets and counts
abstentions and run errors as misses. Primary accuracy compares only the
primary stage and category. The report also shows accuracy among answered
cases and the abstention rate. `injection_coverage` measures how many cases
actually received their fault. `diagnostic_accuracy` scores only those applied
cases. The per-case `injection_applied` and `injection_reason` explain unmet
preconditions. A successful clean execution records the baseline answer and
retrieval behavior. Its answer quality still requires semantic assessment.

If a run stops after completing clean indexing, resume the same output directory.
The runner reuses the completed clean Chroma collection and does not repeat embedding:

```bash
rag-diagnose fault-test-runner \
  --manifest manifests/faults.example.yaml \
  --corpus-dir data/open_ragbench/pdf/arxiv/corpus \
  --config configs/open-ragbench-diagnostic.json \
  --output-dir data/fault_runs/diagnostic_benchmark \
  --resume
```

On macOS, if the output directory is managed by iCloud and the runner reports that
`ground_truth.json` is `dataless`, download that small registry before resuming:

```bash
brctl download data/fault_runs/diagnostic_benchmark/ground_truth.json
```

Index-build telemetry is intentionally compact in benchmark mode; it records counts
and model/collection metadata without writing or re-hashing every embedding vector.

The UI backend can read `report.json` using `schema_version`, `benchmark_id`,
`status`, timestamps, suite and diagnostic accuracy, coverage, per-fault metrics,
stage and category confusion, and `cases` with baseline/fault trace paths.
The report is updated after each query so completed cases remain available
during a run. Large full-corpus snapshots may exceed the judge's corpus budget.
The judge then sees an explicitly incomplete view, which can cause FP1 to
abstain. Each case records `judge_corpus_complete` when available.

Re-score a saved run after changing the judge prompt or recovering from a provider
error. This reuses the seven saved traces and writes a fresh report.

```bash
rag-diagnose fault-test-runner \
  --replay-dir data/fault_runs/diagnostic_benchmark \
  --config configs/open-ragbench-diagnostic.json \
  --output-dir data/fault_runs/diagnostic_benchmark_replayed
```

Gemini requests retry transient HTTP 429/500/502/503/504 responses up to three
attempts. Reports retain API failures as errors, include them in overall accuracy,
and expose per-category validation reasons when a diagnosis abstains.
````

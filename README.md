# Open RAGBench RAG Pipeline

## Code Milestone

## Project Overview

This project is a command-line retrieval-augmented generation (RAG) application that answers questions using documents from Open RAGBench.

The application loads document text, splits it into smaller chunks, creates embeddings, and stores them in ChromaDB. When a user asks a question, it retrieves relevant chunks and sends them to `gpt-5.6 Sol` to generate an answer. The answer and retrieved source paper titles appear in the terminal.

This pipeline is the starting point for a larger project focused on diagnosing RAG failures.

## Requirements

- Python 3.11 is recommended. The project supports Python 3.10–3.13.
- Internet access for downloading the dataset and embedding model.
- An OpenAI API key with API access for generating answers.
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
OPENAI_API_KEY=your_openai_api_key
```

The application loads this file when running the indexing and question-answering commands. An OpenAI API key is required for answer generation, but not for local embedding or indexing.

## 3. Download the Dataset

From the project root, run:

```bash
python download-dataset.py
```

The script downloads `vectara/open_ragbench` into:

```text
data/open_ragbench
```

To change the dataset repository or download location:

```bash
python download-dataset.py --repo-id vectara/open_ragbench --output data/open_ragbench
```

For authenticated Hugging Face downloads, create a read token at:

https://huggingface.co/settings/tokens

Set it in the terminal before running the download script:

```bash
read -s -p "Hugging Face token: " HF_TOKEN
echo
export HF_TOKEN
python download-dataset.py
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
| Generation model     | `gpt-4o-mini`                            |
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
```

To index this corpus with the existing pipeline:

```bash
rag-pipeline index --corpus-dir data/fault_runs/example/corpus \
  --chroma-dir data/chroma_faults --collection fault_example --reset
```

The example manifest uses four real query IDs from the downloaded dataset.
Each case specifies a `query_id`, `failure_type`, and optional `injection_config`.
The query text and clean answer come from the dataset; `qrels.json` supplies the
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

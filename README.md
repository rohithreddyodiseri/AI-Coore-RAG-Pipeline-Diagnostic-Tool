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

The application retrieves relevant chunks, sends them to the language model, and prints the answer followed by the retrieved source paper titles.

The model is instructed to answer using only the supplied context and to say it does not know when the context does not contain the answer. This is a prompt instruction, not a guarantee of answer accuracy.

## Configuration

Default settings:

| Setting | Default |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Generation model | `gpt-4o-mini` |
| Vector database | Persistent ChromaDB |
| Similarity measure | Cosine distance |
| Collection | `open_ragbench` |
| Chunk size | 800 characters |
| Chunk overlap | 120 characters |
| Embedding batch size | 64 |
| Retrieved chunks | 5 |

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
```

Use the same embedding model when indexing and asking questions. If the embedding model or chunking settings change, rebuild the index.

## Tests

Run the included tests:

```bash
python -m pytest
```

The repository includes three tests covering:

- Loading abstracts, section text, and tables while excluding image data.
- Limiting the number of loaded corpus files.
- Generating stable chunk identifiers.

These tests do not measure retrieval accuracy or validate live model answers.

## Release notes

### Initial RAG Pipeline Submission

This release implements the basic command-line RAG workflow:

- Downloading the Open RAGBench dataset.
- Loading abstracts, section text, and supported table content from JSON files.
- Splitting text with `RecursiveCharacterTextSplitter`.
- Creating local embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Storing embeddings, text, and metadata in ChromaDB.
- Retrieving relevant chunks using vector similarity.
- Generating answers through the OpenAI API.
- Displaying retrieved source paper titles.
- Configuring settings through command-line options and environment variables.
- Providing basic automated tests.

The implementation indexes the dataset’s JSON representations of papers. It does not parse raw PDFs or process images.

Hybrid retrieval, reranking, controlled fault injection, stage-level diagnostic reports, and question-answer benchmark evaluation are not included in this release. Displayed source titles identify retrieved papers; the code does not verify that individual answer statements are supported by those sources.


## Additional Branches

All code is in the `main` branch and its `origin/main` tracking reference.

## Future Work

Planned work includes recording intermediate pipeline outputs, creating controlled failures with known causes, evaluating retrieval and answer quality, and developing a tool that identifies likely failure stages with supporting evidence.
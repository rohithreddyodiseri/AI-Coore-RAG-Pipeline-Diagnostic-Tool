# Open RAGBench RAG Pipeline

A small command-line RAG application that:

1. reads the Open RAGBench corpus;
2. splits the text with `RecursiveCharacterTextSplitter`;
3. creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`;
4. stores the embeddings in ChromaDB; and
5. answers questions with `gpt-4o-mini`.

## 1. Create the virtual environment

Use Python 3.11 because the machine's newer Python version may not be supported
by all machine-learning packages.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The project already includes `huggingface-hub`, but it can also be installed
directly with the same compatible version range:

```bash
pip install "huggingface-hub<2,>=0.30"
```

## 2. Set the Hugging Face token

Using a free Hugging Face read token prevents unauthenticated-download warnings
and rate-limit problems.

1. Open [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens).
2. Create a token with `read` permission.
3. Export it in the same terminal:

```bash
export HF_TOKEN="your_hf_token_here"
```

## 3. Download the dataset

```bash
python download-dataset.py
```

By default, the script downloads `vectara/open_ragbench` into
`data/open_ragbench`. Both values can be changed:

```bash
python download-dataset.py --repo-id vectara/open_ragbench --output data/open_ragbench
```

## 4. Set the OpenAI API key

The application reads `OPENAI_API_KEY` from the environment or from the `.env`
file in this project or its parent directory.

```env
OPENAI_API_KEY=your_openai_api_key
```

## 5. Build the index

```bash
rag-pipeline index --reset
```

This reads the default corpus directory:

```text
data/open_ragbench/pdf/arxiv/corpus
```

To use another compatible corpus folder:

```bash
rag-pipeline index --corpus-dir /path/to/corpus --reset
```

Each JSON file should contain a paper title, abstract, and a `sections` list.
Section text and Markdown tables are indexed. Image data is skipped because the
embedding model is text-only.

## 6. Ask a question

```bash
rag-pipeline ask "What methods are used to improve retrieval quality?"
```

The answer and the source paper titles are printed in the terminal.

## Common options

```bash
rag-pipeline index --help
rag-pipeline ask --help
```

Models, paths, collection name, chunk size, overlap, and result count can all be
changed through command-line options. They can also be set with these environment
variables:

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

The indexer accepts an iterable of `Document` objects, so another dataset can be
used by writing a small loader without changing the retrieval or generation code.

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gemini_client import GeminiInferenceEngine
from rag_pipeline.config import Settings
from rag_pipeline.corpus import Document
from rag_pipeline.diagnostics import PipelineTrace


@dataclass(frozen=True)
class SearchResult:
    text: str
    metadata: dict[str, Any]
    chunk_id: str = ""
    score: float | None = None
    rank: int = 0


class RAGPipeline:
    generation_instructions = (
        "Answer the question using only the provided context. "
        "If the answer is not in the context, say you do not know."
    )
    generation_temperature = 0.2

    def __init__(self, settings: Settings, client=None):
        settings.validate()
        self.settings = settings
        self._inference = GeminiInferenceEngine(settings.llm_model, client=client)
        self._embedder: Any = None
        self._collection: Any = None
        self.last_trace_path: Path | None = None

    def _embedding_model(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.settings.embedding_model)
        return self._embedder

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._embedding_model().encode(
            list(texts),
            batch_size=self.settings.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _open_collection(self, create: bool = True) -> Any:
        if self._collection is not None:
            return self._collection

        import chromadb

        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        if create:
            self._collection = client.get_or_create_collection(
                name=self.settings.collection,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._collection = client.get_collection(
                name=self.settings.collection,
                embedding_function=None,
            )
        return self._collection

    def _reset_collection(self) -> None:
        import chromadb

        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        try:
            client.delete_collection(self.settings.collection)
        except Exception as error:
            message = str(error).lower()
            if "does not exist" not in message and "not found" not in message:
                raise
        self._collection = None

    @staticmethod
    def _chunk_id(document: Document, chunk_number: int, text: str) -> str:
        value = f"{document.metadata.get('source')}\0{document.metadata.get('section')}"
        value += f"\0{chunk_number}\0{text}"
        return hashlib.sha256(value.encode()).hexdigest()

    def index(
        self,
        documents: Iterable[Document],
        reset: bool = False,
        progress: Callable[[int], None] | None = None,
    ) -> int:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if reset:
            self._reset_collection()
        collection = self._open_collection()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        batch: list[tuple[str, str, dict[str, str | int]]] = []
        indexed = 0

        def save_batch() -> None:
            nonlocal indexed
            if not batch:
                return
            texts = [item[1] for item in batch]
            collection.upsert(
                ids=[item[0] for item in batch],
                documents=texts,
                metadatas=[item[2] for item in batch],
                embeddings=self._embed(texts),
            )
            indexed += len(batch)
            batch.clear()
            if progress:
                progress(indexed)

        for document in documents:
            for chunk_number, text in enumerate(splitter.split_text(document.text)):
                metadata = {**document.metadata, "chunk": chunk_number}
                batch.append((self._chunk_id(document, chunk_number, text), text, metadata))
                if len(batch) >= self.settings.batch_size:
                    save_batch()
        save_batch()
        return indexed

    def search(self, query: str, k: int, trace: PipelineTrace | None = None) -> list[SearchResult]:
        """The actual vector retrieval implementation, also used for inspection probes."""
        if k < 1:
            raise ValueError("retrieval k must be positive")
        collection = self._open_collection(create=False)
        item_count = collection.count()
        if item_count == 0:
            raise RuntimeError("The index is empty. Run `rag-pipeline index --reset` first.")
        def embed() -> list[list[float]]:
            return self._embed([query])

        embeddings = (
            trace.capture(
                "query_embedding",
                {"texts": [query], "model": self.settings.embedding_model},
                embed,
            )
            if trace
            else embed()
        )

        retrieval_inputs = {
            "collection": self.settings.collection,
            "query_embeddings": embeddings,
            "n_results": min(self.settings.top_k, item_count),
            "include": ["documents", "metadatas"],
        }

        def search() -> list[SearchResult]:
            response = collection.query(
                query_embeddings=embeddings,
                n_results=min(self.settings.top_k, item_count),
                include=["documents", "metadatas"],
            )
            documents = (response.get("documents") or [[]])[0]
            metadata = (response.get("metadatas") or [[]])[0]
            return [
                SearchResult(text=str(text), metadata=dict(item or {}))
                for text, item in zip(documents, metadata, strict=True)
            ]
            raise RuntimeError("The index is empty. Run `rag-pipeline index` first.")
        response = collection.query(
            query_embeddings=self._embed([query]),
            n_results=min(k, item_count),
            include=["documents", "metadatas", "distances"],
        )
        ids = response["ids"][0]
        documents = response["documents"][0]
        metadata = response["metadatas"][0]
        distances = response["distances"][0]
        return [
            SearchResult(str(text), dict(meta or {}), str(chunk_id), 1.0 - float(distance), rank)
            for rank, (chunk_id, text, meta, distance) in enumerate(
                zip(ids, documents, metadata, distances, strict=True), start=1
            )
        ]

    def retrieve(self, query: str) -> list[SearchResult]:
        return self.search(query, self.settings.top_k)

    def assemble_context(self, query: str, results: list[SearchResult]) -> str:
        return "\n\n".join(f"[{n}] {result.text}" for n, result in enumerate(results, start=1))

    def generate(self, query: str, context: str) -> str:
        return self._inference.generate(
            prompt=f"Question:\n{query}\n\nContext:\n{context}",
            system_instruction=self.generation_instructions,
            temperature=self.generation_temperature,
        )

    def execute(self, query: str) -> tuple[str, list[SearchResult], dict[str, Any]]:
        """Execute once and record the exact inputs/outputs of the real query stages."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        results = self.retrieve(query)
        retrieved = [
            {
                **asdict(result),
                "doc_id": str(result.metadata.get("doc_id", "")),
                "rank": result.rank or n,
                "chunk_id": result.chunk_id or f"retrieval:{n}",
            }
            for n, result in enumerate(results, start=1)
        ]
        context = self.assemble_context(query, results)
        prompt = f"Question:\n{query}\n\nContext:\n{context}"
        answer = self.generate(query, context)
        trace = {
            "query": query,
            "generation_instructions": self.generation_instructions,
            "reranker_used": False,
            "stages": {
                "retrieval": {
                    "input_data": {"query": query, "k": self.settings.top_k},
                    "k": self.settings.top_k,
                    "output_data": retrieved,
                },
                "reranking": {"status": "not_used"},
                "context_assembly": {
                    "input_data": retrieved,
                    "output_data": {
                        "chunk_ids": [r["chunk_id"] for r in retrieved],
                        "text": context,
                    },
                },
                "generation": {
                    "input_data": {
                        "prompt": prompt,
                        "system_instruction": self.generation_instructions,
                        "model": self.settings.llm_model,
                        "temperature": self.generation_temperature,
                    },
                    "output_data": answer,
                },
            },
            "final_answer": answer,
        }
        return answer, results, trace

    def run_with_trace(self, query: str) -> dict[str, Any]:
        return self.execute(query)[2]

        def serialize_results(results: list[SearchResult]) -> list[dict[str, Any]]:
            return [
                {"text": result.text, "metadata": result.metadata} for result in results
            ]

        return (
            trace.capture("retrieval", retrieval_inputs, search, serialize_results)
            if trace
            else search()
        )

    def ask(self, query: str) -> tuple[str, list[SearchResult]]:
        trace = PipelineTrace(self.settings.diagnostics_dir)
        self.last_trace_path = trace.path
        trace.record(
            "query",
            {
                "query": query,
                "collection": self.settings.collection,
                "top_k": self.settings.top_k,
                "embedding_model": self.settings.embedding_model,
                "llm_model": self.settings.llm_model,
            },
            status="started",
        )
        try:
            results = self.retrieve(query, trace=trace)

            def build_prompt() -> dict[str, str]:
                context = "\n\n".join(
                    f"[{number}] {result.text}"
                    for number, result in enumerate(results, start=1)
                )
                return {
                    "instructions": (
                        "Answer the question using only the provided context. "
                        "If the answer is not in the context, say you do not know."
                    ),
                    "input": f"Question:\n{query}\n\nContext:\n{context}",
                }

            prompt = trace.capture(
                "prompt_construction",
                {
                    "query": query,
                    "retrieved_results": [
                        {"text": result.text, "metadata": result.metadata}
                        for result in results
                    ],
                },
                build_prompt,
            )

            load_environment()
            from openai import OpenAI

            response = trace.capture(
                "generation",
                {"model": self.settings.llm_model, **prompt},
                lambda: OpenAI().responses.create(
                    model=self.settings.llm_model,
                    instructions=prompt["instructions"],
                    input=prompt["input"],
                ),
                lambda item: {
                    "response_id": getattr(item, "id", None),
                    "output_text": item.output_text,
                    "usage": getattr(item, "usage", None),
                },
            )
            answer = response.output_text.strip()
            trace.record(
                "query",
                {"query": query},
                outputs={
                    "answer": answer,
                    "sources": [
                        {"text": result.text, "metadata": result.metadata}
                        for result in results
                    ],
                },
            )
            return answer, results
        except Exception as error:
            trace.record("query", {"query": query}, status="failed", error=error)
            raise
        answer, sources, _ = self.execute(query)
        return answer, sources

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from rag_pipeline.config import Settings, load_environment
from rag_pipeline.corpus import Document
from rag_pipeline.telemetry import execution_span, stage_span

ANSWER_INSTRUCTIONS = (
    "Answer the question using only the provided context. "
    "If the answer is not in the context, say you do not know."
)


@dataclass(frozen=True)
class SearchResult:
    text: str
    metadata: dict[str, Any]
    chunk_id: str = ""
    score: float | None = None


def _ranked_chunk(result: SearchResult, rank: int) -> dict[str, Any]:
    """Retrieval evidence shape from the producer contract; unknown fields are omitted."""
    chunk: dict[str, Any] = {"chunk_id": result.chunk_id, "text": result.text, "rank": rank}
    if result.metadata.get("doc_id"):
        chunk["doc_id"] = str(result.metadata["doc_id"])
    if result.score is not None:
        chunk["score"] = result.score
    return chunk


class RAGPipeline:
    def __init__(self, settings: Settings):
        settings.validate()
        self.settings = settings
        self._embedder: Any = None
        self._collection: Any = None
        self.last_trace_id: str | None = None

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

    def retrieve(self, query: str) -> list[SearchResult]:
        with stage_span("query_embedding", input_data=[query]):
            query_vector = self._embed([query])

        with stage_span("retrieval", k=self.settings.top_k) as stage:
            collection = self._open_collection(create=False)
            item_count = collection.count()
            if item_count == 0:
                raise RuntimeError("The index is empty. Run `rag-pipeline index --reset` first.")
            response = collection.query(
                query_embeddings=query_vector,
                n_results=min(self.settings.top_k, item_count),
                include=["documents", "metadatas", "distances"],
            )
            ids = (response.get("ids") or [[]])[0]
            documents = (response.get("documents") or [[]])[0]
            metadata = (response.get("metadatas") or [[]])[0]
            distances = (response.get("distances") or [[None] * len(ids)])[0]
            results = [
                SearchResult(
                    text=str(text),
                    metadata=dict(item or {}),
                    chunk_id=str(chunk_id),
                    # The collection uses cosine distance; similarity = 1 - distance.
                    score=None if distance is None else 1 - float(distance),
                )
                for chunk_id, text, item, distance in zip(
                    ids, documents, metadata, distances, strict=True
                )
            ]
            stage.output([_ranked_chunk(r, rank) for rank, r in enumerate(results, start=1)])
        return results

    def ask(self, query: str, flag_reason: str | None = None) -> tuple[str, list[SearchResult]]:
        """Answer a query. ``flag_reason`` marks the query for diagnosis."""
        with execution_span(query, flag_reason) as execution:
            self.last_trace_id = execution.trace_id
            results = self.retrieve(query)

            with stage_span("context_assembly") as stage:
                context = "\n\n".join(
                    f"[{number}] {result.text}" for number, result in enumerate(results, start=1)
                )
                stage.output({"text": context, "chunk_ids": [r.chunk_id for r in results]})

            prompt = f"Question:\n{query}\n\nContext:\n{context}"
            generation_input = {
                "prompt": prompt,
                "system_instruction": ANSWER_INSTRUCTIONS,
                "model": self.settings.llm_model,
            }
            with stage_span("generation", input_data=generation_input) as stage:
                load_environment()
                from openai import OpenAI

                response = OpenAI().responses.create(
                    model=self.settings.llm_model,
                    instructions=ANSWER_INSTRUCTIONS,
                    input=prompt,
                )
                stage.output(response.output_text)

            answer = response.output_text.strip()
            execution.finish(answer=answer, instructions=ANSWER_INSTRUCTIONS)
        return answer, results

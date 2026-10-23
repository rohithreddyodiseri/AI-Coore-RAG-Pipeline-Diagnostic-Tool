from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from rag_pipeline.config import Settings, load_environment
from rag_pipeline.corpus import Document


@dataclass(frozen=True)
class SearchResult:
    text: str
    metadata: dict[str, Any]


class RAGPipeline:
    def __init__(self, settings: Settings):
        settings.validate()
        self.settings = settings
        self._embedder: Any = None
        self._collection: Any = None

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
        collection = self._open_collection(create=False)
        item_count = collection.count()
        if item_count == 0:
            raise RuntimeError("The index is empty. Run `rag-pipeline index --reset` first.")
        response = collection.query(
            query_embeddings=self._embed([query]),
            n_results=min(self.settings.top_k, item_count),
            include=["documents", "metadatas"],
        )
        documents = (response.get("documents") or [[]])[0]
        metadata = (response.get("metadatas") or [[]])[0]
        return [
            SearchResult(text=str(text), metadata=dict(item or {}))
            for text, item in zip(documents, metadata, strict=True)
        ]

    def ask(self, query: str) -> tuple[str, list[SearchResult]]:
        results = self.retrieve(query)
        context = "\n\n".join(
            f"[{number}] {result.text}" for number, result in enumerate(results, start=1)
        )

        load_environment()
        from openai import OpenAI

        response = OpenAI().responses.create(
            model=self.settings.llm_model,
            instructions=(
                "Answer the question using only the provided context. "
                "If the answer is not in the context, say you do not know."
            ),
            input=f"Question:\n{query}\n\nContext:\n{context}",
        )
        return response.output_text.strip(), results

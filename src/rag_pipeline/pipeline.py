from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gemini_client import GeminiInferenceEngine
from rag_pipeline.config import Settings
from rag_pipeline.corpus import Document
from rag_pipeline.diagnostics import PipelineTrace
from rag_pipeline.index_telemetry import IndexTelemetry, index_reference


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
        self._active_trace: ContextVar[PipelineTrace | None] = ContextVar(
            "rag_pipeline_trace", default=None
        )

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

        telemetry = IndexTelemetry(self.settings)

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
            telemetry.chunks(batch)
            vectors = self._embed(texts)
            telemetry.embeddings(batch, vectors)
            collection.upsert(
                ids=[item[0] for item in batch],
                documents=texts,
                metadatas=[item[2] for item in batch],
                embeddings=vectors,
            )
            indexed += len(batch)
            batch.clear()
            if progress:
                progress(indexed)

        for document in documents:
            telemetry.parsed(document)
            for chunk_number, text in enumerate(splitter.split_text(document.text)):
                metadata = {**document.metadata, "chunk": chunk_number}
                batch.append((self._chunk_id(document, chunk_number, text), text, metadata))
                if len(batch) >= self.settings.batch_size:
                    save_batch()
        save_batch()
        telemetry.complete()
        return indexed

    @staticmethod
    def _serialize_results(results: list[SearchResult]) -> list[dict[str, Any]]:
        return [
            {
                **asdict(result),
                "doc_id": str(result.metadata.get("doc_id", "")),
                "rank": result.rank or number,
                "chunk_id": result.chunk_id or f"retrieval:{number}",
            }
            for number, result in enumerate(results, start=1)
        ]

    def search(self, query: str, k: int, trace: PipelineTrace | None = None) -> list[SearchResult]:
        """Run one embedding and vector lookup, using the requested probe cutoff."""
        if type(k) is not int or k < 1:
            raise ValueError("retrieval k must be positive")
        collection = self._open_collection(create=False)
        item_count = collection.count()
        if item_count == 0:
            raise RuntimeError("The index is empty. Run `rag-pipeline index --reset` first.")
        active_trace = trace if trace is not None else self._active_trace.get()
        embeddings = (
            active_trace.capture(
                "query_embedding",
                {"texts": [query], "model": self.settings.embedding_model},
                lambda: self._embed([query]),
            )
            if active_trace is not None
            else self._embed([query])
        )
        retrieval_inputs = {
            "collection": self.settings.collection,
            "query_embeddings": embeddings,
            "n_results": min(k, item_count),
            "include": ["documents", "metadatas", "distances"],
        }

        def lookup() -> list[SearchResult]:
            response = collection.query(
                **{key: value for key, value in retrieval_inputs.items() if key != "collection"}
            )
            return [
                SearchResult(
                    str(text), dict(meta or {}), str(chunk_id), 1.0 - float(distance), rank
                )
                for rank, (chunk_id, text, meta, distance) in enumerate(
                    zip(
                        response["ids"][0],
                        response["documents"][0],
                        response["metadatas"][0],
                        response["distances"][0],
                        strict=True,
                    ),
                    start=1,
                )
            ]

        # execute() captures the retrieval hook so custom retrieve() implementations
        # are observed too. Standalone search probes may supply their own trace.
        return (
            trace.capture("retrieval", retrieval_inputs, lookup, self._serialize_results)
            if trace is not None
            else lookup()
        )

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

    def _telemetry_metadata(self):
        settings = {
            name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(self.settings).items()
        }
        metadata = {
            "name": "rag_pipeline",
            "provider": "gemini",
            "settings": settings,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        try:
            metadata["index_snapshot"] = index_reference(self.settings)
        except (OSError, ValueError, KeyError) as error:
            metadata["index_snapshot"] = {"status": "unavailable", "error": type(error).__name__}
        return metadata

    def execute(
        self, query: str, *, failure_reason: str | None = None
    ) -> tuple[str, list[SearchResult], dict[str, Any]]:
        """Execute once through the same hooks for CLI queries and diagnostic runs."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if failure_reason is not None and (
            not isinstance(failure_reason, str) or not failure_reason.strip()
        ):
            raise ValueError("failure_reason must be a nonempty string")
        metadata = self._telemetry_metadata()
        events = PipelineTrace(self.settings.diagnostics_dir)
        self.last_trace_path = events.path
        events.record(
            "query",
            {
                "query": query,
                "collection": self.settings.collection,
                "top_k": self.settings.top_k,
                "embedding_model": self.settings.embedding_model,
                "llm_model": self.settings.llm_model,
                "provider": "gemini",
            },
            status="started",
        )
        token = self._active_trace.set(events)
        try:
            results = events.capture(
                "retrieval",
                {
                    "query": query,
                    "k": self.settings.top_k,
                    "collection": self.settings.collection,
                    "embedding_model": self.settings.embedding_model,
                },
                lambda: self.retrieve(query),
                self._serialize_results,
            )
            retrieved = self._serialize_results(results)

            def build_prompt() -> dict[str, str]:
                context = self.assemble_context(query, results)
                return {
                    "instructions": self.generation_instructions,
                    "context": context,
                    "input": f"Question:\n{query}\n\nContext:\n{context}",
                }

            prompt = events.capture(
                "prompt_construction",
                {"query": query, "retrieved_results": retrieved},
                build_prompt,
            )
            generation_inputs = {
                "model": self.settings.llm_model,
                "provider": "gemini",
                "temperature": self.generation_temperature,
                "instructions": prompt["instructions"],
                "input": prompt["input"],
            }
            answer = events.capture(
                "generation",
                generation_inputs,
                lambda: self.generate(query, prompt["context"]),
                lambda text: {"output_text": text},
            )
            events.record(
                "query", {"query": query}, outputs={"answer": answer, "sources": retrieved}
            )
            trace = events.execution_payload(query, metadata)
            trace = {
                **trace,
                "execution_status": "completed",
                "failure_signal": {"flagged": bool(failure_reason), "reason": failure_reason},
            }
            events.publish(trace)
            if failure_reason:
                PipelineTrace.flag(events.path, failure_reason)
            return answer, results, trace
        except Exception as error:
            events.record("query", {"query": query}, status="failed", error=error)
            events.publish(events.execution_payload(query, metadata), failed=True)
            raise
        finally:
            self._active_trace.reset(token)

    def flag_last_run(self, reason: str) -> None:
        if self.last_trace_path is None:
            raise ValueError("No execution has been captured")
        PipelineTrace.flag(self.last_trace_path, reason)

    def run_with_trace(self, query: str, *, failure_reason: str | None = None) -> dict[str, Any]:
        return self.execute(query, failure_reason=failure_reason)[2]

    def ask(
        self, query: str, *, failure_reason: str | None = None
    ) -> tuple[str, list[SearchResult]]:
        answer, sources, _ = self.execute(query, failure_reason=failure_reason)
        return answer, sources

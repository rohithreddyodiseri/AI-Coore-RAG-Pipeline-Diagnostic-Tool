"""Connects the tool to the team's rag_pipeline package.

It reads the pipeline without changing it. Since ask() runs retrieval and
generation together, a few small pieces are copied here and marked MIRRORS.
Once each stage is its own function in rag_pipeline, call those instead and
delete the copies.

Generation runs at temperature 0 here so results repeat. ask() should do the
same, or the two will give different answers.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from rag_diagnostic.adapter import PipelineAdapter
from rag_diagnostic.trace import Chunk, Context, ScoredChunk

_INSTRUCTIONS = (  # MIRRORS RAGPipeline.ask()
    "Answer the question using only the provided context. "
    "If the answer is not in the context, say you do not know."
)


def _strings(value: Any, skip_keys: frozenset[str] = frozenset({"images"})) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() if k not in skip_keys for s in _strings(v, skip_keys)]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item, skip_keys)]
    return []


class OpenRAGBenchAdapter(PipelineAdapter):
    def __init__(self, settings=None, pipeline=None, temperature: float = 0.0, client=None):
        from rag_pipeline.config import Settings
        from rag_pipeline.pipeline import RAGPipeline

        self.settings = settings or Settings.from_env()
        self.pipeline = pipeline or RAGPipeline(self.settings)
        self.retrieval_k = self.settings.top_k
        self.context_k = self.settings.top_k  # ask() puts every retrieved chunk in the context
        self.temperature = temperature
        self._client = client
        self._paths: dict[str, Path] | None = None

    # ---- Documents and the index ----------------------------------------------

    def describe(self) -> dict[str, Any]:
        info = {f.name: str(getattr(self.settings, f.name)) for f in fields(self.settings)}
        return {"adapter": "open_ragbench", "temperature": self.temperature, **info}

    def _path(self, doc_id: str) -> Path | None:
        direct = self.settings.corpus_dir / f"{doc_id}.json"
        if direct.is_file():
            return direct
        if self._paths is None:
            self._paths = {}
            for path in sorted(self.settings.corpus_dir.glob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                self._paths[str(data.get("id") or path.stem)] = path
        return self._paths.get(doc_id)

    def _documents(self, doc_id: str):
        """Load one paper the same way the pipeline does. MIRRORS OpenRAGBenchLoader.load()."""
        from rag_pipeline.corpus import Document, OpenRAGBenchLoader

        path = self._path(doc_id)
        if path is None:
            return []
        paper = OpenRAGBenchLoader._read_json(path)
        title = str(paper.get("title") or path.stem)
        base = {"doc_id": str(paper.get("id") or path.stem), "title": title, "source": str(path)}
        docs = []
        abstract = str(paper.get("abstract") or "").strip()
        if abstract:
            docs.append(Document(f"{title}\n\nAbstract\n{abstract}", {**base, "section": -1}))
        for number, section in enumerate(paper.get("sections") or []):
            if isinstance(section, dict):
                text = OpenRAGBenchLoader._section_text(section)
                if text:
                    docs.append(Document(f"{title}\n\n{text}", {**base, "section": number}))
        return docs

    def raw_text(self, doc_id: str) -> str | None:
        path = self._path(doc_id)
        if path is None:
            return None
        return "\n".join(_strings(json.loads(path.read_text(encoding="utf-8"))))

    def parsed_text(self, doc_id: str) -> str | None:
        docs = self._documents(doc_id)
        return "\n\n".join(d.text for d in docs) if docs else None

    def chunks(self, doc_id: str) -> list[Chunk]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(  # MIRRORS RAGPipeline.index()
            chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap
        )
        out = []
        for document in self._documents(doc_id):
            for number, text in enumerate(splitter.split_text(document.text)):
                chunk_id = self.pipeline._chunk_id(document, number, text)
                metadata = {**document.metadata, "chunk": number}
                out.append(Chunk(chunk_id, doc_id, text, metadata))
        return out

    def indexed_ids(self, chunk_ids: list[str]) -> set[str]:
        if not chunk_ids:
            return set()
        collection = self.pipeline._open_collection(create=False)
        return set(collection.get(ids=list(chunk_ids), include=[])["ids"])

    # ---- Stages that run for each question -------------------------------------

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        collection = self.pipeline._open_collection(create=False)
        count = collection.count()
        if count == 0:
            return []
        response = collection.query(
            query_embeddings=self.pipeline._embed([query]),
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        ids = response["ids"][0]
        texts = response["documents"][0]
        metas = response["metadatas"][0]
        distances = response["distances"][0]
        return [
            ScoredChunk(
                chunk_id, str((meta or {}).get("doc_id", "")), str(text), 1.0 - float(dist), rank
            )
            for rank, (chunk_id, text, meta, dist) in enumerate(
                zip(ids, texts, metas, distances, strict=True), start=1
            )
        ]

    def assemble_context(self, query: str, ranked: list[ScoredChunk]) -> Context:
        top = list(ranked)[: self.context_k]  # MIRRORS RAGPipeline.ask()
        text = "\n\n".join(f"[{n}] {c.text}" for n, c in enumerate(top, start=1))
        return Context(tuple(c.chunk_id for c in top), text)

    def generate(self, query: str, context: Context) -> str:
        if self._client is None:  # MIRRORS RAGPipeline.ask()
            from openai import OpenAI

            from rag_pipeline.config import load_environment

            load_environment()
            self._client = OpenAI()
        response = self._client.responses.create(
            model=self.settings.llm_model,
            instructions=_INSTRUCTIONS,
            input=f"Question:\n{query}\n\nContext:\n{context.text}",
            temperature=self.temperature,
        )
        return response.output_text.strip()


def build(pipeline_settings: dict[str, Any]) -> OpenRAGBenchAdapter:
    """Used by the command line. A case's pipeline settings override the defaults,
    for example {"chunk_size": 200, "collection": "small_chunks"}."""
    from rag_pipeline.config import Settings

    settings = Settings.from_env()
    known = {f.name: f.type for f in fields(settings)}
    overrides = {}
    for key, value in pipeline_settings.items():
        if key not in known:
            raise ValueError(f"unknown pipeline setting: {key}")
        current = getattr(settings, key)
        overrides[key] = Path(value) if isinstance(current, Path) else type(current)(value)
    return OpenRAGBenchAdapter(replace(settings, **overrides))

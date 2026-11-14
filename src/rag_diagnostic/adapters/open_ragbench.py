"""Connect the attribution engine directly to this project's RAGPipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from rag_diagnostic.adapter import PipelineAdapter
from rag_diagnostic.trace import ScoredChunk


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for key, nested in value.items() if key != "images" for s in _strings(nested)]
    if isinstance(value, list):
        return [s for nested in value for s in _strings(nested)]
    return []


class OpenRAGBenchAdapter(PipelineAdapter):
    def __init__(self, settings=None, pipeline=None, client=None):
        from rag_pipeline.config import Settings
        from rag_pipeline.pipeline import RAGPipeline

        self.settings = settings or (pipeline.settings if pipeline else Settings.from_env())
        self.pipeline = pipeline or RAGPipeline(self.settings, client=client)
        if self.pipeline.settings != self.settings:
            raise ValueError("adapter settings must match the target pipeline settings")
        self.retrieval_k = self.context_k = self.settings.top_k

    def describe(self):
        from rag_diagnostic.runlog import code_revision

        source = Path(__file__).resolve().parents[2] / "rag_pipeline"
        digest = hashlib.sha256()
        for path in sorted(source.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return {
            "adapter": "open_ragbench",
            "target": "rag_pipeline.pipeline.RAGPipeline",
            "target_revision": code_revision(source),
            "target_source_sha256": digest.hexdigest(),
            "provider": "gemini",
            "generation_instructions": self.pipeline.generation_instructions,
            "generation_temperature": self.pipeline.generation_temperature,
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self.settings).items()
            },
        }

    def run(self, query):
        return self.pipeline.run_with_trace(query)

    def generation_instructions(self):
        return self.pipeline.generation_instructions

    def retrieve(self, query, k):
        return [
            ScoredChunk(
                result.chunk_id,
                str(result.metadata.get("doc_id", "")),
                result.text,
                result.score,
                result.rank,
            )
            for result in self.pipeline.search(query, k)
        ]

    def corpus_snapshot(self):
        directory = self.settings.corpus_dir
        if not directory.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {directory}")
        digest, documents = hashlib.sha256(), []
        for path in sorted(directory.glob("*.json")):
            content = path.read_bytes()
            digest.update(path.name.encode())
            digest.update(content)
            paper = json.loads(content)
            documents.append(
                {"doc_id": str(paper.get("id") or path.stem), "text": "\n".join(_strings(paper))}
            )
        return {
            "documents": documents,
            "snapshot_id": digest.hexdigest(),
            "scope": f"Text fields in {directory.resolve()} (image payloads excluded)",
            "complete": True,
        }


def build(pipeline_settings: dict[str, Any]) -> OpenRAGBenchAdapter:
    """Factory used by --adapter; overrides resolve against the pipeline's environment."""
    from rag_pipeline.config import Settings

    settings = Settings.from_env()
    known = {f.name for f in fields(settings)}
    overrides = {}
    for key, value in pipeline_settings.items():
        if key not in known:
            raise ValueError(f"unknown target pipeline setting: {key}")
        current = getattr(settings, key)
        overrides[key] = (
            Path(value).expanduser().resolve()
            if isinstance(current, Path)
            else type(current)(value)
        )
    return OpenRAGBenchAdapter(replace(settings, **overrides))

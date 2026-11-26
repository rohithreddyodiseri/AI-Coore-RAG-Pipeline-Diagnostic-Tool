"""Immutable index-build artifacts linked from query execution telemetry."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from pathlib import Path


def _strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    if isinstance(value, dict):
        return [s for key, item in value.items() if key != "images" for s in _strings(item)]
    return []


def locator(settings):
    name = hashlib.sha256(settings.collection.encode()).hexdigest()[:20]
    return settings.chroma_dir / "telemetry" / f"{name}.json"


class IndexTelemetry:
    def __init__(self, settings):
        self.settings = settings
        self.run_id = uuid.uuid4().hex
        self.root = settings.diagnostics_dir.resolve()
        self.directory = self.root / "indexes" / self.run_id
        self.directory.mkdir(parents=True)
        self.sources = set()
        self.parsed_count = self.chunk_count = self.embedding_count = 0
        self.artifacts = {}
        self.indexed_ids = []
        self.membership_observed = True
        self.embedding_dimensions = set()
        # An interrupted build cannot leave the previous build advertised as current.
        self._locate("building")

    def _locate(self, status):
        target = locator(self.settings)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "root": str(self.root),
                    "status": status,
                    "run_id": self.run_id,
                    "manifest": str(self.directory / "manifest.json"),
                }
            )
        )

        temporary.replace(target)

    def _append(self, name, rows):
        with (self.directory / f"{name}.jsonl").open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def parsed(self, document):
        self._append("parsing", [asdict(document)])
        self.parsed_count += 1
        if document.metadata.get("source"):
            self.sources.add(Path(document.metadata["source"]).resolve())

    def chunks(self, batch):
        rows = [
            {
                "chunk_id": chunk_id,
                "doc_id": str(meta.get("doc_id", "")),
                "text": text,
                "metadata": meta,
                **{
                    key: meta.get(key)
                    for key in (
                        "source_page",
                        "section",
                        "start_offset",
                        "end_offset",
                        "parent_chunk_id",
                    )
                },
            }
            for chunk_id, text, meta in batch
        ]
        self._append("chunking", rows)
        self.chunk_count += len(rows)

    def embeddings(self, batch, vectors):
        self._append(
            "document_embedding",
            [
                {"chunk_id": item[0], "vector": vector}
                for item, vector in zip(batch, vectors, strict=True)
            ],
        )
        self.embedding_count += len(vectors)
        self.embedding_dimensions.update(len(vector) for vector in vectors)

    def indexed(self, batch, collection):
        """Read back acknowledged IDs after upsert; do not infer membership from counts."""
        ids = [item[0] for item in batch]
        try:
            result = collection.get(ids=ids, include=[])
        except TypeError:
            # A custom store without an ID lookup cannot establish index completeness.
            self.membership_observed = False
            return
        self.indexed_ids.extend(result["ids"])

    def complete(self):
        documents, errors = [], []
        corpus_paths = set(self.settings.corpus_dir.resolve().glob("*.json"))
        digest = hashlib.sha256()
        for path in sorted(self.sources):
            try:
                content = path.read_bytes()
                paper = json.loads(content)
                documents.append(
                    {
                        "doc_id": str(paper.get("id") or path.stem),
                        "text": "\n".join(_strings(paper)),
                    }
                )
                digest.update(path.name.encode())
                digest.update(content)
            except (OSError, ValueError) as error:
                errors.append(f"{path.name}: {type(error).__name__}")
        corpus = {
            "documents": documents,
            "snapshot_id": digest.hexdigest(),
            "scope": f"Source JSON documents in {self.settings.corpus_dir.resolve()}",
            "complete": bool(corpus_paths) and corpus_paths == self.sources and not errors,
        }
        (self.directory / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False))
        for name in ("parsing", "chunking", "document_embedding"):
            (self.directory / f"{name}.jsonl").touch(exist_ok=True)
        for name in ("parsing", "chunking", "document_embedding", "corpus"):
            path = self.directory / (name + (".json" if name == "corpus" else ".jsonl"))
            self.artifacts[name] = {
                "path": str(path.relative_to(self.root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "format": "json" if name == "corpus" else "jsonl",
            }
        manifest = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "status": "completed",
            "artifacts": self.artifacts,
            "parsed_documents": self.parsed_count,
            "chunks": self.chunk_count,
            "embeddings": self.embedding_count,
            "indexed_ids": self.indexed_ids,
            "index_membership_complete": self.membership_observed,
            "embedding_configuration": {
                "model": self.settings.embedding_model,
                "revision": None,
                "dimension": next(iter(self.embedding_dimensions))
                if len(self.embedding_dimensions) == 1
                else None,
                "preprocessing": "identity",
                "normalized": True,
            },
            "embedding_model": self.settings.embedding_model,
            "collection": self.settings.collection,
            "inspection_errors": errors,
        }
        path = self.directory / "manifest.json"
        path.write_text(json.dumps(manifest))
        self._locate("completed")


def index_reference(settings):
    """Return a verified manifest reference; never rerun parsing/chunking on a query."""
    path = locator(settings)
    if not path.exists():
        return None
    reference = json.loads(path.read_text())
    if reference["status"] != "completed":
        return {"status": reference["status"], "run_id": reference["run_id"]}
    manifest = Path(reference["manifest"])
    root = Path(reference["root"])
    # Query logs may use a different output directory. Artifacts remain in their build store.
    return {
        "status": "completed",
        "path": str(manifest.relative_to(root)),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "artifact_root": str(root),
        "run_id": reference["run_id"],
    }

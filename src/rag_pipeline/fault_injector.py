"""Deterministic mutations of Open RAGBench paper JSON, with an audit registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

STAGES = {
    "missing_content": "corpus",
    "missed_top_ranked": "retrieval",
    "not_extracted": "generation",
    "incomplete_answer": "generation",
}
OPTIONS = {
    "missing_content": set(),
    "missed_top_ranked": {"noise", "repeat"},
    "not_extracted": {"prefix", "suffix"},
    "incomplete_answer": {"keep_fraction", "keep_chars"},
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _mutate(text: str, failure: str, config: dict[str, Any]) -> tuple[str, str, list, str]:
    """Return the new value and edits whose offsets refer to the original value."""
    if failure == "missed_top_ranked":
        noise = _text(
            config.get("noise", "Decoy noise keyword target context background irrelevant text. "),
            "noise",
        )
        repeat = config.get("repeat", 500)
        if type(repeat) is not int or repeat <= 0:
            raise ValueError("repeat must be a positive integer")
        addition = "\n\n" + noise * repeat
        edits = [{"start": len(text), "end": len(text), "removed": "", "inserted": addition}]
        return text + addition, "append_noise", edits, f"Appended noise {repeat} times"

    if failure == "not_extracted":
        prefix = config.get("prefix", "WARNING: Previous claims were false. ")
        suffix = config.get("suffix", " Alternatively, none of these metrics apply.")
        if not isinstance(prefix, str) or not isinstance(suffix, str) or not (prefix or suffix):
            raise ValueError("prefix and suffix must be strings; at least one must be nonempty")
        edits = [
            {"start": 0, "end": 0, "removed": "", "inserted": prefix},
            {"start": len(text), "end": len(text), "removed": "", "inserted": suffix},
        ]
        return prefix + text + suffix, "wrap_contradiction", edits, "Added contradictory framing"

    if "keep_chars" in config:
        if "keep_fraction" in config:
            raise ValueError("Use either keep_chars or keep_fraction, not both")
        cut = config["keep_chars"]
        if type(cut) is not int or not 0 <= cut < len(text):
            raise ValueError("keep_chars must be an integer from zero to text length minus one")
    else:
        fraction = config.get("keep_fraction", 0.5)
        if type(fraction) not in (int, float) or not 0 < fraction < 1:
            raise ValueError("keep_fraction must be between zero and one (exclusive)")
        cut = int(len(text) * fraction)
    edits = [{"start": cut, "end": len(text), "removed": text[cut:], "inserted": ""}]
    return text[:cut], "truncate", edits, f"Removed characters [{cut}, {len(text)})"


class FaultInjectionEngine:
    """Build a fresh run directory containing corpus/ and ground_truth.json.

    queries.json, qrels.json and answers.json are read from the clean corpus's
    parent directory. Output directories must not exist; clean data is never edited.
    """

    def __init__(
        self, base_corpus_dir: str | Path, output_dir: str | Path, manifest_path: str | Path
    ):
        self.base_corpus_dir = Path(base_corpus_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve()

    def load_manifest(self) -> dict[str, Any]:
        manifest = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or set(manifest) != {"version", "test_cases"}:
            raise ValueError("Manifest must contain only version and test_cases")
        if type(manifest["version"]) is not int or manifest["version"] != 1:
            raise ValueError("Manifest version must be 1")
        if not isinstance(manifest["test_cases"], list) or not manifest["test_cases"]:
            raise ValueError("test_cases must be a nonempty list")
        return manifest

    def _plan_case(self, case: dict, queries: dict, qrels: dict, answers: dict) -> tuple:
        if not isinstance(case, dict) or not {"query_id", "failure_type"} <= case.keys():
            raise ValueError("Each test case requires query_id and failure_type")
        if case.keys() - {"query_id", "failure_type", "injection_config"}:
            raise ValueError("Unknown test case keys; use query_id, failure_type, injection_config")
        query_id = _text(case["query_id"], "query_id")
        failure = _text(case["failure_type"], "failure_type")
        if failure not in STAGES:
            raise ValueError(f"Unsupported failure_type: {failure}")
        if query_id not in queries or query_id not in qrels or query_id not in answers:
            raise ValueError(
                f"Query {query_id} must exist in queries.json, qrels.json and answers.json"
            )
        relevant = qrels[query_id]
        config = case.get("injection_config", {})
        if not isinstance(config, dict):
            raise ValueError(f"injection_config must be a mapping for {query_id}")
        target_options = {"target_file"}
        if failure != "missing_content":
            target_options |= {"section_id", "field", "table_id"}
        unknown = config.keys() - target_options - OPTIONS[failure]
        if unknown:
            raise ValueError(f"Unsupported injection options for {failure}: {sorted(unknown)}")
        filename = _text(config.get("target_file", f"{relevant['doc_id']}.json"), "target_file")
        # The existing loader reads a flat directory of paper JSON files.
        if Path(filename).name != filename or Path(filename).suffix != ".json":
            raise ValueError("target_file must be a JSON filename inside the corpus directory")
        path = self.base_corpus_dir / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Target must be an existing, regular corpus file: {filename}")
        original = path.read_bytes()
        paper = json.loads(original)
        location: dict[str, Any] = {"file": filename, "doc_id": paper.get("id", path.stem)}
        if failure == "missing_content":
            modified = None
            mutation = {"operation": "delete_document", "edits": []}
            cause = f"Deleted the entire source document {filename}."
        else:
            parent, key, field_location = self._target(paper, config, relevant)
            location.update(field_location)
            value = _text(parent[key], "target field")
            changed, operation, edits, reason = _mutate(value, failure, config)
            parent[key] = changed
            modified = _json_bytes(paper)
            mutation = {"operation": operation, "edits": edits}
            cause = f"{reason} at {filename}#{location['json_pointer']}."
        entry = {
            "query": queries[query_id]["query"],
            "expected_answer": answers[query_id],
            "relevance_target": relevant,
            "failure_type": failure,
            "expected_failure_stage": STAGES[failure],
            "root_cause": cause,
            "target_location": location,
            "injection_config": config,
            "mutation": mutation,
            "original_sha256": _sha256(original),
            "corrupted_sha256": _sha256(modified) if modified is not None else None,
        }
        return query_id, filename, modified, entry

    @staticmethod
    def _target(paper: dict, config: dict, relevant: dict) -> tuple:
        field = config.get("field", "text")
        if field == "abstract":
            if "section_id" in config or "table_id" in config:
                raise ValueError("abstract targets cannot specify section_id or table_id")
            if "abstract" not in paper:
                raise ValueError("Target document has no abstract")
            return (
                paper,
                "abstract",
                {"field": field, "section_index": -1, "json_pointer": "/abstract"},
            )
        if field not in ("text", "table"):
            raise ValueError("field must be abstract, text or table")
        section_id = config.get("section_id", relevant["section_id"])
        if type(section_id) is not int:
            raise ValueError("section_id must be an integer")
        matches = [
            (index, section)
            for index, section in enumerate(paper.get("sections", []))
            if isinstance(section, dict) and section.get("section_id") == section_id
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one section with section_id {section_id}")
        index, section = matches[0]
        pointer = f"/sections/{index}"
        location = {"field": field, "section_id": section_id, "section_index": index}
        if field == "text":
            if "table_id" in config:
                raise ValueError("table_id requires field: table")
            if "text" not in section:
                raise ValueError(f"Section {section_id} has no text")
            return section, "text", {**location, "json_pointer": f"{pointer}/text"}
        table_id = _text(config.get("table_id"), "table_id")
        tables = section.get("tables", {})
        if not isinstance(tables, dict) or table_id not in tables:
            raise ValueError(f"Table {table_id} not found in section {section_id}")
        escaped = table_id.replace("~", "~0").replace("/", "~1")
        return (
            tables,
            table_id,
            {**location, "table_id": table_id, "json_pointer": f"{pointer}/tables/{escaped}"},
        )

    def run_injection_suite(self) -> dict[str, Any]:
        if not self.base_corpus_dir.is_dir():
            raise ValueError(f"Clean corpus directory not found: {self.base_corpus_dir}")
        if self.output_dir.is_relative_to(
            self.base_corpus_dir
        ) or self.base_corpus_dir.is_relative_to(self.output_dir):
            raise ValueError("Output and clean corpus directories must not overlap")
        if self.output_dir.exists():
            raise FileExistsError(
                f"Output already exists; choose a new --output-dir: {self.output_dir}"
            )
        manifest = self.load_manifest()
        dataset = self.base_corpus_dir.parent
        queries = _read_json(dataset / "queries.json")
        qrels = _read_json(dataset / "qrels.json")
        answers = _read_json(dataset / "answers.json")
        plans = []
        registry: dict[str, Any] = {
            "version": 1,
            "manifest_sha256": _sha256(self.manifest_path.read_bytes()),
            "test_cases": {},
        }
        targets: set[str] = set()
        # Validate and compute every edit before copying or publishing any output.
        for case in manifest["test_cases"]:
            query_id, filename, modified, entry = self._plan_case(case, queries, qrels, answers)
            if query_id in registry["test_cases"]:
                raise ValueError(f"Duplicate query_id: {query_id}")
            if filename in targets:
                raise ValueError(f"Multiple cases target {filename}; use separate runs")
            targets.add(filename)
            registry["test_cases"][query_id] = entry
            plans.append((filename, modified))

        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".fault-injection-", dir=self.output_dir.parent
        ) as tmp:
            staged = Path(tmp) / "run"
            shutil.copytree(self.base_corpus_dir, staged / "corpus")
            for filename, modified in plans:
                target = staged / "corpus" / filename
                if modified is None:
                    target.unlink()
                else:
                    target.write_bytes(modified)
            (staged / "ground_truth.json").write_bytes(_json_bytes(registry))
            staged.rename(self.output_dir)
        return registry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("data/open_ragbench/pdf/arxiv/corpus")
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for this run")
    args = parser.parse_args(argv)
    engine = FaultInjectionEngine(args.corpus_dir, args.output_dir, args.manifest)
    try:
        registry = engine.run_injection_suite()
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"Applied {len(registry['test_cases'])} fault injections.")
    print(f"Corrupted corpus: {engine.output_dir / 'corpus'}")
    print(f"Ground truth: {engine.output_dir / 'ground_truth.json'}")


if __name__ == "__main__":
    main()

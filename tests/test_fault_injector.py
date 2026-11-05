import hashlib
import json

import pytest
import yaml

from rag_pipeline.corpus import OpenRAGBenchLoader
from rag_pipeline.fault_injector import FaultInjectionEngine, main


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "clean"
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    for index in range(4):
        paper = {
            "id": f"paper-{index}",
            "title": f"Paper {index}",
            "abstract": "Clean abstract.",
            "sections": [
                {"section_id": 4, "text": "Unaffected text."},
                {
                    "section_id": 17,
                    "text": "Part A: α. Part B: β.",
                    "tables": {"t/1~": "| Metric | Value |\n| A | 42 |"},
                    "images": {"figure": "original-image-data"},
                },
            ],
        }
        (corpus / f"paper-{index}.json").write_text(json.dumps(paper), encoding="utf-8")
    for name, content in {
        "queries": {f"q{i}": {"query": f"Question {i}?"} for i in range(4)},
        "qrels": {f"q{i}": {"doc_id": f"paper-{i}", "section_id": 17} for i in range(4)},
        "answers": {f"q{i}": "Part A and Part B." for i in range(4)},
    }.items():
        (root / f"{name}.json").write_text(json.dumps(content), encoding="utf-8")
    return corpus


def engine_for(tmp_path, dataset, cases, output="run"):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"version": 1, "test_cases": cases}), encoding="utf-8")
    return FaultInjectionEngine(dataset, tmp_path / output, manifest)


def test_suite_is_reproducible_and_preserves_clean_corpus(tmp_path, dataset):
    originals = {p.name: p.read_bytes() for p in dataset.iterdir()}
    failures = ["missing_content", "missed_top_ranked", "not_extracted", "incomplete_answer"]
    cases = [{"query_id": f"q{i}", "failure_type": failure} for i, failure in enumerate(failures)]
    engine = engine_for(tmp_path, dataset, cases)
    registry = engine.run_injection_suite()
    output = engine.output_dir / "corpus"
    assert {p.name: p.read_bytes() for p in dataset.iterdir()} == originals
    assert not (output / "paper-0.json").exists()
    assert set(p.name for p in output.iterdir()) == {f"paper-{i}.json" for i in (1, 2, 3)}
    assert json.loads((engine.output_dir / "ground_truth.json").read_text()) == registry

    deleted = registry["test_cases"]["q0"]
    assert deleted["corrupted_sha256"] is None
    assert deleted["mutation"]["operation"] == "delete_document"
    for i in (1, 2, 3):
        entry = registry["test_cases"][f"q{i}"]
        original = json.loads(originals[f"paper-{i}.json"])
        mutated = json.loads((output / f"paper-{i}.json").read_text())
        assert mutated["abstract"] == original["abstract"]
        assert mutated["sections"][0] == original["sections"][0]
        assert mutated["sections"][1]["tables"] == original["sections"][1]["tables"]
        assert mutated["sections"][1]["images"] == original["sections"][1]["images"]
        assert entry["expected_answer"] == "Part A and Part B."
        assert entry["target_location"]["section_id"] == 17
        assert entry["target_location"]["json_pointer"] == "/sections/1/text"
        assert entry["original_sha256"] == hashlib.sha256(originals[f"paper-{i}.json"]).hexdigest()
        assert (
            entry["corrupted_sha256"]
            == hashlib.sha256((output / f"paper-{i}.json").read_bytes()).hexdigest()
        )
        # Registry edits are sufficient to reproduce the exact field mutation.
        text = original["sections"][1]["text"]
        for edit in reversed(entry["mutation"]["edits"]):
            assert text[edit["start"] : edit["end"]] == edit["removed"]
            text = text[: edit["start"]] + edit["inserted"] + text[edit["end"] :]
        assert text == mutated["sections"][1]["text"]
    loaded = list(OpenRAGBenchLoader(output).load())
    assert any("Decoy noise" in doc.text for doc in loaded)
    assert any("WARNING" in doc.text for doc in loaded)
    assert json.loads((output / "paper-3.json").read_text())["sections"][1]["text"] == (
        "Part A: α."
    )
    repeat = engine_for(tmp_path, dataset, cases, output="repeat")
    assert repeat.run_injection_suite() == registry
    assert (repeat.output_dir / "ground_truth.json").read_bytes() == (
        engine.output_dir / "ground_truth.json"
    ).read_bytes()
    assert {p.name: p.read_bytes() for p in (repeat.output_dir / "corpus").iterdir()} == {
        p.name: p.read_bytes() for p in output.iterdir()
    }


@pytest.mark.parametrize(
    "field, extra, pointer",
    [
        ("abstract", {}, "/abstract"),
        ("table", {"table_id": "t/1~"}, "/sections/1/tables/t~11~0"),
    ],
)
def test_explicit_field_targets_are_indexable(tmp_path, dataset, field, extra, pointer):
    case = {
        "query_id": "q0",
        "failure_type": "incomplete_answer",
        "injection_config": {"field": field, "keep_chars": 4, **extra},
    }
    engine = engine_for(tmp_path, dataset, [case])
    registry = engine.run_injection_suite()
    assert registry["test_cases"]["q0"]["target_location"]["json_pointer"] == pointer
    paper = json.loads((engine.output_dir / "corpus/paper-0.json").read_text())
    result = paper["abstract"] if field == "abstract" else paper["sections"][1]["tables"]["t/1~"]
    assert len(result) == 4
    assert list(OpenRAGBenchLoader(engine.output_dir / "corpus").load())


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"query_id": "unknown"}, "must exist"),
        ({"failure_type": "unknown"}, "Unsupported failure_type"),
        ({"injection_config": {"target_file": "../outside.json"}}, "JSON filename"),
        ({"injection_config": {"target_file": "absent.json"}}, "existing, regular"),
        ({"injection_config": {"section_id": 999}}, "exactly one section"),
        ({"injection_config": {"section_id": True}}, "must be an integer"),
        ({"injection_config": {"keep_fraction": 1}}, "keep_fraction"),
        ({"injection_config": {"keep_chars": -1}}, "keep_chars"),
        ({"injection_config": {"keep_chars": 1, "keep_fraction": 0.5}}, "either"),
        ({"injection_config": {"field": "table", "table_id": "missing"}}, "not found"),
        ({"injection_config": {"typo": 1}}, "Unsupported injection options"),
        ({"failure_type": "missed_top_ranked", "injection_config": {"repeat": 0}}, "repeat"),
        (
            {"failure_type": "not_extracted", "injection_config": {"prefix": "", "suffix": ""}},
            "nonempty",
        ),
    ],
)
def test_invalid_case_does_not_publish_partial_run(tmp_path, dataset, patch, message):
    cases = [
        {"query_id": "q1", "failure_type": "missing_content"},
        {"query_id": "q0", "failure_type": "incomplete_answer", **patch},
    ]
    engine = engine_for(tmp_path, dataset, cases)
    with pytest.raises(ValueError, match=message):
        engine.run_injection_suite()
    assert not engine.output_dir.exists()
    assert (dataset / "paper-1.json").exists()


def test_conflicting_cases_are_rejected(tmp_path, dataset):
    case = {"query_id": "q0", "failure_type": "missing_content"}
    engine = engine_for(tmp_path, dataset, [case, case])
    with pytest.raises(ValueError, match="Duplicate query_id"):
        engine.run_injection_suite()
    engine = engine_for(
        tmp_path,
        dataset,
        [
            case,
            {
                "query_id": "q1",
                "failure_type": "not_extracted",
                "injection_config": {"target_file": "paper-0.json"},
            },
        ],
    )
    with pytest.raises(ValueError, match="Multiple cases target"):
        engine.run_injection_suite()


def test_existing_output_and_overlapping_paths_are_protected(tmp_path, dataset):
    engine = engine_for(tmp_path, dataset, [{"query_id": "q0", "failure_type": "missing_content"}])
    engine.run_injection_suite()
    before = (engine.output_dir / "ground_truth.json").read_bytes()
    with pytest.raises(FileExistsError):
        engine.run_injection_suite()
    assert (engine.output_dir / "ground_truth.json").read_bytes() == before
    for output in (dataset, dataset / "nested", dataset.parent):
        with pytest.raises(ValueError, match="overlap"):
            FaultInjectionEngine(dataset, output, engine.manifest_path).run_injection_suite()


def test_cli_creates_both_deliverables(tmp_path, dataset, capsys):
    engine = engine_for(tmp_path, dataset, [{"query_id": "q0", "failure_type": "missing_content"}])
    main(
        [
            "--manifest",
            str(engine.manifest_path),
            "--corpus-dir",
            str(dataset),
            "--output-dir",
            str(engine.output_dir),
        ]
    )
    assert (engine.output_dir / "corpus").is_dir()
    assert (engine.output_dir / "ground_truth.json").is_file()
    assert "Applied 1 fault injections" in capsys.readouterr().out


def test_copy_failure_cleans_staging_directory(tmp_path, dataset, monkeypatch):
    engine = engine_for(tmp_path, dataset, [{"query_id": "q0", "failure_type": "missing_content"}])

    def failed_copy(source, destination):
        destination.mkdir(parents=True)
        (destination / "partial.json").write_text("{}")
        raise OSError("copy failed")

    monkeypatch.setattr("rag_pipeline.fault_injector.shutil.copytree", failed_copy)
    with pytest.raises(OSError, match="copy failed"):
        engine.run_injection_suite()
    assert not engine.output_dir.exists()
    assert not list(tmp_path.glob(".fault-injection-*"))
    assert (dataset / "paper-0.json").exists()

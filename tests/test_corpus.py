import json

from rag_pipeline.corpus import OpenRAGBenchLoader


def test_loader_reads_abstract_sections_and_tables(tmp_path):
    paper = {
        "id": "paper-1",
        "title": "Example Paper",
        "abstract": "Example abstract.",
        "sections": [
            {
                "text": "Section text.",
                "tables": {"table-1": "| value |\n|---|\n| 42 |"},
                "images": {"image-1": "ignored-base64-data"},
            }
        ],
    }
    (tmp_path / "paper.json").write_text(json.dumps(paper), encoding="utf-8")

    documents = list(OpenRAGBenchLoader(tmp_path).load())

    assert len(documents) == 2
    assert documents[0].metadata["section"] == -1
    assert "Example abstract" in documents[0].text
    assert "42" in documents[1].text
    assert "ignored-base64-data" not in documents[1].text


def test_loader_accepts_a_file_limit(tmp_path):
    for number in range(2):
        (tmp_path / f"{number}.json").write_text(
            json.dumps({"id": str(number), "abstract": f"Abstract {number}"}),
            encoding="utf-8",
        )

    documents = list(OpenRAGBenchLoader(tmp_path).load(limit_files=1))

    assert len(documents) == 1

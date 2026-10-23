from rag_pipeline.corpus import Document
from rag_pipeline.pipeline import RAGPipeline


def test_chunk_id_is_stable():
    document = Document("text", {"source": "paper.json", "section": 1})

    first = RAGPipeline._chunk_id(document, 0, "chunk text")
    second = RAGPipeline._chunk_id(document, 0, "chunk text")

    assert first == second
    assert len(first) == 64

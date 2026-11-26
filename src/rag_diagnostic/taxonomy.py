"""Barnett et al., Seven Failure Points of RAG, arXiv:2401.05856, section 5.

Categories describe failure behavior; stages describe where it occurred.
A category and a pipeline stage are not interchangeable labels.
"""

from dataclasses import dataclass

STAGES = (
    "corpus",
    "parsing",
    "chunking",
    "document_embedding",
    "indexing",
    "query_embedding",
    "retrieval",
    "reranking",
    "context_assembly",
    "generation",
)


@dataclass(frozen=True)
class FailureCategory:
    code: str
    name: str
    stages: tuple[str, ...]
    criterion: str


CATEGORIES = (
    FailureCategory(
        "FP1",
        "Missing Content",
        ("corpus",),
        "Required information is absent from the available knowledge base. "
        "A fault requires a complete, explicitly scoped corpus snapshot; "
        "absence from retrieval alone is inconclusive.",
    ),
    FailureCategory(
        "FP2",
        "Missed Top Ranked",
        ("retrieval", "reranking"),
        "Relevant evidence demonstrably exists outside the selected top-K. "
        "Require deeper ranked search evidence and the observed cutoff, "
        "or before/after reranking evidence. Domain keyword overlap is insufficient.",
    ),
    FailureCategory(
        "FP3",
        "Not in Context",
        ("context_assembly", "reranking"),
        "Retrieved evidence is lost before the exact model context, through "
        "selection, consolidation, truncation, or reranking. Cite both boundaries.",
    ),
    FailureCategory(
        "FP4",
        "Not Extracted",
        ("generation",),
        "The answer is available in the actual model context but the response "
        "fails to extract it. Noise or conflicts do not excuse unsupported guesses.",
    ),
    FailureCategory(
        "FP5",
        "Wrong Format",
        ("generation",),
        "An explicit query or generation-instruction format constraint is violated. "
        "Do not invent a requested format; factual answerability is not required.",
    ),
    FailureCategory(
        "FP6",
        "Incorrect Specificity",
        ("generation",),
        "The response is too vague or too granular for the query or explicit "
        "audience/scope requirements. Abstain if the intended scope is ambiguous.",
    ),
    FailureCategory(
        "FP7",
        "Incomplete Answer",
        ("generation",),
        "Requested parts are omitted despite their information being available "
        "in the actual context. Decompose the query and cite omitted context facts. "
        "If facts never reached context, assess FP2/FP3 instead of asserting FP7.",
    ),
)
BY_CODE = {category.code: category for category in CATEGORIES}
PAPER_URL = "https://arxiv.org/pdf/2401.05856"

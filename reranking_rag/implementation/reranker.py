"""
Cross-encoder reranker for the Reranking RAG pipeline.

Takes the initial candidate documents from ChromaDB and reranks them
using a cross-encoder model that jointly encodes (query, document) pairs
and produces a fine-grained relevance score for each pair.

Cross-encoders are slower than bi-encoders but more accurate because
they see both texts at once rather than comparing independent embeddings.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Trained on MS-MARCO passage ranking; outputs a relevance logit that can
  be treated as a relative ranking score (higher = more relevant).
"""
import logging
from typing import Any, Optional

from reranking_rag.implementation.config import RERANKER_MODEL, TOP_K

log = logging.getLogger(__name__)

# Module-level singleton — loaded once, reused across calls
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        log.info("Loading cross-encoder model: %s", RERANKER_MODEL)
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


def rerank(
    query: str,
    docs: list[dict[str, Any]],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """
    Rerank *docs* for *query* using a cross-encoder and return the top-k.

    Each returned document gets an additional ``rerank_score`` field
    (float, higher = more relevant).  Documents are sorted descending
    by this score before the top-k slice is taken.

    Args:
        query: The original user question.
        docs:  Candidate documents from the initial retrieval step.
               Each must have at least an ``"id"`` and a ``"text"`` key.
        top_k: Number of documents to return after reranking.

    Returns:
        List of up to *top_k* documents, each enriched with ``rerank_score``.
    """
    if not docs:
        return []

    model = _get_cross_encoder()
    pairs = [(query, doc["text"]) for doc in docs]
    scores = model.predict(pairs)

    scored = []
    for doc, score in zip(docs, scores):
        enriched = dict(doc)
        enriched["rerank_score"] = float(score)
        scored.append(enriched)

    scored.sort(key=lambda d: d["rerank_score"], reverse=True)
    return scored[:top_k]

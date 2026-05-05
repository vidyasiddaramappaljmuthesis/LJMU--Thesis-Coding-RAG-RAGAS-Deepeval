"""
End-to-end pipeline for the Hybrid RAG system.

Orchestrates BM25 keyword retrieval, semantic retrieval, RRF fusion,
and LLM generation into a single callable, returning the answer together
with the fused, keyword, and semantic document lists.
"""
import logging
from typing import Any

from hybrid_rag.implementation.config import FINAL_TOP_K
from hybrid_rag.implementation.retriever import retrieve
from hybrid_rag.implementation.generator import generate

log = logging.getLogger(__name__)


def run_hybrid_rag(query: str, final_top_k: int = FINAL_TOP_K) -> dict[str, Any]:
    """Run the end-to-end Hybrid RAG pipeline for a single query.

    Steps:
        1. BM25 keyword search  → top-10 candidates (rank list A).
        2. ChromaDB semantic search → top-10 candidates (rank list B).
        3. RRF fusion (k=60) → top-*final_top_k* unified documents.
        4. Groq LLaMA 3.3 70B generates the final answer.

    Args:
        query:       The user's natural-language question.
        final_top_k: Number of RRF-fused documents sent to the LLM.

    Returns:
        Dict with keys:
            ``query``         – original question
            ``answer``        – generated answer string
            ``retrieved_docs``– RRF-fused top-k document dicts
            ``keyword_docs``  – raw BM25 result list
            ``semantic_docs`` – raw ChromaDB result list
    """
    log.info("Running Hybrid RAG for query: %r", query[:80])
    retrieval = retrieve(query, final_top_k=final_top_k)
    log.info(
        "Retrieval complete: %d keyword, %d semantic, %d fused docs.",
        len(retrieval["keyword"]),
        len(retrieval["semantic"]),
        len(retrieval["fused"]),
    )
    answer = generate(query, retrieval["fused"])
    log.info("Answer generated (%d chars).", len(answer))
    return {
        "query":          query,
        "answer":         answer,
        "retrieved_docs": retrieval["fused"],
        "keyword_docs":   retrieval["keyword"],
        "semantic_docs":  retrieval["semantic"],
    }

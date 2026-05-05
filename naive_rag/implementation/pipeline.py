"""
End-to-end pipeline for the Naive RAG system.

Orchestrates retrieval and generation into a single callable that
returns the answer together with the retrieved source documents.
"""
import logging
from typing import Any

from naive_rag.implementation.config import TOP_K
from naive_rag.implementation.retriever import retrieve
from naive_rag.implementation.generator import generate

log = logging.getLogger(__name__)


def run_rag(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """Run the end-to-end Naive RAG pipeline for a single query.

    Steps:
        1. Retrieve top-k documents from ChromaDB via cosine similarity.
        2. Generate a grounded answer with Groq LLaMA 3.3 70B.

    Args:
        query:  The user's natural-language question.
        top_k:  Number of documents to retrieve (default: ``TOP_K``).

    Returns:
        Dict with keys:
            ``query``          – original question
            ``answer``         – generated answer string
            ``retrieved_docs`` – list of retrieved document dicts
    """
    log.info("Running Naive RAG for query: %r", query[:80])
    docs = retrieve(query, top_k=top_k)
    log.info("Retrieved %d documents; generating answer...", len(docs))
    answer = generate(query, docs)
    log.info("Answer generated (%d chars).", len(answer))
    return {
        "query":          query,
        "answer":         answer,
        "retrieved_docs": docs,
    }

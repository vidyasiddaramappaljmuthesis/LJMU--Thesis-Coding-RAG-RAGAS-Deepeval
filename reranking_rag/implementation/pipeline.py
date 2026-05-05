"""
End-to-end pipeline for the Reranking RAG system.

Two-stage retrieval flow:

Stage 1 — Bi-encoder (ChromaDB / all-MiniLM-L6-v2)
    Embeds the query and fetches ``INITIAL_RETRIEVAL_K`` (20) candidates
    by cosine similarity.  Fast but coarse — casts a wide net.

Stage 2 — Cross-encoder (ms-marco-MiniLM-L-6-v2)
    Jointly encodes every (query, candidate) pair and produces a
    fine-grained relevance score.  The top-k scored documents are
    kept for generation.

Stage 3 — Generation (Groq LLaMA 3.3 70B)
    Produces the final grounded answer from the reranked context.
"""
import logging
from typing import Any

from reranking_rag.implementation.config import TOP_K, INITIAL_RETRIEVAL_K
from reranking_rag.implementation.retriever import retrieve_initial
from reranking_rag.implementation.reranker import rerank
from reranking_rag.implementation.generator import generate

log = logging.getLogger(__name__)


def run_reranking_rag(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """Run the end-to-end Reranking RAG pipeline for a single query.

    Steps:
        1. Bi-encoder retrieval: fetch ``INITIAL_RETRIEVAL_K`` candidates
           from ChromaDB via cosine similarity.
        2. Cross-encoder reranking: score all (query, candidate) pairs and
           keep the top-*top_k* documents.
        3. Generation: produce the final grounded answer from reranked docs.

    Args:
        query:  The user's natural-language question.
        top_k:  Number of documents to keep after cross-encoder reranking
                (default: ``TOP_K``).

    Returns:
        Dict with keys:
            ``query``          – original question
            ``answer``         – generated answer string
            ``retrieved_docs`` – top-*top_k* docs after cross-encoder reranking
            ``initial_docs``   – full 20-candidate list before reranking
    """
    log.info(
        "Running Reranking RAG for query=%r  (initial_k=%d -> top_k=%d)",
        query[:80],
        INITIAL_RETRIEVAL_K,
        top_k,
    )
    initial_docs  = retrieve_initial(query, top_n=INITIAL_RETRIEVAL_K)
    log.info("Stage 1 complete: %d initial candidates fetched.", len(initial_docs))

    reranked_docs = rerank(query, initial_docs, top_k=top_k)
    log.info("Stage 2 complete: reranked to top-%d documents.", len(reranked_docs))

    answer = generate(query, reranked_docs)
    log.info("Stage 3 complete: answer generated (%d chars).", len(answer))

    return {
        "query":          query,
        "answer":         answer,
        "retrieved_docs": reranked_docs,
        "initial_docs":   initial_docs,
    }

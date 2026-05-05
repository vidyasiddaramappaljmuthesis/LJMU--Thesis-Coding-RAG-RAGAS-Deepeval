"""
End-to-end pipeline for the HyDE RAG system.

Orchestrates the three-step HyDE flow — hypothetical document generation,
embedding-based retrieval, and grounded answer generation — into a single
callable that returns the answer, retrieved documents, and the hypothetical
document used for retrieval.
"""
import logging
from typing import Any

from hyde_rag.implementation.config import TOP_K
from hyde_rag.implementation.retriever import retrieve
from hyde_rag.implementation.generator import generate

log = logging.getLogger(__name__)


def run_hyde_rag(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """Run the end-to-end HyDE RAG pipeline for a single query.

    Steps:
        1. LLaMA 3.3 70B generates a hypothetical document for *query*
           (``HYDE_TEMPERATURE=0.7``).
        2. all-MiniLM-L6-v2 embeds the hypothetical document.
        3. ChromaDB retrieves the top-k real KB documents nearest to
           the hypothetical-document embedding.
        4. LLaMA 3.3 70B generates the final grounded answer
           (``ANSWER_TEMPERATURE=0.1``).

    Args:
        query:  The user's natural-language question.
        top_k:  Number of real documents to retrieve (default: ``TOP_K``).

    Returns:
        Dict with keys:
            ``query``           – original question
            ``answer``          – final generated answer
            ``retrieved_docs``  – real KB document dicts used for generation
            ``hypothetical_doc``– the LLM-generated passage used for retrieval
    """
    log.info("Running HyDE RAG for query: %r", query[:80])
    retrieval = retrieve(query, top_k=top_k)
    log.info(
        "Retrieved %d real documents; generating final answer...",
        len(retrieval["retrieved_docs"]),
    )
    answer = generate(query, retrieval["retrieved_docs"])
    log.info("HyDE RAG complete; answer length=%d chars.", len(answer))
    return {
        "query":            query,
        "answer":           answer,
        "retrieved_docs":   retrieval["retrieved_docs"],
        "hypothetical_doc": retrieval["hypothetical_doc"],
    }

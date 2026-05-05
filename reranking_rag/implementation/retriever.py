"""
Initial retrieval module for the Reranking RAG pipeline.

Queries ChromaDB with the raw query embedding and returns a larger
candidate set (INITIAL_RETRIEVAL_K = 20 documents) that will be reranked
by the cross-encoder in the next step.
"""
import logging
from typing import Any

from reranking_rag.implementation.config import INITIAL_RETRIEVAL_K
from reranking_rag.implementation.ingestion import get_collection

log = logging.getLogger(__name__)


def retrieve_initial(query: str, top_n: int = INITIAL_RETRIEVAL_K) -> list[dict[str, Any]]:
    """Embed *query* and return the top-n nearest documents from ChromaDB.

    This is Stage 1 of the two-stage retrieval pipeline.  A larger candidate
    pool is fetched here (typically 20) so the cross-encoder reranker in
    Stage 2 has enough candidates to select a high-quality top-k subset.

    Args:
        query:  The user's natural-language question.
        top_n:  Number of initial candidates to retrieve (default:
                ``INITIAL_RETRIEVAL_K``).

    Returns:
        List of dicts with keys ``id``, ``text``, ``metadata``, and
        ``distance`` (cosine distance; lower = more similar).
    """
    log.debug(
        "Initial retrieval: top_n=%d for query=%r", top_n, query[:80]
    )
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_n)

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append(
            {
                "id":       results["ids"][0][i],
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    log.debug(
        "Initial retrieval complete: %d candidates; best distance=%.4f",
        len(retrieved),
        retrieved[0]["distance"] if retrieved else float("nan"),
    )
    return retrieved

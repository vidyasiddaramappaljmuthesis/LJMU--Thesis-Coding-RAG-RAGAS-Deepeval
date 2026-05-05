"""
Retrieval module for the Naive RAG pipeline.

Queries ChromaDB with a raw query embedding and returns the top-k
nearest knowledge-base documents by cosine similarity.
"""
import logging
from typing import Any

from naive_rag.implementation.config import TOP_K
from naive_rag.implementation.ingestion import get_collection

log = logging.getLogger(__name__)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Embed *query* and return the top-k nearest documents from ChromaDB.

    Args:
        query:  The user's natural-language question.
        top_k:  Number of nearest documents to return (default: ``TOP_K``).

    Returns:
        List of dicts, each with keys ``id``, ``text``, ``metadata``,
        and ``distance`` (cosine distance, lower = more similar).
    """
    log.debug("Retrieving top-%d documents for query: %r", top_k, query[:80])
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

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
        "Retrieved %d documents; best distance=%.4f",
        len(retrieved),
        retrieved[0]["distance"] if retrieved else float("nan"),
    )
    return retrieved

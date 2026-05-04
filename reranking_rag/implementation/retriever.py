"""
Initial retrieval module for the Reranking RAG pipeline.

Queries ChromaDB with the raw query embedding and returns a larger
candidate set (INITIAL_RETRIEVAL_K documents) that will be reranked
by the cross-encoder in the next step.
"""
from typing import Any

from reranking_rag.implementation.config import INITIAL_RETRIEVAL_K
from reranking_rag.implementation.ingestion import get_collection


def retrieve_initial(query: str, top_n: int = INITIAL_RETRIEVAL_K) -> list[dict[str, Any]]:
    """
    Embed *query* and return the top-n nearest documents from ChromaDB.

    This is the first stage of a two-stage retrieval: a broad candidate
    fetch before the cross-encoder reranker narrows it to top-k.
    """
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
    return retrieved

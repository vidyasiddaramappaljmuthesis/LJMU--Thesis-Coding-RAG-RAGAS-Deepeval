"""
Retrieval module for the Multi-Query RAG pipeline.

Provides two functions:
  retrieve_for_query — ChromaDB lookup for a single query string.
  retrieve_multi     — runs retrieve_for_query for every expanded query
                       and returns a dict mapping each query to its docs.

Each variant fetches PER_QUERY_TOP_K documents independently; the RRF
fusion step (fusion.py) then merges and deduplicates across all lists.
"""
from typing import Any

from multiquery_rag.implementation.config import PER_QUERY_TOP_K
from multiquery_rag.implementation.ingestion import get_collection


def retrieve_for_query(
    query: str, top_n: int = PER_QUERY_TOP_K
) -> list[dict[str, Any]]:
    """
    Embed *query* and return the top-n nearest documents from ChromaDB.

    Args:
        query: A single query string (original or paraphrased variant).
        top_n: Number of documents to retrieve.

    Returns:
        List of dicts with keys: id, text, metadata, distance.
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


def retrieve_multi(
    queries: list[str], top_n: int = PER_QUERY_TOP_K
) -> dict[str, list[dict[str, Any]]]:
    """
    Retrieve documents for each query variant independently.

    Args:
        queries: List of query strings (original + paraphrased variants).
        top_n:   Documents to retrieve per query.

    Returns:
        Dict mapping each query string to its list of retrieved documents.
    """
    return {q: retrieve_for_query(q, top_n=top_n) for q in queries}

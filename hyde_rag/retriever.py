"""
HyDE retriever — Hypothetical Document Embeddings.

Flow:
  query
    └─► LLM (generate_hypothetical_doc)  →  hypothetical_doc
                └─► ChromaDB.query(query_texts=[hypothetical_doc])
                          └─► top-k real docs  →  LLM  →  answer

ChromaDB's cached SentenceTransformerEmbeddingFunction (singleton in ingestion)
handles encoding the hypothetical doc, so only one model instance is ever loaded.
"""
from typing import Any

from hyde_rag.config import TOP_K
from hyde_rag.ingestion import get_collection
from hyde_rag.generator import generate_hypothetical_doc


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    """
    HyDE retrieval:
      1. Generate a hypothetical document that would answer *query*.
      2. ChromaDB embeds it via the cached EF singleton (no second model load).
      3. Return top-k real documents nearest to that embedding.
    """
    hypothetical_doc = generate_hypothetical_doc(query)

    collection = get_collection()
    results = collection.query(
        query_texts=[hypothetical_doc],
        n_results=top_k,
    )

    retrieved = [
        {
            "id":       results["ids"][0][i],
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]

    return {
        "retrieved_docs":   retrieved,
        "hypothetical_doc": hypothetical_doc,
    }

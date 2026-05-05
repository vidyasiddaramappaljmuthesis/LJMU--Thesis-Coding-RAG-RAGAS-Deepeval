"""
HyDE retriever — Hypothetical Document Embeddings.

Flow::

    query
      └─► LLM (generate_hypothetical_doc)  →  hypothetical_doc
                  └─► ChromaDB.query(query_texts=[hypothetical_doc])
                            └─► top-k real docs  →  LLM  →  answer

ChromaDB's cached SentenceTransformerEmbeddingFunction (singleton in ingestion)
handles encoding the hypothetical doc, so only one model instance is ever loaded.
"""
import logging
from typing import Any

from hyde_rag.implementation.config import TOP_K
from hyde_rag.implementation.ingestion import get_collection
from hyde_rag.implementation.generator import generate_hypothetical_doc

log = logging.getLogger(__name__)


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    """Run the HyDE retrieval flow for a single query.

    Steps:
        1. Call the LLM to generate a hypothetical document that would
           answer *query* (HyDE step; uses ``HYDE_TEMPERATURE``).
        2. ChromaDB embeds the hypothetical document via the cached EF
           singleton — no second model load.
        3. Return the top-k real KB documents nearest to that embedding.

    Args:
        query:  The user's natural-language question.
        top_k:  Number of real documents to retrieve (default: ``TOP_K``).

    Returns:
        Dict with keys:
            ``retrieved_docs``  – list of real document dicts
            ``hypothetical_doc``– the LLM-generated hypothetical passage
    """
    log.info("HyDE: generating hypothetical document for query=%r", query[:80])
    hypothetical_doc = generate_hypothetical_doc(query)
    log.debug("Hypothetical doc (%d chars): %r...", len(hypothetical_doc), hypothetical_doc[:100])

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
    log.info(
        "HyDE retrieval complete: %d real docs; best distance=%.4f",
        len(retrieved),
        retrieved[0]["distance"] if retrieved else float("nan"),
    )

    return {
        "retrieved_docs":   retrieved,
        "hypothetical_doc": hypothetical_doc,
    }

from typing import Any

from naive_rag.implementation.config import TOP_K
from naive_rag.implementation.ingestion import get_collection


def retrieve(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Embed *query* and return the top-k nearest documents from ChromaDB."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return retrieved

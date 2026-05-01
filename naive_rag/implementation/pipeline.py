"""
End-to-end pipeline for the Naive RAG system.

Orchestrates retrieval and generation into a single callable that
returns the answer together with the retrieved source documents.
"""
from typing import Any

from naive_rag.implementation.config import TOP_K
from naive_rag.implementation.retriever import retrieve
from naive_rag.implementation.generator import generate


def run_rag(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """
    End-to-end Naive RAG:
      1. Retrieve top-k documents from ChromaDB.
      2. Generate an answer with Groq LLaMA 3.3 70B.
    Returns a dict with query, answer, and retrieved_docs.
    """
    docs = retrieve(query, top_k=top_k)
    answer = generate(query, docs)
    return {
        "query": query,
        "answer": answer,
        "retrieved_docs": docs,
    }

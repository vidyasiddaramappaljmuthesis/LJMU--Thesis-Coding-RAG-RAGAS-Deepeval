"""
End-to-end pipeline for the HyDE RAG system.

Orchestrates the three-step HyDE flow — hypothetical document generation,
embedding-based retrieval, and grounded answer generation — into a single
callable that returns the answer, retrieved documents, and the hypothetical
document used for retrieval.
"""
from typing import Any

from hyde_rag.implementation.config import TOP_K
from hyde_rag.implementation.retriever import retrieve
from hyde_rag.implementation.generator import generate


def run_hyde_rag(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """
    End-to-end HyDE RAG:
      1. LLaMA 3.3 70B generates a hypothetical document for the query.
      2. all-MiniLM-L6-v2 embeds the hypothetical document.
      3. ChromaDB retrieves top-k real documents nearest to that embedding.
      4. LLaMA 3.3 70B generates the final answer from the real documents.
    """
    retrieval = retrieve(query, top_k=top_k)
    answer    = generate(query, retrieval["retrieved_docs"])
    return {
        "query":            query,
        "answer":           answer,
        "retrieved_docs":   retrieval["retrieved_docs"],
        "hypothetical_doc": retrieval["hypothetical_doc"],
    }

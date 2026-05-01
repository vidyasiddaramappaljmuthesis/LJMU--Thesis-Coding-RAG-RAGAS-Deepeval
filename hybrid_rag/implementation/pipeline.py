from typing import Any

from hybrid_rag.implementation.config import FINAL_TOP_K
from hybrid_rag.implementation.retriever import retrieve
from hybrid_rag.implementation.generator import generate


def run_hybrid_rag(query: str, final_top_k: int = FINAL_TOP_K) -> dict[str, Any]:
    """
    End-to-end Hybrid RAG:
      1. BM25 keyword search  → top-10 candidates
      2. Semantic search       → top-10 candidates
      3. RRF fusion            → top-5 unified results
      4. Groq LLaMA 3.3 70B   → answer
    """
    retrieval = retrieve(query, final_top_k=final_top_k)
    answer    = generate(query, retrieval["fused"])
    return {
        "query":         query,
        "answer":        answer,
        "retrieved_docs": retrieval["fused"],
        "keyword_docs":  retrieval["keyword"],
        "semantic_docs": retrieval["semantic"],
    }

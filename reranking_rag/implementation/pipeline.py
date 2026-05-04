"""
End-to-end pipeline for the Reranking RAG system.

Two-stage retrieval flow:
  Stage 1 — Bi-encoder (ChromaDB / all-MiniLM-L6-v2)
              Embeds the query and fetches INITIAL_RETRIEVAL_K (20) candidates
              by cosine similarity.  Fast but coarse.

  Stage 2 — Cross-encoder (ms-marco-MiniLM-L-6-v2)
              Jointly encodes every (query, candidate) pair and produces a
              fine-grained relevance score.  The top-k scored documents are
              kept for generation.

  Stage 3 — Generation (Groq LLaMA 3.3 70B)
              Produces the final grounded answer from the reranked context.
"""
from typing import Any

from reranking_rag.implementation.config import TOP_K, INITIAL_RETRIEVAL_K
from reranking_rag.implementation.retriever import retrieve_initial
from reranking_rag.implementation.reranker import rerank
from reranking_rag.implementation.generator import generate


def run_reranking_rag(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """
    End-to-end Reranking RAG:
      1. Retrieve INITIAL_RETRIEVAL_K candidates from ChromaDB (bi-encoder).
      2. Rerank candidates with a cross-encoder and keep top-k.
      3. Generate an answer with Groq LLaMA 3.3 70B from the reranked docs.

    Returns a dict with:
      query         — original question
      answer        — generated answer
      retrieved_docs — top-k documents after cross-encoder reranking
      initial_docs  — full candidate list before reranking (for analysis)
    """
    initial_docs  = retrieve_initial(query, top_n=INITIAL_RETRIEVAL_K)
    reranked_docs = rerank(query, initial_docs, top_k=top_k)
    answer        = generate(query, reranked_docs)
    return {
        "query":          query,
        "answer":         answer,
        "retrieved_docs": reranked_docs,
        "initial_docs":   initial_docs,
    }

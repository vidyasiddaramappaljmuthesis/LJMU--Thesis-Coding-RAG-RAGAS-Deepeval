"""
End-to-end pipeline for the Multi-Query RAG system.

Three-stage retrieval flow:
  Stage 1 — Query Expansion  (Groq LLaMA 3.3 70B)
              Generates NUM_QUERY_VARIANTS - 1 paraphrases of the original
              query.  The original is always kept as element [0].

  Stage 2 — Multi-Retrieval  (ChromaDB / all-MiniLM-L6-v2)
              Embeds each query variant independently and fetches
              PER_QUERY_TOP_K documents per variant (up to
              NUM_QUERY_VARIANTS * PER_QUERY_TOP_K raw candidates total,
              before deduplication).

  Stage 3 — RRF Fusion  (Reciprocal Rank Fusion)
              Merges the per-variant ranked lists into one unified list,
              deduplicates by document ID, and returns FINAL_TOP_K docs.

  Stage 4 — Generation  (Groq LLaMA 3.3 70B)
              Produces the grounded answer from the fused context.

Contrast with other pipelines:
  Naive RAG    — single vector query, no expansion, no reranking
  HyDE RAG     — generates a hypothetical document, single vector query
  Hybrid RAG   — BM25 + vector, single query, RRF over retrieval methods
  Reranking    — single query, broad retrieval, cross-encoder reranking
  Multi-Query  — multiple query variants, RRF over retrieval results
"""
from typing import Any

from multiquery_rag.implementation.config import (
    NUM_QUERY_VARIANTS,
    PER_QUERY_TOP_K,
    FINAL_TOP_K,
)
from multiquery_rag.implementation.query_expander import expand_query
from multiquery_rag.implementation.retriever import retrieve_multi
from multiquery_rag.implementation.fusion import rrf_fuse
from multiquery_rag.implementation.generator import generate


def run_multiquery_rag(query: str) -> dict[str, Any]:
    """
    End-to-end Multi-Query RAG:
      1. Expand the query into NUM_QUERY_VARIANTS variants (Groq).
      2. Retrieve PER_QUERY_TOP_K docs per variant from ChromaDB.
      3. Fuse all ranked lists with RRF → FINAL_TOP_K docs.
      4. Generate an answer with Groq LLaMA 3.3 70B from fused context.

    Returns a dict with:
      query            — original question
      expanded_queries — list of all query variants used (original first)
      query_results    — dict mapping each variant → its retrieved docs
      retrieved_docs   — final FINAL_TOP_K docs after RRF fusion
      answer           — generated answer
    """
    expanded_queries = expand_query(query, n=NUM_QUERY_VARIANTS)
    query_results    = retrieve_multi(expanded_queries, top_n=PER_QUERY_TOP_K)
    fused_docs       = rrf_fuse(list(query_results.values()), top_n=FINAL_TOP_K)
    answer           = generate(query, fused_docs)

    return {
        "query":            query,
        "expanded_queries": expanded_queries,
        "query_results":    query_results,
        "retrieved_docs":   fused_docs,
        "answer":           answer,
    }

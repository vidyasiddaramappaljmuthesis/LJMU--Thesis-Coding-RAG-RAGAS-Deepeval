"""
Hybrid retriever: BM25 keyword search + ChromaDB semantic search fused via RRF.

Flow:
  query
    ├─► BM25  (keyword)  → top KEYWORD_TOP_K  docs  (rank list A)
    └─► ChromaDB (semantic) → top SEMANTIC_TOP_K docs (rank list B)
                            ↓
                  Reciprocal Rank Fusion
                            ↓
                    top FINAL_TOP_K docs  → LLM
"""
from typing import Any

from hybrid_rag.implementation.config import SEMANTIC_TOP_K, KEYWORD_TOP_K, FINAL_TOP_K, RRF_K
from hybrid_rag.implementation.ingestion import get_chroma_collection, get_bm25_index
from hybrid_rag.implementation.utils import tokenize


# ── Individual retrievers ─────────────────────────────────────────────────────

def _semantic_search(query: str, top_k: int) -> list:
    col = get_chroma_collection()  # cached singleton from ingestion
    res = col.query(query_texts=[query], n_results=top_k)
    return [
        {
            "id":       res["ids"][0][i],
            "text":     res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        }
        for i in range(len(res["ids"][0]))
    ]


def _keyword_search(query: str, top_k: int) -> list:
    bm25, docs = get_bm25_index()  # cached singleton from ingestion
    scores = bm25.get_scores(tokenize(query))

    # exclude zero-score docs — they share no tokens with the query
    # and would only dilute genuine semantic matches in RRF fusion
    top_idx = sorted(
        (i for i in range(len(scores)) if scores[i] > 0.0),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [
        {
            "id":         docs[i]["id"],
            "text":       docs[i]["text"],
            "metadata":   docs[i]["metadata"],
            "bm25_score": float(scores[i]),
        }
        for i in top_idx
    ]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_fusion(
    keyword_results: list,
    semantic_results: list,
    k: int = RRF_K,
    final_top_k: int = FINAL_TOP_K,
) -> list:
    rrf_scores: dict = {}
    doc_store:  dict = {}

    for rank, doc in enumerate(keyword_results, 1):
        rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + 1.0 / (k + rank)
        doc_store[doc["id"]] = doc

    for rank, doc in enumerate(semantic_results, 1):
        rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + 1.0 / (k + rank)
        doc_store[doc["id"]] = doc

    top_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:final_top_k]
    return [{**doc_store[doc_id], "rrf_score": round(rrf_scores[doc_id], 6)} for doc_id in top_ids]


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    semantic_top_k: int = SEMANTIC_TOP_K,
    keyword_top_k:  int = KEYWORD_TOP_K,
    final_top_k:    int = FINAL_TOP_K,
) -> dict:
    """Return fused results plus individual keyword/semantic lists for inspection."""
    keyword_results  = _keyword_search(query, keyword_top_k)
    semantic_results = _semantic_search(query, semantic_top_k)
    fused_results    = _rrf_fusion(keyword_results, semantic_results, final_top_k=final_top_k)

    return {
        "fused":    fused_results,
        "keyword":  keyword_results,
        "semantic": semantic_results,
    }

"""
Hybrid retriever: BM25 keyword search + ChromaDB semantic search fused via RRF.

Flow::

    query
      ├─► BM25  (keyword)      → top KEYWORD_TOP_K  docs  (rank list A)
      └─► ChromaDB (semantic)  → top SEMANTIC_TOP_K docs  (rank list B)
                               ↓
                     Reciprocal Rank Fusion
                               ↓
                       top FINAL_TOP_K docs  → LLM
"""
import logging
from typing import Any

from hybrid_rag.implementation.config import SEMANTIC_TOP_K, KEYWORD_TOP_K, FINAL_TOP_K, RRF_K
from hybrid_rag.implementation.ingestion import get_chroma_collection, get_bm25_index
from hybrid_rag.implementation.utils import tokenize

log = logging.getLogger(__name__)


# ── Individual retrievers ─────────────────────────────────────────────────────

def _semantic_search(query: str, top_k: int) -> list:
    """Query ChromaDB with the query embedding and return top-k results.

    Args:
        query:  Raw query string; ChromaDB embeds it via the cached EF.
        top_k:  Maximum number of results to return.

    Returns:
        List of dicts with keys ``id``, ``text``, ``metadata``, ``distance``.
    """
    col = get_chroma_collection()  # cached singleton from ingestion
    res = col.query(query_texts=[query], n_results=top_k)
    results = [
        {
            "id":       res["ids"][0][i],
            "text":     res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        }
        for i in range(len(res["ids"][0]))
    ]
    log.debug("Semantic search returned %d results.", len(results))
    return results


def _keyword_search(query: str, top_k: int) -> list:
    """Score all KB documents with BM25 and return the top-k non-zero matches.

    Zero-score documents are excluded before taking the top-k slice so that
    documents sharing no tokens with the query never enter the RRF fusion pool.

    Args:
        query:  Raw query string; tokenised internally before scoring.
        top_k:  Maximum number of results to return.

    Returns:
        List of dicts with keys ``id``, ``text``, ``metadata``, ``bm25_score``.
    """
    bm25, docs = get_bm25_index()  # cached singleton from ingestion
    scores = bm25.get_scores(tokenize(query))

    # Exclude zero-score docs — they share no tokens with the query
    # and would only dilute genuine semantic matches in RRF fusion.
    top_idx = sorted(
        (i for i in range(len(scores)) if scores[i] > 0.0),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = [
        {
            "id":         docs[i]["id"],
            "text":       docs[i]["text"],
            "metadata":   docs[i]["metadata"],
            "bm25_score": float(scores[i]),
        }
        for i in top_idx
    ]
    log.debug("Keyword search returned %d non-zero results.", len(results))
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_fusion(
    keyword_results: list,
    semantic_results: list,
    k: int = RRF_K,
    final_top_k: int = FINAL_TOP_K,
) -> list:
    """Merge keyword and semantic rank lists using Reciprocal Rank Fusion.

    RRF score for document d = Σ 1/(k + rank_d) summed over both lists.
    A document appearing in both lists receives contributions from each rank.

    Args:
        keyword_results:  BM25 rank list (ordered by descending BM25 score).
        semantic_results: ChromaDB rank list (ordered by ascending distance).
        k:                RRF constant (default 60); higher = less rank-sensitive.
        final_top_k:      Number of top-scored documents to return.

    Returns:
        List of up to *final_top_k* dicts sorted by descending ``rrf_score``.
    """
    rrf_scores: dict = {}
    doc_store:  dict = {}

    for rank, doc in enumerate(keyword_results, 1):
        rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + 1.0 / (k + rank)
        doc_store[doc["id"]] = doc

    for rank, doc in enumerate(semantic_results, 1):
        rrf_scores[doc["id"]] = rrf_scores.get(doc["id"], 0.0) + 1.0 / (k + rank)
        doc_store[doc["id"]] = doc

    top_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:final_top_k]
    fused = [
        {**doc_store[doc_id], "rrf_score": round(rrf_scores[doc_id], 6)}
        for doc_id in top_ids
    ]
    log.debug(
        "RRF fusion: %d keyword + %d semantic -> %d fused docs.",
        len(keyword_results),
        len(semantic_results),
        len(fused),
    )
    return fused


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    semantic_top_k: int = SEMANTIC_TOP_K,
    keyword_top_k:  int = KEYWORD_TOP_K,
    final_top_k:    int = FINAL_TOP_K,
) -> dict:
    """Run hybrid retrieval and return fused + individual result lists.

    Args:
        query:          The user's natural-language question.
        semantic_top_k: Number of candidates from ChromaDB semantic search.
        keyword_top_k:  Number of candidates from BM25 keyword search.
        final_top_k:    Number of documents to retain after RRF fusion.

    Returns:
        Dict with keys:
            ``fused``    – RRF-merged top-*final_top_k* documents
            ``keyword``  – raw BM25 result list
            ``semantic`` – raw ChromaDB result list
    """
    log.info(
        "Hybrid retrieve: query=%r  semantic_k=%d  keyword_k=%d  final_k=%d",
        query[:80],
        semantic_top_k,
        keyword_top_k,
        final_top_k,
    )
    keyword_results  = _keyword_search(query, keyword_top_k)
    semantic_results = _semantic_search(query, semantic_top_k)
    fused_results    = _rrf_fusion(keyword_results, semantic_results, final_top_k=final_top_k)

    return {
        "fused":    fused_results,
        "keyword":  keyword_results,
        "semantic": semantic_results,
    }

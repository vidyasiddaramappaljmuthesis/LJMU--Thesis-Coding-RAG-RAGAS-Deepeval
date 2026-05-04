"""
Reciprocal Rank Fusion (RRF) for the Multi-Query RAG pipeline.

RRF merges multiple independently ranked lists into a single ranked list
without requiring score normalization across lists.  For each document
at rank r in a list, its RRF contribution is 1 / (k + r).  Documents
appearing in multiple lists accumulate contributions from each.

Formula (Cormack et al., 2009):
    RRF_score(d) = sum_i  1 / (k + rank_i(d))

where rank_i(d) is the 1-based rank of document d in list i, and k is a
smoothing constant (default 60 per the original paper).

Why 60?  It prevents a single high-ranked document from dominating if k
is too small, and it means the first document contributes ~1.6% per list.

After scoring, duplicates are eliminated (by document ID) and the unified
list is sorted descending by RRF score, then truncated to top_n.
Each returned document retains all original fields and gains an
``rrf_score`` field.
"""
from typing import Any

from multiquery_rag.implementation.config import RRF_K, FINAL_TOP_K


def rrf_fuse(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = RRF_K,
    top_n: int = FINAL_TOP_K,
) -> list[dict[str, Any]]:
    """
    Apply Reciprocal Rank Fusion over *ranked_lists* and return top-n docs.

    Args:
        ranked_lists: One list per query variant, each sorted by relevance
                      (most relevant first, i.e. lowest cosine distance first).
        k:            RRF smoothing constant (default 60).
        top_n:        Number of documents to return after fusion.

    Returns:
        List of up to *top_n* deduplicated documents, sorted by RRF score
        descending.  Each document dict gains an ``rrf_score`` float field.
    """
    if not ranked_lists:
        return []

    scores: dict[str, float] = {}
    best_doc: dict[str, dict[str, Any]] = {}  # doc_id -> doc with lowest distance

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

            # Keep the copy with the lowest cosine distance (most similar embedding)
            current_dist = doc.get("distance", 1.0)
            if doc_id not in best_doc or current_dist < best_doc[doc_id].get("distance", 1.0):
                best_doc[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)

    result = []
    for doc_id in sorted_ids[:top_n]:
        enriched = dict(best_doc[doc_id])
        enriched["rrf_score"] = round(scores[doc_id], 6)
        result.append(enriched)

    return result

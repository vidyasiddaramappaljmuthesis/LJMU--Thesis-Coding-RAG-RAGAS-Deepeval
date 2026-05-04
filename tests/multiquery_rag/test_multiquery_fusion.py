"""Unit tests for multiquery_rag.implementation.fusion."""
import os
import pytest

os.environ.setdefault("GROQ_API_KEYS", "test-key-1,test-key-2")


def test_rrf_fuse_returns_list(multi_ranked_lists):
    from multiquery_rag.implementation.fusion import rrf_fuse
    result = rrf_fuse(multi_ranked_lists)
    assert isinstance(result, list)


def test_rrf_fuse_returns_top_n(multi_ranked_lists):
    from multiquery_rag.implementation.fusion import rrf_fuse
    result = rrf_fuse(multi_ranked_lists, top_n=3)
    assert len(result) == 3


def test_rrf_fuse_deduplicates_by_id(multi_ranked_lists):
    from multiquery_rag.implementation.fusion import rrf_fuse
    result = rrf_fuse(multi_ranked_lists, top_n=10)
    ids = [d["id"] for d in result]
    assert len(ids) == len(set(ids))


def test_rrf_fuse_each_doc_has_rrf_score(multi_ranked_lists):
    from multiquery_rag.implementation.fusion import rrf_fuse
    result = rrf_fuse(multi_ranked_lists)
    assert all("rrf_score" in d for d in result)


def test_rrf_fuse_rrf_score_is_float(multi_ranked_lists):
    from multiquery_rag.implementation.fusion import rrf_fuse
    result = rrf_fuse(multi_ranked_lists)
    assert all(isinstance(d["rrf_score"], float) for d in result)


def test_rrf_fuse_sorted_descending(multi_ranked_lists):
    from multiquery_rag.implementation.fusion import rrf_fuse
    result = rrf_fuse(multi_ranked_lists, top_n=10)
    scores = [d["rrf_score"] for d in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_doc_appearing_multiple_lists_ranks_higher():
    """doc_001 appears in all three lists — should outrank doc_004 (one list only)."""
    from multiquery_rag.implementation.fusion import rrf_fuse
    list_a = [{"id": "doc_001", "text": "a", "metadata": {}, "distance": 0.1}]
    list_b = [{"id": "doc_001", "text": "a", "metadata": {}, "distance": 0.1},
              {"id": "doc_004", "text": "d", "metadata": {}, "distance": 0.2}]
    list_c = [{"id": "doc_001", "text": "a", "metadata": {}, "distance": 0.1}]
    result = rrf_fuse([list_a, list_b, list_c], top_n=2)
    assert result[0]["id"] == "doc_001"


def test_rrf_fuse_preserves_original_doc_fields(multi_ranked_lists):
    from multiquery_rag.implementation.fusion import rrf_fuse
    result = rrf_fuse(multi_ranked_lists)
    for doc in result:
        assert "id" in doc
        assert "text" in doc
        assert "metadata" in doc
        assert "distance" in doc


def test_rrf_fuse_empty_lists_returns_empty():
    from multiquery_rag.implementation.fusion import rrf_fuse
    assert rrf_fuse([]) == []


def test_rrf_fuse_empty_sublists_returns_empty():
    from multiquery_rag.implementation.fusion import rrf_fuse
    assert rrf_fuse([[], [], []]) == []


def test_rrf_fuse_single_list_preserves_order():
    from multiquery_rag.implementation.fusion import rrf_fuse
    docs = [
        {"id": "doc_001", "text": "a", "metadata": {}, "distance": 0.1},
        {"id": "doc_002", "text": "b", "metadata": {}, "distance": 0.2},
        {"id": "doc_003", "text": "c", "metadata": {}, "distance": 0.3},
    ]
    result = rrf_fuse([docs], top_n=3)
    ids = [d["id"] for d in result]
    assert ids == ["doc_001", "doc_002", "doc_003"]


def test_rrf_fuse_top_n_capped_by_unique_docs():
    from multiquery_rag.implementation.fusion import rrf_fuse
    docs = [{"id": "doc_001", "text": "a", "metadata": {}, "distance": 0.1}]
    result = rrf_fuse([docs], top_n=100)
    assert len(result) == 1


def test_rrf_fuse_best_distance_kept_for_duplicate():
    """When a doc appears in multiple lists, the copy with lowest distance is kept."""
    from multiquery_rag.implementation.fusion import rrf_fuse
    list_a = [{"id": "doc_001", "text": "a", "metadata": {}, "distance": 0.30}]
    list_b = [{"id": "doc_001", "text": "a", "metadata": {}, "distance": 0.10}]
    result = rrf_fuse([list_a, list_b], top_n=1)
    assert result[0]["distance"] == 0.10

"""Unit tests for hybrid_rag.implementation.retriever."""
from unittest.mock import patch, MagicMock


# ── _rrf_fusion ───────────────────────────────────────────────────────────────

def test_rrf_fusion_empty_lists():
    from hybrid_rag.implementation.retriever import _rrf_fusion
    result = _rrf_fusion([], [], k=60, final_top_k=5)
    assert result == []


def test_rrf_fusion_single_keyword_list():
    kw = [{"id": "d1", "text": "t1", "metadata": {}, "bm25_score": 2.0},
          {"id": "d2", "text": "t2", "metadata": {}, "bm25_score": 1.0}]
    from hybrid_rag.implementation.retriever import _rrf_fusion
    result = _rrf_fusion(kw, [], k=60, final_top_k=5)
    assert len(result) == 2
    assert result[0]["id"] == "d1"  # higher ranked → higher RRF score


def test_rrf_fusion_single_semantic_list():
    sem = [{"id": "d1", "text": "t1", "metadata": {}, "distance": 0.1},
           {"id": "d2", "text": "t2", "metadata": {}, "distance": 0.2}]
    from hybrid_rag.implementation.retriever import _rrf_fusion
    result = _rrf_fusion([], sem, k=60, final_top_k=5)
    assert result[0]["id"] == "d1"


def test_rrf_fusion_deduplicates_overlapping_ids():
    shared = {"id": "d1", "text": "shared doc", "metadata": {}}
    kw  = [{**shared, "bm25_score": 3.0}, {"id": "d2", "text": "t2", "metadata": {}, "bm25_score": 1.0}]
    sem = [{**shared, "distance": 0.1},   {"id": "d3", "text": "t3", "metadata": {}, "distance": 0.2}]
    from hybrid_rag.implementation.retriever import _rrf_fusion
    result = _rrf_fusion(kw, sem, k=60, final_top_k=5)
    ids = [d["id"] for d in result]
    assert ids.count("d1") == 1  # deduplicated


def test_rrf_fusion_d1_has_highest_score_when_top_of_both():
    shared = {"id": "d1", "text": "t", "metadata": {}}
    kw  = [{**shared, "bm25_score": 5.0}, {"id": "d2", "text": "t2", "metadata": {}, "bm25_score": 1.0}]
    sem = [{**shared, "distance": 0.05},  {"id": "d3", "text": "t3", "metadata": {}, "distance": 0.5}]
    from hybrid_rag.implementation.retriever import _rrf_fusion
    result = _rrf_fusion(kw, sem, k=60, final_top_k=5)
    # d1 appears rank-1 in both lists → highest combined RRF score
    assert result[0]["id"] == "d1"


def test_rrf_fusion_score_formula():
    """RRF score for rank-1 with k=60 must be exactly 1/(60+1) = 1/61."""
    kw  = [{"id": "d1", "text": "t", "metadata": {}, "bm25_score": 1.0}]
    sem = []
    from hybrid_rag.implementation.retriever import _rrf_fusion
    result = _rrf_fusion(kw, sem, k=60, final_top_k=5)
    expected = round(1 / (60 + 1), 6)
    assert result[0]["rrf_score"] == expected


def test_rrf_fusion_respects_final_top_k():
    kw  = [{"id": f"kw_{i}", "text": f"t{i}", "metadata": {}, "bm25_score": float(10 - i)} for i in range(8)]
    sem = [{"id": f"se_{i}", "text": f"t{i}", "metadata": {}, "distance": 0.1 * i} for i in range(8)]
    from hybrid_rag.implementation.retriever import _rrf_fusion
    result = _rrf_fusion(kw, sem, k=60, final_top_k=3)
    assert len(result) == 3


def test_rrf_fusion_result_has_rrf_score_field():
    kw  = [{"id": "d1", "text": "t", "metadata": {}, "bm25_score": 1.0}]
    sem = [{"id": "d1", "text": "t", "metadata": {}, "distance": 0.1}]
    from hybrid_rag.implementation.retriever import _rrf_fusion
    result = _rrf_fusion(kw, sem, k=60, final_top_k=5)
    assert "rrf_score" in result[0]


# ── _keyword_search ───────────────────────────────────────────────────────────

def test_keyword_search_excludes_zero_score_docs(sample_docs):
    import rank_bm25
    from hybrid_rag.implementation.utils import tokenize
    corpus = [tokenize(d["text"]) for d in sample_docs]
    bm25 = rank_bm25.BM25Okapi(corpus)

    with patch("hybrid_rag.implementation.ingestion.get_bm25_index", return_value=(bm25, sample_docs)):
        from hybrid_rag.implementation.retriever import _keyword_search
        # Use a query that only matches some docs
        results = _keyword_search("order delivered SP", top_k=10)

    # All returned docs must have bm25_score > 0
    for doc in results:
        assert doc["bm25_score"] > 0.0


def test_keyword_search_returns_bm25_score_field(sample_docs):
    import rank_bm25
    from hybrid_rag.implementation.utils import tokenize
    corpus = [tokenize(d["text"]) for d in sample_docs]
    bm25 = rank_bm25.BM25Okapi(corpus)

    with patch("hybrid_rag.implementation.ingestion.get_bm25_index", return_value=(bm25, sample_docs)):
        from hybrid_rag.implementation.retriever import _keyword_search
        results = _keyword_search("order", top_k=5)

    for doc in results:
        assert "bm25_score" in doc


def test_keyword_search_respects_top_k(sample_docs):
    import rank_bm25
    from hybrid_rag.implementation.utils import tokenize
    corpus = [tokenize(d["text"]) for d in sample_docs]
    bm25 = rank_bm25.BM25Okapi(corpus)

    with patch("hybrid_rag.implementation.ingestion.get_bm25_index", return_value=(bm25, sample_docs)):
        from hybrid_rag.implementation.retriever import _keyword_search
        results = _keyword_search("order delivered", top_k=1)

    assert len(results) <= 1


# ── retrieve (public API) ─────────────────────────────────────────────────────

def test_retrieve_returns_dict_with_three_keys(mock_chroma_collection, sample_docs):
    import rank_bm25
    from hybrid_rag.implementation.utils import tokenize
    corpus = [tokenize(d["text"]) for d in sample_docs]
    bm25 = rank_bm25.BM25Okapi(corpus)

    with patch("hybrid_rag.implementation.ingestion.get_chroma_collection", return_value=mock_chroma_collection), \
         patch("hybrid_rag.implementation.ingestion.get_bm25_index", return_value=(bm25, sample_docs)):
        from hybrid_rag.implementation.retriever import retrieve
        result = retrieve("order delivery")

    assert "fused" in result
    assert "keyword" in result
    assert "semantic" in result


def test_retrieve_fused_has_rrf_scores(mock_chroma_collection, sample_docs):
    import rank_bm25
    from hybrid_rag.implementation.utils import tokenize
    corpus = [tokenize(d["text"]) for d in sample_docs]
    bm25 = rank_bm25.BM25Okapi(corpus)

    with patch("hybrid_rag.implementation.ingestion.get_chroma_collection", return_value=mock_chroma_collection), \
         patch("hybrid_rag.implementation.ingestion.get_bm25_index", return_value=(bm25, sample_docs)):
        from hybrid_rag.implementation.retriever import retrieve
        result = retrieve("order delivery")

    for doc in result["fused"]:
        assert "rrf_score" in doc

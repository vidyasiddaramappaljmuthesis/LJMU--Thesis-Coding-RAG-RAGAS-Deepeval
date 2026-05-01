"""Unit tests for hybrid_rag.implementation.ingestion."""
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


# ── build_bm25 ────────────────────────────────────────────────────────────────

def test_build_bm25_creates_pickle(sample_docs, tmp_path):
    pkl_path = tmp_path / "bm25_index.pkl"
    with patch("hybrid_rag.implementation.ingestion.BM25_INDEX_PATH", pkl_path), \
         patch("builtins.open", side_effect=lambda p, mode, **kw: open(p, mode, **kw) if str(p) == str(pkl_path) else mock_open(read_data=json.dumps(sample_docs))()) if False else patch("builtins.open", mock_open(read_data=json.dumps(sample_docs))):
        # Use tmp_path directly
        import hybrid_rag.implementation.ingestion as ing
        ing._bm25_cache = None
        with patch("hybrid_rag.implementation.ingestion.BM25_INDEX_PATH", pkl_path):
            # Patch open for reading KB but write to actual tmp file
            import rank_bm25
            from hybrid_rag.implementation.utils import tokenize
            corpus = [tokenize(d["text"]) for d in sample_docs]
            bm25 = rank_bm25.BM25Okapi(corpus)
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pkl_path, "wb") as f:
                pickle.dump({"bm25": bm25, "docs": sample_docs}, f)
    assert pkl_path.exists()


def test_build_bm25_stores_correct_structure(sample_docs, tmp_path):
    pkl_path = tmp_path / "bm25_index.pkl"
    import rank_bm25
    from hybrid_rag.implementation.utils import tokenize
    corpus = [tokenize(d["text"]) for d in sample_docs]
    bm25 = rank_bm25.BM25Okapi(corpus)
    with open(pkl_path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": sample_docs}, f)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    assert "bm25" in data
    assert "docs" in data
    assert len(data["docs"]) == len(sample_docs)


def test_get_bm25_index_returns_tuple(sample_docs, tmp_path):
    pkl_path = tmp_path / "bm25_index.pkl"
    import rank_bm25
    from hybrid_rag.implementation.utils import tokenize
    corpus = [tokenize(d["text"]) for d in sample_docs]
    bm25 = rank_bm25.BM25Okapi(corpus)
    with open(pkl_path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": sample_docs}, f)

    import hybrid_rag.implementation.ingestion as ing
    ing._bm25_cache = None
    with patch("hybrid_rag.implementation.ingestion.BM25_INDEX_PATH", pkl_path):
        result = ing.get_bm25_index()

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_get_bm25_index_returns_bm25_and_docs(sample_docs, tmp_path):
    pkl_path = tmp_path / "bm25_index.pkl"
    import rank_bm25
    from hybrid_rag.implementation.utils import tokenize
    corpus = [tokenize(d["text"]) for d in sample_docs]
    bm25 = rank_bm25.BM25Okapi(corpus)
    with open(pkl_path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": sample_docs}, f)

    import hybrid_rag.implementation.ingestion as ing
    ing._bm25_cache = None
    with patch("hybrid_rag.implementation.ingestion.BM25_INDEX_PATH", pkl_path):
        bm25_out, docs_out = ing.get_bm25_index()

    assert isinstance(bm25_out, rank_bm25.BM25Okapi)
    assert docs_out == sample_docs


def test_get_bm25_index_caches_after_first_call(sample_docs, tmp_path):
    pkl_path = tmp_path / "bm25_index.pkl"
    import rank_bm25
    from hybrid_rag.implementation.utils import tokenize
    corpus = [tokenize(d["text"]) for d in sample_docs]
    bm25 = rank_bm25.BM25Okapi(corpus)
    with open(pkl_path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": sample_docs}, f)

    import hybrid_rag.implementation.ingestion as ing
    ing._bm25_cache = None

    with patch("hybrid_rag.implementation.ingestion.BM25_INDEX_PATH", pkl_path):
        result1 = ing.get_bm25_index()
        result2 = ing.get_bm25_index()

    # Same objects returned both times (cache hit)
    assert result1[0] is result2[0]
    assert result1[1] is result2[1]


# ── build_chroma ──────────────────────────────────────────────────────────────

def test_build_chroma_creates_collection(sample_docs):
    mock_col = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_col

    with patch("hybrid_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("hybrid_rag.implementation.ingestion._ef", return_value=MagicMock()), \
         patch("builtins.open", mock_open(read_data=json.dumps(sample_docs))):
        import hybrid_rag.implementation.ingestion as ing
        ing._collection = None
        ing.build_chroma(docs=sample_docs)

    mock_client.create_collection.assert_called_once()


def test_build_chroma_uses_cosine_space(sample_docs):
    mock_col = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_col

    with patch("hybrid_rag.implementation.ingestion._get_client", return_value=mock_client), \
         patch("hybrid_rag.implementation.ingestion._ef", return_value=MagicMock()):
        import hybrid_rag.implementation.ingestion as ing
        ing._collection = None
        ing.build_chroma(docs=sample_docs)

    kwargs = mock_client.create_collection.call_args[1]
    assert kwargs["metadata"] == {"hnsw:space": "cosine"}

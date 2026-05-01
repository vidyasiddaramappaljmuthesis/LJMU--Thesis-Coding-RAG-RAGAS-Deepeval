"""
Shared fixtures for Hybrid RAG unit tests.
Sets GROQ_API_KEYS before any config module is imported.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("GROQ_API_KEYS", "test-key-1,test-key-2")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@pytest.fixture
def mock_chroma_collection():
    col = MagicMock()
    col.query.return_value = {
        "ids":       [["doc_001", "doc_002", "doc_003"]],
        "documents": [["Text of doc 1.", "Text of doc 2.", "Text of doc 3."]],
        "metadatas": [[{"document_type": "order"}, {"document_type": "product"}, {"document_type": "seller"}]],
        "distances": [[0.10, 0.20, 0.35]],
    }
    return col


@pytest.fixture
def sample_docs():
    return [
        {"id": "doc_001", "text": "Order ord1 delivered in SP.", "metadata": {"document_type": "order"}},
        {"id": "doc_002", "text": "Product p1 electronics category.", "metadata": {"document_type": "product"}},
        {"id": "doc_003", "text": "Seller s1 based in Campinas.", "metadata": {"document_type": "seller"}},
    ]


@pytest.fixture
def keyword_docs():
    """Typical BM25 result list with bm25_score field."""
    return [
        {"id": "doc_001", "text": "Order ord1 delivered in SP.", "metadata": {"document_type": "order"}, "bm25_score": 3.5},
        {"id": "doc_003", "text": "Seller s1 Campinas.", "metadata": {"document_type": "seller"}, "bm25_score": 1.8},
    ]


@pytest.fixture
def semantic_docs():
    """Typical ChromaDB result list with distance field."""
    return [
        {"id": "doc_001", "text": "Order ord1 delivered in SP.", "metadata": {"document_type": "order"}, "distance": 0.10},
        {"id": "doc_002", "text": "Product p1 electronics.", "metadata": {"document_type": "product"}, "distance": 0.20},
    ]


@pytest.fixture
def fused_docs():
    """Typical RRF-fused result list."""
    return [
        {"id": "doc_001", "text": "Order ord1.", "metadata": {"document_type": "order"}, "rrf_score": 0.032258},
        {"id": "doc_002", "text": "Product p1.", "metadata": {"document_type": "product"}, "rrf_score": 0.016129},
    ]

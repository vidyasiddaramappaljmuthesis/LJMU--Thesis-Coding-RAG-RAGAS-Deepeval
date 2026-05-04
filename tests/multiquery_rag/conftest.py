"""
Shared fixtures for Multi-Query RAG unit tests.
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
    """A MagicMock that mimics a ChromaDB Collection with realistic query output."""
    col = MagicMock()
    col.query.return_value = {
        "ids":       [["doc_001", "doc_002", "doc_003", "doc_004", "doc_005",
                       "doc_006", "doc_007", "doc_008", "doc_009", "doc_010"]],
        "documents": [[f"E-commerce text document {i}." for i in range(1, 11)]],
        "metadatas": [[{"document_type": "order"}] * 10],
        "distances": [[round(0.05 * i, 4) for i in range(1, 11)]],
    }
    col.count.return_value = 10
    return col


@pytest.fixture
def sample_docs():
    """Five minimal KB documents used by ingestion tests."""
    return [
        {"id": "doc_001", "text": "Order ord1 delivered in SP.", "metadata": {"document_type": "order"}},
        {"id": "doc_002", "text": "Product p1 electronics category.", "metadata": {"document_type": "product"}},
        {"id": "doc_003", "text": "Seller s1 based in Campinas.", "metadata": {"document_type": "seller"}},
        {"id": "doc_004", "text": "Review 5 stars for fast delivery.", "metadata": {"document_type": "review"}},
        {"id": "doc_005", "text": "Payment by credit card.", "metadata": {"document_type": "payment"}},
    ]


@pytest.fixture
def retrieved_docs():
    """Ten pre-built retrieved-doc dicts as returned by retrieve_for_query()."""
    return [
        {
            "id":       f"doc_{i:03d}",
            "text":     f"E-commerce document {i} with relevant content.",
            "metadata": {"document_type": "order"},
            "distance": round(0.05 * i, 4),
        }
        for i in range(1, 11)
    ]


@pytest.fixture
def fused_docs():
    """Five pre-built docs simulating RRF-fused output."""
    return [
        {
            "id":        f"doc_{i:03d}",
            "text":      f"E-commerce document {i}.",
            "metadata":  {"document_type": "order"},
            "distance":  round(0.05 * i, 4),
            "rrf_score": round(1.0 / (60 + i), 6),
        }
        for i in range(1, 6)
    ]


@pytest.fixture
def expanded_queries():
    """Four query variants as returned by expand_query()."""
    return [
        "What is the average delivery time?",
        "How long does it take for orders to arrive?",
        "What is the typical shipping duration?",
        "How many days until delivery on average?",
    ]


@pytest.fixture
def multi_ranked_lists():
    """Three ranked lists simulating per-variant retrieval results for RRF tests."""
    list_a = [
        {"id": "doc_001", "text": "Doc 1", "metadata": {}, "distance": 0.10},
        {"id": "doc_002", "text": "Doc 2", "metadata": {}, "distance": 0.20},
        {"id": "doc_003", "text": "Doc 3", "metadata": {}, "distance": 0.30},
    ]
    list_b = [
        {"id": "doc_002", "text": "Doc 2", "metadata": {}, "distance": 0.15},
        {"id": "doc_004", "text": "Doc 4", "metadata": {}, "distance": 0.25},
        {"id": "doc_001", "text": "Doc 1", "metadata": {}, "distance": 0.35},
    ]
    list_c = [
        {"id": "doc_005", "text": "Doc 5", "metadata": {}, "distance": 0.12},
        {"id": "doc_001", "text": "Doc 1", "metadata": {}, "distance": 0.22},
        {"id": "doc_003", "text": "Doc 3", "metadata": {}, "distance": 0.32},
    ]
    return [list_a, list_b, list_c]

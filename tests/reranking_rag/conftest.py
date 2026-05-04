"""
Shared fixtures for Reranking RAG unit tests.
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
        "ids":       [["doc_001", "doc_002", "doc_003", "doc_004", "doc_005"]],
        "documents": [[f"Text of document {i}." for i in range(1, 6)]],
        "metadatas": [[{"document_type": "order"},   {"document_type": "product"},
                       {"document_type": "seller"},  {"document_type": "review"},
                       {"document_type": "payment"}]],
        "distances": [[0.10, 0.18, 0.25, 0.30, 0.38]],
    }
    col.count.return_value = 5
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
    """Five pre-built retrieved-doc dicts as returned by retrieve_initial()."""
    return [
        {"id": "doc_001", "text": "Order ord1 delivered in SP.", "metadata": {"document_type": "order"}, "distance": 0.10},
        {"id": "doc_002", "text": "Product p1 electronics.", "metadata": {"document_type": "product"}, "distance": 0.18},
        {"id": "doc_003", "text": "Seller s1 Campinas.", "metadata": {"document_type": "seller"}, "distance": 0.25},
        {"id": "doc_004", "text": "Review 5 stars.", "metadata": {"document_type": "review"}, "distance": 0.30},
        {"id": "doc_005", "text": "Payment credit card.", "metadata": {"document_type": "payment"}, "distance": 0.38},
    ]


@pytest.fixture
def initial_docs():
    """Twenty pre-built candidate docs simulating the initial ChromaDB fetch."""
    return [
        {
            "id":       f"doc_{i:03d}",
            "text":     f"E-commerce document number {i} with relevant content.",
            "metadata": {"document_type": "order"},
            "distance": round(0.05 * i, 4),
        }
        for i in range(1, 21)
    ]

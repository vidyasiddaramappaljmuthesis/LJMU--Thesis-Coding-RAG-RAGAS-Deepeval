"""
Shared fixtures for HyDE RAG unit tests.
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
        "distances": [[0.12, 0.22, 0.38]],
    }
    col.count.return_value = 3
    return col


@pytest.fixture
def sample_docs():
    return [
        {"id": "doc_001", "text": "Order ord1 delivered in SP.", "metadata": {"document_type": "order"}},
        {"id": "doc_002", "text": "Product p1 electronics category.", "metadata": {"document_type": "product"}},
        {"id": "doc_003", "text": "Seller s1 based in Campinas.", "metadata": {"document_type": "seller"}},
    ]


@pytest.fixture
def retrieved_docs():
    return [
        {"id": "doc_001", "text": "Order ord1 delivered in SP.", "metadata": {"document_type": "order"}, "distance": 0.12},
        {"id": "doc_002", "text": "Product p1 electronics.", "metadata": {"document_type": "product"}, "distance": 0.22},
        {"id": "doc_003", "text": "Seller s1 Campinas.", "metadata": {"document_type": "seller"}, "distance": 0.38},
    ]


@pytest.fixture
def hypothetical_doc():
    return "In Q4 2017, the average delivery time for electronics orders in São Paulo was 8.3 days."

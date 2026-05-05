"""Unit tests for preprocessing/step4_build_knowledge_base.py.

Covers the six KB document layers (order, category, seller, customer-state,
month, delivery-status), utility helpers (_s, _f, _ts, _pct), document
structure validation, and the JSON persistence helper (_save).
All tests use the ``enriched_df`` fixture from conftest — no CSVs needed.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import preprocessing.step4_build_knowledge_base as kb_mod
from preprocessing.step4_build_knowledge_base import (
    _s,
    _f,
    _ts,
    _pct,
    build_order_documents,
    build_category_documents,
    build_seller_documents,
    build_customer_state_documents,
    build_month_documents,
    build_delivery_status_documents,
)

import pandas as pd


# ── Utility helper tests ──────────────────────────────────────────────────────

def test_s_returns_default_for_none():
    """_s(None) must return the default 'N/A' string."""
    assert _s(None) == "N/A"


def test_s_returns_default_for_nan():
    """_s(NaN) must return the default 'N/A' string."""
    import numpy as np
    assert _s(float("nan")) == "N/A"


def test_s_returns_string():
    """_s with a valid value must return its string representation."""
    assert _s("hello") == "hello"


def test_f_returns_default_for_none():
    """_f(None) must return the default 0.0."""
    assert _f(None) == 0.0


def test_f_returns_float():
    """_f must coerce a numeric string to float."""
    assert _f("3.14") == pytest.approx(3.14)


def test_ts_returns_na_for_nat():
    """_ts(pd.NaT) must return 'N/A'."""
    assert _ts(pd.NaT) == "N/A"


def test_ts_formats_timestamp():
    """_ts must return 'YYYY-MM-DD HH:MM:SS' and strip sub-second precision."""
    ts = pd.Timestamp("2017-10-02 10:30:00.123456")
    assert _ts(ts) == "2017-10-02 10:30:00"


def test_pct_returns_na_for_zero_denominator():
    """_pct must guard against division-by-zero and return 'N/A'."""
    assert _pct(5, 0) == "N/A"


def test_pct_computes_percentage():
    """_pct(1, 4) must produce '25.00%'."""
    assert _pct(1, 4) == "25.00%"


# ── Document structure assertion helper ───────────────────────────────────────

def _assert_doc_structure(doc: dict, expected_id_prefix: str) -> None:
    """Assert the three required keys exist and the id has the expected prefix."""
    assert "id" in doc
    assert "text" in doc
    assert "metadata" in doc
    assert doc["id"].startswith(expected_id_prefix)
    assert isinstance(doc["text"], str)
    assert len(doc["text"]) > 0
    assert isinstance(doc["metadata"], dict)


# ── Layer 1: Order documents ──────────────────────────────────────────────────

def test_order_documents_structure(enriched_df):
    """Each order document must have id, text, metadata with correct id prefix."""
    docs = build_order_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "order_")


def test_order_documents_count(enriched_df):
    """5 unique orders, all below ORDER_KB_SAMPLE → exactly 5 order documents."""
    docs = build_order_documents(enriched_df)
    assert len(docs) == 5


def test_order_documents_text_header(enriched_df):
    """Each order document text must begin with the expected document-type header."""
    docs = build_order_documents(enriched_df)
    assert all("Document Type: Order-Level Summary" in d["text"] for d in docs)


def test_order_documents_metadata_keys(enriched_df):
    """Order document metadata must contain all required analytical keys."""
    docs = build_order_documents(enriched_df)
    required = {
        "document_type", "source_id", "order_id",
        "order_status", "customer_state", "delivery_status",
        "delivery_bucket", "purchase_month", "review_bucket",
    }
    for doc in docs:
        assert required.issubset(doc["metadata"].keys())


def test_order_document_type_value(enriched_df):
    """metadata['document_type'] must be 'order_level' for all order docs."""
    docs = build_order_documents(enriched_df)
    assert all(d["metadata"]["document_type"] == "order_level" for d in docs)


# ── Layer 2: Category documents ───────────────────────────────────────────────

def test_category_documents_structure(enriched_df):
    """Category documents must have id/text/metadata with 'category_' id prefix."""
    docs = build_category_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "category_")


def test_category_documents_count(enriched_df):
    """minimal_master has 5 unique product_category_final values → 5 docs."""
    docs = build_category_documents(enriched_df)
    assert len(docs) == 5


def test_category_documents_text_header(enriched_df):
    """Category document text must include the product-category header."""
    docs = build_category_documents(enriched_df)
    assert all("Document Type: Product Category Summary" in d["text"] for d in docs)


def test_category_documents_metadata_keys(enriched_df):
    """Category document metadata must contain the required aggregate keys."""
    docs = build_category_documents(enriched_df)
    required = {"document_type", "source_id", "total_orders", "avg_review", "total_revenue"}
    for doc in docs:
        assert required.issubset(doc["metadata"].keys())


# ── Layer 3: Seller documents ─────────────────────────────────────────────────

def test_seller_documents_structure(enriched_df):
    """Seller documents must have id/text/metadata with 'seller_' id prefix."""
    docs = build_seller_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "seller_")


def test_seller_documents_count(enriched_df):
    """minimal_master has sellers s1..s5 → exactly 5 seller documents."""
    docs = build_seller_documents(enriched_df)
    assert len(docs) == 5


def test_seller_documents_metadata_keys(enriched_df):
    """Seller document metadata must include city, state, and order-count fields."""
    docs = build_seller_documents(enriched_df)
    required = {"document_type", "source_id", "seller_city", "seller_state", "total_orders"}
    for doc in docs:
        assert required.issubset(doc["metadata"].keys())


# ── Layer 4: Customer-state documents ────────────────────────────────────────

def test_state_documents_structure(enriched_df):
    """State documents must have id/text/metadata with 'state_' id prefix."""
    docs = build_customer_state_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "state_")


def test_state_documents_count(enriched_df):
    """minimal_master has states SP, RJ, MG → exactly 3 state documents."""
    docs = build_customer_state_documents(enriched_df)
    assert len(docs) == 3


def test_state_documents_text_header(enriched_df):
    """State document text must include the customer-state header."""
    docs = build_customer_state_documents(enriched_df)
    assert all("Document Type: Customer-State Summary" in d["text"] for d in docs)


# ── Layer 5: Month documents ──────────────────────────────────────────────────

def test_month_documents_structure(enriched_df):
    """Month documents must have id/text/metadata with 'month_' id prefix."""
    docs = build_month_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "month_")


def test_month_documents_count(enriched_df):
    """All 5 orders are in 2017-10 → exactly 1 month document."""
    docs = build_month_documents(enriched_df)
    assert len(docs) == 1


def test_month_document_id_format(enriched_df):
    """Month document id must follow the 'month_YYYY_MM' format."""
    docs = build_month_documents(enriched_df)
    assert docs[0]["id"] == "month_2017_10"


def test_month_document_metadata_keys(enriched_df):
    """Month document metadata must include year, month, and order-count fields."""
    docs = build_month_documents(enriched_df)
    required = {"document_type", "source_id", "year", "month", "total_orders"}
    for doc in docs:
        assert required.issubset(doc["metadata"].keys())


# ── Layer 6: Delivery-status documents ───────────────────────────────────────

def test_delivery_status_documents_structure(enriched_df):
    """Delivery-status docs must have id/text/metadata with 'delivery_status_' prefix."""
    docs = build_delivery_status_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "delivery_status_")


def test_delivery_status_documents_count(enriched_df):
    """minimal_master has early/late/on_time/not_delivered → exactly 4 docs."""
    docs = build_delivery_status_documents(enriched_df)
    assert len(docs) == 4


def test_delivery_status_all_statuses_present(enriched_df):
    """All four delivery statuses in the fixture must each have a document."""
    docs = build_delivery_status_documents(enriched_df)
    statuses = {d["metadata"]["delivery_status"] for d in docs}
    assert statuses == {"early", "late", "on_time", "not_delivered"}


def test_delivery_status_text_header(enriched_df):
    """Delivery-status document text must include the insight header."""
    docs = build_delivery_status_documents(enriched_df)
    assert all("Document Type: Delivery-Status Insight" in d["text"] for d in docs)


# ── _save persists valid JSON ─────────────────────────────────────────────────

def test_save_writes_valid_json(tmp_path, monkeypatch):
    """_save must write a file that deserialises back to the original list."""
    monkeypatch.setattr(kb_mod, "DATA_KB", tmp_path)
    docs = [{"id": "test_1", "text": "hello", "metadata": {"k": "v"}}]
    kb_mod._save(docs, "test_output.json")

    out = tmp_path / "test_output.json"
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == docs

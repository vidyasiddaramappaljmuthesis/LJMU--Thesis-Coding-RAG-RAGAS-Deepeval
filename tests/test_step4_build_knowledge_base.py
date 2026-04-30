"""Unit tests for preprocessing/step4_build_knowledge_base.py."""
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


# ── Helper utilities ──────────────────────────────────────────────────────────

def test_s_returns_default_for_none():
    assert _s(None) == "N/A"


def test_s_returns_default_for_nan():
    import numpy as np
    assert _s(float("nan")) == "N/A"


def test_s_returns_string():
    assert _s("hello") == "hello"


def test_f_returns_default_for_none():
    assert _f(None) == 0.0


def test_f_returns_float():
    assert _f("3.14") == pytest.approx(3.14)


def test_ts_returns_na_for_nat():
    assert _ts(pd.NaT) == "N/A"


def test_ts_formats_timestamp():
    ts = pd.Timestamp("2017-10-02 10:30:00.123456")
    assert _ts(ts) == "2017-10-02 10:30:00"


def test_pct_returns_na_for_zero_denominator():
    assert _pct(5, 0) == "N/A"


def test_pct_computes_percentage():
    assert _pct(1, 4) == "25.00%"


# ── Document structure helpers ────────────────────────────────────────────────

def _assert_doc_structure(doc: dict, expected_id_prefix: str) -> None:
    assert "id" in doc
    assert "text" in doc
    assert "metadata" in doc
    assert doc["id"].startswith(expected_id_prefix)
    assert isinstance(doc["text"], str)
    assert len(doc["text"]) > 0
    assert isinstance(doc["metadata"], dict)


# ── Layer 1: Order documents ──────────────────────────────────────────────────

def test_order_documents_structure(enriched_df):
    docs = build_order_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "order_")


def test_order_documents_count(enriched_df):
    docs = build_order_documents(enriched_df)
    # 5 unique orders, all below ORDER_KB_SAMPLE → 5 docs
    assert len(docs) == 5


def test_order_documents_text_header(enriched_df):
    docs = build_order_documents(enriched_df)
    assert all("Document Type: Order-Level Summary" in d["text"] for d in docs)


def test_order_documents_metadata_keys(enriched_df):
    docs = build_order_documents(enriched_df)
    required = {
        "document_type", "source_id", "order_id",
        "order_status", "customer_state", "delivery_status",
        "delivery_bucket", "purchase_month", "review_bucket",
    }
    for doc in docs:
        assert required.issubset(doc["metadata"].keys())


def test_order_document_type_value(enriched_df):
    docs = build_order_documents(enriched_df)
    assert all(d["metadata"]["document_type"] == "order_level" for d in docs)


# ── Layer 2: Category documents ───────────────────────────────────────────────

def test_category_documents_structure(enriched_df):
    docs = build_category_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "category_")


def test_category_documents_count(enriched_df):
    docs = build_category_documents(enriched_df)
    # minimal_master has 5 unique product_category_final values
    assert len(docs) == 5


def test_category_documents_text_header(enriched_df):
    docs = build_category_documents(enriched_df)
    assert all("Document Type: Product Category Summary" in d["text"] for d in docs)


def test_category_documents_metadata_keys(enriched_df):
    docs = build_category_documents(enriched_df)
    required = {"document_type", "source_id", "total_orders", "avg_review", "total_revenue"}
    for doc in docs:
        assert required.issubset(doc["metadata"].keys())


# ── Layer 3: Seller documents ─────────────────────────────────────────────────

def test_seller_documents_structure(enriched_df):
    docs = build_seller_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "seller_")


def test_seller_documents_count(enriched_df):
    docs = build_seller_documents(enriched_df)
    # minimal_master has sellers s1..s5 → 5 seller docs
    assert len(docs) == 5


def test_seller_documents_metadata_keys(enriched_df):
    docs = build_seller_documents(enriched_df)
    required = {"document_type", "source_id", "seller_city", "seller_state", "total_orders"}
    for doc in docs:
        assert required.issubset(doc["metadata"].keys())


# ── Layer 4: Customer-state documents ────────────────────────────────────────

def test_state_documents_structure(enriched_df):
    docs = build_customer_state_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "state_")


def test_state_documents_count(enriched_df):
    docs = build_customer_state_documents(enriched_df)
    # minimal_master: SP, RJ, MG → 3 unique states
    assert len(docs) == 3


def test_state_documents_text_header(enriched_df):
    docs = build_customer_state_documents(enriched_df)
    assert all("Document Type: Customer-State Summary" in d["text"] for d in docs)


# ── Layer 5: Month documents ──────────────────────────────────────────────────

def test_month_documents_structure(enriched_df):
    docs = build_month_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "month_")


def test_month_documents_count(enriched_df):
    docs = build_month_documents(enriched_df)
    # all 5 orders have BASE = 2017-10-02 → single month 2017-10 → 1 doc
    assert len(docs) == 1


def test_month_document_id_format(enriched_df):
    docs = build_month_documents(enriched_df)
    assert docs[0]["id"] == "month_2017_10"


def test_month_document_metadata_keys(enriched_df):
    docs = build_month_documents(enriched_df)
    required = {"document_type", "source_id", "year", "month", "total_orders"}
    for doc in docs:
        assert required.issubset(doc["metadata"].keys())


# ── Layer 6: Delivery-status documents ───────────────────────────────────────

def test_delivery_status_documents_structure(enriched_df):
    docs = build_delivery_status_documents(enriched_df)
    assert len(docs) > 0
    _assert_doc_structure(docs[0], "delivery_status_")


def test_delivery_status_documents_count(enriched_df):
    docs = build_delivery_status_documents(enriched_df)
    # minimal_master has: early, late, on_time, not_delivered → 4 docs
    assert len(docs) == 4


def test_delivery_status_all_statuses_present(enriched_df):
    docs = build_delivery_status_documents(enriched_df)
    statuses = {d["metadata"]["delivery_status"] for d in docs}
    assert statuses == {"early", "late", "on_time", "not_delivered"}


def test_delivery_status_text_header(enriched_df):
    docs = build_delivery_status_documents(enriched_df)
    assert all("Document Type: Delivery-Status Insight" in d["text"] for d in docs)


# ── _save writes valid JSON ───────────────────────────────────────────────────

def test_save_writes_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(kb_mod, "DATA_KB", tmp_path)
    docs = [{"id": "test_1", "text": "hello", "metadata": {"k": "v"}}]
    kb_mod._save(docs, "test_output.json")

    out = tmp_path / "test_output.json"
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == docs

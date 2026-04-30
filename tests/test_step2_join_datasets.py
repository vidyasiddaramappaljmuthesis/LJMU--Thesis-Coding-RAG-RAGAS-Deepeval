"""Unit tests for preprocessing/step2_join_datasets.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from preprocessing.step2_join_datasets import (
    aggregate_payments,
    aggregate_reviews,
    create_master_dataset,
)


# ── aggregate_payments ────────────────────────────────────────────────────────

def test_aggregate_payments_one_row_per_order(payments_raw):
    result = aggregate_payments(payments_raw)
    assert len(result) == 2
    assert result["order_id"].nunique() == 2


def test_aggregate_payments_sums_values(payments_raw):
    result = aggregate_payments(payments_raw)
    ord1 = result.loc[result["order_id"] == "ord1", "total_payment_value"].iloc[0]
    assert ord1 == pytest.approx(100.0)  # 80 + 20


def test_aggregate_payments_pipe_joins_sorted_types(payments_raw):
    result = aggregate_payments(payments_raw)
    ord1_types = result.loc[result["order_id"] == "ord1", "payment_types"].iloc[0]
    # sorted unique: credit_card < voucher alphabetically
    assert ord1_types == "credit_card|voucher"


def test_aggregate_payments_single_type(payments_raw):
    result = aggregate_payments(payments_raw)
    ord2_types = result.loc[result["order_id"] == "ord2", "payment_types"].iloc[0]
    assert ord2_types == "boleto"


def test_aggregate_payments_max_installments(payments_raw):
    result = aggregate_payments(payments_raw)
    ord1_inst = result.loc[result["order_id"] == "ord1", "max_installments"].iloc[0]
    assert ord1_inst == 3


def test_aggregate_payments_methods_count(payments_raw):
    result = aggregate_payments(payments_raw)
    ord1_count = result.loc[result["order_id"] == "ord1", "payment_methods_count"].iloc[0]
    ord2_count = result.loc[result["order_id"] == "ord2", "payment_methods_count"].iloc[0]
    assert ord1_count == 2
    assert ord2_count == 1


# ── aggregate_reviews ─────────────────────────────────────────────────────────

def test_aggregate_reviews_one_row_per_order(reviews_raw):
    result = aggregate_reviews(reviews_raw)
    assert len(result) == 2
    assert result["order_id"].nunique() == 2


def test_aggregate_reviews_keeps_latest(reviews_raw):
    result = aggregate_reviews(reviews_raw)
    ord1 = result.loc[result["order_id"] == "ord1"]
    # r2 (score=5, answered 2017-11-11) is later than r1 (score=3, answered 2017-11-02)
    assert ord1["review_score"].iloc[0] == 5
    assert ord1["review_id"].iloc[0] == "r2"


def test_aggregate_reviews_single_review_unchanged(reviews_raw):
    result = aggregate_reviews(reviews_raw)
    ord2 = result.loc[result["order_id"] == "ord2"]
    assert ord2["review_score"].iloc[0] == 4
    assert ord2["review_id"].iloc[0] == "r3"


# ── create_master_dataset ─────────────────────────────────────────────────────

def test_create_master_row_count(minimal_datasets):
    master = create_master_dataset(minimal_datasets)
    # ord1 has 2 items → 2 rows; ord2 has 1 item → 1 row
    assert len(master) == 3


def test_create_master_has_order_columns(minimal_datasets):
    master = create_master_dataset(minimal_datasets)
    for col in ["order_id", "order_status", "order_purchase_timestamp"]:
        assert col in master.columns


def test_create_master_has_payment_columns(minimal_datasets):
    master = create_master_dataset(minimal_datasets)
    for col in ["total_payment_value", "payment_types", "max_installments"]:
        assert col in master.columns


def test_create_master_has_review_columns(minimal_datasets):
    master = create_master_dataset(minimal_datasets)
    for col in ["review_score", "review_id"]:
        assert col in master.columns


def test_create_master_has_customer_columns(minimal_datasets):
    master = create_master_dataset(minimal_datasets)
    for col in ["customer_state", "customer_city", "customer_unique_id"]:
        assert col in master.columns


def test_create_master_has_product_columns(minimal_datasets):
    master = create_master_dataset(minimal_datasets)
    for col in ["product_id", "product_category_name", "product_category_name_english"]:
        assert col in master.columns

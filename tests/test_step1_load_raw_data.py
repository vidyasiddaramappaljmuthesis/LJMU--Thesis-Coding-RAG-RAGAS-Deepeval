"""Unit tests for preprocessing/step1_load_raw_data.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

import preprocessing.step1_load_raw_data as step1_mod


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_orders_csv(path: Path) -> None:
    df = pd.DataFrame({
        "order_id":                      ["o1"],
        "customer_id":                   ["c1"],
        "order_status":                  ["delivered"],
        "order_purchase_timestamp":      ["2017-10-02 10:00:00"],
        "order_approved_at":             ["2017-10-02 11:00:00"],
        "order_delivered_carrier_date":  ["2017-10-04 12:00:00"],
        "order_delivered_customer_date": ["2017-10-10 12:00:00"],
        "order_estimated_delivery_date": ["2017-10-18 00:00:00"],
    })
    df.to_csv(path, index=False, encoding="utf-8")


def _write_reviews_csv(path: Path) -> None:
    df = pd.DataFrame({
        "review_id":    ["r1"],
        "order_id":     ["o1"],
        "review_score": [5],
        "review_comment_title":   [None],
        "review_comment_message": [None],
        "review_creation_date":   ["2017-11-01"],
        "review_answer_timestamp":["2017-11-02"],
    })
    df.to_csv(path, index=False, encoding="utf-8")


def _write_category_csv_with_bom(path: Path) -> None:
    df = pd.DataFrame({
        "product_category_name":         ["eletronicos"],
        "product_category_name_english": ["electronics"],
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_simple_csv(path: Path) -> None:
    pd.DataFrame({"col": [1, 2]}).to_csv(path, index=False, encoding="utf-8")


# ── tests ─────────────────────────────────────────────────────────────────────

def test_load_dataset_returns_dataframe(tmp_path, monkeypatch):
    csv = tmp_path / "orders.csv"
    _write_orders_csv(csv)
    monkeypatch.setitem(step1_mod.RAW_FILES, "orders", csv)

    df = step1_mod.load_dataset("orders")
    assert isinstance(df, pd.DataFrame)


def test_load_dataset_parses_order_datetimes(tmp_path, monkeypatch):
    csv = tmp_path / "orders.csv"
    _write_orders_csv(csv)
    monkeypatch.setitem(step1_mod.RAW_FILES, "orders", csv)

    df = step1_mod.load_dataset("orders")
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        assert pd.api.types.is_datetime64_any_dtype(df[col]), (
            f"Column {col} should be datetime"
        )


def test_load_dataset_parses_review_datetimes(tmp_path, monkeypatch):
    csv = tmp_path / "reviews.csv"
    _write_reviews_csv(csv)
    monkeypatch.setitem(step1_mod.RAW_FILES, "reviews", csv)

    df = step1_mod.load_dataset("reviews")
    assert pd.api.types.is_datetime64_any_dtype(df["review_creation_date"])
    assert pd.api.types.is_datetime64_any_dtype(df["review_answer_timestamp"])


def test_load_dataset_strips_bom(tmp_path, monkeypatch):
    csv = tmp_path / "cat.csv"
    _write_category_csv_with_bom(csv)
    monkeypatch.setitem(step1_mod.RAW_FILES, "category_translation", csv)

    df = step1_mod.load_dataset("category_translation")
    # If BOM is not stripped, the first column name would start with '﻿'
    assert "product_category_name" in df.columns


def test_load_dataset_raises_on_missing_file(tmp_path, monkeypatch):
    missing = tmp_path / "does_not_exist.csv"
    monkeypatch.setitem(step1_mod.RAW_FILES, "sellers", missing)

    with pytest.raises(FileNotFoundError):
        step1_mod.load_dataset("sellers")


def test_load_all_datasets_returns_all_keys(tmp_path, monkeypatch):
    keys = list(step1_mod.RAW_FILES.keys())
    for key in keys:
        p = tmp_path / f"{key}.csv"
        _write_simple_csv(p)
        monkeypatch.setitem(step1_mod.RAW_FILES, key, p)

    datasets = step1_mod.load_all_datasets()
    assert set(datasets.keys()) == set(keys)


def test_load_all_datasets_values_are_dataframes(tmp_path, monkeypatch):
    keys = list(step1_mod.RAW_FILES.keys())
    for key in keys:
        p = tmp_path / f"{key}.csv"
        _write_simple_csv(p)
        monkeypatch.setitem(step1_mod.RAW_FILES, key, p)

    datasets = step1_mod.load_all_datasets()
    for key, df in datasets.items():
        assert isinstance(df, pd.DataFrame), f"{key} should be a DataFrame"

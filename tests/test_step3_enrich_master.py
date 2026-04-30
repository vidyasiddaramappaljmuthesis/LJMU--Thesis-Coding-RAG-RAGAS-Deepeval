"""Unit tests for preprocessing/step3_enrich_master.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from preprocessing.step3_enrich_master import enrich_master_dataset

BASE = pd.Timestamp("2017-10-02 10:00:00")


# ── Fixtures ──────────────────────────────────────────────────────────────────
# All tests use the conftest `enriched_df` fixture (enrich_master_dataset run on minimal_master)


def _row(enriched_df, order_id: str) -> pd.Series:
    return enriched_df.loc[enriched_df["order_id"] == order_id].iloc[0]


# ── Time features ─────────────────────────────────────────────────────────────

def test_purchase_month(enriched_df):
    assert _row(enriched_df, "ord1")["purchase_month"] == 10


def test_purchase_month_name(enriched_df):
    assert _row(enriched_df, "ord1")["purchase_month_name"] == "October"


def test_purchase_year(enriched_df):
    assert _row(enriched_df, "ord1")["purchase_year"] == 2017


def test_purchase_day_name(enriched_df):
    # BASE = 2017-10-02, which is a Monday
    assert _row(enriched_df, "ord1")["purchase_day_name"] == "Monday"


def test_purchase_hour(enriched_df):
    assert _row(enriched_df, "ord1")["purchase_hour"] == 10


# ── Delivery status ───────────────────────────────────────────────────────────

def test_delivery_status_early(enriched_df):
    # ord1: delivered 8 days, estimated 15 days → diff = -7 → early
    assert _row(enriched_df, "ord1")["delivery_status"] == "early"


def test_delivery_status_late(enriched_df):
    # ord2: diff = +5 → late
    assert _row(enriched_df, "ord2")["delivery_status"] == "late"


def test_delivery_status_on_time(enriched_df):
    # ord3: diff = 0 → on_time
    assert _row(enriched_df, "ord3")["delivery_status"] == "on_time"


def test_delivery_status_not_delivered(enriched_df):
    # ord4: NaT delivery date → not_delivered
    assert _row(enriched_df, "ord4")["delivery_status"] == "not_delivered"


# ── Delivery bucket ───────────────────────────────────────────────────────────

def test_bucket_slightly_early(enriched_df):
    # ord1: diff = -7 → slightly_early (>= -7 and < 0)
    assert _row(enriched_df, "ord1")["delivery_bucket"] == "slightly_early"


def test_bucket_slightly_late(enriched_df):
    # ord2: diff = +5 → slightly_late (> 0 and <= 7)
    assert _row(enriched_df, "ord2")["delivery_bucket"] == "slightly_late"


def test_bucket_on_time(enriched_df):
    # ord3: diff = 0 → on_time
    assert _row(enriched_df, "ord3")["delivery_bucket"] == "on_time"


def test_bucket_not_delivered(enriched_df):
    # ord4: NaT → not_delivered
    assert _row(enriched_df, "ord4")["delivery_bucket"] == "not_delivered"


def test_bucket_very_early(enriched_df):
    # ord5: diff = -15 → very_early (< -14)
    assert _row(enriched_df, "ord5")["delivery_bucket"] == "very_early"


# ── Product features ──────────────────────────────────────────────────────────

def test_product_category_final_uses_english(enriched_df):
    # ord1 has product_category_name_english="electronics"
    assert _row(enriched_df, "ord1")["product_category_final"] == "electronics"


def test_product_category_final_falls_back_to_portuguese(enriched_df):
    # ord2 has product_category_name_english=None, product_category_name="moveis"
    assert _row(enriched_df, "ord2")["product_category_final"] == "moveis"


def test_item_total_value(enriched_df):
    # ord1: price=100.0, freight_value=10.0 → 110.0
    assert _row(enriched_df, "ord1")["item_total_value"] == pytest.approx(110.0)


# ── Review features ───────────────────────────────────────────────────────────

def test_review_bucket_positive(enriched_df):
    # ord1: review_score=5 → positive
    assert _row(enriched_df, "ord1")["review_bucket"] == "positive"


def test_review_bucket_negative(enriched_df):
    # ord2: review_score=2 → negative
    assert _row(enriched_df, "ord2")["review_bucket"] == "negative"


def test_review_bucket_neutral(enriched_df):
    # ord3: review_score=3 → neutral
    assert _row(enriched_df, "ord3")["review_bucket"] == "neutral"


def test_review_bucket_no_review(enriched_df):
    # ord4: review_score=None → no_review
    assert _row(enriched_df, "ord4")["review_bucket"] == "no_review"


# ── Column presence ───────────────────────────────────────────────────────────

def test_all_enriched_columns_present(enriched_df):
    expected = [
        "purchase_month", "purchase_month_name", "purchase_year",
        "purchase_day_name", "purchase_hour",
        "approval_hours", "carrier_handover_days",
        "delivery_days", "estimated_delivery_days", "delivery_difference_days",
        "delivery_status", "delivery_bucket",
        "product_category_final", "item_total_value", "review_bucket",
    ]
    for col in expected:
        assert col in enriched_df.columns, f"Missing column: {col}"


def test_enrich_does_not_drop_original_columns(enriched_df):
    originals = ["order_id", "order_purchase_timestamp", "price", "review_score"]
    for col in originals:
        assert col in enriched_df.columns

"""Unit tests for preprocessing/step3_enrich_master.py.

Validates time-decomposition, delivery status/bucket logic, product-category
resolution, item value calculation, and review sentiment bucketing.
Uses the ``enriched_df`` fixture from conftest (runs the real enricher on
the synthetic ``minimal_master`` DataFrame — no CSV files required).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from preprocessing.step3_enrich_master import enrich_master_dataset

BASE = pd.Timestamp("2017-10-02 10:00:00")


# ── Helper ────────────────────────────────────────────────────────────────────

def _row(enriched_df, order_id: str) -> pd.Series:
    """Return the single enriched row for *order_id*."""
    return enriched_df.loc[enriched_df["order_id"] == order_id].iloc[0]


# ── Time features ─────────────────────────────────────────────────────────────

def test_purchase_month(enriched_df):
    """purchase_month must be the integer month of order_purchase_timestamp."""
    assert _row(enriched_df, "ord1")["purchase_month"] == 10


def test_purchase_month_name(enriched_df):
    """purchase_month_name must be the full English month name."""
    assert _row(enriched_df, "ord1")["purchase_month_name"] == "October"


def test_purchase_year(enriched_df):
    """purchase_year must be the 4-digit calendar year."""
    assert _row(enriched_df, "ord1")["purchase_year"] == 2017


def test_purchase_day_name(enriched_df):
    """purchase_day_name must be the full English weekday name (BASE = Monday)."""
    assert _row(enriched_df, "ord1")["purchase_day_name"] == "Monday"


def test_purchase_hour(enriched_df):
    """purchase_hour must be the 24-hour integer hour of the purchase timestamp."""
    assert _row(enriched_df, "ord1")["purchase_hour"] == 10


# ── Delivery status ───────────────────────────────────────────────────────────

def test_delivery_status_early(enriched_df):
    """ord1: actual 8 days, estimated 15 → diff=-7 → delivery_status='early'."""
    assert _row(enriched_df, "ord1")["delivery_status"] == "early"


def test_delivery_status_late(enriched_df):
    """ord2: diff=+5 → delivery_status='late'."""
    assert _row(enriched_df, "ord2")["delivery_status"] == "late"


def test_delivery_status_on_time(enriched_df):
    """ord3: diff=0 → delivery_status='on_time'."""
    assert _row(enriched_df, "ord3")["delivery_status"] == "on_time"


def test_delivery_status_not_delivered(enriched_df):
    """ord4: NaT delivery date → delivery_status='not_delivered'."""
    assert _row(enriched_df, "ord4")["delivery_status"] == "not_delivered"


# ── Delivery bucket ───────────────────────────────────────────────────────────

def test_bucket_slightly_early(enriched_df):
    """ord1: diff=-7 falls in [-7, 0) → delivery_bucket='slightly_early'."""
    assert _row(enriched_df, "ord1")["delivery_bucket"] == "slightly_early"


def test_bucket_slightly_late(enriched_df):
    """ord2: diff=+5 falls in (0, 7] → delivery_bucket='slightly_late'."""
    assert _row(enriched_df, "ord2")["delivery_bucket"] == "slightly_late"


def test_bucket_on_time(enriched_df):
    """ord3: diff=0 → delivery_bucket='on_time'."""
    assert _row(enriched_df, "ord3")["delivery_bucket"] == "on_time"


def test_bucket_not_delivered(enriched_df):
    """ord4: NaT delivery date → delivery_bucket='not_delivered'."""
    assert _row(enriched_df, "ord4")["delivery_bucket"] == "not_delivered"


def test_bucket_very_early(enriched_df):
    """ord5: diff=-15 is < -14 → delivery_bucket='very_early'."""
    assert _row(enriched_df, "ord5")["delivery_bucket"] == "very_early"


# ── Product features ──────────────────────────────────────────────────────────

def test_product_category_final_uses_english(enriched_df):
    """When the English category name is available, it should be used."""
    assert _row(enriched_df, "ord1")["product_category_final"] == "electronics"


def test_product_category_final_falls_back_to_portuguese(enriched_df):
    """When English translation is missing, fall back to the Portuguese name."""
    assert _row(enriched_df, "ord2")["product_category_final"] == "moveis"


def test_item_total_value(enriched_df):
    """item_total_value must equal price + freight_value (ord1: 100 + 10 = 110)."""
    assert _row(enriched_df, "ord1")["item_total_value"] == pytest.approx(110.0)


# ── Review features ───────────────────────────────────────────────────────────

def test_review_bucket_positive(enriched_df):
    """review_score >= 4 → review_bucket='positive'."""
    assert _row(enriched_df, "ord1")["review_bucket"] == "positive"


def test_review_bucket_negative(enriched_df):
    """review_score <= 2 → review_bucket='negative'."""
    assert _row(enriched_df, "ord2")["review_bucket"] == "negative"


def test_review_bucket_neutral(enriched_df):
    """review_score == 3 → review_bucket='neutral'."""
    assert _row(enriched_df, "ord3")["review_bucket"] == "neutral"


def test_review_bucket_no_review(enriched_df):
    """NaN review_score → review_bucket='no_review'."""
    assert _row(enriched_df, "ord4")["review_bucket"] == "no_review"


# ── Column presence ───────────────────────────────────────────────────────────

def test_all_enriched_columns_present(enriched_df):
    """All 15 enrichment columns must exist after running the enricher."""
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
    """The enricher must not drop any pre-existing master columns."""
    originals = ["order_id", "order_purchase_timestamp", "price", "review_score"]
    for col in originals:
        assert col in enriched_df.columns

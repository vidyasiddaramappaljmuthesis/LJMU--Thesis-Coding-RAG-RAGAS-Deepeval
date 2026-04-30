"""
Step 3 – Enrich the master dataset with derived features.

All new columns are appended; originals are left untouched.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Feature groups ───────────────────────────────────────────────────────────

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose order_purchase_timestamp into calendar parts."""
    ts = df["order_purchase_timestamp"]
    df["purchase_month"]      = ts.dt.month
    df["purchase_month_name"] = ts.dt.strftime("%B")
    df["purchase_year"]       = ts.dt.year
    df["purchase_day_name"]   = ts.dt.strftime("%A")
    df["purchase_hour"]       = ts.dt.hour
    return df


def _add_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all timing and delivery-status columns.

    delivery_difference_days = actual_delivery - estimated_delivery
        positive → late, negative → early, 0 → on_time
    """
    purchase   = df["order_purchase_timestamp"]
    approved   = df["order_approved_at"]
    to_carrier = df["order_delivered_carrier_date"]
    delivered  = df["order_delivered_customer_date"]
    estimated  = df["order_estimated_delivery_date"]

    # Time-to-* in hours / days
    df["approval_hours"]         = ((approved   - purchase  ).dt.total_seconds() / 3_600 ).round(2)
    df["carrier_handover_days"]  = ((to_carrier - approved  ).dt.total_seconds() / 86_400).round(2)
    df["delivery_days"]          = ((delivered  - purchase  ).dt.total_seconds() / 86_400).round(2)
    df["estimated_delivery_days"]= ((estimated  - purchase  ).dt.total_seconds() / 86_400).round(2)

    # Integer-day difference (positive = late, negative = early)
    df["delivery_difference_days"] = (delivered - estimated).dt.days

    # ── delivery_status ──────────────────────────────────────────────────────
    d   = df["delivery_difference_days"]
    has = delivered.notna()
    conditions = [~has, has & (d > 0), has & (d < 0), has & (d == 0)]
    choices    = ["not_delivered", "late", "early", "on_time"]
    df["delivery_status"] = np.select(conditions, choices, default="unknown")

    # ── delivery_bucket ──────────────────────────────────────────────────────
    bucket_conditions = [
        ~has,
        has & (d < -14),
        has & (d >= -14) & (d < -7),
        has & (d >= -7)  & (d < 0),
        has & (d == 0),
        has & (d > 0)    & (d <= 7),
        has & (d > 7)    & (d <= 14),
        has & (d > 14),
    ]
    bucket_labels = [
        "not_delivered",
        "very_early", "early", "slightly_early",
        "on_time",
        "slightly_late", "late", "very_late",
    ]
    df["delivery_bucket"] = np.select(bucket_conditions, bucket_labels, default="unknown")

    return df


def _add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve English category name and compute item-level total value."""
    df["product_category_final"] = (
        df["product_category_name_english"]
        .fillna(df["product_category_name"])
        .fillna("unknown")
    )
    df["item_total_value"] = (df["price"] + df["freight_value"]).round(2)
    return df


def _add_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map numeric review scores to sentiment buckets."""
    def _bucket(score):
        if pd.isna(score):
            return "no_review"
        s = float(score)
        if s >= 4:
            return "positive"
        if s == 3:
            return "neutral"
        return "negative"

    df["review_bucket"] = df["review_score"].apply(_bucket)
    return df


# ── Public entry point ───────────────────────────────────────────────────────

def enrich_master_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all enrichment steps and return the augmented DataFrame."""
    logger.info("Enriching master dataset...")

    df = _add_time_features(df);     logger.info("  [OK] Time features")
    df = _add_delivery_features(df); logger.info("  [OK] Delivery features")
    df = _add_product_features(df);  logger.info("  [OK] Product features")
    df = _add_review_features(df);   logger.info("  [OK] Review features")

    new_cols = [
        "purchase_month", "purchase_month_name", "purchase_year",
        "purchase_day_name", "purchase_hour",
        "approval_hours", "carrier_handover_days",
        "delivery_days", "estimated_delivery_days", "delivery_difference_days",
        "delivery_status", "delivery_bucket",
        "product_category_final", "item_total_value", "review_bucket",
    ]
    logger.info(
        f"  Enriched: {len(df):,} rows x {df.shape[1]} columns"
        f"  (+{len(new_cols)} new)"
    )
    return df

"""
Step 2 – Aggregate payments & reviews, then join all tables into a single
         master DataFrame (order-item grain: one row per order × item).
"""
import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


# ── Aggregation helpers ──────────────────────────────────────────────────────

def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple payment rows per order into one row.

    Columns produced
    ----------------
    total_payment_value   – sum of all payment_value for the order
    payment_types         – '|'-joined sorted unique payment methods
    max_installments      – maximum installments across payment methods
    payment_methods_count – number of distinct payment methods
    """
    logger.info("  Aggregating payments -> 1 row per order...")
    agg = (
        payments
        .groupby("order_id")
        .agg(
            total_payment_value   = ("payment_value",      "sum"),
            payment_types         = ("payment_type",       lambda x: "|".join(sorted(set(x)))),
            max_installments      = ("payment_installments","max"),
            payment_methods_count = ("payment_type",       "nunique"),
        )
        .reset_index()
    )
    logger.info(f"  -> {len(agg):,} aggregated payment rows")
    return agg


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple review rows per order to the *latest* review.

    Some orders were re-reviewed; we keep the one with the latest
    review_answer_timestamp so the score reflects final customer sentiment.
    """
    logger.info("  Aggregating reviews  -> 1 row per order (latest review)...")
    keep_cols = [
        "order_id", "review_id", "review_score",
        "review_comment_title", "review_comment_message",
        "review_creation_date", "review_answer_timestamp",
    ]
    agg = (
        reviews
        .sort_values("review_answer_timestamp", ascending=False)
        .groupby("order_id")
        .first()
        .reset_index()
        [keep_cols]
    )
    logger.info(f"  -> {len(agg):,} aggregated review rows\n")
    return agg


# ── Main join ────────────────────────────────────────────────────────────────

def create_master_dataset(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join all datasets into a single denormalised master DataFrame.

    Join chain
    ----------
    orders
      <- order_items       (order_id)
      <- payments_agg      (order_id)
      <- reviews_agg       (order_id)
      <- customers         (customer_id)
      <- products          (product_id)
      <- sellers           (seller_id)
      <- category_translation (product_category_name)

    Grain: one row per (order, item).  Orders with multiple items appear on
    multiple rows; order-level columns (payment, review, dates) are repeated.
    """
    logger.info("Building master dataset...")

    orders       = datasets["orders"]
    order_items  = datasets["order_items"]
    payments     = datasets["payments"]
    reviews      = datasets["reviews"]
    customers    = datasets["customers"]
    products     = datasets["products"]
    sellers      = datasets["sellers"]
    cat_trans    = datasets["category_translation"]

    payments_agg = aggregate_payments(payments)
    reviews_agg  = aggregate_reviews(reviews)

    steps = [
        ("order_items",          "order_id",               order_items),
        ("payments (aggregated)","order_id",               payments_agg),
        ("reviews  (aggregated)","order_id",               reviews_agg),
        ("customers",            "customer_id",            customers),
        ("products",             "product_id",             products),
        ("sellers",              "seller_id",              sellers),
        ("category_translation", "product_category_name",  cat_trans),
    ]

    df = orders.copy()
    logger.info(f"  Start  (orders)                   -> {len(df):>8,} rows")

    for label, key, right in steps:
        df = df.merge(right, on=key, how="left")
        logger.info(f"  + {label:<30} -> {len(df):>8,} rows")

    logger.info(f"\n  Master dataset ready: {len(df):,} rows × {df.shape[1]} columns\n")
    return df

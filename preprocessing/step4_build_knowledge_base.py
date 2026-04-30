"""
Step 4 – Build 6 KB document layers from final_olist_master_enriched.csv.

Layers
------
1. Order-level          (sampled to ORDER_KB_SAMPLE = 10,000)
2. Product-category-level
3. Seller-level
4. Customer-state-level
5. Month-level temporal
6. Delivery-status insight

Every document follows:
    {
        "id":       "<layer>_<source_id>",
        "text":     "Document Type: ...\\nKey: Value\\n...",
        "metadata": { "document_type": "...", "source_id": "...", ... }
    }

Aggregation rule
----------------
Revenue / freight / item counts  → item-level (full DataFrame)
Delivery / review / payment      → order-level (deduplicated by order_id per entity)
to avoid double-counting multi-item orders.
"""
import json
import logging
from typing import List

import numpy as np
import pandas as pd

from .config import DATA_KB, ORDER_KB_SAMPLE, TOP_SELLERS_FOR_KB, RANDOM_SEED

logger = logging.getLogger(__name__)


# ── Tiny helpers ─────────────────────────────────────────────────────────────

def _s(val, default: str = "N/A") -> str:
    """Safe string – returns default for NaN/None/empty."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    s = str(val).strip()
    return s if s else default


def _f(val, default: float = 0.0) -> float:
    """Safe float."""
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _ts(val) -> str:
    """Format a timestamp to 'YYYY-MM-DD HH:MM:SS', or 'N/A'."""
    if pd.isna(val):
        return "N/A"
    return str(val)[:19]


def _pct(num, den) -> str:
    """Safe percentage string."""
    if den == 0:
        return "N/A"
    return f"{num / den * 100:.2f}%"


def _save(docs: List[dict], filename: str) -> None:
    path = DATA_KB / filename
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(docs, fh, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Saved {len(docs):>6,} docs -> {path.name}")


# ── Shared: order-grain helper ────────────────────────────────────────────────

def _order_grain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the item-level DataFrame to one row per order_id.
    Used for delivery / review / payment statistics to avoid
    counting a 3-item order three times.
    """
    return (
        df.groupby("order_id")
        .agg(
            delivery_status           = ("delivery_status",           "first"),
            delivery_days             = ("delivery_days",             "first"),
            delivery_difference_days  = ("delivery_difference_days",  "first"),
            total_payment_value       = ("total_payment_value",       "first"),
            review_score              = ("review_score",              "first"),
            customer_state            = ("customer_state",            "first"),
            seller_state              = ("seller_state",              "first"),
            product_category_final    = ("product_category_final",    "first"),
            purchase_month            = ("purchase_month",            "first"),
            purchase_year             = ("purchase_year",             "first"),
            purchase_month_name       = ("purchase_month_name",       "first"),
        )
        .reset_index()
    )


# ── Layer 1: Order-level ──────────────────────────────────────────────────────

def build_order_documents(df: pd.DataFrame) -> List[dict]:
    """
    One document per order (sampled to ORDER_KB_SAMPLE).
    All fields from the user specification are included.
    """
    logger.info("Layer 1: Building order documents...")

    # Full order-grain aggregation
    agg = (
        df.groupby("order_id")
        .agg(
            order_status                  = ("order_status",                  "first"),
            customer_id                   = ("customer_id",                   "first"),
            customer_city                 = ("customer_city",                 "first"),
            customer_state                = ("customer_state",                "first"),
            order_purchase_timestamp      = ("order_purchase_timestamp",      "first"),
            order_approved_at             = ("order_approved_at",             "first"),
            order_delivered_carrier_date  = ("order_delivered_carrier_date",  "first"),
            order_delivered_customer_date = ("order_delivered_customer_date", "first"),
            order_estimated_delivery_date = ("order_estimated_delivery_date", "first"),
            purchase_month_name           = ("purchase_month_name",           "first"),
            purchase_year                 = ("purchase_year",                 "first"),
            purchase_month                = ("purchase_month",                "first"),
            delivery_days                 = ("delivery_days",                 "first"),
            estimated_delivery_days       = ("estimated_delivery_days",       "first"),
            delivery_status               = ("delivery_status",               "first"),
            delivery_bucket               = ("delivery_bucket",               "first"),
            total_items                   = ("order_item_id",                 "count"),
            unique_products               = ("product_id",                   "nunique"),
            unique_sellers                = ("seller_id",                    "nunique"),
            product_categories            = ("product_category_final",
                                              lambda x: "|".join(sorted(set(x.dropna())))),
            seller_states                 = ("seller_state",
                                              lambda x: "|".join(sorted(set(x.dropna())))),
            total_price                   = ("price",                         "sum"),
            total_freight                 = ("freight_value",                 "sum"),
            payment_types                 = ("payment_types",                 "first"),
            total_payment_value           = ("total_payment_value",           "first"),
            max_installments              = ("max_installments",              "first"),
            review_score                  = ("review_score",                  "first"),
            review_bucket                 = ("review_bucket",                 "first"),
            review_comment_title          = ("review_comment_title",          "first"),
            review_comment_message        = ("review_comment_message",        "first"),
        )
        .reset_index()
    )
    agg["total_item_value"] = (agg["total_price"] + agg["total_freight"]).round(2)

    # Sample
    if ORDER_KB_SAMPLE and len(agg) > ORDER_KB_SAMPLE:
        sample = agg.sample(n=ORDER_KB_SAMPLE, random_state=RANDOM_SEED)
        logger.info(f"  Sampled {ORDER_KB_SAMPLE:,} / {len(agg):,} orders")
    else:
        sample = agg

    docs: List[dict] = []
    for _, r in sample.iterrows():
        yr = int(r["purchase_year"]) if not pd.isna(r["purchase_year"]) else "N/A"
        mo = int(r["purchase_month"]) if not pd.isna(r["purchase_month"]) else 0
        purchase_month_str = f"{yr}-{mo:02d}" if yr != "N/A" else "N/A"
        month_name_str     = f"{_s(r['purchase_month_name'])} {yr}"

        # Fractional delivery difference (matches the user's example)
        dd = _f(r["delivery_days"])
        ed = _f(r["estimated_delivery_days"])
        diff_str = f"{dd - ed:.2f}" if (dd and ed) else "N/A"

        rv_score = (
            str(int(r["review_score"]))
            if not pd.isna(r["review_score"])
            else "No review"
        )

        text = "\n".join([
            "Document Type: Order-Level Summary",
            f"Order ID: {r['order_id']}",
            f"Customer ID: {_s(r['customer_id'])}",
            f"Customer City: {_s(r['customer_city'])}",
            f"Customer State: {_s(r['customer_state'])}",
            "",
            f"Order Status: {_s(r['order_status'])}",
            f"Purchase Month: {month_name_str}",
            f"Purchase Timestamp: {_ts(r['order_purchase_timestamp'])}",
            f"Approved At: {_ts(r['order_approved_at'])}",
            f"Delivered to Carrier Date: {_ts(r['order_delivered_carrier_date'])}",
            f"Delivered to Customer Date: {_ts(r['order_delivered_customer_date'])}",
            f"Estimated Delivery Date: {_ts(r['order_estimated_delivery_date'])}",
            "",
            f"Delivery Days: {dd:.2f}",
            f"Estimated Delivery Days: {ed:.2f}",
            f"Delivery Difference Days: {diff_str}",
            f"Delivery Status: {_s(r['delivery_status'])}",
            f"Delivery Bucket: {_s(r['delivery_bucket'])}",
            "",
            f"Total Items: {int(r['total_items'])}",
            f"Unique Products: {int(r['unique_products'])}",
            f"Unique Sellers: {int(r['unique_sellers'])}",
            f"Product Categories: {_s(r['product_categories'])}",
            f"Seller States: {_s(r['seller_states'])}",
            "",
            f"Total Product Price: {_f(r['total_price']):.2f}",
            f"Total Freight Value: {_f(r['total_freight']):.2f}",
            f"Total Item Value: {_f(r['total_item_value']):.2f}",
            f"Payment Types: {_s(r['payment_types'])}",
            f"Total Payment Value: {_f(r['total_payment_value']):.2f}",
            f"Maximum Payment Installments: {int(_f(r['max_installments']))}",
            "",
            f"Review Score: {rv_score}",
            f"Review Bucket: {_s(r['review_bucket'])}",
            f"Review Comment Title: {_s(r['review_comment_title'], default='No comment title')}",
            f"Review Comment Message: {_s(r['review_comment_message'], default='No comment message')}",
            "",
            "This document summarises one customer order including delivery, seller,"
            " product, payment, and review information.",
        ])

        docs.append({
            "id":   f"order_{r['order_id']}",
            "text": text,
            "metadata": {
                "document_type":  "order_level",
                "source_id":      r["order_id"],
                "order_id":       r["order_id"],
                "order_status":   _s(r["order_status"]),
                "customer_state": _s(r["customer_state"]),
                "delivery_status":_s(r["delivery_status"]),
                "delivery_bucket":_s(r["delivery_bucket"]),
                "purchase_month": purchase_month_str,
                "review_bucket":  _s(r["review_bucket"]),
            },
        })

    logger.info(f"  -> {len(docs):,} order documents")
    return docs


# ── Layer 2: Product category-level ──────────────────────────────────────────

def build_category_documents(df: pd.DataFrame) -> List[dict]:
    """One document per product_category_final."""
    logger.info("Layer 2: Building category documents...")

    # Item-level stats (revenue, freight, items)
    item_stats = (
        df.groupby("product_category_final")
        .agg(
            total_items   = ("order_item_id",  "count"),
            total_revenue = ("price",          "sum"),
            total_freight = ("freight_value",  "sum"),
            avg_price     = ("price",          "mean"),
            avg_freight   = ("freight_value",  "mean"),
        )
        .reset_index()
    )

    # Order-level stats per category (deduplicated to avoid multi-item double-count)
    cat_ord = (
        df.groupby(["product_category_final", "order_id"])
        .agg(
            total_payment_value = ("total_payment_value", "first"),
            review_score        = ("review_score",        "first"),
            delivery_status     = ("delivery_status",     "first"),
            delivery_days       = ("delivery_days",       "first"),
        )
        .reset_index()
    )

    ord_stats = (
        cat_ord.groupby("product_category_final")
        .agg(
            total_orders     = ("order_id",            "nunique"),
            avg_payment      = ("total_payment_value", "mean"),
            avg_review       = ("review_score",        "mean"),
            avg_delivery     = ("delivery_days",       "mean"),
            positive_reviews = ("review_score",        lambda x: (x >= 4).sum()),
            negative_reviews = ("review_score",        lambda x: (x <= 2).sum()),
            late_count       = ("delivery_status",     lambda x: (x == "late").sum()),
            ontime_count     = ("delivery_status",     lambda x: x.isin(["early","on_time"]).sum()),
            delivered_count  = ("delivery_status",     lambda x: (x != "not_delivered").sum()),
        )
        .reset_index()
    )

    # Top customer state and seller state
    top_cust = (
        df.groupby(["product_category_final","customer_state"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("product_category_final")["customer_state"].first()
        .reset_index().rename(columns={"customer_state":"top_customer_state"})
    )
    top_sell = (
        df.groupby(["product_category_final","seller_state"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("product_category_final")["seller_state"].first()
        .reset_index().rename(columns={"seller_state":"top_seller_state"})
    )

    full = (
        item_stats
        .merge(ord_stats, on="product_category_final", how="left")
        .merge(top_cust,  on="product_category_final", how="left")
        .merge(top_sell,  on="product_category_final", how="left")
    )

    docs: List[dict] = []
    for _, r in full.iterrows():
        cat           = r["product_category_final"]
        total_orders  = int(_f(r["total_orders"]))
        late_count    = int(_f(r["late_count"]))
        ontime_count  = int(_f(r["ontime_count"]))
        delivered     = int(_f(r["delivered_count"]))
        late_rate_str = _pct(late_count, delivered)

        text = "\n".join([
            "Document Type: Product Category Summary",
            f"Product Category: {cat}",
            "",
            f"Total Orders: {total_orders:,}",
            f"Total Items Sold: {int(_f(r['total_items'])):,}",
            f"Total Product Revenue: {_f(r['total_revenue']):.2f}",
            f"Total Freight Value: {_f(r['total_freight']):.2f}",
            f"Average Product Price: {_f(r['avg_price']):.2f}",
            f"Average Freight Value: {_f(r['avg_freight']):.2f}",
            f"Average Payment Value: {_f(r['avg_payment']):.2f}",
            "",
            f"Average Review Score: {_f(r['avg_review']):.2f}",
            f"Positive Review Count: {int(_f(r['positive_reviews'])):,}",
            f"Negative Review Count: {int(_f(r['negative_reviews'])):,}",
            "",
            f"Late Delivery Count: {late_count:,}",
            f"Early or On-Time Delivery Count: {ontime_count:,}",
            f"Late Delivery Rate: {late_rate_str}",
            f"Average Delivery Days: {_f(r['avg_delivery']):.2f}",
            "",
            f"Top Customer State: {_s(r.get('top_customer_state'))}",
            f"Top Seller State: {_s(r.get('top_seller_state'))}",
            "",
            f"This document summarises sales, delivery, freight, and review"
            f" performance for the {cat} product category.",
        ])

        docs.append({
            "id":   f"category_{cat.replace(' ','_')}",
            "text": text,
            "metadata": {
                "document_type": "category_level",
                "source_id":     cat,
                "total_orders":  total_orders,
                "avg_review":    round(_f(r["avg_review"]), 2),
                "late_rate_pct": round(late_count / delivered * 100, 2) if delivered else 0,
                "total_revenue": round(_f(r["total_revenue"]), 2),
            },
        })

    logger.info(f"  -> {len(docs):,} category documents")
    return docs


# ── Layer 3: Seller-level ─────────────────────────────────────────────────────

def build_seller_documents(df: pd.DataFrame) -> List[dict]:
    """One document per seller_id."""
    logger.info("Layer 3: Building seller documents...")

    # Item-level stats
    item_stats = (
        df.groupby("seller_id")
        .agg(
            seller_city   = ("seller_city",   "first"),
            seller_state  = ("seller_state",  "first"),
            total_items   = ("order_item_id", "count"),
            total_revenue = ("price",         "sum"),
            total_freight = ("freight_value", "sum"),
            avg_freight   = ("freight_value", "mean"),
        )
        .reset_index()
    )

    # Order-level stats (deduplicated)
    sell_ord = (
        df.groupby(["seller_id","order_id"])
        .agg(
            review_score    = ("review_score",    "first"),
            delivery_status = ("delivery_status", "first"),
            delivery_days   = ("delivery_days",   "first"),
            customer_state  = ("customer_state",  "first"),
        )
        .reset_index()
    )
    ord_stats = (
        sell_ord.groupby("seller_id")
        .agg(
            total_orders     = ("order_id",        "nunique"),
            avg_review       = ("review_score",    "mean"),
            avg_delivery     = ("delivery_days",   "mean"),
            positive_reviews = ("review_score",    lambda x: (x >= 4).sum()),
            negative_reviews = ("review_score",    lambda x: (x <= 2).sum()),
            late_count       = ("delivery_status", lambda x: (x == "late").sum()),
            delivered_count  = ("delivery_status", lambda x: (x != "not_delivered").sum()),
        )
        .reset_index()
    )

    # Top product category and customer state
    top_cat = (
        df.groupby(["seller_id","product_category_final"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("seller_id")["product_category_final"].first()
        .reset_index().rename(columns={"product_category_final":"top_category"})
    )
    top_cust = (
        sell_ord.groupby(["seller_id","customer_state"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("seller_id")["customer_state"].first()
        .reset_index().rename(columns={"customer_state":"top_customer_state"})
    )

    full = (
        item_stats
        .merge(ord_stats, on="seller_id", how="left")
        .merge(top_cat,   on="seller_id", how="left")
        .merge(top_cust,  on="seller_id", how="left")
    )

    if TOP_SELLERS_FOR_KB:
        full = full.nlargest(TOP_SELLERS_FOR_KB, "total_orders")

    docs: List[dict] = []
    for _, r in full.iterrows():
        sid          = r["seller_id"]
        late_count   = int(_f(r["late_count"]))
        delivered    = int(_f(r["delivered_count"]))
        late_rate_str= _pct(late_count, delivered)
        total_orders = int(_f(r["total_orders"]))

        text = "\n".join([
            "Document Type: Seller-Level Summary",
            f"Seller ID: {sid}",
            f"Seller City: {_s(r['seller_city'])}",
            f"Seller State: {_s(r['seller_state'])}",
            "",
            f"Total Orders: {total_orders:,}",
            f"Total Items Sold: {int(_f(r['total_items'])):,}",
            f"Total Product Revenue: {_f(r['total_revenue']):.2f}",
            f"Total Freight Value: {_f(r['total_freight']):.2f}",
            f"Average Freight Value: {_f(r['avg_freight']):.2f}",
            "",
            f"Average Review Score: {_f(r['avg_review']):.2f}",
            f"Positive Review Count: {int(_f(r['positive_reviews'])):,}",
            f"Negative Review Count: {int(_f(r['negative_reviews'])):,}",
            "",
            f"Late Delivery Count: {late_count:,}",
            f"Late Delivery Rate: {late_rate_str}",
            f"Average Delivery Days: {_f(r['avg_delivery']):.2f}",
            "",
            f"Top Product Category: {_s(r.get('top_category'))}",
            f"Top Customer State: {_s(r.get('top_customer_state'))}",
            "",
            "This document summarises seller fulfilment, revenue, freight,"
            " delivery, and review performance.",
        ])

        docs.append({
            "id":   f"seller_{sid}",
            "text": text,
            "metadata": {
                "document_type": "seller_level",
                "source_id":     sid,
                "seller_city":   _s(r["seller_city"]),
                "seller_state":  _s(r["seller_state"]),
                "total_orders":  total_orders,
                "avg_review":    round(_f(r["avg_review"]), 2),
                "total_revenue": round(_f(r["total_revenue"]), 2),
                "late_rate_pct": round(late_count / delivered * 100, 2) if delivered else 0,
            },
        })

    logger.info(f"  -> {len(docs):,} seller documents")
    return docs


# ── Layer 4: Customer-state-level ─────────────────────────────────────────────

def build_customer_state_documents(df: pd.DataFrame) -> List[dict]:
    """One document per customer_state."""
    logger.info("Layer 4: Building customer-state documents...")

    # Order-level stats (deduplicated)
    state_ord = (
        df.groupby(["customer_state","order_id"])
        .agg(
            customer_unique_id  = ("customer_unique_id",  "first"),
            total_payment_value = ("total_payment_value", "first"),
            review_score        = ("review_score",        "first"),
            delivery_status     = ("delivery_status",     "first"),
            delivery_days       = ("delivery_days",       "first"),
            payment_types       = ("payment_types",       "first"),
        )
        .reset_index()
    )
    state_stats = (
        state_ord.groupby("customer_state")
        .agg(
            total_orders      = ("order_id",            "nunique"),
            unique_customers  = ("customer_unique_id",  "nunique"),
            total_payment     = ("total_payment_value", "sum"),
            avg_payment       = ("total_payment_value", "mean"),
            avg_review        = ("review_score",        "mean"),
            avg_delivery      = ("delivery_days",       "mean"),
            positive_reviews  = ("review_score",        lambda x: (x >= 4).sum()),
            negative_reviews  = ("review_score",        lambda x: (x <= 2).sum()),
            late_count        = ("delivery_status",     lambda x: (x == "late").sum()),
            delivered_count   = ("delivery_status",     lambda x: (x != "not_delivered").sum()),
        )
        .reset_index()
    )

    # Top product category
    top_cat = (
        df.groupby(["customer_state","product_category_final"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("customer_state")["product_category_final"].first()
        .reset_index().rename(columns={"product_category_final":"top_category"})
    )
    # Top payment type
    top_pay = (
        state_ord.groupby(["customer_state","payment_types"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("customer_state")["payment_types"].first()
        .reset_index().rename(columns={"payment_types":"top_payment_type"})
    )

    full = (
        state_stats
        .merge(top_cat, on="customer_state", how="left")
        .merge(top_pay, on="customer_state", how="left")
    )

    docs: List[dict] = []
    for _, r in full.iterrows():
        state        = r["customer_state"]
        late_count   = int(_f(r["late_count"]))
        delivered    = int(_f(r["delivered_count"]))
        late_rate_str= _pct(late_count, delivered)
        total_orders = int(_f(r["total_orders"]))

        text = "\n".join([
            "Document Type: Customer-State Summary",
            f"Customer State: {state}",
            "",
            f"Total Orders: {total_orders:,}",
            f"Unique Customers: {int(_f(r['unique_customers'])):,}",
            f"Total Payment Value: {_f(r['total_payment']):.2f}",
            f"Average Payment Value: {_f(r['avg_payment']):.2f}",
            "",
            f"Average Review Score: {_f(r['avg_review']):.2f}",
            f"Positive Review Count: {int(_f(r['positive_reviews'])):,}",
            f"Negative Review Count: {int(_f(r['negative_reviews'])):,}",
            "",
            f"Late Delivery Count: {late_count:,}",
            f"Late Delivery Rate: {late_rate_str}",
            f"Average Delivery Days: {_f(r['avg_delivery']):.2f}",
            "",
            f"Top Product Category: {_s(r.get('top_category'))}",
            f"Top Payment Type: {_s(r.get('top_payment_type'))}",
            "",
            f"This document summarises order volume, payment value, delivery"
            f" performance, and review behaviour for customers in {state}.",
        ])

        docs.append({
            "id":   f"state_{state}",
            "text": text,
            "metadata": {
                "document_type": "customer_state_level",
                "source_id":     state,
                "total_orders":  total_orders,
                "avg_review":    round(_f(r["avg_review"]), 2),
                "late_rate_pct": round(late_count / delivered * 100, 2) if delivered else 0,
                "avg_delivery_days": round(_f(r["avg_delivery"]), 2),
            },
        })

    logger.info(f"  -> {len(docs):,} customer-state documents")
    return docs


# ── Layer 5: Month-level temporal ─────────────────────────────────────────────

def build_month_documents(df: pd.DataFrame) -> List[dict]:
    """One document per purchase_month (format: YYYY-MM)."""
    logger.info("Layer 5: Building month documents...")

    # Create 'YYYY-MM' key
    df = df.copy()
    df["ym_key"] = (
        df["purchase_year"].astype("Int64").astype(str)
        + "-"
        + df["purchase_month"].astype("Int64").apply(lambda m: f"{m:02d}")
    )

    # Order-level per month (deduplicated)
    mo_ord = (
        df.groupby(["ym_key","order_id"])
        .agg(
            order_status        = ("order_status",        "first"),
            total_payment_value = ("total_payment_value", "first"),
            freight_value       = ("freight_value",       "first"),
            review_score        = ("review_score",        "first"),
            delivery_status     = ("delivery_status",     "first"),
            delivery_days       = ("delivery_days",       "first"),
            customer_state      = ("customer_state",      "first"),
            seller_state        = ("seller_state",        "first"),
            purchase_month_name = ("purchase_month_name", "first"),
            purchase_year       = ("purchase_year",       "first"),
            purchase_month      = ("purchase_month",      "first"),
        )
        .reset_index()
    )
    mo_stats = (
        mo_ord.groupby("ym_key")
        .agg(
            purchase_month_name = ("purchase_month_name", "first"),
            purchase_year       = ("purchase_year",       "first"),
            purchase_month      = ("purchase_month",      "first"),
            total_orders        = ("order_id",            "nunique"),
            delivered_orders    = ("delivery_status",     lambda x: (x != "not_delivered").sum()),
            canceled_orders     = ("order_status",        lambda x: (x == "canceled").sum()),
            late_count          = ("delivery_status",     lambda x: (x == "late").sum()),
            ontime_count        = ("delivery_status",     lambda x: x.isin(["early","on_time"]).sum()),
            delivered_count     = ("delivery_status",     lambda x: (x != "not_delivered").sum()),
            total_payment       = ("total_payment_value", "sum"),
            avg_payment         = ("total_payment_value", "mean"),
            avg_delivery        = ("delivery_days",       "mean"),
            avg_review          = ("review_score",        "mean"),
            negative_reviews    = ("review_score",        lambda x: (x <= 2).sum()),
        )
        .reset_index()
    )

    # Average freight (item-level is fine here)
    avg_freight = (
        df.groupby("ym_key")["freight_value"].mean()
        .reset_index().rename(columns={"freight_value":"avg_freight"})
    )

    # Top category, customer state, seller state
    top_cat = (
        df.groupby(["ym_key","product_category_final"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("ym_key")["product_category_final"].first()
        .reset_index().rename(columns={"product_category_final":"top_category"})
    )
    top_cust = (
        mo_ord.groupby(["ym_key","customer_state"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("ym_key")["customer_state"].first()
        .reset_index().rename(columns={"customer_state":"top_customer_state"})
    )
    top_sell = (
        mo_ord.groupby(["ym_key","seller_state"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("ym_key")["seller_state"].first()
        .reset_index().rename(columns={"seller_state":"top_seller_state"})
    )

    full = (
        mo_stats
        .merge(avg_freight, on="ym_key", how="left")
        .merge(top_cat,     on="ym_key", how="left")
        .merge(top_cust,    on="ym_key", how="left")
        .merge(top_sell,    on="ym_key", how="left")
        .sort_values("ym_key")
    )

    docs: List[dict] = []
    for _, r in full.iterrows():
        ym           = r["ym_key"]
        yr           = int(_f(r["purchase_year"]))
        mo           = int(_f(r["purchase_month"]))
        mo_name      = _s(r["purchase_month_name"])
        late_count   = int(_f(r["late_count"]))
        delivered    = int(_f(r["delivered_count"]))
        late_rate_str= _pct(late_count, delivered)

        text = "\n".join([
            "Document Type: Month-Level Temporal Summary",
            f"Purchase Month: {ym}",
            f"Month Name: {mo_name} {yr}",
            "",
            f"Total Orders: {int(_f(r['total_orders'])):,}",
            f"Delivered Orders: {int(_f(r['delivered_orders'])):,}",
            f"Canceled Orders: {int(_f(r['canceled_orders'])):,}",
            "",
            f"Late Delivery Count: {late_count:,}",
            f"Early or On-Time Delivery Count: {int(_f(r['ontime_count'])):,}",
            f"Late Delivery Rate: {late_rate_str}",
            f"Average Delivery Days: {_f(r['avg_delivery']):.2f}",
            "",
            f"Total Payment Value: {_f(r['total_payment']):.2f}",
            f"Average Payment Value: {_f(r['avg_payment']):.2f}",
            f"Average Freight Value: {_f(r['avg_freight']):.2f}",
            "",
            f"Average Review Score: {_f(r['avg_review']):.2f}",
            f"Negative Review Count: {int(_f(r['negative_reviews'])):,}",
            "",
            f"Most Popular Product Category: {_s(r.get('top_category'))}",
            f"Top Customer State: {_s(r.get('top_customer_state'))}",
            f"Top Seller State: {_s(r.get('top_seller_state'))}",
            "",
            "This document summarises monthly e-commerce fulfilment, revenue,"
            " delivery, and review performance.",
        ])

        docs.append({
            "id":   f"month_{yr}_{mo:02d}",
            "text": text,
            "metadata": {
                "document_type":     "month_level",
                "source_id":         ym,
                "year":              yr,
                "month":             mo,
                "month_name":        mo_name,
                "total_orders":      int(_f(r["total_orders"])),
                "late_rate_pct":     round(late_count / delivered * 100, 2) if delivered else 0,
                "avg_review":        round(_f(r["avg_review"]), 2),
                "total_payment":     round(_f(r["total_payment"]), 2),
            },
        })

    logger.info(f"  -> {len(docs):,} month documents")
    return docs


# ── Layer 6: Delivery-status insight ─────────────────────────────────────────

def build_delivery_status_documents(df: pd.DataFrame) -> List[dict]:
    """
    One document per delivery_status value.
    Gives high-level insight into each delivery outcome group.
    """
    logger.info("Layer 6: Building delivery-status insight documents...")

    # Order-level (deduplicated)
    ord_lvl = _order_grain(df)

    stats = (
        ord_lvl.groupby("delivery_status")
        .agg(
            total_orders   = ("order_id",          "nunique"),
            avg_delivery   = ("delivery_days",     "mean"),
            avg_review     = ("review_score",      "mean"),
            avg_payment    = ("total_payment_value","mean"),
            avg_diff_days  = ("delivery_difference_days","mean"),
        )
        .reset_index()
    )

    top_cat = (
        df.groupby(["delivery_status","product_category_final"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("delivery_status")["product_category_final"].first()
        .reset_index().rename(columns={"product_category_final":"top_category"})
    )
    top_cust = (
        ord_lvl.groupby(["delivery_status","customer_state"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("delivery_status")["customer_state"].first()
        .reset_index().rename(columns={"customer_state":"top_customer_state"})
    )
    top_sell = (
        ord_lvl.groupby(["delivery_status","seller_state"])["order_id"]
        .nunique().reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("delivery_status")["seller_state"].first()
        .reset_index().rename(columns={"seller_state":"top_seller_state"})
    )

    full = (
        stats
        .merge(top_cat,  on="delivery_status", how="left")
        .merge(top_cust, on="delivery_status", how="left")
        .merge(top_sell, on="delivery_status", how="left")
    )

    total_delivered = int(ord_lvl[ord_lvl["delivery_status"] != "not_delivered"]["order_id"].nunique())

    docs: List[dict] = []
    for _, r in full.iterrows():
        status       = r["delivery_status"]
        total_orders = int(_f(r["total_orders"]))
        share_str    = _pct(total_orders, total_delivered) if status != "not_delivered" else "N/A (undelivered)"
        diff_val     = _f(r["avg_diff_days"])
        diff_str     = f"{diff_val:.2f} days {'early' if diff_val < 0 else 'late'}" if status not in ("not_delivered",) else "N/A"

        text = "\n".join([
            "Document Type: Delivery-Status Insight",
            f"Delivery Status: {status}",
            "",
            f"Total Orders in Group: {total_orders:,}",
            f"Share of Delivered Orders: {share_str}",
            f"Average Delivery Days: {_f(r['avg_delivery']):.2f}",
            f"Average Difference vs Estimated: {diff_str}",
            "",
            f"Average Review Score: {_f(r['avg_review']):.2f}",
            f"Average Payment Value: {_f(r['avg_payment']):.2f}",
            "",
            f"Top Product Category: {_s(r.get('top_category'))}",
            f"Top Customer State: {_s(r.get('top_customer_state'))}",
            f"Top Seller State: {_s(r.get('top_seller_state'))}",
            "",
            f"This document provides aggregate insight into orders with"
            f" delivery status '{status}'.",
        ])

        docs.append({
            "id":   f"delivery_status_{status}",
            "text": text,
            "metadata": {
                "document_type": "delivery_status_insight",
                "source_id":     status,
                "delivery_status": status,
                "total_orders":  total_orders,
                "avg_review":    round(_f(r["avg_review"]), 2),
                "avg_delivery_days": round(_f(r["avg_delivery"]), 2),
            },
        })

    logger.info(f"  -> {len(docs):,} delivery-status documents")
    return docs


# ── Orchestrator ──────────────────────────────────────────────────────────────

def build_knowledge_base(df: pd.DataFrame) -> List[dict]:
    """Build all 6 layers, save individual JSON files, and combine."""
    DATA_KB.mkdir(parents=True, exist_ok=True)
    logger.info("\nBuilding knowledge base (6 layers)...")

    order_docs    = build_order_documents(df)
    cat_docs      = build_category_documents(df)
    seller_docs   = build_seller_documents(df)
    state_docs    = build_customer_state_documents(df)
    month_docs    = build_month_documents(df)
    deliv_docs    = build_delivery_status_documents(df)

    _save(order_docs,  "kb_order_documents.json")
    _save(cat_docs,    "kb_category_documents.json")
    _save(seller_docs, "kb_seller_documents.json")
    _save(state_docs,  "kb_customer_state_documents.json")
    _save(month_docs,  "kb_month_documents.json")
    _save(deliv_docs,  "kb_delivery_status_documents.json")

    all_docs = order_docs + cat_docs + seller_docs + state_docs + month_docs + deliv_docs
    _save(all_docs, "kb_all_documents.json")

    logger.info(
        f"\n  KB complete:"
        f"\n    Layer 1 Orders:          {len(order_docs):>6,}"
        f"\n    Layer 2 Categories:      {len(cat_docs):>6,}"
        f"\n    Layer 3 Sellers:         {len(seller_docs):>6,}"
        f"\n    Layer 4 Customer States: {len(state_docs):>6,}"
        f"\n    Layer 5 Months:          {len(month_docs):>6,}"
        f"\n    Layer 6 Delivery Status: {len(deliv_docs):>6,}"
        f"\n    -------------------------"
        f"\n    Total:                   {len(all_docs):>6,}\n"
    )
    return all_docs

"""
Shared fixtures for all unit tests.

All fixtures use synthetic in-memory DataFrames — no raw CSV files are read.
The base timestamp is 2017-10-02 10:00:00 (Monday, October, hour=10).
"""
import sys
from pathlib import Path

# Ensure project root is importable regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


# ── Timestamp anchor ──────────────────────────────────────────────────────────
BASE = pd.Timestamp("2017-10-02 10:00:00")


# ── Step 1 / Step 2 raw-table fixtures ───────────────────────────────────────

@pytest.fixture
def payments_raw():
    """
    3 payment rows covering 2 orders.
    ord1 pays with credit_card (R$80, 3 instalments) + voucher (R$20, 1 instalment).
    ord2 pays with boleto only (R$50, 1 instalment).
    """
    return pd.DataFrame({
        "order_id":             ["ord1", "ord1", "ord2"],
        "payment_sequential":   [1,      2,      1],
        "payment_type":         ["credit_card", "voucher", "boleto"],
        "payment_installments": [3,      1,      1],
        "payment_value":        [80.0,   20.0,   50.0],
    })


@pytest.fixture
def reviews_raw():
    """
    3 review rows covering 2 orders.
    ord1 was reviewed twice; the second (r2, score=5) is the latest.
    ord2 has one review (r3, score=4).
    """
    return pd.DataFrame({
        "review_id":    ["r1",   "r2",   "r3"],
        "order_id":     ["ord1", "ord1", "ord2"],
        "review_score": [3,      5,      4],
        "review_comment_title":   [None,   "Great!", None],
        "review_comment_message": [None,   "Fast delivery", None],
        "review_creation_date":   pd.to_datetime(["2017-11-01", "2017-11-10", "2017-11-05"]),
        "review_answer_timestamp":pd.to_datetime(["2017-11-02", "2017-11-11", "2017-11-06"]),
    })


@pytest.fixture
def minimal_datasets():
    """
    Dict of 9 minimal DataFrames matching the exact column set expected by
    create_master_dataset().  Two orders, two items (ord1 has 2 items).
    """
    orders = pd.DataFrame({
        "order_id":    ["ord1", "ord2"],
        "customer_id": ["c1",   "c2"],
        "order_status":["delivered", "delivered"],
        "order_purchase_timestamp":      pd.to_datetime(["2017-10-02 10:00:00", "2017-11-01 08:00:00"]),
        "order_approved_at":             pd.to_datetime(["2017-10-02 11:00:00", "2017-11-01 09:00:00"]),
        "order_delivered_carrier_date":  pd.to_datetime(["2017-10-04 12:00:00", "2017-11-03 12:00:00"]),
        "order_delivered_customer_date": pd.to_datetime(["2017-10-10 12:00:00", "2017-11-15 12:00:00"]),
        "order_estimated_delivery_date": pd.to_datetime(["2017-10-18 00:00:00", "2017-11-10 00:00:00"]),
    })

    # ord1 has 2 items; ord2 has 1 item → expect 3 rows after join
    order_items = pd.DataFrame({
        "order_id":       ["ord1", "ord1", "ord2"],
        "order_item_id":  [1,      2,      1],
        "product_id":     ["p1",   "p2",   "p3"],
        "seller_id":      ["s1",   "s1",   "s2"],
        "shipping_limit_date": pd.to_datetime(["2017-10-05", "2017-10-05", "2017-11-05"]),
        "price":          [100.0, 50.0, 200.0],
        "freight_value":  [10.0,  5.0,  20.0],
    })

    payments = pd.DataFrame({
        "order_id":             ["ord1", "ord2"],
        "payment_sequential":   [1,      1],
        "payment_type":         ["credit_card", "boleto"],
        "payment_installments": [2,      1],
        "payment_value":        [165.0,  220.0],
    })

    reviews = pd.DataFrame({
        "review_id":    ["r1", "r2"],
        "order_id":     ["ord1", "ord2"],
        "review_score": [4, 5],
        "review_comment_title":   [None,   None],
        "review_comment_message": [None,   None],
        "review_creation_date":   pd.to_datetime(["2017-10-15", "2017-11-20"]),
        "review_answer_timestamp":pd.to_datetime(["2017-10-16", "2017-11-21"]),
    })

    customers = pd.DataFrame({
        "customer_id":          ["c1",      "c2"],
        "customer_unique_id":   ["uc1",     "uc2"],
        "customer_zip_code_prefix": ["01310", "20040"],
        "customer_city":        ["sao paulo","rio de janeiro"],
        "customer_state":       ["SP",       "RJ"],
    })

    products = pd.DataFrame({
        "product_id":               ["p1",          "p2",      "p3"],
        "product_category_name":    ["eletronicos",  "moveis",  "saude_beleza"],
        "product_name_lenght":      [30,  25,  40],
        "product_description_lenght":[200, 150, 300],
        "product_photos_qty":       [2,   1,   3],
        "product_weight_g":         [500, 800, 200],
        "product_length_cm":        [20,  30,  15],
        "product_height_cm":        [10,  15,  8],
        "product_width_cm":         [15,  20,  12],
    })

    sellers = pd.DataFrame({
        "seller_id":               ["s1",      "s2"],
        "seller_zip_code_prefix":  ["13023",   "04018"],
        "seller_city":             ["campinas", "sao paulo"],
        "seller_state":            ["SP",       "SP"],
    })

    geolocation = pd.DataFrame({
        "geolocation_zip_code_prefix": ["01310"],
        "geolocation_lat":             [-23.5],
        "geolocation_lng":             [-46.6],
        "geolocation_city":            ["sao paulo"],
        "geolocation_state":           ["SP"],
    })

    category_translation = pd.DataFrame({
        "product_category_name":         ["eletronicos",  "moveis",   "saude_beleza"],
        "product_category_name_english": ["electronics",  "furniture","health_beauty"],
    })

    return {
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "reviews": reviews,
        "customers": customers,
        "products": products,
        "sellers": sellers,
        "geolocation": geolocation,
        "category_translation": category_translation,
    }


# ── Step 3 enricher fixture ───────────────────────────────────────────────────

@pytest.fixture
def minimal_master():
    """
    5-row master-like DataFrame with deterministic delivery outcomes:

    ord1  early        delivered 8 days after purchase, est 15 → diff = -7 (slightly_early)
    ord2  late         delivered 20 days,              est 15 → diff = +5 (slightly_late)
    ord3  on_time      delivered 15 days,              est 15 → diff =  0 (on_time)
    ord4  not_delivered NaT delivery date                      (not_delivered)
    ord5  very_early   delivered 0 days (same day),   est 15 → diff = -15 (very_early)
    """
    est = BASE + pd.Timedelta(days=15)
    return pd.DataFrame({
        "order_id":      ["ord1",                    "ord2",                    "ord3",  "ord4", "ord5"],
        "order_item_id": [1,                          1,                         1,       1,      1],
        "product_id":    ["p1",                       "p2",                      "p3",    "p4",   "p5"],
        "seller_id":     ["s1",                       "s2",                      "s3",    "s4",   "s5"],
        "order_status":  ["delivered", "delivered", "delivered", "canceled", "delivered"],
        "order_purchase_timestamp":      [BASE, BASE, BASE, BASE, BASE],
        "order_approved_at":             [
            BASE + pd.Timedelta(hours=1),
            BASE + pd.Timedelta(hours=2),
            BASE + pd.Timedelta(hours=1),
            BASE + pd.Timedelta(hours=3),
            pd.NaT,                        # ord5: no approval
        ],
        "order_delivered_carrier_date":  [
            BASE + pd.Timedelta(days=2),
            BASE + pd.Timedelta(days=3),
            BASE + pd.Timedelta(days=2),
            pd.NaT,
            pd.NaT,
        ],
        "order_delivered_customer_date": [
            BASE + pd.Timedelta(days=8),   # diff = -7
            BASE + pd.Timedelta(days=20),  # diff = +5
            BASE + pd.Timedelta(days=15),  # diff =  0
            pd.NaT,
            BASE,                          # diff = -15
        ],
        "order_estimated_delivery_date": [est, est, est, est, est],
        "price":         [100.0, 200.0,  50.0, 75.0, 90.0],
        "freight_value": [10.0,   20.0,   5.0,  7.5,  9.0],
        "review_score":  [5.0,    2.0,    3.0, None,  4.0],
        "product_category_name_english": ["electronics", None, "furniture", "sports", "health_beauty"],
        "product_category_name":         ["eletronicos", "moveis", "moveis", "esportes", "saude_beleza"],
        "customer_state": ["SP", "RJ", "MG", "SP", "RJ"],
        "customer_city":  ["sao paulo", "rio", "bh", "sp", "rio"],
        "seller_city":    ["campinas", "rio", "bh", "sp", "curitiba"],
        "seller_state":   ["SP", "RJ", "MG", "SP", "PR"],
        "customer_id":    ["c1", "c2", "c3", "c4", "c5"],
        "customer_unique_id": ["uc1", "uc2", "uc3", "uc4", "uc5"],
        "total_payment_value": [110.0, 220.0, 55.0, 82.5, 99.0],
        "payment_types":       ["credit_card", "boleto", "credit_card", "voucher", "credit_card"],
        "max_installments":    [2, 1, 3, 1, 4],
        "payment_methods_count": [1, 1, 1, 1, 1],
        "review_id": ["r1", "r2", "r3", None, "r5"],
        "review_comment_title":   [None] * 5,
        "review_comment_message": [None] * 5,
        "review_creation_date":   [BASE] * 5,
        "review_answer_timestamp":[BASE + pd.Timedelta(days=1)] * 5,
    })


# ── Step 4 KB builder fixture ─────────────────────────────────────────────────

@pytest.fixture
def enriched_df(minimal_master):
    """
    Fully enriched DataFrame produced by running the real enricher on minimal_master.
    Used by all step4 KB builder tests.
    """
    from preprocessing.step3_enrich_master import enrich_master_dataset
    return enrich_master_dataset(minimal_master)

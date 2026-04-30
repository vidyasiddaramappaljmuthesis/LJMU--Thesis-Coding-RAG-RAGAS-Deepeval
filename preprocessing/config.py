"""
Central configuration: file paths, sampling limits, and constants.
"""
from pathlib import Path

# ── Directory layout ────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent
DATA_RAW       = BASE_DIR / "dataset" / "raw"
DATA_PROCESSED = BASE_DIR / "dataset" / "processed"
DATA_KB        = BASE_DIR / "dataset" / "knowledge_base"
DATA_GOLDEN    = BASE_DIR / "dataset" / "golden"

# ── Raw file registry ───────────────────────────────────────────────────────
RAW_FILES = {
    "customers":          DATA_RAW / "olist_customers_dataset.csv",
    "geolocation":        DATA_RAW / "olist_geolocation_dataset.csv",
    "order_items":        DATA_RAW / "olist_order_items_dataset.csv",
    "payments":           DATA_RAW / "olist_order_payments_dataset.csv",
    "reviews":            DATA_RAW / "olist_order_reviews_dataset.csv",
    "orders":             DATA_RAW / "olist_orders_dataset.csv",
    "products":           DATA_RAW / "olist_products_dataset.csv",
    "sellers":            DATA_RAW / "olist_sellers_dataset.csv",
    "category_translation": DATA_RAW / "product_category_name_translation.csv",
}

# ── KB sampling ─────────────────────────────────────────────────────────────
ORDER_KB_SAMPLE   = 10_000  # number of orders sampled into kb_order_documents.json
TOP_SELLERS_FOR_KB = None  # None = all sellers; set int to cap

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

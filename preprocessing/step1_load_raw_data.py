"""
Step 1 – Load all raw CSV files into a dictionary of DataFrames.

Datetime columns are parsed automatically; utf-8-sig encoding strips the BOM
that appears in the category-translation file.
"""
import logging
from typing import Dict

import pandas as pd

from .config import RAW_FILES

logger = logging.getLogger(__name__)

# Columns that must be parsed as datetime per dataset
_DATETIME_COLS: Dict[str, list] = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "reviews":     ["review_creation_date", "review_answer_timestamp"],
    "order_items": ["shipping_limit_date"],
}


def load_dataset(name: str) -> pd.DataFrame:
    """Load a single CSV by registry name, parse datetimes, and return a DataFrame."""
    path = RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    logger.info(f"  Loading {name:<25} from {path.name}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    for col in _DATETIME_COLS.get(name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    logger.info(f"  -> {name:<25} {len(df):>8,} rows  x  {df.shape[1]} cols")
    return df


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """Load every file in RAW_FILES and return a name→DataFrame mapping."""
    logger.info("Loading all raw datasets...")
    datasets = {name: load_dataset(name) for name in RAW_FILES}
    logger.info(f"All {len(datasets)} datasets loaded successfully.")
    return datasets

"""Unit tests for preprocessing/config.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.config import (
    BASE_DIR,
    DATA_RAW,
    DATA_PROCESSED,
    DATA_KB,
    DATA_GOLDEN,
    RAW_FILES,
    ORDER_KB_SAMPLE,
    TOP_SELLERS_FOR_KB,
    RANDOM_SEED,
)


def test_base_dir_is_path():
    assert isinstance(BASE_DIR, Path)


def test_data_dirs_are_under_base():
    assert DATA_RAW.parent.parent       == BASE_DIR
    assert DATA_PROCESSED.parent.parent == BASE_DIR
    assert DATA_KB.parent.parent        == BASE_DIR
    assert DATA_GOLDEN.parent.parent    == BASE_DIR


def test_data_dir_names():
    assert DATA_RAW.name       == "raw"
    assert DATA_PROCESSED.name == "processed"
    assert DATA_KB.name        == "knowledge_base"
    assert DATA_GOLDEN.name    == "golden"


def test_raw_files_has_nine_keys():
    expected = {
        "customers", "geolocation", "order_items", "payments",
        "reviews", "orders", "products", "sellers", "category_translation",
    }
    assert set(RAW_FILES.keys()) == expected


def test_raw_files_all_path_objects():
    for key, path in RAW_FILES.items():
        assert isinstance(path, Path), f"{key} value is not a Path"


def test_order_kb_sample():
    assert ORDER_KB_SAMPLE == 10_000


def test_random_seed():
    assert RANDOM_SEED == 42


def test_top_sellers_default_none():
    assert TOP_SELLERS_FOR_KB is None

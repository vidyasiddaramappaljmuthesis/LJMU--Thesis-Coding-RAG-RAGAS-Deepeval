from __future__ import annotations

"""
Olist E-commerce RAG — Data Preparation Script
===============================================
Scope: preprocessing only (load -> join -> enrich -> KB -> golden dataset).
RAG pipelines and evaluation runners are separate scripts in rag/ and evaluation/.

Usage
-----
    python -m preprocessing.data_preparation                 # full preprocessing pipeline
    python -m preprocessing.data_preparation --steps enrich  # produce master CSVs only
    python -m preprocessing.data_preparation --steps kb      # load enriched CSV, build KB JSON files
    python -m preprocessing.data_preparation --steps golden  # load enriched CSV + KB, build golden Q&A
    python -m preprocessing.data_preparation --steps all     # same as default

Steps that load from disk (kb / golden) skip re-running the expensive join+enrich
so iteration is fast when only the KB or golden dataset needs to be rebuilt.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Olist RAG data preparation pipeline")
    p.add_argument(
        "--steps",
        choices=["enrich", "kb", "golden", "all"],
        default="all",
        help=(
            "enrich  -> produce final_olist_master_enriched.csv\n"
            "kb      -> load enriched CSV, build all KB JSON files\n"
            "golden  -> load enriched CSV + KB, build golden_dataset.csv\n"
            "all     -> full pipeline (default)"
        ),
    )
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_enriched(data_processed: Path) -> "pd.DataFrame":
    """Load the pre-built enriched CSV (fast path for kb / golden steps)."""
    import pandas as pd
    path = data_processed / "final_olist_master_enriched.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run with --steps enrich first."
        )
    logger.info(f"Loading enriched dataset from {path.name} ...")
    df = pd.read_csv(path, low_memory=False, parse_dates=[
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ])
    logger.info(f"  Loaded: {len(df):,} rows x {df.shape[1]} cols")
    return df


def _run_enrich(data_processed: Path) -> "pd.DataFrame":
    """Steps 1-3: load raw CSVs, join, enrich, save both master CSVs."""
    from .step1_load_raw_data import load_all_datasets
    from .step2_join_datasets import create_master_dataset
    from .step3_enrich_master import enrich_master_dataset

    logger.info("[Step 1/3]  Loading raw CSV files")
    datasets = load_all_datasets()

    logger.info("[Step 2/3]  Joining all datasets into master")
    master = create_master_dataset(datasets)
    out_master = data_processed / "final_olist_master.csv"
    master.to_csv(out_master, index=False)
    logger.info(f"  Saved -> {out_master.name}  ({len(master):,} rows x {master.shape[1]} cols)")

    logger.info("[Step 3/3]  Enriching with derived features")
    enriched = enrich_master_dataset(master)
    out_enriched = data_processed / "final_olist_master_enriched.csv"
    enriched.to_csv(out_enriched, index=False)
    logger.info(f"  Saved -> {out_enriched.name}  ({len(enriched):,} rows x {enriched.shape[1]} cols)")
    return enriched


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(steps: str = "all") -> None:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("  Olist E-commerce RAG Data Preparation Pipeline")
    logger.info(f"  Mode: --steps {steps}")
    logger.info("=" * 60)

    from .config import DATA_PROCESSED, DATA_KB, DATA_GOLDEN

    for d in (DATA_PROCESSED, DATA_KB, DATA_GOLDEN):
        d.mkdir(parents=True, exist_ok=True)

    # ── Obtain enriched DataFrame ─────────────────────────────────────────────
    if steps == "enrich" or steps == "all":
        enriched = _run_enrich(DATA_PROCESSED)
    else:
        # kb / golden: load from disk (skip reprocessing)
        enriched = _load_enriched(DATA_PROCESSED)

    if steps == "enrich":
        _finish(t0)
        return

    # ── Step 4: Knowledge Base ────────────────────────────────────────────────
    from .step4_build_knowledge_base import build_knowledge_base

    logger.info("[Step 4]  Building knowledge-base documents (6 layers)")
    kb_docs = build_knowledge_base(enriched)

    if steps == "kb":
        _finish(t0)
        return

    # ── Step 5: Golden Dataset ────────────────────────────────────────────────
    from .step5_build_golden_dataset import generate_golden_dataset

    logger.info("[Step 5]  Generating golden evaluation dataset")
    golden_df = generate_golden_dataset(enriched, kb_docs)

    out_golden = DATA_GOLDEN / "golden_dataset.csv"
    golden_df.to_csv(out_golden, index=False)
    logger.info(f"  Saved -> {out_golden.name}  ({len(golden_df):,} questions)")

    _finish(t0)


def _finish(t0: float) -> None:
    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"  Pipeline complete  ({elapsed:.1f}s)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(steps=args.steps)

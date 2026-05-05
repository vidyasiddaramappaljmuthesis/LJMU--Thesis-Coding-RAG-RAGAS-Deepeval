"""
Root-level entry point for the data preparation pipeline.

Delegates to preprocessing.data_preparation so the script can be invoked
from the project root without needing -m or manual sys.path manipulation.

Usage:
    python data_preparation.py                  # full pipeline
    python data_preparation.py --steps enrich   # Steps 1-3 only
    python data_preparation.py --steps kb       # Step 4 only
    python data_preparation.py --steps golden   # Step 5 only
    python data_preparation.py --steps all      # full pipeline (explicit)
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so package imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing.data_preparation import parse_args, run_pipeline

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(steps=args.steps)

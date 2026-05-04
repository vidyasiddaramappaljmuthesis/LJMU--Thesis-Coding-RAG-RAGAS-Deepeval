# -*- coding: utf-8 -*-
"""
Quick end-to-end smoke-test for the Reranking RAG pipeline.
Runs 4 representative e-commerce queries and prints the full result.

Usage:
    python test_reranking_e2e.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # avoid OpenMP double-init on Windows

import sys
import io
import time
import textwrap
from pathlib import Path

# Force UTF-8 stdout so Windows cp1252 never crashes on accented chars in docs
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

TEST_QUERIES = [
    "What is the average delivery time for orders in Sao Paulo?",
    "Which product categories have the highest review scores?",
    "Who are the top sellers by total revenue?",
    "What percentage of orders are delivered late?",
]

W = 88


def _sep(char="="):
    return char * W


def _header(title):
    pad = max(0, W - len(title) - 4)
    return "+-- " + title + " " + "-" * pad + "+"


def _footer():
    return "+" + "-" * W + "+"


def main():
    print()
    print(_sep("="))
    print("  RERANKING RAG -- END-TO-END SMOKE TEST -- " + str(len(TEST_QUERIES)) + " queries")
    print("  Pipeline: ChromaDB(top-20) -> CrossEncoder -> top-5 -> Groq LLaMA 3.3 70B")
    print(_sep("="))
    print()

    # ── Step 0: ensure vector store ────────────────────────────────────────────
    print("[Setup] Checking ChromaDB vector store ...")
    try:
        from reranking_rag.implementation.ingestion import get_collection
        col   = get_collection()
        count = col.count()
        if count == 0:
            print("[Setup] Collection is empty -- building index now (~2 min) ...")
            from reranking_rag.implementation.ingestion import build_vector_store
            t0 = time.time()
            build_vector_store()
            print("[Setup] Index built in %.1fs" % (time.time() - t0))
        else:
            print("[Setup] Vector store OK -- %d documents indexed." % count)
    except Exception:
        print("[Setup] Collection not found -- building index now (~2 min) ...")
        from reranking_rag.implementation.ingestion import build_vector_store
        t0 = time.time()
        build_vector_store()
        print("[Setup] Index built in %.1fs" % (time.time() - t0))

    # ── Step 1: import pipeline (cross-encoder loads on first rerank call) ─────
    print("[Setup] Loading pipeline ...")
    from reranking_rag.implementation.pipeline import run_reranking_rag
    print("[Setup] Ready.\n")

    overall_start = time.time()

    for qi, query in enumerate(TEST_QUERIES, 1):
        print(_header("QUERY %d/%d" % (qi, len(TEST_QUERIES))))
        print("|  Q : " + query)
        print("|" + "-" * W)

        t_start = time.time()
        try:
            result  = run_reranking_rag(query)
            elapsed = time.time() - t_start

            # ── Stage 1a: initial retrieval ───────────────────────────────────
            initial = result["initial_docs"]
            print("|  +- Stage 1a : ChromaDB initial retrieval  (%d candidates)" % len(initial))
            for i, d in enumerate(initial[:3], 1):
                dist     = d.get("distance", 0)
                doc_type = d.get("metadata", {}).get("document_type", "?")
                snip     = d["text"].replace("\n", " ")[:68]
                print("|  |  [%d] dist=%.4f  type=%-12s  %s ..." % (i, dist, doc_type, snip))
            if len(initial) > 3:
                print("|  |  ... and %d more candidates" % (len(initial) - 3))
            print("|  |")

            # ── Stage 1b: reranked top-k ──────────────────────────────────────
            reranked = result["retrieved_docs"]
            print("|  +- Stage 1b : Cross-encoder reranked  top-%d" % len(reranked))
            for i, d in enumerate(reranked, 1):
                rscore   = d.get("rerank_score", 0)
                dist     = d.get("distance", 0)
                doc_type = d.get("metadata", {}).get("document_type", "?")
                snip     = d["text"].replace("\n", " ")[:60]
                print("|  |  [%d] rerank=%+.4f  dist=%.4f  type=%-12s  %s ..." % (
                    i, rscore, dist, doc_type, snip))
            print("|  |")

            # ── Generated answer ──────────────────────────────────────────────
            print("|  +- Answer  (%.1fs total)" % elapsed)
            for line in textwrap.wrap(result["answer"], width=W - 7):
                print("|     " + line)
            print("|")
            print("|  STATUS : OK")

        except Exception as exc:
            elapsed = time.time() - t_start
            print("|  STATUS : ERROR after %.1fs -- %s" % (elapsed, exc))

        print(_footer())
        print()

        if qi < len(TEST_QUERIES):
            print("  [sleep] Waiting 5s before next query ...\n")
            time.sleep(5)

    total = time.time() - overall_start
    print(_sep("="))
    print("  All %d queries completed in %.1fs" % (len(TEST_QUERIES), total))
    print(_sep("="))
    print()


if __name__ == "__main__":
    main()

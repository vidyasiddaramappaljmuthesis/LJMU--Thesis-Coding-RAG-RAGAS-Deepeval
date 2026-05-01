"""
Hybrid RAG — single entry point.

  python run_hybrid_rag.py          # auto-setup + interactive Q&A
  python run_hybrid_rag.py --ingest # force re-index both stores
"""
import sys
import chromadb

from hybrid_rag.implementation.config import CHROMA_DB_PATH, COLLECTION_NAME, BM25_INDEX_PATH
from hybrid_rag.implementation.ingestion import build_all, build_chroma, build_bm25
from hybrid_rag.implementation.pipeline import run_hybrid_rag


def _chroma_exists() -> bool:
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        return client.get_collection(COLLECTION_NAME).count() > 0
    except Exception:
        return False


def _bm25_exists() -> bool:
    return BM25_INDEX_PATH.exists()


def _ensure_setup() -> None:
    chroma_ok = _chroma_exists()
    bm25_ok   = _bm25_exists()

    if chroma_ok and bm25_ok:
        print("[Setup] ChromaDB + BM25 index found — skipping ingestion.")
        return

    if not chroma_ok:
        print("[Setup] ChromaDB not found. Building semantic index...")
        build_chroma()

    if not bm25_ok:
        print("[Setup] BM25 index not found. Building keyword index...")
        build_bm25()

    print("[Setup] Done.\n")


def _interactive() -> None:
    print("=" * 60)
    print("  E-Commerce Hybrid RAG  |  LLaMA 3.3 70B via Groq")
    print("  Keyword  : BM25 (rank_bm25)")
    print("  Semantic : sentence-transformers/all-MiniLM-L6-v2")
    print("  Fusion   : Reciprocal Rank Fusion (RRF k=60)")
    print("  VectorDB : ChromaDB (cosine)")
    print("  Type 'exit' to quit.")
    print("=" * 60)
    print()

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        print()
        result = run_hybrid_rag(query)

        print(f"Answer:\n{result['answer']}")
        print()
        print(f"Top-{len(result['retrieved_docs'])} docs after RRF fusion:")
        for doc in result["retrieved_docs"]:
            print(
                f"  [{doc['id']}]  rrf={doc['rrf_score']:.5f}  "
                f"type={doc['metadata'].get('document_type', 'n/a')}"
            )
        print()


if __name__ == "__main__":
    if "--ingest" in sys.argv:
        print("[Setup] Force re-indexing both stores...")
        build_all()
    else:
        _ensure_setup()
        _interactive()

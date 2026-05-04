"""
Multi-Query RAG — single entry point.

  python -m multiquery_rag.run_multiquery_rag          # auto-setup + interactive Q&A
  python -m multiquery_rag.run_multiquery_rag --ingest # force re-index only (no Q&A)
"""
import sys
import chromadb

from .implementation.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    NUM_QUERY_VARIANTS,
    PER_QUERY_TOP_K,
    FINAL_TOP_K,
)
from .implementation.ingestion import build_vector_store
from .implementation.pipeline import run_multiquery_rag


def _vector_store_exists() -> bool:
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        col = client.get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False


def _ensure_vector_store():
    if _vector_store_exists():
        print("[Setup] Vector store already exists — skipping ingestion.")
    else:
        print("[Setup] Vector store not found. Building now (this runs once)...")
        build_vector_store()
        print("[Setup] Done.\n")


def _interactive():
    print("=" * 65)
    print("  E-Commerce Multi-Query RAG  |  LLaMA 3.3 70B via Groq")
    print("  Embedding    : sentence-transformers/all-MiniLM-L6-v2")
    print(f"  Query variants : {NUM_QUERY_VARIANTS}  |  Per-variant k : {PER_QUERY_TOP_K}")
    print(f"  Final top-k  : {FINAL_TOP_K}  (after RRF fusion)")
    print("  Vector DB    : ChromaDB (cosine)")
    print("  Type 'exit' to quit.")
    print("=" * 65)
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
        result = run_multiquery_rag(query)

        print(f"Answer:\n{result['answer']}")
        print()

        print(f"Expanded queries ({NUM_QUERY_VARIANTS} variants):")
        for i, q in enumerate(result["expanded_queries"], 1):
            tag = " [original]" if i == 1 else ""
            print(f"  {i}. {q}{tag}")
        print()

        total_candidates = sum(len(v) for v in result["query_results"].values())
        unique_after_rrf = len(result["retrieved_docs"])
        print(
            f"Fused documents (top-{unique_after_rrf} from RRF, "
            f"~{total_candidates} raw candidates across {NUM_QUERY_VARIANTS} queries):"
        )
        for doc in result["retrieved_docs"]:
            print(
                f"  [{doc['id']}]  "
                f"rrf_score={doc.get('rrf_score', 0):.6f}  "
                f"dist={doc.get('distance', 0):.4f}  "
                f"type={doc['metadata'].get('document_type', 'n/a')}"
            )
        print()


if __name__ == "__main__":
    if "--ingest" in sys.argv:
        print("[Setup] Force re-indexing...")
        build_vector_store()
        print("[Setup] Done.")
    else:
        _ensure_vector_store()
        _interactive()

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent

KB_ALL_DOCS    = BASE_DIR / "dataset" / "knowledge_base" / "kb_all_documents.json"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "ecommerce_kb"
BM25_INDEX_PATH = BASE_DIR / "dataset" / "bm25_index" / "bm25_index.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_raw = os.getenv("GROQ_API_KEYS", "")
GROQ_API_KEYS = [k.strip() for k in _raw.split(",") if k.strip()]
if not GROQ_API_KEYS:
    raise EnvironmentError(
        "GROQ_API_KEYS is not set. Copy .env.example to .env and add your keys."
    )

GROQ_MODEL = "llama-3.3-70b-versatile"

# How many results each method fetches before fusion
SEMANTIC_TOP_K = 10
KEYWORD_TOP_K  = 10
# Final docs sent to LLM after RRF
FINAL_TOP_K = 5
RRF_K = 60          # RRF constant — higher = less sensitive to rank differences

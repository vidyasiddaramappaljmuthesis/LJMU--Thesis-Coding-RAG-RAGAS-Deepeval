"""
Documentation generator for Naive RAG Pipeline.
Produces:  docs/Naive_RAG_Documentation.pdf
Run:       python naive_rag/generate_docs.py
"""
import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).parent.parent / "docs"
PDF_PATH = DOCS_DIR / "Naive_RAG_Documentation.pdf"

# ── Colours ───────────────────────────────────────────────────────────────────
C_DARK_BLUE  = colors.HexColor("#1A237E")
C_ACCENT     = colors.HexColor("#1565C0")
C_LIGHT_BLUE = colors.HexColor("#E3F2FD")
C_TEAL       = colors.HexColor("#00695C")
C_ORANGE     = colors.HexColor("#E65100")
C_HEADER_BG  = colors.HexColor("#1A237E")
C_ROW_EVEN   = colors.HexColor("#F5F5F5")
C_ROW_ODD    = colors.white
C_CODE_BG    = colors.HexColor("#F8F8F8")
C_CODE_BORD  = colors.HexColor("#CCCCCC")
C_BG         = colors.HexColor("#E3F2FD")
C_BD         = colors.HexColor("#1565C0")

W, H = A4


# ── Styles ────────────────────────────────────────────────────────────────────
def _styles():
    s = {}
    s["title"] = ParagraphStyle("T", fontSize=26, leading=34, alignment=TA_CENTER,
        textColor=colors.white, spaceAfter=6, fontName="Helvetica-Bold")
    s["subtitle"] = ParagraphStyle("ST", fontSize=13, leading=18, alignment=TA_CENTER,
        textColor=colors.HexColor("#BBDEFB"), spaceAfter=4, fontName="Helvetica")
    s["h1"] = ParagraphStyle("H1", fontSize=18, leading=24, spaceBefore=18, spaceAfter=8,
        textColor=C_DARK_BLUE, fontName="Helvetica-Bold")
    s["h2"] = ParagraphStyle("H2", fontSize=14, leading=20, spaceBefore=14, spaceAfter=6,
        textColor=C_ACCENT, fontName="Helvetica-Bold")
    s["h3"] = ParagraphStyle("H3", fontSize=12, leading=17, spaceBefore=10, spaceAfter=4,
        textColor=C_TEAL, fontName="Helvetica-Bold")
    s["body"] = ParagraphStyle("B", fontSize=10, leading=15, spaceAfter=6,
        textColor=colors.HexColor("#212121"), fontName="Helvetica", alignment=TA_JUSTIFY)
    s["body_l"] = ParagraphStyle("BL", fontSize=10, leading=15, spaceAfter=4,
        textColor=colors.HexColor("#212121"), fontName="Helvetica")
    s["bullet"] = ParagraphStyle("BU", fontSize=10, leading=15, spaceAfter=3,
        textColor=colors.HexColor("#212121"), fontName="Helvetica", leftIndent=16, bulletIndent=6)
    s["code"] = ParagraphStyle("C", fontSize=8.5, leading=13, spaceAfter=0,
        textColor=colors.HexColor("#212121"), fontName="Courier", leftIndent=8)
    s["code_c"] = ParagraphStyle("CC", fontSize=8.5, leading=13, spaceAfter=0,
        textColor=colors.HexColor("#5D6D7E"), fontName="Courier-Oblique", leftIndent=8)
    s["caption"] = ParagraphStyle("CAP", fontSize=8.5, leading=12, spaceAfter=6, spaceBefore=2,
        textColor=colors.HexColor("#757575"), fontName="Helvetica-Oblique", alignment=TA_CENTER)
    return s

ST = _styles()


# ── Helpers ───────────────────────────────────────────────────────────────────
def sp(h=0.3):  return Spacer(1, h * cm)
def hr(c=C_ACCENT, t=0.5): return HRFlowable(width="100%", thickness=t, color=c, spaceAfter=6, spaceBefore=4)

def h1(txt):
    return [sp(0.3), hr(C_DARK_BLUE, 1.5), Paragraph(txt, ST["h1"]), hr(C_ACCENT, 0.5)]
def h2(txt): return [sp(0.2), Paragraph(txt, ST["h2"])]
def h3(txt): return [sp(0.1), Paragraph(txt, ST["h3"])]
def body(t):   return Paragraph(t, ST["body"])
def body_l(t): return Paragraph(t, ST["body_l"])

def bullet(t, level=1):
    st = ParagraphStyle(f"B{level}", parent=ST["bullet"],
                        leftIndent=16*level, bulletIndent=16*level-10)
    return Paragraph(f"• {t}", st)

def code_block(lines):
    rows = [[Paragraph(ln, ST["code_c"] if cmt else ST["code"])] for ln, cmt in lines]
    tbl = Table(rows, colWidths=[16.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_CODE_BG),
        ("BOX",           (0,0),(-1,-1), 0.5, C_CODE_BORD),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]))
    return tbl

def info_box(title, lines, bg=C_BG, border=C_BD):
    rows = [[Paragraph(f"<b>{title}</b>", ST["body_l"])]] + \
           [[Paragraph(f"  {l}", ST["body_l"])] for l in lines]
    tbl = Table(rows, colWidths=[16.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), bg),
        ("BOX",           (0,0),(-1,-1), 1.0, border),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("BACKGROUND",    (0,0),(-1, 0), border),
        ("TEXTCOLOR",     (0,0),(-1, 0), colors.white),
    ]))
    return tbl

def flow_box(steps, bg=C_BG, border=C_BD):
    rows = [[Paragraph(s, ST["body_l"])] for s in steps]
    tbl = Table(rows, colWidths=[16.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), bg),
        ("BOX",           (0,0),(-1,-1), 1.2, border),
        ("LEFTPADDING",   (0,0),(-1,-1), 14),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LINEBELOW",     (0,0),(-1,-2), 0.3, colors.HexColor("#CCCCCC")),
    ]))
    return tbl

def data_table(headers, rows, col_widths=None):
    data = [headers] + rows
    n = len(headers)
    if col_widths is None:
        col_widths = [16.5*cm/n]*n
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",    (0,0),(-1, 0), C_HEADER_BG),
        ("TEXTCOLOR",     (0,0),(-1, 0), colors.white),
        ("FONTNAME",      (0,0),(-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1, 0), 9),
        ("ALIGN",         (0,0),(-1,-1), "LEFT"),
        ("FONTSIZE",      (0,1),(-1,-1), 9),
        ("FONTNAME",      (0,1),(-1,-1), "Helvetica"),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ("RIGHTPADDING",  (0,0),(-1,-1), 6),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
    ]
    for i in range(1, len(data)):
        style.append(("BACKGROUND", (0,i),(-1,i), C_ROW_EVEN if i%2==0 else C_ROW_ODD))
    tbl.setStyle(TableStyle(style))
    return tbl


# ── Cover ─────────────────────────────────────────────────────────────────────
def cover_page():
    e = [sp(2.5)]
    banner = Table([
        [Paragraph("Naive RAG Pipeline", ST["title"])],
        [Paragraph("End-to-End Implementation Documentation", ST["subtitle"])],
        [Paragraph("sentence-transformers/all-MiniLM-L6-v2  •  ChromaDB  •  Groq LLaMA 3.3 70B", ST["subtitle"])],
    ], colWidths=[17*cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_DARK_BLUE),
        ("TOPPADDING",    (0,0),(-1,-1), 16),
        ("BOTTOMPADDING", (0,0),(-1,-1), 16),
        ("LEFTPADDING",   (0,0),(-1,-1), 20),
        ("RIGHTPADDING",  (0,0),(-1,-1), 20),
        ("BOX",           (0,0),(-1,-1), 2, C_ACCENT),
    ]))
    e.append(banner)
    e.append(sp(1.2))

    meta = Table([
        [Paragraph(f"<b>Date:</b>  {datetime.datetime.now().strftime('%B %Y')}", ST["body_l"])],
        [Paragraph("<b>Pipeline:</b>  Naive RAG — pure semantic vector retrieval", ST["body_l"])],
        [Paragraph("<b>Embedding:</b>  sentence-transformers/all-MiniLM-L6-v2 (384-dim, cosine)", ST["body_l"])],
        [Paragraph("<b>Vector DB:</b>  ChromaDB PersistentClient", ST["body_l"])],
        [Paragraph("<b>LLM:</b>  Groq — llama-3.3-70b-versatile", ST["body_l"])],
        [Paragraph("<b>API Keys:</b>  13 Groq keys with automatic round-robin rotation", ST["body_l"])],
        [Paragraph("<b>Dataset:</b>  Olist Brazilian E-Commerce — 13,225 KB documents", ST["body_l"])],
    ], colWidths=[17*cm])
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_LIGHT_BLUE),
        ("BOX",           (0,0),(-1,-1), 1, C_ACCENT),
        ("LEFTPADDING",   (0,0),(-1,-1), 16),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
    ]))
    e.append(meta)
    e.append(sp(1.5))

    stats = [("5", "Python\nFiles"), ("384", "Embedding\nDimensions"),
             ("13,225", "KB\nDocuments"), ("13", "Groq API\nKeys")]
    cells = []
    for val, lbl in stats:
        c = Table([
            [Paragraph(f"<b>{val}</b>", ParagraphStyle("SV", fontSize=20,
              textColor=colors.white, alignment=TA_CENTER, fontName="Helvetica-Bold"))],
            [Paragraph(lbl, ParagraphStyle("SL", fontSize=8,
              textColor=colors.HexColor("#BBDEFB"), alignment=TA_CENTER,
              fontName="Helvetica", leading=11))],
        ], colWidths=[3.8*cm])
        c.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),C_DARK_BLUE),
            ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
            ("BOX",(0,0),(-1,-1),1,C_ACCENT)]))
        cells.append(c)
    sr = Table([cells], colWidths=[4.0*cm]*4)
    sr.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))
    e.append(sr)
    e.append(PageBreak())
    return e


# ── Sections ──────────────────────────────────────────────────────────────────
def sec_overview():
    e = []
    e += h1("1. What is Naive RAG?")
    e.append(body(
        "Naive RAG is the simplest form of Retrieval-Augmented Generation. The user query is "
        "embedded into a vector and compared against pre-indexed document vectors in ChromaDB "
        "using cosine similarity. The most similar documents are retrieved and passed as context "
        "to an LLM which generates the final grounded answer. There is no keyword search, no "
        "re-ranking, and no query transformation — just embed, retrieve, generate."
    ))
    e.append(sp())
    e += h2("1.1 Core Idea")
    e.append(flow_box([
        "  User Query  →  Embed (all-MiniLM-L6-v2)  →  384-dim Query Vector",
        "      ↓",
        "  ChromaDB Cosine Similarity Search  →  Top-5 Documents",
        "      ↓",
        "  Prompt = System Message + [Doc 1] ... [Doc 5] + Question",
        "      ↓",
        "  Groq LLaMA 3.3 70B  →  Final Answer",
    ]))
    e.append(sp())
    e += h2("1.2 When to Use Naive RAG")
    for item in [
        "Questions about trends, patterns, and general performance summaries.",
        "Queries where meaning matters more than exact keyword matches.",
        "Fast prototyping — minimal setup, single index, simple query path.",
        "Baseline for comparing against more complex RAG variants.",
    ]:
        e.append(bullet(item))
    e.append(sp())
    e += h2("1.3 Limitations")
    e.append(data_table(
        ["Limitation", "Example", "Solved By"],
        [
            ["Misses exact keyword matches",
             "Order ID 'abc123' may not rank #1 if semantically similar IDs exist",
             "Hybrid RAG (adds BM25)"],
            ["Single retrieval signal",
             "Only semantic similarity — no term frequency weighting",
             "Hybrid RAG (BM25 + RRF)"],
            ["No query expansion",
             "Short queries may retrieve off-topic docs",
             "HyDE RAG / Multi-Query RAG"],
        ],
        col_widths=[4*cm, 6*cm, 6.5*cm],
    ))
    return e


def sec_libraries():
    e = []
    e += h1("2. Libraries Used")
    e.append(data_table(
        ["Library", "Version", "Purpose in Naive RAG"],
        [
            ["chromadb",              "≥ 0.5.0",  "Persistent vector database. Stores 384-dim embeddings on disk. Provides cosine similarity search via HNSW index."],
            ["sentence-transformers", "≥ 3.0.0",  "Loads all-MiniLM-L6-v2. Used by ChromaDB's SentenceTransformerEmbeddingFunction to embed documents at index time and queries at retrieval time."],
            ["groq",                  "≥ 0.11.0", "Official Groq Python SDK. Calls llama-3.3-70b-versatile. Raises RateLimitError and AuthenticationError which trigger key rotation."],
        ],
        col_widths=[4*cm, 2.2*cm, 10.3*cm],
    ))
    e.append(sp(0.3))
    e += h2("2.1 Install")
    e.append(code_block([
        ("pip install chromadb sentence-transformers groq", False),
    ]))
    e.append(sp(0.3))
    e += h2("2.2 Embedding Model Details")
    e.append(data_table(
        ["Property", "Value"],
        [
            ["Model name",        "sentence-transformers/all-MiniLM-L6-v2"],
            ["Model size",        "22 MB"],
            ["Output dimensions", "384"],
            ["Max input tokens",  "256 tokens (longer texts are truncated)"],
            ["Similarity metric", "Cosine similarity — ChromaDB configured with hnsw:space=cosine"],
            ["Download",          "Auto-downloaded from HuggingFace Hub on first use"],
            ["Usage",             "ChromaDB's SentenceTransformerEmbeddingFunction handles both indexing and querying"],
        ],
        col_widths=[4.5*cm, 12*cm],
    ))
    return e


def sec_architecture():
    e = []
    e += h1("3. End-to-End Architecture")
    e += h2("3.1 Indexing Phase (runs once)")
    e.append(body("All 13,225 KB documents are embedded and stored. This is a one-time operation persisted to disk."))
    e.append(code_block([
        ("kb_all_documents.json", False),
        ("    [13,225 docs]  { id, text, metadata }", True),
        ("         |", False),
        ("         |  SentenceTransformerEmbeddingFunction(all-MiniLM-L6-v2)", True),
        ("         |  embed doc['text']  →  384-dim vector", True),
        ("         ↓", False),
        ("    ChromaDB PersistentClient", False),
        ("    collection: 'ecommerce_kb'  hnsw:space=cosine", False),
        ("         |", False),
        ("         ↓", False),
        ("    chroma_db/   (persisted to disk)", False),
    ]))
    e.append(sp(0.3))
    e += h2("3.2 Query Phase (every request)")
    e.append(code_block([
        ("User Query (string)", False),
        ("         |", False),
        ("         |  SentenceTransformerEmbeddingFunction  →  384-dim vector", True),
        ("         ↓", False),
        ("    ChromaDB.query(query_texts=[query], n_results=5)", False),
        ("         |  HNSW cosine similarity search across 13,225 vectors", True),
        ("         ↓", False),
        ("    Top-5 Documents  { id, text, metadata, distance }", False),
        ("         |", False),
        ("         |  format: [Document 1]\\ntext\\n\\n[Document 2]\\ntext...", True),
        ("         ↓", False),
        ("    Groq API  (llama-3.3-70b-versatile)", False),
        ("    System: 'Answer using only the provided context.'", False),
        ("    User:   context_block + '\\n\\nQuestion: ' + query", False),
        ("         ↓", False),
        ("    Final Answer (string)", False),
    ]))
    e.append(sp())
    e += h2("3.3 File Dependency Map")
    e.append(data_table(
        ["File", "Imports From", "Used By"],
        [
            ["config.py",    "pathlib, (nothing internal)",       "ingestion, retriever, generator, pipeline"],
            ["ingestion.py", "config.py",                         "pipeline.py, run_naive_rag.py"],
            ["retriever.py", "config.py, ingestion.py",           "pipeline.py"],
            ["generator.py", "config.py",                         "pipeline.py"],
            ["pipeline.py",  "config.py, retriever.py, generator.py", "run_naive_rag.py"],
        ],
        col_widths=[3.5*cm, 6*cm, 7*cm],
    ))
    return e


def sec_file_structure():
    e = []
    e += h1("4. File Structure")
    e.append(code_block([
        ("naive_rag/", False),
        ("|-- __init__.py        # empty — marks the directory as a Python package", False),
        ("|-- config.py          # all constants: paths, API keys, model names, TOP_K=5", False),
        ("|-- ingestion.py       # load JSON → embed → store in ChromaDB (run once)", False),
        ("|-- retriever.py       # embed query → ChromaDB query → return top-k docs", False),
        ("|-- generator.py       # Groq LLaMA 3.3 70B + 13-key rotation logic", False),
        ("|-- pipeline.py        # run_naive_rag(query) — orchestrates retriever + generator", False),
        ("|-- generate_docs.py   # this script — generates Naive_RAG_Documentation.pdf", False),
        ("", False),
        ("run_naive_rag.py       # single entry point: auto-setup + interactive CLI", False),
        ("chroma_db/             # ChromaDB persistent storage (created on first run)", False),
    ]))
    return e


def sec_config():
    e = []
    e += h1("5. naive_rag/config.py")
    e.append(body(
        "Central configuration file. All other modules import from here. "
        "No other file hardcodes paths, keys, or model names."
    ))
    e.append(code_block([
        ("from pathlib import Path", False),
        ("", False),
        ("BASE_DIR        = Path(__file__).parent.parent", False),
        ("", False),
        ("# Paths", True),
        ("KB_ALL_DOCS     = BASE_DIR / 'dataset' / 'knowledge_base' / 'kb_all_documents.json'", False),
        ("CHROMA_DB_PATH  = BASE_DIR / 'chroma_db'", False),
        ("COLLECTION_NAME = 'ecommerce_kb'", False),
        ("", False),
        ("# Models", True),
        ("EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'", False),
        ("GROQ_MODEL      = 'llama-3.3-70b-versatile'", False),
        ("", False),
        ("# 13 Groq API keys — rotated automatically on rate limit / auth error", True),
        ("GROQ_API_KEYS = [", False),
        ("    'gsk_<YOUR_GROQ_API_KEY_1>',", False),
        ("    'gsk_<YOUR_GROQ_API_KEY_2>',", False),
        ("    # ... 11 more keys (total 13)", True),
        ("]", False),
        ("", False),
        ("TOP_K = 5    # number of documents returned to the LLM as context", False),
    ]))
    e.append(sp(0.3))
    e.append(data_table(
        ["Constant", "Value", "Description"],
        [
            ["KB_ALL_DOCS",     "dataset/knowledge_base/kb_all_documents.json", "Source of all 13,225 documents"],
            ["CHROMA_DB_PATH",  "chroma_db/",                   "Where ChromaDB stores vectors on disk"],
            ["COLLECTION_NAME", "ecommerce_kb",                  "ChromaDB collection name (shared with hybrid_rag)"],
            ["EMBEDDING_MODEL", "all-MiniLM-L6-v2",             "384-dim sentence embedding model"],
            ["GROQ_MODEL",      "llama-3.3-70b-versatile",      "Groq LLaMA model ID"],
            ["GROQ_API_KEYS",   "List of 13 strings",           "API keys tried in round-robin on exhaustion"],
            ["TOP_K",           "5",                             "Documents sent to LLM per query"],
        ],
        col_widths=[4*cm, 5*cm, 7.5*cm],
    ))
    return e


def sec_ingestion():
    e = []
    e += h1("6. naive_rag/ingestion.py — Building the Vector Store")
    e.append(body(
        "Reads kb_all_documents.json, embeds every document's text field using "
        "all-MiniLM-L6-v2, and stores the vectors in a persistent ChromaDB collection. "
        "Documents are added in batches of 500 to avoid memory spikes. "
        "After this runs once, the chroma_db/ folder contains everything needed for retrieval."
    ))
    e.append(sp(0.3))
    e += h2("6.1 build_vector_store() — Full Code")
    e.append(code_block([
        ("import json", False),
        ("import chromadb", False),
        ("from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction", False),
        ("from naive_rag.config import KB_ALL_DOCS, CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL", False),
        ("", False),
        ("def build_vector_store(batch_size: int = 500) -> chromadb.Collection:", False),
        ("    with open(KB_ALL_DOCS, 'r', encoding='utf-8') as f:", False),
        ("        docs = json.load(f)         # list of { id, text, metadata }", True),
        ("", False),
        ("    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))", False),
        ("", False),
        ("    # Drop existing collection to allow clean re-index", True),
        ("    try:", False),
        ("        client.delete_collection(COLLECTION_NAME)", False),
        ("    except Exception:", False),
        ("        pass", False),
        ("", False),
        ("    collection = client.create_collection(", False),
        ("        name             = COLLECTION_NAME,", False),
        ("        embedding_function = SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL),", False),
        ("        metadata         = {'hnsw:space': 'cosine'},  # cosine distance metric", True),
        ("    )", False),
        ("", False),
        ("    for i in range(0, len(docs), batch_size):", False),
        ("        batch = docs[i : i + batch_size]", False),
        ("        collection.add(", False),
        ("            ids       = [d['id']       for d in batch],", False),
        ("            documents = [d['text']     for d in batch],  # ChromaDB auto-embeds", True),
        ("            metadatas = [d['metadata'] for d in batch],", False),
        ("        )", False),
        ("        print(f'  Indexed {min(i+batch_size, len(docs))}/{len(docs)}')", False),
        ("    return collection", False),
    ]))
    e.append(sp(0.3))
    e += h2("6.2 get_collection() — Fast Load")
    e.append(body("Used at query time to load the already-built collection without re-embedding."))
    e.append(code_block([
        ("def get_collection() -> chromadb.Collection:", False),
        ("    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))", False),
        ("    return client.get_collection(", False),
        ("        name             = COLLECTION_NAME,", False),
        ("        embedding_function = SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL),", False),
        ("    )", False),
    ]))
    e.append(sp(0.3))
    e.append(data_table(
        ["Design Choice", "Reason"],
        [
            ["batch_size=500",         "Avoids ChromaDB's internal limit; prints progress per batch for large indexes"],
            ["hnsw:space=cosine",       "Normalises vector magnitude so only direction matters — better for semantic text similarity"],
            ["SentenceTransformerEmbeddingFunction", "ChromaDB uses this to embed both documents (add time) and queries (query time) with the same model, ensuring consistent vector space"],
            ["delete_collection first", "Ensures a clean re-index when --ingest is forced; prevents duplicate IDs"],
        ],
        col_widths=[5*cm, 11.5*cm],
    ))
    return e


def sec_retriever():
    e = []
    e += h1("7. naive_rag/retriever.py — Semantic Search")
    e.append(body(
        "At query time the retriever loads the ChromaDB collection, "
        "embeds the query using the same model used at index time, "
        "and returns the top-k documents by cosine similarity. "
        "Lower distance = more similar (0 = identical, 2 = opposite)."
    ))
    e.append(sp(0.3))
    e += h2("7.1 retrieve() — Full Code")
    e.append(code_block([
        ("from typing import Any", False),
        ("from naive_rag.config   import TOP_K", False),
        ("from naive_rag.ingestion import get_collection", False),
        ("", False),
        ("def retrieve(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:", False),
        ("    collection = get_collection()", False),
        ("    results    = collection.query(", False),
        ("        query_texts = [query],   # ChromaDB embeds this with same model", True),
        ("        n_results   = top_k,", False),
        ("    )", False),
        ("", False),
        ("    return [", False),
        ("        {", False),
        ("            'id':       results['ids'][0][i],", False),
        ("            'text':     results['documents'][0][i],", False),
        ("            'metadata': results['metadatas'][0][i],", False),
        ("            'distance': results['distances'][0][i],  # cosine distance 0–2", True),
        ("        }", False),
        ("        for i in range(len(results['ids'][0]))", False),
        ("    ]", False),
    ]))
    e.append(sp(0.3))
    e += h2("7.2 Output Schema")
    e.append(data_table(
        ["Field", "Type", "Description"],
        [
            ["id",       "str",   "Document ID, e.g. 'order_8704f37bae751578'"],
            ["text",     "str",   "Full document text — the context sent to the LLM"],
            ["metadata", "dict",  "Structured fields: document_type, order_id, delivery_status, purchase_month, etc."],
            ["distance", "float", "Cosine distance (0=identical, 2=opposite). Lower = more relevant."],
        ],
        col_widths=[2.5*cm, 1.5*cm, 12.5*cm],
    ))
    return e


def sec_generator():
    e = []
    e += h1("8. naive_rag/generator.py — LLM Generation")
    e.append(body(
        "Calls Groq's LLaMA 3.3 70B API with a RAG prompt. "
        "Implements automatic round-robin rotation across 13 API keys — "
        "if the current key raises RateLimitError or AuthenticationError, "
        "the next key is tried immediately without user interruption."
    ))
    e.append(sp(0.3))
    e += h2("8.1 Key Rotation Logic")
    e.append(code_block([
        ("import groq", False),
        ("from naive_rag.config import GROQ_API_KEYS, GROQ_MODEL", False),
        ("", False),
        ("_current_key_idx: int = 0              # persists across calls", True),
        ("_ROTATABLE = (groq.RateLimitError, groq.AuthenticationError)", False),
        ("", False),
        ("def _call_groq(messages: list[dict], temperature: float = 0.1) -> str:", False),
        ("    global _current_key_idx", False),
        ("    for _ in range(len(GROQ_API_KEYS)):  # try each key at most once", True),
        ("        try:", False),
        ("            client = groq.Groq(api_key=GROQ_API_KEYS[_current_key_idx])", False),
        ("            resp   = client.chat.completions.create(", False),
        ("                model=GROQ_MODEL, messages=messages, temperature=temperature)", False),
        ("            return resp.choices[0].message.content   # success", True),
        ("        except _ROTATABLE as exc:", False),
        ("            print(f'  Key [{_current_key_idx}] exhausted ({type(exc).__name__}). Rotating...')", False),
        ("            _current_key_idx = (_current_key_idx + 1) % len(GROQ_API_KEYS)", False),
        ("    raise RuntimeError('All 13 Groq API keys exhausted.')", False),
    ]))
    e.append(sp(0.3))
    e += h2("8.2 RAG Prompt Construction")
    e.append(code_block([
        ("SYSTEM_PROMPT = (", False),
        ("    'You are a helpful e-commerce data assistant. '", False),
        ("    'Answer questions using only the provided context. '", False),
        ("    'If the answer cannot be found in the context, say so clearly.'", False),
        (")", False),
        ("", False),
        ("def generate(query: str, context_docs: list[dict], temperature: float = 0.1) -> str:", False),
        ("    context_block = '\\n\\n'.join(", False),
        ("        f'[Document {i+1}]\\n{doc[\"text\"]}' for i, doc in enumerate(context_docs)", False),
        ("    )", False),
        ("    messages = [", False),
        ("        {'role': 'system', 'content': SYSTEM_PROMPT},", False),
        ("        {'role': 'user', 'content':", False),
        ("            f'Context:\\n{context_block}\\n\\nQuestion: {query}\\n\\nAnswer:'},", False),
        ("    ]", False),
        ("    return _call_groq(messages, temperature)", False),
    ]))
    e.append(sp(0.3))
    e.append(data_table(
        ["Key Rotation Step", "What Happens"],
        [
            ["1. Start",        "_current_key_idx = 0 on first import; persists in module scope across all calls"],
            ["2. Try current",  "groq.Groq(api_key=GROQ_API_KEYS[_current_key_idx]) then call completions.create()"],
            ["3a. Success",     "Return response immediately; _current_key_idx stays on the working key"],
            ["3b. RateLimitError", "Print warning, advance index: (_current_key_idx + 1) % 13"],
            ["3c. AuthError",   "Same — advance index and retry"],
            ["4. All 13 spent", "Loop exits after 13 iterations; raise RuntimeError"],
        ],
        col_widths=[4*cm, 12.5*cm],
    ))
    return e


def sec_pipeline():
    e = []
    e += h1("9. naive_rag/pipeline.py — Orchestration")
    e.append(body(
        "The pipeline module provides the single public function run_naive_rag(). "
        "It calls retrieve() then generate() and packages the results into a dict "
        "that includes the query, answer, and full retrieved document list for inspection."
    ))
    e.append(sp(0.3))
    e.append(code_block([
        ("from typing import Any", False),
        ("from naive_rag.config    import TOP_K", False),
        ("from naive_rag.retriever import retrieve", False),
        ("from naive_rag.generator import generate", False),
        ("", False),
        ("def run_naive_rag(query: str, top_k: int = TOP_K) -> dict[str, Any]:", False),
        ("    \"\"\"", False),
        ("    End-to-end Naive RAG:", False),
        ("      1. Retrieve top-k documents from ChromaDB (cosine similarity).", False),
        ("      2. Generate an answer with Groq LLaMA 3.3 70B.", False),
        ("    Returns dict: query, answer, retrieved_docs.", False),
        ("    \"\"\"", False),
        ("    docs   = retrieve(query, top_k=top_k)  # list of dicts", True),
        ("    answer = generate(query, docs)          # string", True),
        ("    return {", False),
        ("        'query':          query,", False),
        ("        'answer':         answer,", False),
        ("        'retrieved_docs': docs,   # includes id, text, metadata, distance", True),
        ("    }", False),
    ]))
    e.append(sp(0.3))
    e.append(info_box(
        "Return Value Schema",
        [
            "query          (str)  — the original user question",
            "answer         (str)  — LLM-generated answer grounded in context",
            "retrieved_docs (list) — top-k dicts: { id, text, metadata, distance }",
        ],
    ))
    return e


def sec_entry_point():
    e = []
    e += h1("10. run_naive_rag.py — Entry Point & Auto-Setup")
    e.append(body(
        "The entry point handles setup automatically. On first run it detects that "
        "ChromaDB does not exist and builds it. On all subsequent runs it skips ingestion "
        "and goes straight to the interactive Q&A loop. A --ingest flag forces a rebuild."
    ))
    e.append(sp(0.3))
    e += h2("10.1 Auto-Setup Logic")
    e.append(code_block([
        ("def _vector_store_exists() -> bool:", False),
        ("    try:", False),
        ("        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))", False),
        ("        return client.get_collection(COLLECTION_NAME).count() > 0", False),
        ("    except Exception:", False),
        ("        return False", False),
        ("", False),
        ("def _ensure_vector_store():", False),
        ("    if _vector_store_exists():", False),
        ("        print('[Setup] Vector store already exists — skipping ingestion.')", False),
        ("    else:", False),
        ("        print('[Setup] Building vector store (this runs once)...')", False),
        ("        build_vector_store()", False),
    ]))
    e.append(sp(0.3))
    e += h2("10.2 Commands")
    e.append(data_table(
        ["Command", "Action", "Expected Duration"],
        [
            ["python run_naive_rag.py",
             "Auto-detects ChromaDB.\nBuilds it if missing, then starts Q&A loop.",
             "First run ~5 min\nSubsequent: < 3 sec"],
            ["python run_naive_rag.py --ingest",
             "Force re-indexes all 13,225 documents.\nUse after updating knowledge base.",
             "~5 min"],
        ],
        col_widths=[5.5*cm, 7.5*cm, 3.5*cm],
    ))
    e.append(sp(0.3))
    e += h2("10.3 Interactive Session Example")
    e.append(code_block([
        ("[Setup] Vector store already exists — skipping ingestion.", False),
        ("", False),
        ("============================================================", False),
        ("  E-Commerce Naive RAG  |  LLaMA 3.3 70B via Groq", False),
        ("  Embedding : sentence-transformers/all-MiniLM-L6-v2", False),
        ("  Vector DB : ChromaDB (cosine)", False),
        ("  Type 'exit' to quit.", False),
        ("============================================================", False),
        ("", False),
        ("Question: Which product category had the highest late delivery rate?", False),
        ("", False),
        ("Answer:", False),
        ("Based on the provided context, the office_furniture category had the", False),
        ("highest late delivery rate at 12.3%, followed by ...", False),
        ("", False),
        ("Retrieved documents:", False),
        ("  [category_office_furniture]  distance=0.1823  type=category_level", False),
        ("  [category_computers]         distance=0.2104  type=category_level", False),
        ("  ...", False),
    ]))
    return e


def sec_how_to_run():
    e = []
    e += h1("11. Complete Run Guide")
    e += h2("11.1 Prerequisites")
    e.append(code_block([
        ("# Python 3.10+", True),
        ("pip install chromadb sentence-transformers groq", False),
        ("", False),
        ("# Knowledge base must exist at:", True),
        ("dataset/knowledge_base/kb_all_documents.json", False),
    ]))
    e.append(sp(0.3))
    e += h2("11.2 Step-by-Step")
    e.append(data_table(
        ["Step", "Command", "Output"],
        [
            ["1 — Run (first time)", "python run_naive_rag.py",
             "Builds chroma_db/ automatically,\nthen starts Q&A CLI"],
            ["2 — Ask questions", "Type at the 'Question:' prompt",
             "Answer + retrieved doc IDs with distances"],
            ["3 — Exit", "Type 'exit' or Ctrl+C",
             "Session ends"],
            ["4 — Force re-index", "python run_naive_rag.py --ingest",
             "Drops and rebuilds chroma_db/"],
        ],
        col_widths=[3.5*cm, 6*cm, 7*cm],
    ))
    e.append(sp(0.3))
    e += h2("11.3 File Outputs Created")
    e.append(data_table(
        ["File / Folder", "Created By", "Contents"],
        [
            ["chroma_db/",           "build_vector_store()",  "Persistent ChromaDB: 13,225 vectors + metadata"],
        ],
        col_widths=[5*cm, 4.5*cm, 7*cm],
    ))
    return e


# ── Footer ────────────────────────────────────────────────────────────────────
def on_first_page(canvas, doc): pass

def on_later_pages(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#757575"))
    canvas.drawString(2*cm, 1.2*cm, "Naive RAG Pipeline — End-to-End Implementation")
    canvas.drawRightString(W-2*cm, 1.2*cm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 1.5*cm, W-2*cm, 1.5*cm)
    canvas.restoreState()


# ── Build PDF ─────────────────────────────────────────────────────────────────
def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
        title="Naive RAG Pipeline — End-to-End Documentation",
    )

    story = []
    story += cover_page()

    # TOC
    story += h1("Table of Contents")
    toc = [
        ("1. What is Naive RAG?",                     False),
        ("2. Libraries Used",                          False),
        ("3. End-to-End Architecture",                 False),
        ("4. File Structure",                          False),
        ("5. naive_rag/config.py",                     False),
        ("6. naive_rag/ingestion.py — Vector Store",   False),
        ("7. naive_rag/retriever.py — Semantic Search",False),
        ("8. naive_rag/generator.py — LLM + Key Rotation",False),
        ("9. naive_rag/pipeline.py — Orchestration",   False),
        ("10. run_naive_rag.py — Entry Point",          False),
        ("11. Complete Run Guide",                     False),
    ]
    for title, _ in toc:
        story.append(Paragraph(title, ParagraphStyle(
            "TOC", fontSize=11, leading=17, leftIndent=0,
            textColor=C_DARK_BLUE, fontName="Helvetica-Bold", spaceAfter=3)))
    story.append(PageBreak())

    for sec in [sec_overview, sec_libraries, sec_architecture, sec_file_structure,
                sec_config, sec_ingestion, sec_retriever, sec_generator,
                sec_pipeline, sec_entry_point, sec_how_to_run]:
        story += sec()
        story.append(PageBreak())

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    print(f"PDF written  : {PDF_PATH}")
    print(f"File size    : {PDF_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build_pdf()

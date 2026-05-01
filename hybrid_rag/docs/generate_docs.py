"""
Documentation generator for Hybrid RAG Pipeline.
Produces:  docs/Hybrid_RAG_Documentation.pdf
Run:       python hybrid_rag/generate_docs.py
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
PDF_PATH = DOCS_DIR / "Hybrid_RAG_Documentation.pdf"

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
C_BG         = colors.HexColor("#E8F5E9")
C_BD         = colors.HexColor("#2E7D32")
C_DARK_GREEN = colors.HexColor("#1B5E20")

W, H = A4


# ── Styles ────────────────────────────────────────────────────────────────────
def _styles():
    s = {}
    s["title"] = ParagraphStyle("T", fontSize=26, leading=34, alignment=TA_CENTER,
        textColor=colors.white, spaceAfter=6, fontName="Helvetica-Bold")
    s["subtitle"] = ParagraphStyle("ST", fontSize=13, leading=18, alignment=TA_CENTER,
        textColor=colors.HexColor("#C8E6C9"), spaceAfter=4, fontName="Helvetica")
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
        [Paragraph("Hybrid RAG Pipeline", ST["title"])],
        [Paragraph("End-to-End Implementation Documentation", ST["subtitle"])],
        [Paragraph("BM25 Keyword  +  Semantic Search  →  Reciprocal Rank Fusion  →  Groq LLaMA 3.3 70B", ST["subtitle"])],
    ], colWidths=[17*cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_DARK_GREEN),
        ("TOPPADDING",    (0,0),(-1,-1), 16),
        ("BOTTOMPADDING", (0,0),(-1,-1), 16),
        ("LEFTPADDING",   (0,0),(-1,-1), 20),
        ("RIGHTPADDING",  (0,0),(-1,-1), 20),
        ("BOX",           (0,0),(-1,-1), 2, C_BD),
    ]))
    e.append(banner)
    e.append(sp(1.2))

    meta = Table([
        [Paragraph(f"<b>Date:</b>  {datetime.datetime.now().strftime('%B %Y')}", ST["body_l"])],
        [Paragraph("<b>Pipeline:</b>  Hybrid RAG — BM25 keyword + semantic vector + RRF fusion", ST["body_l"])],
        [Paragraph("<b>Keyword Index:</b>  BM25Okapi (rank_bm25), tokenised, persisted as .pkl", ST["body_l"])],
        [Paragraph("<b>Semantic Index:</b>  sentence-transformers/all-MiniLM-L6-v2 + ChromaDB", ST["body_l"])],
        [Paragraph("<b>Fusion:</b>  Reciprocal Rank Fusion (RRF, k=60)", ST["body_l"])],
        [Paragraph("<b>LLM:</b>  Groq — llama-3.3-70b-versatile", ST["body_l"])],
        [Paragraph("<b>API Keys:</b>  13 Groq keys with automatic round-robin rotation", ST["body_l"])],
        [Paragraph("<b>Dataset:</b>  Olist Brazilian E-Commerce — 13,225 KB documents", ST["body_l"])],
    ], colWidths=[17*cm])
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#F1F8E9")),
        ("BOX",           (0,0),(-1,-1), 1, C_BD),
        ("LEFTPADDING",   (0,0),(-1,-1), 16),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
    ]))
    e.append(meta)
    e.append(sp(1.5))

    stats = [("2", "Indexes\n(BM25+Vector)"), ("60", "RRF\nConstant k"),
             ("13,225", "KB\nDocuments"), ("13", "Groq API\nKeys")]
    cells = []
    for val, lbl in stats:
        c = Table([
            [Paragraph(f"<b>{val}</b>", ParagraphStyle("SV", fontSize=20,
              textColor=colors.white, alignment=TA_CENTER, fontName="Helvetica-Bold"))],
            [Paragraph(lbl, ParagraphStyle("SL", fontSize=8,
              textColor=colors.HexColor("#C8E6C9"), alignment=TA_CENTER,
              fontName="Helvetica", leading=11))],
        ], colWidths=[3.8*cm])
        c.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),C_DARK_GREEN),
            ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
            ("BOX",(0,0),(-1,-1),1,C_BD)]))
        cells.append(c)
    sr = Table([cells], colWidths=[4.0*cm]*4)
    sr.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))
    e.append(sr)
    e.append(PageBreak())
    return e


# ── Sections ──────────────────────────────────────────────────────────────────
def sec_overview():
    e = []
    e += h1("1. What is Hybrid RAG?")
    e.append(body(
        "Hybrid RAG improves on Naive RAG by combining two fundamentally different retrieval "
        "signals: BM25 keyword search and semantic vector search. BM25 ranks documents by exact "
        "term frequency and inverse document frequency — it excels when the query contains precise "
        "terms like order IDs, state codes, or category names. Semantic search captures meaning and "
        "context. Reciprocal Rank Fusion (RRF) merges the two ranked lists into one unified "
        "ranking without needing to normalise scores from different systems."
    ))
    e.append(sp())
    e += h2("1.1 Core Idea")
    e.append(flow_box([
        "  INDEXING (once):",
        "    kb_all_documents.json  →  BM25Okapi(tokenised corpus)  →  bm25_index.pkl",
        "    kb_all_documents.json  →  all-MiniLM-L6-v2 embed       →  ChromaDB",
        "",
        "  QUERY TIME:",
        "    User Query",
        "        ├─►  BM25 tokenise + score  →  Keyword Top-10  (rank list A)",
        "        └─►  ChromaDB embed + cosine →  Semantic Top-10 (rank list B)",
        "                          ↓",
        "              Reciprocal Rank Fusion  score(d) = Σ 1/(60 + rank)",
        "                          ↓",
        "              Unified Top-5 Documents",
        "                          ↓",
        "              Groq LLaMA 3.3 70B  →  Final Answer",
    ]))
    e.append(sp())
    e += h2("1.2 Why Hybrid Outperforms Naive RAG")
    e.append(data_table(
        ["Scenario", "Naive RAG", "Hybrid RAG"],
        [
            ["Exact order ID in query",
             "May miss — semantic similarity to similar IDs is unreliable",
             "BM25 scores the exact token match at rank 1"],
            ["State code 'SP' or 'RJ'",
             "May cluster with semantically related content, not exact match",
             "BM25 finds exact abbreviation; semantic confirms context"],
            ["Category name 'health_beauty'",
             "Semantic captures general beauty/health — may retrieve wrong docs",
             "BM25 exact match + semantic boost via RRF"],
            ["General trend question",
             "Semantic handles this perfectly",
             "Same — semantic dominates, BM25 adds nothing, RRF still correct"],
        ],
        col_widths=[4*cm, 6*cm, 6.5*cm],
    ))
    return e


def sec_libraries():
    e = []
    e += h1("2. Libraries Used")
    e.append(data_table(
        ["Library", "Version", "Purpose in Hybrid RAG"],
        [
            ["chromadb",              "≥ 0.5.0",  "Semantic index. Same ChromaDB collection as Naive RAG (shared, no re-embedding). Provides cosine similarity search."],
            ["sentence-transformers", "≥ 3.0.0",  "Loads all-MiniLM-L6-v2 for embedding documents and queries into 384-dim vectors for ChromaDB."],
            ["rank_bm25",             "≥ 0.2.2",  "BM25Okapi keyword index. Tokenises all 13,225 documents, builds an in-memory inverted index, and scores documents against a tokenised query. Pickled to disk."],
            ["groq",                  "≥ 0.11.0", "Official Groq Python SDK. Calls llama-3.3-70b-versatile. RateLimitError and AuthenticationError trigger key rotation."],
        ],
        col_widths=[4*cm, 2.2*cm, 10.3*cm],
    ))
    e.append(sp(0.3))
    e += h2("2.1 Install")
    e.append(code_block([
        ("pip install chromadb sentence-transformers groq rank_bm25", False),
    ]))
    e.append(sp(0.3))
    e += h2("2.2 BM25 vs Semantic — When Each Wins")
    e.append(data_table(
        ["Signal", "Algorithm", "Strength", "Weakness"],
        [
            ["Keyword",  "BM25Okapi",        "Exact term match, order IDs, state codes", "No semantic understanding"],
            ["Semantic", "all-MiniLM-L6-v2", "Meaning, synonyms, paraphrase",            "May miss exact keywords"],
            ["Fused",    "RRF (k=60)",        "Best of both signals",                     "Slightly higher query latency"],
        ],
        col_widths=[2.5*cm, 3.5*cm, 5.5*cm, 5*cm],
    ))
    return e


def sec_architecture():
    e = []
    e += h1("3. End-to-End Architecture")
    e += h2("3.1 Indexing Phase (runs once)")
    e.append(body(
        "Two indexes are built from the same knowledge base. The ChromaDB collection "
        "is shared with Naive RAG — if already built, only BM25 needs to be constructed."
    ))
    e.append(code_block([
        ("kb_all_documents.json  [13,225 docs]", False),
        ("         |", False),
        ("         +─── BM25 branch ──────────────────────────────────────────────────", False),
        ("         |    tokenize(doc['text'])  →  list of lowercase tokens per doc", True),
        ("         |    BM25Okapi(corpus)      →  inverted index (TF-IDF weights)", True),
        ("         |    pickle.dump(bm25 + docs)  →  dataset/bm25_index/bm25_index.pkl", True),
        ("         |", False),
        ("         +─── Semantic branch ─────────────────────────────────────────────", False),
        ("              SentenceTransformerEmbeddingFunction(all-MiniLM-L6-v2)", True),
        ("              embed doc['text']  →  384-dim vector", True),
        ("              ChromaDB.add(ids, documents, metadatas)  →  chroma_db/", True),
    ]))
    e.append(sp(0.3))
    e += h2("3.2 Query Phase (every request)")
    e.append(code_block([
        ("User Query (string)", False),
        ("         |", False),
        ("         +─── BM25 branch ─────────────────────────────────────────────────", False),
        ("         |    tokenize(query)  →  token list", True),
        ("         |    bm25.get_scores(tokens)  →  float[13225]", True),
        ("         |    sort desc  →  Keyword Top-10  [(doc, bm25_score), ...]", True),
        ("         |", False),
        ("         +─── Semantic branch ─────────────────────────────────────────────", False),
        ("              embed query  →  384-dim vector", True),
        ("              ChromaDB.query(n_results=10)  →  Semantic Top-10  [(doc, distance), ...]", True),
        ("         |", False),
        ("         ↓", False),
        ("    RRF Fusion", False),
        ("    for each doc: rrf_score += 1 / (60 + rank)  from keyword list", True),
        ("    for each doc: rrf_score += 1 / (60 + rank)  from semantic list", True),
        ("    sort by rrf_score desc  →  Top-5 Fused Documents", True),
        ("         |", False),
        ("         ↓", False),
        ("    Groq API  (llama-3.3-70b-versatile)", False),
        ("    context = [Doc 1] ... [Doc 5] + Question", True),
        ("         ↓", False),
        ("    Final Answer", False),
    ]))
    return e


def sec_file_structure():
    e = []
    e += h1("4. File Structure")
    e.append(code_block([
        ("hybrid_rag/", False),
        ("|-- __init__.py        # empty — marks the directory as a Python package", False),
        ("|-- config.py          # paths, API keys, SEMANTIC_TOP_K=10, KEYWORD_TOP_K=10, RRF_K=60", False),
        ("|-- ingestion.py       # build ChromaDB (shared) + BM25 index (new)", False),
        ("|-- retriever.py       # BM25 keyword search + semantic search + RRF fusion", False),
        ("|-- generator.py       # Groq LLaMA 3.3 70B + 13-key rotation logic", False),
        ("|-- pipeline.py        # run_hybrid_rag(query) — full end-to-end orchestration", False),
        ("|-- generate_docs.py   # this script — generates Hybrid_RAG_Documentation.pdf", False),
        ("", False),
        ("run_hybrid_rag.py      # single entry point: auto-setup + interactive CLI", False),
        ("chroma_db/             # ChromaDB — SHARED with naive_rag (no re-embedding)", False),
        ("dataset/bm25_index/", False),
        ("└-- bm25_index.pkl     # pickled BM25Okapi + raw doc list", False),
    ]))
    return e


def sec_config():
    e = []
    e += h1("5. hybrid_rag/config.py")
    e.append(body("Extends the naive_rag config with BM25 path and separate top-k constants for each retrieval signal."))
    e.append(code_block([
        ("from pathlib import Path", False),
        ("", False),
        ("BASE_DIR        = Path(__file__).parent.parent", False),
        ("KB_ALL_DOCS     = BASE_DIR / 'dataset' / 'knowledge_base' / 'kb_all_documents.json'", False),
        ("", False),
        ("# ChromaDB — SHARED with naive_rag, same collection name", True),
        ("CHROMA_DB_PATH  = BASE_DIR / 'chroma_db'", False),
        ("COLLECTION_NAME = 'ecommerce_kb'", False),
        ("", False),
        ("# BM25 index — new for hybrid_rag", True),
        ("BM25_INDEX_PATH = BASE_DIR / 'dataset' / 'bm25_index' / 'bm25_index.pkl'", False),
        ("", False),
        ("EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'", False),
        ("GROQ_MODEL      = 'llama-3.3-70b-versatile'", False),
        ("GROQ_API_KEYS   = [ ... ]    # same 13 keys", True),
        ("", False),
        ("# Retrieval parameters", True),
        ("SEMANTIC_TOP_K  = 10   # ChromaDB returns this many before fusion", False),
        ("KEYWORD_TOP_K   = 10   # BM25 returns this many before fusion", False),
        ("FINAL_TOP_K     = 5    # docs sent to LLM after RRF", False),
        ("RRF_K           = 60   # RRF smoothing constant", False),
    ]))
    e.append(sp(0.3))
    e.append(data_table(
        ["Constant", "Value", "Description"],
        [
            ["CHROMA_DB_PATH",  "chroma_db/",         "Shared with naive_rag — no re-embedding if already built"],
            ["BM25_INDEX_PATH", "dataset/bm25_index/bm25_index.pkl", "Pickled BM25 index — built once, loaded at query time"],
            ["SEMANTIC_TOP_K",  "10",                 "Wider net before fusion — more candidates for RRF"],
            ["KEYWORD_TOP_K",   "10",                 "Same — more BM25 candidates improves recall"],
            ["FINAL_TOP_K",     "5",                  "Context window for LLM — top-5 after fusion"],
            ["RRF_K",           "60",                 "Standard value. Higher = gentler rank penalisation"],
        ],
        col_widths=[4*cm, 4.5*cm, 8*cm],
    ))
    return e


def sec_ingestion():
    e = []
    e += h1("6. hybrid_rag/ingestion.py — Building Both Indexes")
    e.append(body(
        "Ingestion builds two separate indexes. The ChromaDB collection is shared with "
        "naive_rag. If naive_rag was already set up, only the BM25 index needs to be built. "
        "Both are built by build_all(); or individually via build_chroma() / build_bm25()."
    ))
    e.append(sp(0.3))
    e += h2("6.1 Tokenisation")
    e.append(body("BM25 requires tokenised text. The tokeniser lowercases and strips punctuation."))
    e.append(code_block([
        ("import re", False),
        ("", False),
        ("def _tokenize(text: str) -> list[str]:", False),
        ("    # lowercase, replace punctuation with space, split on whitespace", True),
        ("    return re.sub(r'[^\\w\\s]', ' ', text.lower()).split()", False),
        ("", False),
        ("# Example:", True),
        ("# _tokenize('Order ID: abc-123, State: SP')", True),
        ("# → ['order', 'id', 'abc', '123', 'state', 'sp']", True),
    ]))
    e.append(sp(0.3))
    e += h2("6.2 build_bm25() — Building and Persisting the Keyword Index")
    e.append(code_block([
        ("import json, pickle", False),
        ("from rank_bm25 import BM25Okapi", False),
        ("from hybrid_rag.config import KB_ALL_DOCS, BM25_INDEX_PATH", False),
        ("", False),
        ("def build_bm25() -> None:", False),
        ("    docs   = json.load(open(KB_ALL_DOCS, encoding='utf-8'))", False),
        ("    corpus = [_tokenize(d['text']) for d in docs]  # list of token lists", True),
        ("    bm25   = BM25Okapi(corpus)                     # builds inverted index in memory", True),
        ("", False),
        ("    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)", False),
        ("    with open(BM25_INDEX_PATH, 'wb') as f:", False),
        ("        pickle.dump({'bm25': bm25, 'docs': docs}, f)  # persist both", True),
        ("    print(f'  BM25 index saved — {len(docs)} documents.')", False),
    ]))
    e.append(sp(0.3))
    e += h2("6.3 get_bm25_index() — Fast Load at Query Time")
    e.append(code_block([
        ("def get_bm25_index() -> tuple[BM25Okapi, list[dict]]:", False),
        ("    with open(BM25_INDEX_PATH, 'rb') as f:", False),
        ("        data = pickle.load(f)", False),
        ("    return data['bm25'], data['docs']  # both needed for keyword search", True),
    ]))
    e.append(sp(0.3))
    e.append(data_table(
        ["Design Choice", "Reason"],
        [
            ["pickle for BM25",         "BM25Okapi is not serialisable to JSON — pickle preserves the full Python object including the inverted index structure"],
            ["Store docs alongside bm25","At search time we need doc text and metadata; avoids re-loading JSON separately"],
            ["Shared ChromaDB",          "naive_rag and hybrid_rag embed once. If chroma_db/ exists, hybrid_rag reuses it — no wasted compute"],
            ["build_all() convenience",  "Single function rebuilds both indexes for a clean re-index (--ingest flag)"],
        ],
        col_widths=[5*cm, 11.5*cm],
    ))
    return e


def sec_retriever():
    e = []
    e += h1("7. hybrid_rag/retriever.py — BM25 + Semantic + RRF")
    e.append(body(
        "The retriever runs BM25 and ChromaDB searches independently, then merges their "
        "ranked lists using Reciprocal Rank Fusion. It returns the fused top-k results "
        "plus the raw keyword and semantic lists for debugging and evaluation."
    ))
    e.append(sp(0.3))
    e += h2("7.1 _keyword_search() — BM25")
    e.append(code_block([
        ("def _keyword_search(query: str, top_k: int) -> list[dict]:", False),
        ("    bm25, docs = get_bm25_index()", False),
        ("    scores     = bm25.get_scores(_tokenize(query))  # float array, len=13225", True),
        ("    top_idx    = sorted(range(len(scores)),", False),
        ("                        key=lambda i: scores[i], reverse=True)[:top_k]", False),
        ("    return [", False),
        ("        {", False),
        ("            'id':         docs[i]['id'],", False),
        ("            'text':       docs[i]['text'],", False),
        ("            'metadata':   docs[i]['metadata'],", False),
        ("            'bm25_score': float(scores[i]),", False),
        ("        }", False),
        ("        for i in top_idx", False),
        ("    ]", False),
    ]))
    e.append(sp(0.3))
    e += h2("7.2 _semantic_search() — ChromaDB")
    e.append(code_block([
        ("def _semantic_search(query: str, top_k: int) -> list[dict]:", False),
        ("    col = get_chroma_collection()           # load shared collection", True),
        ("    res = col.query(query_texts=[query], n_results=top_k)", False),
        ("    return [", False),
        ("        {", False),
        ("            'id':       res['ids'][0][i],", False),
        ("            'text':     res['documents'][0][i],", False),
        ("            'metadata': res['metadatas'][0][i],", False),
        ("            'distance': res['distances'][0][i],   # cosine 0–2", True),
        ("        }", False),
        ("        for i in range(len(res['ids'][0]))", False),
        ("    ]", False),
    ]))
    e.append(sp(0.3))
    e += h2("7.3 _rrf_fusion() — Reciprocal Rank Fusion")
    e.append(body(
        "RRF is score-agnostic — it only uses the rank position, not the raw score value. "
        "This means BM25 scores (unbounded floats) and cosine distances (0–2) never need "
        "to be normalised before merging."
    ))
    e.append(code_block([
        ("def _rrf_fusion(keyword_results, semantic_results, k=60, final_top_k=5):", False),
        ("    rrf_scores = {}   # doc_id  →  cumulative RRF score", True),
        ("    doc_store  = {}   # doc_id  →  doc dict (for final reconstruction)", True),
        ("", False),
        ("    # Accumulate keyword ranks  (rank starts at 1)", True),
        ("    for rank, doc in enumerate(keyword_results, 1):", False),
        ("        rrf_scores[doc['id']] = rrf_scores.get(doc['id'], 0.0) + 1.0 / (k + rank)", False),
        ("        doc_store[doc['id']]  = doc", False),
        ("", False),
        ("    # Accumulate semantic ranks", True),
        ("    for rank, doc in enumerate(semantic_results, 1):", False),
        ("        rrf_scores[doc['id']] = rrf_scores.get(doc['id'], 0.0) + 1.0 / (k + rank)", False),
        ("        doc_store[doc['id']]  = doc", False),
        ("", False),
        ("    # Sort by combined RRF score, keep top final_top_k", True),
        ("    top_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:final_top_k]", False),
        ("    return [{**doc_store[d], 'rrf_score': round(rrf_scores[d], 6)} for d in top_ids]", False),
    ]))
    e.append(sp(0.3))
    e += h2("7.4 RRF Formula and Worked Example")
    e.append(body("RRF score formula:  score(d) = Σ  1 / (k + rank_i(d))  for each retrieval list i"))
    e.append(sp(0.2))
    e.append(data_table(
        ["Doc", "Keyword rank", "Semantic rank", "Keyword score", "Semantic score", "RRF Total", "Final rank"],
        [
            ["doc_A", "1", "2", "1/61 = 0.01639", "1/62 = 0.01613", "0.03252", "1st"],
            ["doc_B", "3", "1", "1/63 = 0.01587", "1/61 = 0.01639", "0.03226", "2nd"],
            ["doc_C", "2", "—", "1/62 = 0.01613", "not in list",    "0.01613", "3rd"],
            ["doc_D", "—", "3", "not in list",    "1/63 = 0.01587", "0.01587", "4th"],
        ],
        col_widths=[1.8*cm, 2.5*cm, 2.5*cm, 3*cm, 3*cm, 2.2*cm, 2*cm],
    ))
    e.append(Paragraph(
        "doc_A wins because it ranked highly in both lists. doc_C and doc_D each appeared in only one list "
        "but still make the final top-4. This is RRF's key advantage: cross-list agreement boosts rank.",
        ST["caption"],
    ))
    e.append(sp(0.3))
    e += h2("7.5 retrieve() — Public API")
    e.append(code_block([
        ("def retrieve(query, semantic_top_k=SEMANTIC_TOP_K,", False),
        ("             keyword_top_k=KEYWORD_TOP_K, final_top_k=FINAL_TOP_K):", False),
        ("    keyword_results  = _keyword_search(query, keyword_top_k)", False),
        ("    semantic_results = _semantic_search(query, semantic_top_k)", False),
        ("    fused_results    = _rrf_fusion(keyword_results, semantic_results,", False),
        ("                                   final_top_k=final_top_k)", False),
        ("    return {", False),
        ("        'fused':    fused_results,     # what gets sent to LLM", True),
        ("        'keyword':  keyword_results,   # raw BM25 results (for evaluation)", True),
        ("        'semantic': semantic_results,  # raw ChromaDB results (for evaluation)", True),
        ("    }", False),
    ]))
    return e


def sec_generator():
    e = []
    e += h1("8. hybrid_rag/generator.py — LLM Generation")
    e.append(body(
        "Same Groq key rotation logic as naive_rag. The fused top-5 documents from RRF "
        "are formatted as a numbered context block and injected into the prompt."
    ))
    e.append(sp(0.3))
    e += h2("8.1 generate() — Full Code")
    e.append(code_block([
        ("import groq", False),
        ("from hybrid_rag.config import GROQ_API_KEYS, GROQ_MODEL", False),
        ("", False),
        ("_current_key_idx: int = 0", False),
        ("_ROTATABLE = (groq.RateLimitError, groq.AuthenticationError)", False),
        ("", False),
        ("def _call_groq(messages, temperature=0.1):", False),
        ("    global _current_key_idx", False),
        ("    for _ in range(len(GROQ_API_KEYS)):", False),
        ("        try:", False),
        ("            client = groq.Groq(api_key=GROQ_API_KEYS[_current_key_idx])", False),
        ("            resp   = client.chat.completions.create(", False),
        ("                model=GROQ_MODEL, messages=messages, temperature=temperature)", False),
        ("            return resp.choices[0].message.content", False),
        ("        except _ROTATABLE as exc:", False),
        ("            print(f'  Key [{_current_key_idx}] exhausted. Rotating...')", False),
        ("            _current_key_idx = (_current_key_idx + 1) % len(GROQ_API_KEYS)", False),
        ("    raise RuntimeError('All 13 Groq API keys exhausted.')", False),
        ("", False),
        ("def generate(query: str, context_docs: list[dict], temperature=0.1) -> str:", False),
        ("    context_block = '\\n\\n'.join(", False),
        ("        f'[Document {i+1}]\\n{doc[\"text\"]}' for i, doc in enumerate(context_docs)", False),
        ("    )", False),
        ("    messages = [", False),
        ("        {'role': 'system', 'content': SYSTEM_PROMPT},", False),
        ("        {'role': 'user',   'content':", False),
        ("            f'Context:\\n{context_block}\\n\\nQuestion: {query}\\n\\nAnswer:'},", False),
        ("    ]", False),
        ("    return _call_groq(messages, temperature)", False),
    ]))
    return e


def sec_pipeline():
    e = []
    e += h1("9. hybrid_rag/pipeline.py — Orchestration")
    e.append(code_block([
        ("from hybrid_rag.config    import FINAL_TOP_K", False),
        ("from hybrid_rag.retriever import retrieve", False),
        ("from hybrid_rag.generator import generate", False),
        ("", False),
        ("def run_hybrid_rag(query: str, final_top_k: int = FINAL_TOP_K) -> dict:", False),
        ("    \"\"\"", False),
        ("    End-to-end Hybrid RAG:", False),
        ("      1. BM25 keyword search     → top-10 candidates", False),
        ("      2. ChromaDB semantic search → top-10 candidates", False),
        ("      3. RRF fusion              → top-5 unified results", False),
        ("      4. Groq LLaMA 3.3 70B      → grounded answer", False),
        ("    \"\"\"", False),
        ("    retrieval = retrieve(query, final_top_k=final_top_k)", False),
        ("    answer    = generate(query, retrieval['fused'])", False),
        ("    return {", False),
        ("        'query':          query,", False),
        ("        'answer':         answer,", False),
        ("        'retrieved_docs': retrieval['fused'],     # RRF top-5 (sent to LLM)", True),
        ("        'keyword_docs':   retrieval['keyword'],   # BM25 top-10 (debug/eval)", True),
        ("        'semantic_docs':  retrieval['semantic'],  # ChromaDB top-10 (debug/eval)", True),
        ("    }", False),
    ]))
    e.append(sp(0.3))
    e.append(info_box(
        "Return Value Schema",
        [
            "query          (str)  — original user question",
            "answer         (str)  — LLM-generated answer",
            "retrieved_docs (list) — RRF top-5: { id, text, metadata, rrf_score }",
            "keyword_docs   (list) — BM25 top-10: { id, text, metadata, bm25_score }",
            "semantic_docs  (list) — ChromaDB top-10: { id, text, metadata, distance }",
        ],
    ))
    return e


def sec_entry_point():
    e = []
    e += h1("10. run_hybrid_rag.py — Entry Point & Auto-Setup")
    e.append(body(
        "Two independent existence checks run at startup — one for ChromaDB, one for the "
        "BM25 pickle file. Only missing indexes are built, so if naive_rag already built "
        "ChromaDB, only the BM25 index is created (takes ~30 seconds)."
    ))
    e.append(sp(0.3))
    e += h2("10.1 Auto-Setup Logic")
    e.append(code_block([
        ("def _chroma_exists() -> bool:", False),
        ("    try:", False),
        ("        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))", False),
        ("        return client.get_collection(COLLECTION_NAME).count() > 0", False),
        ("    except Exception:", False),
        ("        return False", False),
        ("", False),
        ("def _bm25_exists() -> bool:", False),
        ("    return BM25_INDEX_PATH.exists()         # just a file check", True),
        ("", False),
        ("def _ensure_setup():", False),
        ("    chroma_ok = _chroma_exists()", False),
        ("    bm25_ok   = _bm25_exists()", False),
        ("    if chroma_ok and bm25_ok:", False),
        ("        print('[Setup] Both indexes found — skipping ingestion.')", False),
        ("        return", False),
        ("    if not chroma_ok:", False),
        ("        print('[Setup] ChromaDB missing. Building semantic index...')", False),
        ("        build_chroma()", False),
        ("    if not bm25_ok:", False),
        ("        print('[Setup] BM25 index missing. Building keyword index...')", False),
        ("        build_bm25()               # ~30 sec for 13,225 docs", True),
    ]))
    e.append(sp(0.3))
    e += h2("10.2 Commands")
    e.append(data_table(
        ["Command", "Action", "Duration"],
        [
            ["python run_hybrid_rag.py",
             "Checks both indexes.\nBuilds only what is missing.\nStarts Q&A loop.",
             "First run ~5–6 min\nIf ChromaDB exists: ~30 sec\nSubsequent: < 3 sec"],
            ["python run_hybrid_rag.py --ingest",
             "Force rebuilds BOTH indexes.\nDrops and recreates ChromaDB.\nRe-pickles BM25.",
             "~6 min"],
        ],
        col_widths=[5.5*cm, 7*cm, 4*cm],
    ))
    e.append(sp(0.3))
    e += h2("10.3 Interactive Session Example")
    e.append(code_block([
        ("[Setup] Both indexes found — skipping ingestion.", False),
        ("", False),
        ("============================================================", False),
        ("  E-Commerce Hybrid RAG  |  LLaMA 3.3 70B via Groq", False),
        ("  Keyword  : BM25 (rank_bm25)", False),
        ("  Semantic : sentence-transformers/all-MiniLM-L6-v2", False),
        ("  Fusion   : Reciprocal Rank Fusion (RRF k=60)", False),
        ("  VectorDB : ChromaDB (cosine)", False),
        ("  Type 'exit' to quit.", False),
        ("============================================================", False),
        ("", False),
        ("Question: Show late orders from SP state in December 2017", False),
        ("", False),
        ("Answer:", False),
        ("Based on the context, several orders from SP state in December 2017", False),
        ("experienced late delivery. Order 8704f37b... arrived 8.88 days early...", False),
        ("", False),
        ("Top-5 docs after RRF fusion:", False),
        ("  [order_8704f37b]  rrf=0.03252  type=order_level", False),
        ("  [state_SP]        rrf=0.02890  type=customer_state_level", False),
        ("  [month_2017-12]   rrf=0.02614  type=month_level", False),
        ("  ...", False),
    ]))
    return e


def sec_how_to_run():
    e = []
    e += h1("11. Complete Run Guide")
    e += h2("11.1 Prerequisites")
    e.append(code_block([
        ("# Python 3.10+", True),
        ("pip install chromadb sentence-transformers groq rank_bm25", False),
        ("", False),
        ("# Knowledge base must exist at:", True),
        ("dataset/knowledge_base/kb_all_documents.json", False),
    ]))
    e.append(sp(0.3))
    e += h2("11.2 Step-by-Step")
    e.append(data_table(
        ["Step", "Command", "Output"],
        [
            ["1 — Run (first time)", "python run_hybrid_rag.py",
             "Builds chroma_db/ + bm25_index.pkl,\nthen starts Q&A CLI"],
            ["2 — Ask questions",   "Type at the 'Question:' prompt",
             "Answer + RRF-fused doc IDs with rrf_score"],
            ["3 — Exit",            "Type 'exit' or Ctrl+C", "Session ends"],
            ["4 — Force re-index",  "python run_hybrid_rag.py --ingest",
             "Rebuilds both chroma_db/ and bm25_index.pkl"],
        ],
        col_widths=[3.5*cm, 6*cm, 7*cm],
    ))
    e.append(sp(0.3))
    e += h2("11.3 File Outputs Created")
    e.append(data_table(
        ["File / Folder", "Created By", "Contents"],
        [
            ["chroma_db/",                 "build_chroma()",  "Persistent ChromaDB: 13,225 vectors + metadata (shared with naive_rag)"],
            ["dataset/bm25_index/bm25_index.pkl", "build_bm25()",   "Pickled BM25Okapi object + raw document list"],
        ],
        col_widths=[5.5*cm, 4*cm, 7*cm],
    ))
    return e


# ── Footer ────────────────────────────────────────────────────────────────────
def on_first_page(canvas, doc): pass

def on_later_pages(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#757575"))
    canvas.drawString(2*cm, 1.2*cm, "Hybrid RAG Pipeline — End-to-End Implementation")
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
        title="Hybrid RAG Pipeline — End-to-End Documentation",
    )

    story = []
    story += cover_page()

    # TOC
    story += h1("Table of Contents")
    toc = [
        ("1. What is Hybrid RAG?",                           False),
        ("2. Libraries Used",                                False),
        ("3. End-to-End Architecture",                       False),
        ("4. File Structure",                                False),
        ("5. hybrid_rag/config.py",                          False),
        ("6. hybrid_rag/ingestion.py — BM25 + ChromaDB",     False),
        ("7. hybrid_rag/retriever.py — BM25 + Semantic + RRF",False),
        ("8. hybrid_rag/generator.py — LLM + Key Rotation",  False),
        ("9. hybrid_rag/pipeline.py — Orchestration",        False),
        ("10. run_hybrid_rag.py — Entry Point",              False),
        ("11. Complete Run Guide",                           False),
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

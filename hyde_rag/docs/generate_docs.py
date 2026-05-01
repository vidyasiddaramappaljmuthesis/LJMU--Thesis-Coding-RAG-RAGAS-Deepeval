"""
Documentation generator for HyDE RAG Pipeline.
Produces:  docs/HyDE_RAG_Documentation.pdf
Run:       python hyde_rag/generate_docs.py
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
PDF_PATH = DOCS_DIR / "HyDE_RAG_Documentation.pdf"

# ── Colours — Purple / Indigo theme ──────────────────────────────────────────
C_DARK       = colors.HexColor("#1A237E")
C_ACCENT     = colors.HexColor("#1565C0")
C_PURPLE     = colors.HexColor("#4A148C")
C_MID_PURPLE = colors.HexColor("#7B1FA2")
C_LIGHT_PURP = colors.HexColor("#F3E5F5")
C_TEAL       = colors.HexColor("#00695C")
C_HEADER_BG  = colors.HexColor("#1A237E")
C_ROW_EVEN   = colors.HexColor("#F5F5F5")
C_ROW_ODD    = colors.white
C_CODE_BG    = colors.HexColor("#F8F8F8")
C_CODE_BORD  = colors.HexColor("#CCCCCC")
C_BG         = colors.HexColor("#F3E5F5")
C_BD         = colors.HexColor("#7B1FA2")
C_STEP1_BG   = colors.HexColor("#E8EAF6")   # indigo tint  — hypothetical step
C_STEP1_BD   = colors.HexColor("#3949AB")
C_STEP2_BG   = colors.HexColor("#E3F2FD")   # blue tint    — embed step
C_STEP2_BD   = colors.HexColor("#1565C0")
C_STEP3_BG   = colors.HexColor("#E8F5E9")   # green tint   — answer step
C_STEP3_BD   = colors.HexColor("#2E7D32")

W, H = A4


# ── Styles ────────────────────────────────────────────────────────────────────
def _styles():
    s = {}
    s["title"]   = ParagraphStyle("T",  fontSize=26, leading=34, alignment=TA_CENTER,
                    textColor=colors.white, spaceAfter=6, fontName="Helvetica-Bold")
    s["subtitle"]= ParagraphStyle("ST", fontSize=13, leading=18, alignment=TA_CENTER,
                    textColor=colors.HexColor("#E1BEE7"), spaceAfter=4, fontName="Helvetica")
    s["h1"]      = ParagraphStyle("H1", fontSize=18, leading=24, spaceBefore=18, spaceAfter=8,
                    textColor=C_DARK, fontName="Helvetica-Bold")
    s["h2"]      = ParagraphStyle("H2", fontSize=14, leading=20, spaceBefore=14, spaceAfter=6,
                    textColor=C_ACCENT, fontName="Helvetica-Bold")
    s["h3"]      = ParagraphStyle("H3", fontSize=12, leading=17, spaceBefore=10, spaceAfter=4,
                    textColor=C_TEAL, fontName="Helvetica-Bold")
    s["body"]    = ParagraphStyle("B",  fontSize=10, leading=15, spaceAfter=6,
                    textColor=colors.HexColor("#212121"), fontName="Helvetica", alignment=TA_JUSTIFY)
    s["body_l"]  = ParagraphStyle("BL", fontSize=10, leading=15, spaceAfter=4,
                    textColor=colors.HexColor("#212121"), fontName="Helvetica")
    s["bullet"]  = ParagraphStyle("BU", fontSize=10, leading=15, spaceAfter=3,
                    textColor=colors.HexColor("#212121"), fontName="Helvetica",
                    leftIndent=16, bulletIndent=6)
    s["code"]    = ParagraphStyle("C",  fontSize=8.5, leading=13, spaceAfter=0,
                    textColor=colors.HexColor("#212121"), fontName="Courier", leftIndent=8)
    s["code_c"]  = ParagraphStyle("CC", fontSize=8.5, leading=13, spaceAfter=0,
                    textColor=colors.HexColor("#5D6D7E"), fontName="Courier-Oblique", leftIndent=8)
    s["caption"] = ParagraphStyle("CAP", fontSize=8.5, leading=12, spaceAfter=6, spaceBefore=2,
                    textColor=colors.HexColor("#757575"), fontName="Helvetica-Oblique",
                    alignment=TA_CENTER)
    return s

ST = _styles()


# ── Helpers ───────────────────────────────────────────────────────────────────
def sp(h=0.3):  return Spacer(1, h * cm)
def hr(c=C_ACCENT, t=0.5):
    return HRFlowable(width="100%", thickness=t, color=c, spaceAfter=6, spaceBefore=4)

def h1(txt):
    return [sp(0.3), hr(C_DARK, 1.5), Paragraph(txt, ST["h1"]), hr(C_ACCENT, 0.5)]
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

def step_box(label, lines, bg, border):
    rows = [[Paragraph(f"<b>{label}</b>", ST["body_l"])]] + \
           [[Paragraph(f"  {l}", ST["body_l"])] for l in lines]
    tbl = Table(rows, colWidths=[16.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), bg),
        ("BOX",           (0,0),(-1,-1), 1.2, border),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
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
        style.append(("BACKGROUND",(0,i),(-1,i), C_ROW_EVEN if i%2==0 else C_ROW_ODD))
    tbl.setStyle(TableStyle(style))
    return tbl


# ── Cover ─────────────────────────────────────────────────────────────────────
def cover_page():
    e = [sp(2.5)]
    banner = Table([
        [Paragraph("HyDE RAG Pipeline", ST["title"])],
        [Paragraph("Hypothetical Document Embeddings — End-to-End Implementation", ST["subtitle"])],
        [Paragraph("Groq LLaMA 3.3 70B  •  all-MiniLM-L6-v2  •  ChromaDB  •  2-LLM-Call Architecture", ST["subtitle"])],
    ], colWidths=[17*cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_PURPLE),
        ("TOPPADDING",    (0,0),(-1,-1), 16),
        ("BOTTOMPADDING", (0,0),(-1,-1), 16),
        ("LEFTPADDING",   (0,0),(-1,-1), 20),
        ("RIGHTPADDING",  (0,0),(-1,-1), 20),
        ("BOX",           (0,0),(-1,-1), 2, C_MID_PURPLE),
    ]))
    e.append(banner)
    e.append(sp(1.2))

    meta = Table([
        [Paragraph(f"<b>Date:</b>  {datetime.datetime.now().strftime('%B %Y')}", ST["body_l"])],
        [Paragraph("<b>Pipeline:</b>  HyDE RAG — Hypothetical Document Embeddings", ST["body_l"])],
        [Paragraph("<b>HyDE Step:</b>  LLM generates hypothetical answer → embedded for retrieval", ST["body_l"])],
        [Paragraph("<b>Embedding:</b>  sentence-transformers/all-MiniLM-L6-v2 (384-dim, cosine)", ST["body_l"])],
        [Paragraph("<b>Vector DB:</b>  ChromaDB PersistentClient (shared with Naive RAG)", ST["body_l"])],
        [Paragraph("<b>LLM:</b>  Groq — llama-3.3-70b-versatile (called twice per query)", ST["body_l"])],
        [Paragraph("<b>API Keys:</b>  13 Groq keys with automatic round-robin rotation", ST["body_l"])],
        [Paragraph("<b>Dataset:</b>  Olist Brazilian E-Commerce — 13,225 KB documents", ST["body_l"])],
    ], colWidths=[17*cm])
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_LIGHT_PURP),
        ("BOX",           (0,0),(-1,-1), 1, C_MID_PURPLE),
        ("LEFTPADDING",   (0,0),(-1,-1), 16),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
    ]))
    e.append(meta)
    e.append(sp(1.5))

    stats = [("2", "LLM Calls\nPer Query"), ("0.7", "HyDE\nTemperature"),
             ("13,225", "KB\nDocuments"),   ("13",  "Groq API\nKeys")]
    cells = []
    for val, lbl in stats:
        c = Table([
            [Paragraph(f"<b>{val}</b>", ParagraphStyle("SV", fontSize=20,
              textColor=colors.white, alignment=TA_CENTER, fontName="Helvetica-Bold"))],
            [Paragraph(lbl, ParagraphStyle("SL", fontSize=8,
              textColor=colors.HexColor("#E1BEE7"), alignment=TA_CENTER,
              fontName="Helvetica", leading=11))],
        ], colWidths=[3.8*cm])
        c.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1), C_PURPLE),
            ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
            ("BOX",(0,0),(-1,-1),1, C_MID_PURPLE)]))
        cells.append(c)
    sr = Table([cells], colWidths=[4.0*cm]*4)
    sr.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2)]))
    e.append(sr)
    e.append(PageBreak())
    return e


# ── Section 1: What is HyDE RAG ───────────────────────────────────────────────
def sec_overview():
    e = []
    e += h1("1. What is HyDE RAG?")
    e.append(body(
        "HyDE (Hypothetical Document Embeddings) is an advanced retrieval technique that solves "
        "a fundamental mismatch in standard RAG: user queries are short and keyword-like, while "
        "knowledge base documents are long and content-rich. Embedding a short query and comparing "
        "it to full-length documents is an asymmetric problem — the vector spaces rarely align well."
    ))
    e.append(body(
        "HyDE's insight: instead of embedding the raw query, use the LLM to first generate a "
        "hypothetical document that would directly answer the question. This hypothetical passage "
        "is written in the same style as real KB documents, so its embedding lands in the same "
        "region of the vector space as real matching documents. Retrieval then becomes a "
        "document-to-document comparison rather than a query-to-document comparison."
    ))
    e.append(sp(0.3))

    e += h2("1.1 The Core Insight — Query vs Hypothetical Document")
    e.append(data_table(
        ["", "Naive RAG (Query)", "HyDE RAG (Hypothetical Doc)"],
        [
            ["What gets embedded",
             "Short user query\n'Which category has most late deliveries?'",
             "Full hypothetical passage\n'The product category with the highest late delivery rate\nis office_furniture at 12.3%. Orders in this category\naveragedelivery days of 18.4...'"],
            ["Vector space match",
             "Query vector ≠ document vector space.\nShort text → different embedding region.",
             "Hypothetical doc vector ≈ real doc vector space.\nSame length, style, and vocabulary."],
            ["Retrieval quality",
             "Good for broad semantic queries.\nWeaker for domain-specific phrasing.",
             "Strong across all query types.\nHypothetical doc mirrors KB document style."],
        ],
        col_widths=[3.5*cm, 6*cm, 7*cm],
    ))
    e.append(sp(0.3))

    e += h2("1.2 Three-Step Pipeline")
    e.append(step_box("Step 1 — Generate Hypothetical Document  (LLM Call 1, temperature=0.7)", [
        "Input  : User query",
        "Model  : Groq LLaMA 3.3 70B",
        "Prompt : 'Write a passage that would directly answer this question.'",
        "Output : ~200-word hypothetical passage in KB document style",
        "Key    : Higher temperature (0.7) for creative, diverse hypothetical content",
    ], C_STEP1_BG, C_STEP1_BD))
    e.append(sp(0.2))
    e.append(step_box("Step 2 — Embed Hypothetical Doc and Retrieve Real Documents", [
        "Input  : Hypothetical document (string)",
        "Embed  : ChromaDB auto-embeds with all-MiniLM-L6-v2  →  384-dim vector",
        "Search : Cosine similarity against 13,225 real KB document vectors",
        "Output : Top-5 real documents (id, text, metadata, distance)",
        "Key    : The query is NOT embedded — only the hypothetical doc is",
    ], C_STEP2_BG, C_STEP2_BD))
    e.append(sp(0.2))
    e.append(step_box("Step 3 — Generate Final Answer  (LLM Call 2, temperature=0.1)", [
        "Input  : Original query + top-5 real retrieved documents",
        "Model  : Groq LLaMA 3.3 70B",
        "Prompt : Standard RAG prompt — context block + question",
        "Output : Grounded, factual answer from real KB data",
        "Key    : Lower temperature (0.1) for deterministic, factual answers",
    ], C_STEP3_BG, C_STEP3_BD))
    e.append(sp(0.3))

    e += h2("1.3 When HyDE Outperforms Naive RAG")
    for item in [
        "<b>Domain-specific queries</b>: 'Which delivery bucket had the lowest review score?' — "
        "Naive RAG embeds a short question; HyDE embeds a full hypothetical analysis passage.",
        "<b>Analytical questions</b>: 'What factors correlate with late deliveries?' — "
        "The hypothetical answer bridges the vocabulary gap between query and KB documents.",
        "<b>Sparse KB coverage</b>: When the exact query wording rarely appears in the KB, "
        "HyDE generates bridging vocabulary that matches document style.",
    ]:
        e.append(bullet(item))
    return e


# ── Section 2: Libraries ──────────────────────────────────────────────────────
def sec_libraries():
    e = []
    e += h1("2. Libraries Used")
    e.append(data_table(
        ["Library", "Version", "Purpose in HyDE RAG"],
        [
            ["groq", "≥ 0.11.0",
             "Called TWICE per query: (1) generate hypothetical document at temperature=0.7; "
             "(2) generate final answer at temperature=0.1. Same 13-key rotation for both calls."],
            ["chromadb", "≥ 0.5.0",
             "Persistent vector store — SHARED with Naive RAG. No re-embedding needed if "
             "Naive RAG already built it. Receives hypothetical doc as query_texts and "
             "returns nearest real KB documents."],
            ["sentence-transformers", "≥ 3.0.0",
             "all-MiniLM-L6-v2 used via ChromaDB's SentenceTransformerEmbeddingFunction. "
             "Embeds both the hypothetical document (query time) and KB documents (index time)."],
        ],
        col_widths=[3.8*cm, 2.2*cm, 10.5*cm],
    ))
    e.append(sp(0.3))
    e += h2("2.1 Install")
    e.append(code_block([
        ("pip install chromadb sentence-transformers groq", False),
        ("# rank_bm25 NOT needed — HyDE uses only semantic search", True),
    ]))
    e.append(sp(0.3))
    e += h2("2.2 Two Groq Calls — Different Roles and Temperatures")
    e.append(data_table(
        ["Call", "Function", "Temperature", "max_tokens", "Role"],
        [
            ["Call 1", "generate_hypothetical_doc()", "0.7 (creative)",
             "300", "Generate diverse, plausible hypothetical answer passage"],
            ["Call 2", "generate()",                  "0.1 (precise)",
             "unlimited", "Generate factual answer grounded in real retrieved context"],
        ],
        col_widths=[1.5*cm, 5*cm, 3*cm, 2.5*cm, 5*cm],
    ))
    e.append(sp(0.2))
    e.append(info_box(
        "Why Different Temperatures?",
        [
            "Call 1 (temp=0.7): Creative generation is desirable. A richer, more varied",
            "  hypothetical doc covers more vocabulary → wider semantic net in retrieval.",
            "Call 2 (temp=0.1): Factual accuracy is critical. Low temperature ensures",
            "  the LLM sticks closely to retrieved context without hallucination.",
        ],
    ))
    return e


# ── Section 3: Architecture ───────────────────────────────────────────────────
def sec_architecture():
    e = []
    e += h1("3. End-to-End Architecture")
    e += h2("3.1 Indexing Phase (runs once — shared with Naive RAG)")
    e.append(body(
        "HyDE RAG shares ChromaDB with Naive RAG. The same 13,225 documents, "
        "the same embedding model, the same collection. No re-indexing needed."
    ))
    e.append(code_block([
        ("kb_all_documents.json  [13,225 docs]", False),
        ("         |", False),
        ("         |  SentenceTransformerEmbeddingFunction(all-MiniLM-L6-v2)", True),
        ("         |  embed doc['text']  →  384-dim vector per document", True),
        ("         ↓", False),
        ("    ChromaDB PersistentClient  →  chroma_db/  (cosine distance)", False),
        ("         |", False),
        ("         └── SHARED with naive_rag and hybrid_rag", True),
        ("             No re-embedding if chroma_db/ already exists.", True),
    ]))
    e.append(sp(0.3))
    e += h2("3.2 Query Phase — Two LLM Calls")
    e.append(code_block([
        ("User Query (string)", False),
        ("    │", False),
        ("    │  ── LLM CALL 1 ──────────────────────────────────────────────────", False),
        ("    │  generate_hypothetical_doc(query)", False),
        ("    │  System: 'Write a passage that would directly answer this question.'", True),
        ("    │  Model:  llama-3.3-70b-versatile  temperature=0.7  max_tokens=300", True),
        ("    ↓", False),
        ("Hypothetical Document (string, ~200 words)", False),
        ("    │", False),
        ("    │  ── EMBEDDING ─────────────────────────────────────────────────────", False),
        ("    │  ChromaDB.query(query_texts=[hypothetical_doc], n_results=5)", False),
        ("    │  → ChromaDB embeds hypothetical_doc with all-MiniLM-L6-v2", True),
        ("    │  → cosine search against 13,225 real KB document vectors", True),
        ("    ↓", False),
        ("Top-5 Real Documents  { id, text, metadata, distance }", False),
        ("    │", False),
        ("    │  ── LLM CALL 2 ──────────────────────────────────────────────────", False),
        ("    │  generate(original_query, top_5_real_docs)", False),
        ("    │  System: 'Answer using only the provided context.'", True),
        ("    │  Model:  llama-3.3-70b-versatile  temperature=0.1", True),
        ("    ↓", False),
        ("Final Answer (string)", False),
    ]))
    e.append(sp(0.3))
    e += h2("3.3 File Dependency Map")
    e.append(data_table(
        ["File", "Imports From", "Used By"],
        [
            ["config.py",       "pathlib only",                            "All other modules"],
            ["ingestion.py",    "config.py",                               "retriever.py, run_hyde_rag.py"],
            ["generator.py",    "config.py",                               "retriever.py, pipeline.py"],
            ["retriever.py",    "config.py, ingestion.py, generator.py",   "pipeline.py"],
            ["pipeline.py",     "config.py, retriever.py, generator.py",   "run_hyde_rag.py"],
        ],
        col_widths=[3.5*cm, 6.5*cm, 6.5*cm],
    ))
    return e


# ── Section 4: File Structure ─────────────────────────────────────────────────
def sec_file_structure():
    e = []
    e += h1("4. File Structure")
    e.append(code_block([
        ("hyde_rag/", False),
        ("|-- __init__.py        # empty — marks the directory as a Python package", False),
        ("|-- config.py          # paths, API keys, TOP_K, HYDE_TEMPERATURE=0.7, ANSWER_TEMPERATURE=0.1", False),
        ("|-- ingestion.py       # load KB → embed → persist ChromaDB (shared with naive_rag)", False),
        ("|-- generator.py       # TWO functions: generate_hypothetical_doc() + generate()", False),
        ("|-- retriever.py       # calls generator for HyDE doc, then searches ChromaDB", False),
        ("|-- pipeline.py        # run_hyde_rag(query) — full 3-step orchestration", False),
        ("|-- generate_docs.py   # this script — generates HyDE_RAG_Documentation.pdf", False),
        ("", False),
        ("run_hyde_rag.py        # single entry point: auto-setup + interactive CLI", False),
        ("chroma_db/             # ChromaDB — SHARED with naive_rag and hybrid_rag", False),
    ]))
    e.append(sp(0.3))
    e.append(info_box(
        "Key Difference from Naive RAG File Structure",
        [
            "generator.py now has TWO public functions instead of one:",
            "  generate_hypothetical_doc(query)  — called by retriever.py (HyDE step)",
            "  generate(query, context_docs)      — called by pipeline.py (answer step)",
            "retriever.py imports from generator.py (not just from ingestion.py).",
            "No new index files — HyDE adds LLM calls, not new storage.",
        ],
    ))
    return e


# ── Section 5: config.py ──────────────────────────────────────────────────────
def sec_config():
    e = []
    e += h1("5. hyde_rag/config.py")
    e.append(body(
        "Identical structure to naive_rag/config.py with two new constants: "
        "HYDE_TEMPERATURE for the hypothetical generation step and ANSWER_TEMPERATURE "
        "for the final answer step."
    ))
    e.append(code_block([
        ("from pathlib import Path", False),
        ("", False),
        ("BASE_DIR        = Path(__file__).parent.parent", False),
        ("KB_ALL_DOCS     = BASE_DIR / 'dataset' / 'knowledge_base' / 'kb_all_documents.json'", False),
        ("CHROMA_DB_PATH  = BASE_DIR / 'chroma_db'     # shared with naive_rag + hybrid_rag", True),
        ("COLLECTION_NAME = 'ecommerce_kb'", False),
        ("", False),
        ("EMBEDDING_MODEL    = 'sentence-transformers/all-MiniLM-L6-v2'", False),
        ("GROQ_MODEL         = 'llama-3.3-70b-versatile'", False),
        ("", False),
        ("GROQ_API_KEYS = [  # 13 keys — rotated on RateLimitError / AuthenticationError", True),
        ("    'gsk_vYX6KOVL2FDn...',", False),
        ("    # ... 12 more keys", True),
        ("]", False),
        ("", False),
        ("TOP_K              = 5     # documents returned to LLM as final context", False),
        ("HYDE_TEMPERATURE   = 0.7   # creative generation for hypothetical document", False),
        ("ANSWER_TEMPERATURE = 0.1   # precise/deterministic for final answer", False),
    ]))
    e.append(sp(0.3))
    e.append(data_table(
        ["Constant", "Value", "Description"],
        [
            ["CHROMA_DB_PATH",       "chroma_db/",    "Shared — reuses existing index from naive_rag if available"],
            ["COLLECTION_NAME",      "ecommerce_kb",  "Same collection name across all pipelines"],
            ["TOP_K",                "5",             "Real documents returned to LLM after HyDE retrieval"],
            ["HYDE_TEMPERATURE",     "0.7",           "Higher creativity for hypothetical doc — wider semantic net"],
            ["ANSWER_TEMPERATURE",   "0.1",           "Deterministic for factual final answer generation"],
        ],
        col_widths=[4.5*cm, 2.5*cm, 9.5*cm],
    ))
    return e


# ── Section 6: ingestion.py ───────────────────────────────────────────────────
def sec_ingestion():
    e = []
    e += h1("6. hyde_rag/ingestion.py — Vector Store")
    e.append(body(
        "Identical to naive_rag/ingestion.py — loads kb_all_documents.json, embeds with "
        "all-MiniLM-L6-v2, and persists to ChromaDB. Since HyDE shares the same collection "
        "name and path, this step is automatically skipped if Naive RAG already ran."
    ))
    e.append(sp(0.3))
    e.append(code_block([
        ("import json, chromadb", False),
        ("from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction", False),
        ("from hyde_rag.config import KB_ALL_DOCS, CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL", False),
        ("", False),
        ("def build_vector_store(batch_size: int = 500) -> chromadb.Collection:", False),
        ("    with open(KB_ALL_DOCS, 'r', encoding='utf-8') as f:", False),
        ("        docs = json.load(f)", False),
        ("    client     = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))", False),
        ("    collection = client.create_collection(", False),
        ("        name=COLLECTION_NAME,", False),
        ("        embedding_function=SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL),", False),
        ("        metadata={'hnsw:space': 'cosine'},", False),
        ("    )", False),
        ("    for i in range(0, len(docs), batch_size):", False),
        ("        batch = docs[i:i+batch_size]", False),
        ("        collection.add(ids=[d['id'] for d in batch],", False),
        ("                       documents=[d['text'] for d in batch],", False),
        ("                       metadatas=[d['metadata'] for d in batch])", False),
        ("    return collection", False),
        ("", False),
        ("def get_collection() -> chromadb.Collection:", False),
        ("    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))", False),
        ("    return client.get_collection(name=COLLECTION_NAME,", False),
        ("        embedding_function=SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL))", False),
    ]))
    return e


# ── Section 7: generator.py ───────────────────────────────────────────────────
def sec_generator():
    e = []
    e += h1("7. hyde_rag/generator.py — Two-Role LLM Module")
    e.append(body(
        "The generator module is the heart of HyDE RAG. It exposes two public functions: "
        "generate_hypothetical_doc() for the HyDE retrieval step (LLM Call 1), "
        "and generate() for the final answer step (LLM Call 2). "
        "Both share the same key-rotation infrastructure."
    ))
    e.append(sp(0.3))
    e += h2("7.1 Shared Key Rotation Infrastructure")
    e.append(code_block([
        ("import groq", False),
        ("from hyde_rag.config import GROQ_API_KEYS, GROQ_MODEL, HYDE_TEMPERATURE, ANSWER_TEMPERATURE", False),
        ("", False),
        ("_current_key_idx: int = 0              # shared across BOTH call types", True),
        ("_ROTATABLE = (groq.RateLimitError, groq.AuthenticationError)", False),
        ("", False),
        ("def _call_groq(messages: list[dict], temperature: float) -> str:", False),
        ("    global _current_key_idx", False),
        ("    for _ in range(len(GROQ_API_KEYS)):", False),
        ("        try:", False),
        ("            client = groq.Groq(api_key=GROQ_API_KEYS[_current_key_idx])", False),
        ("            resp   = client.chat.completions.create(", False),
        ("                model=GROQ_MODEL, messages=messages, temperature=temperature)", False),
        ("            return resp.choices[0].message.content", False),
        ("        except _ROTATABLE as exc:", False),
        ("            print(f'  [Groq] Key [{_current_key_idx}] exhausted. Rotating...')", False),
        ("            _current_key_idx = (_current_key_idx + 1) % len(GROQ_API_KEYS)", False),
        ("    raise RuntimeError('All 13 Groq API keys exhausted.')", False),
    ]))
    e.append(sp(0.3))
    e += h2("7.2 generate_hypothetical_doc() — LLM Call 1 (HyDE Step)")
    e.append(body(
        "Generates a hypothetical passage that would answer the query. The system prompt "
        "instructs the model to write in the same fact-rich, key-value style as the real "
        "knowledge base documents — ensuring the resulting embedding is compatible."
    ))
    e.append(code_block([
        ("_HYDE_SYSTEM_PROMPT = (", False),
        ("    'You are an e-commerce data expert. Given a question, write a concise, '", False),
        ("    'factual passage that would directly answer it. The passage should read '", False),
        ("    'as if extracted from an e-commerce analytics database or report. '", False),
        ("    'Write only the passage — no preamble, no labels.'", False),
        (")", False),
        ("", False),
        ("def generate_hypothetical_doc(query: str) -> str:", False),
        ("    messages = [", False),
        ("        {'role': 'system', 'content': _HYDE_SYSTEM_PROMPT},", False),
        ("        {'role': 'user',   'content': f'Question: {query}\\n\\nPassage:'},", False),
        ("    ]", False),
        ("    return _call_groq(messages, temperature=HYDE_TEMPERATURE)  # 0.7", True),
    ]))
    e.append(sp(0.2))
    e.append(info_box(
        "Hypothetical Document — Example",
        [
            "Query:  'Which product category has the highest late delivery rate?'",
            "",
            "Hypothetical Doc:",
            "  'Late Delivery Rate by Category: office_furniture shows the highest late",
            "  delivery rate at 12.3%, with an average delivery time of 18.4 days compared",
            "  to the estimated 14.2 days. Review Score: 3.1 (below platform average of",
            "  4.1). Total Orders: 1,243. Late Orders: 153. Top Seller State: SP...'",
            "",
            "→ This passage embeds much closer to the real category KB document than",
            "  the raw query 'Which category has highest late delivery rate?' would.",
        ],
    ))
    e.append(sp(0.3))
    e += h2("7.3 generate() — LLM Call 2 (Final Answer Step)")
    e.append(code_block([
        ("_ANSWER_SYSTEM_PROMPT = (", False),
        ("    'You are a helpful e-commerce data assistant. '", False),
        ("    'Answer questions using only the provided context. '", False),
        ("    'If the answer cannot be found in the context, say so clearly.'", False),
        (")", False),
        ("", False),
        ("def generate(query: str, context_docs: list[dict],", False),
        ("             temperature: float = ANSWER_TEMPERATURE) -> str:  # 0.1", True),
        ("    context_block = '\\n\\n'.join(", False),
        ("        f'[Document {i+1}]\\n{doc[\"text\"]}' for i, doc in enumerate(context_docs)", False),
        ("    )", False),
        ("    messages = [", False),
        ("        {'role': 'system', 'content': _ANSWER_SYSTEM_PROMPT},", False),
        ("        {'role': 'user', 'content':", False),
        ("            f'Context:\\n{context_block}\\n\\nQuestion: {query}\\n\\nAnswer:'},", False),
        ("    ]", False),
        ("    return _call_groq(messages, temperature)  # 0.1 — precise", True),
    ]))
    return e


# ── Section 8: retriever.py ───────────────────────────────────────────────────
def sec_retriever():
    e = []
    e += h1("8. hyde_rag/retriever.py — HyDE Retrieval")
    e.append(body(
        "The retriever orchestrates the two-step HyDE retrieval process. "
        "It calls generator to produce a hypothetical document, then passes that "
        "document as the query text to ChromaDB. ChromaDB embeds it with the same "
        "model used for indexing, ensuring vector space consistency. "
        "The original user query is preserved separately for the answer step."
    ))
    e.append(sp(0.3))
    e += h2("8.1 retrieve() — Full Code")
    e.append(code_block([
        ("from typing import Any", False),
        ("from hyde_rag.config    import TOP_K", False),
        ("from hyde_rag.ingestion import get_collection", False),
        ("from hyde_rag.generator import generate_hypothetical_doc", False),
        ("", False),
        ("def retrieve(query: str, top_k: int = TOP_K) -> dict[str, Any]:", False),
        ("    # Step 1: LLM generates a hypothetical passage (LLM Call 1)", True),
        ("    hypothetical_doc = generate_hypothetical_doc(query)", False),
        ("", False),
        ("    # Step 2: Embed the hypothetical doc, NOT the original query", True),
        ("    collection = get_collection()", False),
        ("    results    = collection.query(", False),
        ("        query_texts = [hypothetical_doc],  # ChromaDB embeds this", True),
        ("        n_results   = top_k,", False),
        ("    )", False),
        ("", False),
        ("    retrieved = [", False),
        ("        {", False),
        ("            'id':       results['ids'][0][i],", False),
        ("            'text':     results['documents'][0][i],", False),
        ("            'metadata': results['metadatas'][0][i],", False),
        ("            'distance': results['distances'][0][i],", False),
        ("        }", False),
        ("        for i in range(len(results['ids'][0]))", False),
        ("    ]", False),
        ("    return {", False),
        ("        'retrieved_docs':   retrieved,        # top-k real KB documents", True),
        ("        'hypothetical_doc': hypothetical_doc, # stored for inspection/logging", True),
        ("    }", False),
    ]))
    e.append(sp(0.3))
    e += h2("8.2 Why query_texts Instead of query_embeddings?")
    e.append(body(
        "ChromaDB's query_texts parameter auto-embeds the input using the same "
        "SentenceTransformerEmbeddingFunction that was used at index time. "
        "This guarantees the hypothetical document is embedded in exactly the same "
        "vector space as the KB documents — no manual model loading needed."
    ))
    e.append(data_table(
        ["Approach", "Code", "Result"],
        [
            ["query_texts (used here)",
             "collection.query(query_texts=[hypo_doc])",
             "ChromaDB embeds hypo_doc with same model as index → consistent vector space"],
            ["query_embeddings (alternative)",
             "emb = model.encode(hypo_doc)\ncollection.query(query_embeddings=[emb])",
             "Works but requires loading SentenceTransformer separately → extra dependency"],
        ],
        col_widths=[3.5*cm, 5.5*cm, 7.5*cm],
    ))
    return e


# ── Section 9: pipeline.py ────────────────────────────────────────────────────
def sec_pipeline():
    e = []
    e += h1("9. hyde_rag/pipeline.py — Orchestration")
    e.append(code_block([
        ("from typing import Any", False),
        ("from hyde_rag.config    import TOP_K", False),
        ("from hyde_rag.retriever import retrieve", False),
        ("from hyde_rag.generator import generate", False),
        ("", False),
        ("def run_hyde_rag(query: str, top_k: int = TOP_K) -> dict[str, Any]:", False),
        ("    \"\"\"", False),
        ("    End-to-end HyDE RAG:", False),
        ("      1. LLaMA 3.3 70B generates a hypothetical document (temp=0.7).", False),
        ("      2. ChromaDB embeds it and retrieves top-k real documents.", False),
        ("      3. LLaMA 3.3 70B generates the final answer from real docs (temp=0.1).", False),
        ("    \"\"\"", False),
        ("    retrieval = retrieve(query, top_k=top_k)    # Steps 1 + 2", True),
        ("    answer    = generate(query, retrieval['retrieved_docs'])  # Step 3", True),
        ("    return {", False),
        ("        'query':            query,", False),
        ("        'answer':           answer,", False),
        ("        'retrieved_docs':   retrieval['retrieved_docs'],   # top-k real docs", True),
        ("        'hypothetical_doc': retrieval['hypothetical_doc'], # what was embedded", True),
        ("    }", False),
    ]))
    e.append(sp(0.3))
    e.append(info_box(
        "Return Value Schema",
        [
            "query            (str)  — original user question",
            "answer           (str)  — final LLM answer grounded in real retrieved docs",
            "retrieved_docs   (list) — top-5 real KB docs: { id, text, metadata, distance }",
            "hypothetical_doc (str)  — the generated passage used for embedding/retrieval",
        ],
    ))
    return e


# ── Section 10: entry point ───────────────────────────────────────────────────
def sec_entry_point():
    e = []
    e += h1("10. run_hyde_rag.py — Entry Point & Auto-Setup")
    e.append(body(
        "Same auto-setup pattern as Naive RAG. Checks if ChromaDB exists; "
        "builds it if missing; then starts the interactive Q&A loop. "
        "The interactive loop shows the hypothetical document so users can see "
        "exactly what was embedded during retrieval."
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
        ("def _ensure_vector_store() -> None:", False),
        ("    if _vector_store_exists():", False),
        ("        print('[Setup] Vector store already exists — skipping ingestion.')", False),
        ("    else:", False),
        ("        print('[Setup] Building vector store (runs once)...')", False),
        ("        build_vector_store()", False),
    ]))
    e.append(sp(0.3))
    e += h2("10.2 Commands")
    e.append(data_table(
        ["Command", "Action", "Duration"],
        [
            ["python run_hyde_rag.py",
             "Auto-detects ChromaDB.\nBuilds if missing, then starts Q&A loop.",
             "First run ~5 min\nSubsequent: instant"],
            ["python run_hyde_rag.py --ingest",
             "Force re-indexes all 13,225 docs into ChromaDB.",
             "~5 min"],
        ],
        col_widths=[5.5*cm, 7*cm, 4*cm],
    ))
    e.append(sp(0.3))
    e += h2("10.3 Interactive Session — HyDE Output")
    e.append(code_block([
        ("Question: Which product category has the highest late delivery rate?", False),
        ("", False),
        ("Hypothetical document (used for retrieval):", False),
        ("  Late Delivery Rate by Category: The office_furniture category shows", False),
        ("  a late delivery rate of 12.3%, significantly above the platform average", False),
        ("  of 6.7%. Average delivery days: 18.4 (estimated: 14.2). Review Score:", False),
        ("  3.1. Total Orders: 1,243. Late Orders: 153. Top Seller State: SP...", False),
        ("", False),
        ("Answer:", False),
        ("  Based on the provided context, the computers_accessories category had", False),
        ("  the highest late delivery rate at 11.8%, followed by office_furniture...", False),
        ("", False),
        ("Retrieved documents:", False),
        ("  [category_office_furniture]  distance=0.1234  type=category_level", False),
        ("  [category_computers]         distance=0.1456  type=category_level", False),
        ("  ...", False),
    ]))
    return e


# ── Section 11: HyDE vs Naive comparison ─────────────────────────────────────
def sec_comparison():
    e = []
    e += h1("11. HyDE RAG vs Naive RAG — Comparison")
    e.append(data_table(
        ["Aspect", "Naive RAG", "HyDE RAG"],
        [
            ["What gets embedded",  "Raw user query",              "LLM-generated hypothetical document"],
            ["LLM calls per query", "1 (final answer only)",       "2 (hypothetical doc + final answer)"],
            ["Latency",             "Lower — one LLM call",        "Higher — two LLM calls"],
            ["Retrieval quality",   "Good for broad queries",      "Better for domain-specific, analytical"],
            ["Vector space match",  "Query ≠ document style",      "Hypothetical doc ≈ document style"],
            ["Setup complexity",    "Simple — no extra step",      "Same setup — only query path differs"],
            ["Extra storage",       "None",                        "None — same ChromaDB"],
            ["Best for",            "Simple factual lookups,\nkeyword-like queries",
             "Complex analytical questions,\nvague or paraphrased queries"],
        ],
        col_widths=[4*cm, 6*cm, 6.5*cm],
    ))
    e.append(sp(0.3))
    e += h2("11.1 HyDE Limitation")
    e.append(info_box(
        "When HyDE May Underperform",
        [
            "If the LLM generates a plausible but factually wrong hypothetical document,",
            "  the embedding may point to the wrong region of the vector space.",
            "Example: Query about 'canceled orders' — if LLM generates a passage about",
            "  'delivered orders' instead, retrieval will return wrong documents.",
            "Mitigation: Use HYDE_TEMPERATURE=0.7 (not 1.0) to balance creativity",
            "  with accuracy. Lower temperatures (0.5) trade diversity for precision.",
        ],
    ))
    return e


# ── Section 12: Quick Run Guide ───────────────────────────────────────────────
def sec_how_to_run():
    e = []
    e += h1("12. Complete Run Guide")
    e += h2("12.1 Prerequisites")
    e.append(code_block([
        ("pip install chromadb sentence-transformers groq", False),
        ("", False),
        ("# Knowledge base must exist at:", True),
        ("dataset/knowledge_base/kb_all_documents.json", False),
    ]))
    e.append(sp(0.3))
    e += h2("12.2 Step-by-Step")
    e.append(data_table(
        ["Step", "Command", "What Happens"],
        [
            ["1 — First run", "python run_hyde_rag.py",
             "Builds chroma_db/ (if missing), starts Q&A CLI"],
            ["2 — Ask",       "Type at 'Question:' prompt",
             "LLM generates hypothetical doc → ChromaDB retrieves → LLM answers"],
            ["3 — Inspect",   "Read the 'Hypothetical document:' output",
             "Shows exactly what was embedded for retrieval"],
            ["4 — Exit",      "Type 'exit' or Ctrl+C",   "Session ends"],
            ["5 — Re-index",  "python run_hyde_rag.py --ingest",
             "Rebuilds ChromaDB from scratch"],
        ],
        col_widths=[2.5*cm, 5.5*cm, 8.5*cm],
    ))
    e.append(sp(0.3))
    e += h2("12.3 File Outputs")
    e.append(data_table(
        ["File / Folder", "Created By", "Contents"],
        [
            ["chroma_db/", "build_vector_store()",
             "Persistent ChromaDB: 13,225 vectors + metadata. Shared across all RAG pipelines."],
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
    canvas.drawString(2*cm, 1.2*cm, "HyDE RAG Pipeline — Hypothetical Document Embeddings")
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
        title="HyDE RAG Pipeline — End-to-End Documentation",
    )

    story = []
    story += cover_page()

    story += h1("Table of Contents")
    toc = [
        "1.  What is HyDE RAG?",
        "2.  Libraries Used",
        "3.  End-to-End Architecture",
        "4.  File Structure",
        "5.  hyde_rag/config.py",
        "6.  hyde_rag/ingestion.py — Vector Store",
        "7.  hyde_rag/generator.py — Two-Role LLM Module",
        "8.  hyde_rag/retriever.py — HyDE Retrieval",
        "9.  hyde_rag/pipeline.py — Orchestration",
        "10. run_hyde_rag.py — Entry Point",
        "11. HyDE RAG vs Naive RAG — Comparison",
        "12. Complete Run Guide",
    ]
    for title in toc:
        story.append(Paragraph(title, ParagraphStyle(
            "TOC", fontSize=11, leading=17, leftIndent=0,
            textColor=C_DARK, fontName="Helvetica-Bold", spaceAfter=3)))
    story.append(PageBreak())

    for sec in [sec_overview, sec_libraries, sec_architecture, sec_file_structure,
                sec_config, sec_ingestion, sec_generator, sec_retriever,
                sec_pipeline, sec_entry_point, sec_comparison, sec_how_to_run]:
        story += sec()
        story.append(PageBreak())

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    print(f"PDF written  : {PDF_PATH}")
    print(f"File size    : {PDF_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build_pdf()

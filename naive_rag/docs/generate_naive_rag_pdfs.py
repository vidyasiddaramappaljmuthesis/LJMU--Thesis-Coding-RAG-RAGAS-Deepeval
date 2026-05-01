"""Generate two professional PDFs for Naive RAG documentation and evaluation."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import ListFlowable, ListItem

# ─── Colour palette ───────────────────────────────────────────────────────────
DARK_BLUE   = colors.HexColor("#1A237E")
MID_BLUE    = colors.HexColor("#283593")
ACCENT_BLUE = colors.HexColor("#1565C0")
LIGHT_BLUE  = colors.HexColor("#E3F2FD")
TEAL        = colors.HexColor("#00695C")
LIGHT_TEAL  = colors.HexColor("#E0F2F1")
RED         = colors.HexColor("#B71C1C")
LIGHT_RED   = colors.HexColor("#FFEBEE")
AMBER       = colors.HexColor("#E65100")
LIGHT_AMBER = colors.HexColor("#FFF3E0")
GREY        = colors.HexColor("#424242")
LIGHT_GREY  = colors.HexColor("#F5F5F5")
WHITE       = colors.white
BLACK       = colors.black


def _styles():
    base = getSampleStyleSheet()

    def add(name, **kw):
        if name not in base:
            base.add(ParagraphStyle(name=name, **kw))
        return base[name]

    add("CoverTitle",   parent=base["Title"],    fontSize=28, textColor=WHITE,
        alignment=TA_CENTER, spaceAfter=12, leading=34)
    add("CoverSub",     parent=base["Normal"],   fontSize=14, textColor=LIGHT_BLUE,
        alignment=TA_CENTER, spaceAfter=8,  leading=18)
    add("CoverMeta",    parent=base["Normal"],   fontSize=11, textColor=WHITE,
        alignment=TA_CENTER, spaceAfter=6,  leading=14)

    add("H1",  parent=base["Heading1"], fontSize=18, textColor=DARK_BLUE,
        spaceAfter=10, spaceBefore=16, leading=22)
    add("H2",  parent=base["Heading2"], fontSize=14, textColor=ACCENT_BLUE,
        spaceAfter=8,  spaceBefore=12, leading=18)
    add("H3",  parent=base["Heading3"], fontSize=12, textColor=GREY,
        spaceAfter=6,  spaceBefore=10, leading=15)
    add("Body", parent=base["Normal"],  fontSize=10.5, textColor=BLACK,
        spaceAfter=6, leading=15, alignment=TA_JUSTIFY)
    add("Code", parent=base["Code"],    fontSize=8.5, textColor=colors.HexColor("#212121"),
        backColor=LIGHT_GREY, borderPadding=(4, 6, 4, 6), leading=12)
    add("Bullet", parent=base["Normal"], fontSize=10.5, leftIndent=18,
        bulletIndent=6, spaceAfter=4, leading=14, textColor=BLACK)
    add("Caption", parent=base["Normal"], fontSize=9, textColor=GREY,
        alignment=TA_CENTER, spaceAfter=6, leading=12)
    add("Insight_Pos", parent=base["Normal"], fontSize=10.5, textColor=TEAL,
        backColor=LIGHT_TEAL, borderPadding=(6,8,6,8), leading=14, spaceAfter=8)
    add("Insight_Neg", parent=base["Normal"], fontSize=10.5, textColor=RED,
        backColor=LIGHT_RED, borderPadding=(6,8,6,8), leading=14, spaceAfter=8)
    add("Observation", parent=base["Normal"], fontSize=10.5, textColor=AMBER,
        backColor=LIGHT_AMBER, borderPadding=(6,8,6,8), leading=14, spaceAfter=8)
    return base


def _hr(story, color=ACCENT_BLUE, thickness=1.5):
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=thickness, color=color))
    story.append(Spacer(1, 6))


def _cover_band(story, styles, title_lines, subtitle, meta_lines):
    """Dark-blue cover band with title."""
    cover_data = [[Paragraph("<br/>".join(
        [f'<font color="white"><b>{t}</b></font>' for t in title_lines] +
        [f'<font color="#90CAF9">{subtitle}</font>'] +
        [f'<font color="#BBDEFB">{m}</font>' for m in meta_lines]
    ), styles["CoverTitle"])]]
    tbl = Table(cover_data, colWidths=[17*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 30),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 30),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 20))


def _section(story, styles, title, level=1):
    story.append(Spacer(1, 4))
    story.append(Paragraph(title, styles[f"H{level}"]))
    if level == 1:
        _hr(story, ACCENT_BLUE)


def _body(story, styles, text):
    story.append(Paragraph(text, styles["Body"]))


def _bullet(story, styles, items, bullet="•"):
    for item in items:
        story.append(Paragraph(f"{bullet} {item}", styles["Bullet"]))


def _code(story, styles, lines):
    text = "<br/>".join(lines)
    story.append(Paragraph(text, styles["Code"]))
    story.append(Spacer(1, 6))


def _table(story, data, col_widths=None, header_bg=DARK_BLUE, header_fg=WHITE):
    if col_widths is None:
        n = len(data[0])
        col_widths = [17 * cm / n] * n

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",  (0, 0), (-1, 0), header_fg),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#BDBDBD")),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]
    tbl.setStyle(TableStyle(style))
    story.append(tbl)
    story.append(Spacer(1, 10))


# ══════════════════════════════════════════════════════════════════════════════
#  PDF 1 — IMPLEMENTATION OF NAIVE RAG
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf1(output_path: str):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title="Implementation of Naive RAG"
    )
    S = _styles()
    story = []

    # ── Cover ──
    _cover_band(story, S,
        title_lines=["Implementation of Naive RAG"],
        subtitle="End-to-End Technical Architecture & Pipeline",
        meta_lines=["E-Commerce Analytics System — Olist Brazilian Dataset",
                    "RAGAS & DeepEval Evaluation Framework | May 2026"])
    story.append(Spacer(1, 10))

    # ── Abstract ──
    _section(story, S, "Abstract")
    _body(story, S,
        "This document describes the complete end-to-end implementation of a Naive "
        "Retrieval-Augmented Generation (RAG) system built on the Olist Brazilian "
        "e-commerce dataset. The system ingests 13,225 structured knowledge-base "
        "documents into a ChromaDB vector store, performs semantic retrieval using "
        "the <i>all-MiniLM-L6-v2</i> SentenceTransformer embedding model, and generates "
        "answers with <i>llama-3.3-70b-versatile</i> via the Groq API. The pipeline is "
        "evaluated against a 100-question golden dataset using reference-based metrics "
        "from RAGAS and DeepEval frameworks without any LLM judge calls during "
        "evaluation.")

    # ── 1. System Overview ──
    _section(story, S, "1. System Overview")
    _body(story, S,
        "Naive RAG is the simplest instantiation of retrieval-augmented generation: "
        "a query is embedded, the nearest knowledge-base documents are retrieved by "
        "cosine similarity, and a language model synthesises an answer from those "
        "documents. No query rewriting, hypothetical document generation, or hybrid "
        "retrieval is applied. This baseline provides a performance ceiling against "
        "which more sophisticated variants (Hybrid RAG, HyDE) are compared.")

    _section(story, S, "1.1 Architecture Diagram", 2)
    arch = [
        ["Layer", "Component", "Technology", "Role"],
        ["Data",       "Raw Olist CSV Files",     "9 CSV files / 99K–1M rows each",       "Source data for KB construction"],
        ["Processing", "ETL Pipeline (5 steps)",  "Python / Pandas",                       "Join, enrich, aggregate to KB docs"],
        ["Knowledge",  "kb_all_documents.json",   "13,225 JSON documents",                 "Structured KB across 6 document types"],
        ["Indexing",   "ChromaDB Vector Store",   "all-MiniLM-L6-v2 (384-dim)",            "Cosine-similarity semantic index"],
        ["Retrieval",  "naive_rag/retriever.py",  "ChromaDB query API, top-k=5",           "Dense semantic retrieval"],
        ["Generation", "naive_rag/generator.py",  "llama-3.3-70b-versatile / Groq API",   "Context-grounded answer synthesis"],
        ["Evaluation", "evaluation/ scripts",     "RAGAS + DeepEval + golden dataset",     "Reference-based 0-LLM-judge metrics"],
    ]
    _table(story, arch, col_widths=[2.5*cm, 4*cm, 5*cm, 5.5*cm])

    # ── 2. Data Pipeline ──
    _section(story, S, "2. Data Preparation Pipeline")
    _body(story, S,
        "The Olist dataset comprises nine raw CSV files covering orders, customers, "
        "products, sellers, reviews, geolocation, and payments. A five-step ETL "
        "pipeline transforms these into a structured knowledge base.")

    _section(story, S, "2.1 ETL Steps", 2)
    etl = [
        ["Step", "Script", "Output", "Description"],
        ["1", "step1_load_raw_data.py",     "9 validated DataFrames",   "Load CSVs, validate schema, report nulls"],
        ["2", "step2_join_datasets.py",     "master_joined.csv",        "Star-schema join on order_id / customer_id / product_id"],
        ["3", "step3_enrich_master.py",     "master_enriched.csv",      "Add delivery_days, late_flag, review_score_category"],
        ["4", "step4_build_knowledge_base.py", "kb_all_documents.json", "Aggregate to 6 document-type layers (13,225 docs)"],
        ["5", "step5_build_golden_dataset.py", "golden_dataset.csv",    "Generate 100 Q&A pairs via Gemini Flash LLM"],
    ]
    _table(story, etl, col_widths=[1*cm, 4.5*cm, 4*cm, 7.5*cm])

    _section(story, S, "2.2 Knowledge Base Document Types", 2)
    _body(story, S,
        "The knowledge base is structured into six granularity layers, each "
        "representing a different aggregation level of the e-commerce data:")
    kb_types = [
        ["Layer", "ID Pattern", "Count", "Fields"],
        ["Order",            "order_<uuid>",           "~99K", "Order ID, payment, delivery dates, review, freight"],
        ["Seller",           "seller_<id>",            "~3K",  "Seller city/state, avg review, total revenue, late rate"],
        ["Product Category", "category_<name>",        "~73",  "Category name, total orders, avg review, revenue, late rate"],
        ["Delivery Status",  "delivery_status_<type>", "3",    "early / on_time / late — aggregate stats per group"],
        ["Customer State",   "state_<abbr>",           "~27",  "State-level orders, satisfaction, logistics metrics"],
        ["Temporal",         "month_<YYYY_MM>",        "~25",  "Month-level: total orders, avg payment, top categories"],
    ]
    _table(story, kb_types, col_widths=[3.5*cm, 4.5*cm, 1.5*cm, 7.5*cm])

    _section(story, S, "2.3 Document Format", 2)
    _body(story, S, "Each knowledge-base document follows a consistent JSON schema:")
    _code(story, S, [
        '<font color="#6A1B9A">{</font>',
        '  <font color="#1565C0">"id"</font>: <font color="#2E7D32">"category_health_beauty"</font>,',
        '  <font color="#1565C0">"text"</font>: <font color="#2E7D32">"Document Type: Product Category Summary\\n',
        '           Product Category: health_beauty\\n',
        '           Total Orders: 9,672 | Total Items Sold: 10,934\\n',
        '           Total Revenue: R$ 1,258,044.00 | Late Delivery Rate: 11.23%\\n',
        '           Average Review Score: 4.14 ..."</font>,',
        '  <font color="#1565C0">"metadata"</font>: {',
        '    <font color="#1565C0">"document_type"</font>: <font color="#2E7D32">"product_category"</font>,',
        '    <font color="#1565C0">"product_category"</font>: <font color="#2E7D32">"health_beauty"</font>,',
        '    <font color="#1565C0">"review_score"</font>: 4.14',
        '  }',
        '<font color="#6A1B9A">}</font>',
    ])

    # ── 3. Vector Store ──
    _section(story, S, "3. Vector Store — ChromaDB Ingestion")
    _body(story, S,
        "All 13,225 knowledge-base documents are embedded with "
        "<b>sentence-transformers/all-MiniLM-L6-v2</b> (384-dimensional dense vectors) "
        "and persisted in a ChromaDB PersistentClient using cosine distance as the "
        "similarity metric. The ingestion process uses batch processing of 500 documents "
        "per batch to manage memory efficiently.")

    _section(story, S, "3.1 Ingestion Configuration", 2)
    cfg = [
        ["Parameter", "Value", "Notes"],
        ["Embedding model",     "all-MiniLM-L6-v2",          "384-dim, lightweight, strong semantic quality"],
        ["Distance metric",     "cosine",                      "hnsw:space = cosine in collection metadata"],
        ["Batch size",          "500 documents",               "Avoids memory spikes during large ingestion"],
        ["Persistence path",    "chroma_db/",                  "Persistent across sessions; skip re-embed on restart"],
        ["Collection name",     "ecommerce_kb",                "Dropped and recreated on each build_vector_store() call"],
        ["Total indexed docs",  "13,225",                      "Full KB across all 6 document types"],
    ]
    _table(story, cfg, col_widths=[4*cm, 5*cm, 8*cm])

    _section(story, S, "3.2 Ingestion Code Flow", 2)
    _code(story, S, [
        "<b>build_vector_store(batch_size=500):</b>",
        "  1. Load kb_all_documents.json  →  13,225 dicts",
        "  2. Delete existing collection  (idempotent rebuild)",
        "  3. Create collection with cosine distance",
        "  4. For i in range(0, len(docs), 500):",
        "       col.add(ids=[...], documents=[...], metadatas=[...])",
        "",
        "<b>get_collection():</b>",
        "  → Singleton fast-path; reuses client and collection",
        "  → No re-embedding on subsequent pipeline calls",
    ])

    # ── 4. Retriever ──
    _section(story, S, "4. Retrieval Module")
    _body(story, S,
        "The retriever queries ChromaDB for the top-<i>k</i> nearest documents to an "
        "embedded query vector. All similarity is computed in cosine space, meaning "
        "the returned <i>distance</i> value is 1 − cosine_similarity (lower is more "
        "similar). The default retrieval depth is <b>top-k = 5</b>.")

    _section(story, S, "4.1 Retrieval API", 2)
    _code(story, S, [
        "<b>retrieve(query: str, top_k: int = 5) → list[dict]</b>",
        "",
        "  Result schema per document:",
        "    id        : str   — ChromaDB document ID (e.g. 'category_health_beauty')",
        "    text      : str   — Full document content",
        "    metadata  : dict  — document_type, filters, review_score, ...",
        "    distance  : float — cosine distance (0 = identical, 1 = orthogonal)",
    ])

    _section(story, S, "4.2 Why Cosine Similarity?", 2)
    _body(story, S,
        "Cosine similarity measures the angle between query and document vectors, "
        "making it invariant to document length. This is important because KB documents "
        "vary significantly in length — a single-order summary has ~200 tokens while a "
        "category summary has ~500 tokens. Euclidean (L2) distance would systematically "
        "favour shorter documents.")

    # ── 5. Generator ──
    _section(story, S, "5. Answer Generation Module")
    _body(story, S,
        "The generator receives the original query and the list of retrieved documents, "
        "formats a structured RAG prompt, and calls the <b>llama-3.3-70b-versatile</b> "
        "model via the Groq API. Low temperature (0.1) is used to ensure factual, "
        "deterministic responses.")

    _section(story, S, "5.1 Prompt Template", 2)
    _code(story, S, [
        "<b>System:</b> You are a helpful e-commerce data assistant.",
        "         Answer questions using only the provided context.",
        "         If the answer cannot be found in the context, say so clearly.",
        "",
        "<b>User:</b>   Context:",
        "         [Document 1]: {retrieved_docs[0]['text']}",
        "         [Document 2]: {retrieved_docs[1]['text']}",
        "         ...  (up to top-k documents)",
        "",
        "         Question: {query}",
        "         Answer:",
    ])

    _section(story, S, "5.2 LLM Configuration", 2)
    llm_cfg = [
        ["Parameter", "Value", "Rationale"],
        ["Model",         "llama-3.3-70b-versatile",   "State-of-the-art open model via Groq; fast inference"],
        ["Temperature",   "0.1",                        "Low randomness for factual e-commerce queries"],
        ["Max tokens",    "512",                         "Sufficient for analytical answers; prevents runaway generation"],
        ["API provider",  "Groq",                        "Low-latency inference; 5 parallel API key slots for throughput"],
    ]
    _table(story, llm_cfg, col_widths=[3*cm, 5.5*cm, 8.5*cm])

    # ── 6. Multi-Key API Strategy ──
    _section(story, S, "6. Groq API Key Management")
    _body(story, S,
        "To sustain throughput across 100 queries without hitting rate limits, the "
        "system maintains a pool of Groq API keys loaded from the "
        "<code>GROQ_API_KEYS</code> environment variable (comma-separated). The first "
        "five keys act as primary keys assigned to batches 1–5; additional keys serve "
        "as fallbacks if a primary is exhausted.")

    key_tbl = [
        ["Key Slot", "Query Range", "On Primary Exhaustion"],
        ["Key #1",  "Queries 1–20",   "Falls back to keys #6, #7, ... in order"],
        ["Key #2",  "Queries 21–40",  "Falls back to keys #6, #7, ... in order"],
        ["Key #3",  "Queries 41–60",  "Falls back to keys #6, #7, ... in order"],
        ["Key #4",  "Queries 61–80",  "Falls back to keys #6, #7, ... in order"],
        ["Key #5",  "Queries 81–100", "Falls back to keys #6, #7, ... in order"],
    ]
    _table(story, key_tbl, col_widths=[3*cm, 4.5*cm, 9.5*cm])

    _section(story, S, "6.1 Error Classification", 2)
    err_tbl = [
        ["Error Type", "HTTP Status", "Handling"],
        ["Organization restricted",  "400",  "Permanent ban — key removed from pool"],
        ["Daily token limit (TPD)",  "429",  "Session ban — key skipped for this run"],
        ["Per-minute rate limit",    "429",  "Transient skip — key retried after delay"],
        ["API connection error",     "5xx",  "Transient — rotates to next key"],
    ]
    _table(story, err_tbl, col_widths=[5*cm, 3*cm, 9*cm])

    story.append(PageBreak())

    # ── 7. Pipeline Integration ──
    _section(story, S, "7. End-to-End Pipeline")
    _body(story, S,
        "The <code>naive_rag/pipeline.py</code> module ties retrieval and generation "
        "into a single callable function. The entry-point script <code>run_naive_rag.py</code> "
        "adds vector-store initialisation and an interactive Q&A loop.")

    _section(story, S, "7.1 Pipeline Execution Flow", 2)
    flow = [
        ["Step", "Function", "Input → Output"],
        ["0 — Init",       "build_vector_store()",  "kb_all_documents.json → ChromaDB collection"],
        ["1 — Retrieve",   "retrieve(query, k=5)",  "query string → list[dict{id,text,metadata,distance}]"],
        ["2 — Generate",   "generate(query, docs)", "query + docs → answer string"],
        ["3 — Return",     "run_rag(query)",         "query → {query, answer, retrieved_docs}"],
    ]
    _table(story, flow, col_widths=[2.5*cm, 5*cm, 9.5*cm])

    _section(story, S, "7.2 Interactive Entry Point", 2)
    _code(story, S, [
        "$ python run_naive_rag.py",
        "",
        "  → Checks if ChromaDB index exists; builds if missing",
        "  → Enters interactive loop:",
        "",
        "  Question: What is the late delivery rate for health_beauty?",
        "  Answer  : The late delivery rate for the health_beauty category is 11.23%.",
        "",
        "  Retrieved Documents:",
        "    [1]  dist=0.42  category_health_beauty",
        "    [2]  dist=0.55  seller_3b4c... (health_beauty seller)",
        "    ...",
        "",
        "$ python run_naive_rag.py --ingest   # Force re-index without interactive mode",
    ])

    # ── 8. Project Dependencies ──
    _section(story, S, "8. Key Dependencies")
    deps = [
        ["Library", "Version", "Purpose"],
        ["chromadb",              "≥0.5.0",  "Persistent vector store with HNSW index"],
        ["sentence-transformers", "≥3.0.0",  "all-MiniLM-L6-v2 embedding model"],
        ["groq",                  "≥0.11.0", "Groq API client for LLM inference"],
        ["pandas / numpy",        "≥2.0 / ≥1.24", "Data manipulation and ETL"],
        ["ragas",                 "≥0.2.0",  "RAGAS evaluation framework"],
        ["deepeval",              "≥1.0.0",  "DeepEval evaluation framework"],
        ["scikit-learn",          "latest",  "TF-IDF vectoriser and cosine similarity"],
        ["openpyxl",              "≥3.1.0",  "Excel export for evaluation results"],
        ["reportlab",             "≥4.0.0",  "PDF report generation"],
        ["google-genai",          "≥1.0.0",  "Gemini Flash for golden dataset generation"],
        ["python-dotenv",         "≥1.0.0",  "Environment variable management"],
    ]
    _table(story, deps, col_widths=[5*cm, 3.5*cm, 8.5*cm])

    # ── 9. Configuration Reference ──
    _section(story, S, "9. Configuration Reference")
    conf = [
        ["Config Key", "Default", "Description"],
        ["GROQ_API_KEYS",   "env var",            "Comma-separated list of Groq API keys"],
        ["EMBEDDING_MODEL", "all-MiniLM-L6-v2",   "SentenceTransformer model name"],
        ["COLLECTION_NAME", "ecommerce_kb",        "ChromaDB collection identifier"],
        ["CHROMA_DB_PATH",  "chroma_db/",          "Filesystem path for persistent vector store"],
        ["KB_ALL_DOCS",     "dataset/knowledge_base/kb_all_documents.json", "Knowledge base source file"],
        ["TOP_K",           "5",                   "Number of documents retrieved per query"],
        ["LLM_MODEL",       "llama-3.3-70b-versatile", "Groq LLM model identifier"],
        ["TEMPERATURE",     "0.1",                 "LLM sampling temperature"],
        ["MAX_TOKENS",      "512",                 "Maximum LLM response tokens"],
    ]
    _table(story, conf, col_widths=[5*cm, 4.5*cm, 7.5*cm])

    # ── 10. Summary ──
    _section(story, S, "10. Summary")
    _body(story, S,
        "The Naive RAG pipeline provides a clean, interpretable baseline for "
        "retrieval-augmented question answering over structured e-commerce data. "
        "Its design priorities are simplicity and reproducibility: a single dense "
        "retrieval step, a minimal prompt template, and a low-temperature language "
        "model. This makes it the ideal comparison point for evaluating the marginal "
        "gains from Hybrid retrieval (BM25 + semantic) and Hypothetical Document "
        "Embeddings (HyDE).")

    doc.build(story)
    print(f"[PDF 1] Saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PDF 2 — HOW TO EVALUATE NAIVE RAG
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf2(output_path: str):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title="Evaluation of Naive RAG"
    )
    S = _styles()
    story = []

    # ── Cover ──
    _cover_band(story, S,
        title_lines=["Evaluation of Naive RAG"],
        subtitle="Methodology, Metrics, Results & Research Insights",
        meta_lines=["Reference-Based Evaluation — 100 Queries — Olist E-Commerce Dataset",
                    "RAGAS & DeepEval Frameworks | May 2026"])
    story.append(Spacer(1, 10))

    # ── Abstract ──
    _section(story, S, "Abstract")
    _body(story, S,
        "This document presents a comprehensive evaluation framework for the Naive RAG "
        "system, covering evaluation philosophy, metric definitions, implementation "
        "details, experimental results, and research-grade insights. All 11 metrics "
        "across RAGAS and DeepEval frameworks are computed without an LLM judge — "
        "instead relying on a 100-question golden dataset generated by Gemini Flash as "
        "the reference oracle. The evaluation reveals systematic weaknesses in temporal "
        "and aggregate-level retrieval, LLM verbosity misalignment, and high "
        "hallucination rates that point to concrete opportunities for improvement.")

    # ── 1. Evaluation Philosophy ──
    _section(story, S, "1. Evaluation Philosophy")
    _body(story, S,
        "Traditional RAG evaluation with LLM-as-judge is expensive, slow, and introduces "
        "its own bias (the judge LLM may agree with plausible-sounding but incorrect "
        "answers). This evaluation adopts a <b>reference-based approach</b>: a golden "
        "dataset of 100 high-quality Q&A pairs — each with an expected answer, expected "
        "context, and expected source document IDs — serves as the ground truth. Metrics "
        "are computed using deterministic algorithms (token overlap, TF-IDF, ROUGE-L, "
        "exact ID matching) with zero additional LLM calls at evaluation time.")

    phil = [
        ["Principle", "Implementation"],
        ["No LLM judge",        "All metrics computed with deterministic algorithms; 0 evaluation LLM calls"],
        ["Reference-based",     "Golden dataset (expected_answer + expected_source_ids) as ground truth oracle"],
        ["Framework parity",    "Same underlying computation for RAGAS and DeepEval where metrics overlap"],
        ["Exact ID matching",   "Context precision/recall use ChromaDB document IDs, not fuzzy text overlap"],
        ["Verbosity-robustness","ROUGE-L (LCS) for factual correctness — handles verbose answers fairly"],
    ]
    _table(story, phil, col_widths=[5*cm, 12*cm])

    # ── 2. Golden Dataset ──
    _section(story, S, "2. Golden Dataset")
    _body(story, S,
        "The golden dataset contains 100 question-answer pairs generated by "
        "<b>Gemini Flash</b> from knowledge-base documents. Each record includes "
        "structured metadata enabling fine-grained performance analysis.")

    _section(story, S, "2.1 Schema", 2)
    schema = [
        ["Column", "Type", "Example"],
        ["question_id",         "str",       "q001"],
        ["question",            "str",       "What is the late delivery rate for the portateis_cozinha...?"],
        ["expected_answer",     "str",       "7.69%"],
        ["expected_context",    "JSON list", "[\"Document Type: Product Category Summary\\n...\"]"],
        ["expected_source_ids", "JSON list", "[\"category_portateis_cozinha_e_preparadores_de_alimentos\"]"],
        ["question_type",       "enum",      "factual | analytical | comparison"],
        ["difficulty",          "enum",      "easy | medium | hard"],
        ["best_kb_layer",       "enum",      "category | seller | order | state | month | delivery_status"],
    ]
    _table(story, schema, col_widths=[4.5*cm, 2.5*cm, 10*cm])

    _section(story, S, "2.2 Dataset Distribution", 2)
    dist = [
        ["Dimension", "Breakdown"],
        ["Question type",  "Factual (~40%), Analytical (~45%), Comparison (~15%)"],
        ["Difficulty",     "Easy (~35%), Medium (~40%), Hard (~25%)"],
        ["KB layer",       "Category (~20%), Seller (~20%), Order (~20%), State (~15%), Month (~15%), Delivery (~10%)"],
        ["Total pairs",    "100 questions with unique expected answers and source IDs"],
    ]
    _table(story, dist, col_widths=[4*cm, 13*cm])

    # ── 3. Metric Definitions ──
    story.append(PageBreak())
    _section(story, S, "3. Metric Definitions")
    _body(story, S,
        "Eleven metrics are computed for each query — five from the RAGAS framework "
        "and six from DeepEval. Metrics that overlap between frameworks share the "
        "same computation.")

    _section(story, S, "3.1 RAGAS Metrics", 2)

    _section(story, S, "Faithfulness (RAGAS)", 3)
    _body(story, S,
        "<b>Definition:</b> Fraction of answer sentences whose content is supported "
        "by the retrieved context. A sentence is 'supported' if ≥50% of its content "
        "tokens (length &gt; 1) appear in the combined retrieved context.<br/>"
        "<b>Formula:</b> supported_sentences / total_sentences<br/>"
        "<b>Standard motivation:</b> Whole-answer token recall over-credits shared "
        "structural vocabulary ('Document', 'Total', 'Orders'). Sentence-level "
        "granularity correctly penalises answers that mix correct and hallucinated claims.")

    _section(story, S, "Answer Relevancy (RAGAS)", 3)
    _body(story, S,
        "<b>Definition:</b> How relevant is the generated answer to the original "
        "question? Computed as TF-IDF cosine similarity between the generated answer "
        "and the question text.<br/>"
        "<b>Formula:</b> cosine(TF-IDF(generated_answer), TF-IDF(question))<br/>"
        "<b>Standard motivation:</b> Measures topical alignment with what was asked, "
        "not accuracy against the expected answer. A verbose but on-topic answer "
        "scores higher than an off-topic concise answer.")

    _section(story, S, "Context Precision (RAGAS)", 3)
    _body(story, S,
        "<b>Definition:</b> Average Precision at k (AP@k) — rewards systems that "
        "rank relevant documents higher in the retrieval list.<br/>"
        "<b>Formula:</b> AP@k = (1/R) × Σ(k=1 to K) [P@k × rel(k)] where R = total "
        "relevant docs and rel(k) = 1 if doc at rank k matches an expected_source_id.<br/>"
        "<b>Standard motivation:</b> Simple precision (relevant_retrieved / total_retrieved) "
        "ignores rank order. AP@k penalises systems that find the right document at "
        "rank 5 instead of rank 1.")

    _section(story, S, "Context Recall (RAGAS)", 3)
    _body(story, S,
        "<b>Definition:</b> How much of the expected answer's vocabulary is present "
        "in the retrieved context?<br/>"
        "<b>Formula:</b> |tokens(expected_answer) ∩ tokens(combined_retrieved)| / "
        "|tokens(expected_answer)|<br/>"
        "<b>Standard motivation:</b> Measures whether the retrieved context contains "
        "the information needed to produce the correct answer.")

    _section(story, S, "Factual Correctness (RAGAS)", 3)
    _body(story, S,
        "<b>Definition:</b> How factually close is the generated answer to the "
        "expected answer? Computed as ROUGE-L F1 — the longest common subsequence "
        "between the two texts.<br/>"
        "<b>Formula:</b> ROUGE-L F1 = 2 × prec_LCS × rec_LCS / (prec_LCS + rec_LCS) "
        "where prec_LCS = LCS/len(pred), rec_LCS = LCS/len(ref)<br/>"
        "<b>Standard motivation:</b> Raw token F1 is symmetric and penalises verbose "
        "correct answers (e.g., 'The rate is 7.69%' vs expected '7.69%'). ROUGE-L "
        "handles verbosity via LCS alignment.")

    _section(story, S, "3.2 DeepEval Metrics", 2)

    de_metrics = [
        ["Metric", "Formula", "PASS Threshold"],
        ["Answer Relevancy",       "TF-IDF cosine(generated_answer, question)",             "≥ 0.5"],
        ["Faithfulness",           "Sentence-support fraction (same as RAGAS)",             "≥ 0.5"],
        ["Contextual Precision",   "AP@k with exact ID match (same as RAGAS)",              "≥ 0.5"],
        ["Contextual Recall",      "Token recall of expected_answer in context (same as RAGAS)", "≥ 0.5"],
        ["Contextual Relevancy",   "Mean TF-IDF cosine(each_retrieved_doc, question)",      "≥ 0.5"],
        ["Hallucination",          "1.0 − sentence_faithfulness",                           "≤ 0.5 (lower = better)"],
    ]
    _table(story, de_metrics, col_widths=[4.5*cm, 8*cm, 4.5*cm])

    _body(story, S,
        "<b>Contextual Relevancy</b> is the only metric unique to DeepEval vs RAGAS: "
        "it measures whether each individual retrieved document is topically relevant "
        "to the question (not just whether the correct document was retrieved). This "
        "exposes cases where documents are retrieved because of structural keyword "
        "overlap, not semantic relevance.")

    # ── 4. Evaluation Pipeline ──
    story.append(PageBreak())
    _section(story, S, "4. Evaluation Pipeline Architecture")
    _body(story, S,
        "The evaluation script (<code>evaluation/run_naive_rag_eval.py</code>) "
        "orchestrates five parallel evaluation batches, each driven by a dedicated "
        "Groq API key. Within each batch, queries are processed sequentially with "
        "random 5–20 second delays to respect per-minute rate limits.")

    _section(story, S, "4.1 Per-Query Evaluation Steps", 2)
    steps = [
        ["Step", "Operation", "LLM Calls", "Output"],
        ["1 — Retrieve",  "ChromaDB cosine-similarity search, top-k=5",           "0", "5 documents + IDs + distances"],
        ["2 — Generate",  "Groq LLM call with context + question",                 "1", "generated_answer string"],
        ["3 — RAGAS",     "_compute_metrics() — all 5 RAGAS scores",              "0", "faithfulness, relevancy, precision, recall, correctness"],
        ["4 — DeepEval",  "_compute_metrics() reuse — all 6 DeepEval scores",     "0", "relevancy, faithfulness, precision, recall, relevancy, hallucination"],
    ]
    _table(story, steps, col_widths=[2.5*cm, 5.5*cm, 1.5*cm, 7.5*cm])

    _section(story, S, "4.2 Parallel Batch Architecture", 2)
    _code(story, S, [
        "ThreadPoolExecutor(max_workers=5)",
        "  │",
        "  ├─ Thread 1 → Key #1 → Queries 1-20   (sequential, 5-20s delay)",
        "  ├─ Thread 2 → Key #2 → Queries 21-40  (sequential, 5-20s delay)",
        "  ├─ Thread 3 → Key #3 → Queries 41-60  (sequential, 5-20s delay)",
        "  ├─ Thread 4 → Key #4 → Queries 61-80  (sequential, 5-20s delay)",
        "  └─ Thread 5 → Key #5 → Queries 81-100 (sequential, 5-20s delay)",
        "",
        "Total wall time ≈ max(batch_time) ≈ 20 × 12.5s avg ≈ ~4.5 minutes",
        "Total Groq calls: 100 (1 per query, generation only)",
    ])

    # ── 5. Results ──
    story.append(PageBreak())
    _section(story, S, "5. Experimental Results")
    _body(story, S,
        "The full evaluation was conducted on all 100 golden dataset queries across "
        "five parallel batches on 1 May 2026. Results are presented below as mean "
        "scores across all 100 queries.")

    _section(story, S, "5.1 Aggregate Scores", 2)
    results = [
        ["Framework", "Metric", "Mean Score", "Interpretation"],
        ["RAGAS",    "Faithfulness",         "0.268", "Only 27% of answer sentences grounded in context"],
        ["RAGAS",    "Answer Relevancy",     "0.508", "Moderate — answers are on-topic but verbose"],
        ["RAGAS",    "Context Precision",    "0.322", "~1/3 of queries retrieve the right doc at top ranks"],
        ["RAGAS",    "Context Recall",       "0.334", "~1/3 of expected-answer vocabulary found in context"],
        ["RAGAS",    "Factual Correctness",  "0.116", "Low ROUGE-L due to verbosity mismatch"],
        ["DeepEval", "Answer Relevancy",     "0.508", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Faithfulness",         "0.268", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Contextual Precision", "0.322", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Contextual Recall",    "0.334", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Contextual Relevancy", "0.123", "Very low — retrieved docs not topically aligned"],
        ["DeepEval", "Hallucination",        "0.732", "73% of answer sentences unsupported by context"],
    ]
    _table(story, results, col_widths=[2.5*cm, 4.5*cm, 2.5*cm, 7.5*cm])

    _section(story, S, "5.2 Performance by Query Type", 2)
    qtype = [
        ["Query Type", "Typical Context Precision", "Typical Factual Correctness", "Observation"],
        ["Factual / Easy",      "0.8–1.0",  "0.16–0.20", "Right doc retrieved, but ROUGE-L penalises verbose answer"],
        ["Analytical / Medium", "0.0–0.5",  "0.10–0.20", "Query requires aggregate doc; order-level docs retrieved instead"],
        ["Comparison / Hard",   "0.0–0.33", "0.10–0.30", "Multi-source queries — partial source match common"],
    ]
    _table(story, qtype, col_widths=[3.5*cm, 3.5*cm, 3.5*cm, 6.5*cm])

    _section(story, S, "5.3 Performance by KB Layer", 2)
    layer = [
        ["KB Layer Required", "Context Precision", "Notes"],
        ["category_*",           "High (~0.8)",   "Category docs are rare (73) → semantic search finds them reliably"],
        ["delivery_status_*",    "High (~0.9)",   "Only 3 unique delivery-status docs → easy to retrieve"],
        ["state_*",              "Medium (~0.5)", "27 state docs among 13K+ → occasional confusion"],
        ["month_*",              "Low (~0.15)",   "Temporal queries frequently retrieve order-level docs instead of month summaries"],
        ["seller_*",             "Low (~0.10)",   "3K+ seller docs; without exact seller ID in query, wrong doc retrieved"],
        ["order_*",              "Low (~0.05)",   "99K+ order docs; only queries with exact order ID retrieve correctly"],
    ]
    _table(story, layer, col_widths=[4.5*cm, 3.5*cm, 9*cm])

    # ── 6. Top 5 Positive Insights ──
    story.append(PageBreak())
    _section(story, S, "6. Top 5 Positive Research Insights")
    _body(story, S,
        "Despite modest aggregate scores, the evaluation reveals several noteworthy "
        "strengths of the Naive RAG implementation:")

    pos_insights = [
        ("Category & Delivery-Status Queries Excel",
         "For the 73 product-category documents and 3 delivery-status documents, "
         "Naive RAG achieves Context Precision of 0.8–1.0. When the KB has few documents "
         "of a type, semantic search reliably surfaces the correct one. Queries like "
         "'What is the average review score for health_beauty?' (q008, q028) and delivery "
         "status queries (q011, q016) consistently achieve CP=1.0 and CR=1.0, demonstrating "
         "that the embedding model captures category vocabulary precisely."),

        ("LLM Stays On-Topic (Answer Relevancy = 0.508)",
         "The LLaMA-3.3-70b model, even when the retrieved context is wrong, "
         "generates answers that remain relevant to the question domain. The 0.508 "
         "Answer Relevancy score means the model does not hallucinate entirely "
         "off-topic responses — it attempts to answer the right question. This "
         "indicates the prompt template and low-temperature setting are effective "
         "at keeping generation focused."),

        ("AP@k Captures Ranking Quality",
         "The use of AP@k (Average Precision at k) instead of simple precision reveals "
         "that when the system does retrieve a relevant document, it is often ranked "
         "first (rank 1). For queries q001, q002, q008, q022, q031 the relevant document "
         "is at position 1, giving perfect AP@k=1.0. This confirms the embedding quality "
         "for direct entity queries is strong."),

        ("Reference-Based Evaluation is Consistent and Reproducible",
         "The zero-LLM-judge evaluation architecture produces fully deterministic "
         "scores. Running the same evaluation twice produces identical results, "
         "unlike LLM-as-judge systems where temperature introduces variance. The "
         "golden dataset with exact source IDs enables precise measurement of "
         "retrieval failures vs generation failures separately."),

        ("Correct Factual Retrieval for Direct Entity Queries",
         "For queries asking about a specific named entity (e.g., a product category, "
         "a seller ID, a delivery status), the top-1 retrieved document is correct "
         "~60% of the time. This establishes a solid retrieval baseline that Hybrid "
         "RAG (BM25 + semantic) can build upon by improving exact-match retrieval "
         "for less common entity names."),
    ]

    for i, (title, body) in enumerate(pos_insights, 1):
        story.append(Spacer(1, 6))
        banner = [[Paragraph(f'<font color="white"><b>✓ Insight #{i}: {title}</b></font>', S["Body"])]]
        tbl = Table(banner, colWidths=[17*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), TEAL),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
        ]))
        story.append(tbl)
        _body(story, S, body)

    # ── 7. Top 5 Negative Insights ──
    _section(story, S, "7. Top 5 Negative Research Insights")
    _body(story, S,
        "The evaluation exposes fundamental limitations of naive single-stage "
        "dense retrieval that motivate more advanced RAG architectures:")

    neg_insights = [
        ("Temporal Queries Catastrophically Fail (Context Precision ≈ 0.0)",
         "Queries requiring temporal aggregate documents (e.g., 'What was the total "
         "payment value for March 2017?' → needs month_2017_03) almost never retrieve "
         "the correct document. The KB contains 25 month-level docs among 13,225 total "
         "documents. The cosine similarity of a general time-period query is higher with "
         "individual order-level documents that happen to be from that period than with "
         "the aggregate monthly summary. This is a fundamental limitation of pure "
         "semantic retrieval without metadata filtering."),

        ("Hallucination Rate = 0.732 — LLM Adds Unsupported Content",
         "73.2% of generated answer sentences cannot be traced back to the retrieved "
         "context (sentence faithfulness = 0.268). Even when the correct context is "
         "retrieved, the LLM fabricates additional details, commentary, or cross-document "
         "inferences not present in the top-5 documents. This is particularly acute for "
         "analytical questions where the model is asked to 'compare' or 'relate' metrics "
         "across multiple entities."),

        ("Factual Correctness = 0.116 — Verbosity Penalty is Severe",
         "The golden dataset contains concise numeric answers ('7.69%', '22428.70') "
         "while the LLM generates verbose paragraph explanations. ROUGE-L F1 correctly "
         "does not fully penalise this (it would score 0.20 for 'The rate is 7.69%' vs "
         "expected '7.69%'), but the verbosity gap is still substantial. The system "
         "has no output length control mechanism — a future improvement would be "
         "to add instruction prompting for concise numeric answers on factual queries."),

        ("Exact Seller/Order Retrieval Requires IDs (Context Precision ≈ 0.05)",
         "Queries about specific sellers (by ID) or individual orders (by UUID) "
         "have near-zero context precision unless the exact ID appears in the query. "
         "Semantic similarity between 'What is the revenue for seller 640e21a7...' "
         "and the correct seller document is low because seller UUIDs are not "
         "semantically meaningful tokens. BM25 sparse retrieval would solve this "
         "with exact-match keyword lookup."),

        ("Contextual Relevancy = 0.123 — Structural Vocabulary Dominates Embeddings",
         "All KB documents share structural boilerplate: 'Document Type:', 'Total "
         "Orders:', 'Average Score:', 'Late Delivery Rate:'. This shared vocabulary "
         "causes all documents to cluster together in embedding space, making it "
         "difficult to discriminate between, e.g., a health_beauty category document "
         "and a random order document. The 0.123 Contextual Relevancy score (how "
         "relevant are retrieved docs to the question) confirms that most retrieved "
         "documents are structurally similar but topically mismatched."),
    ]

    for i, (title, body) in enumerate(neg_insights, 1):
        story.append(Spacer(1, 6))
        banner = [[Paragraph(f'<font color="white"><b>✗ Issue #{i}: {title}</b></font>', S["Body"])]]
        tbl = Table(banner, colWidths=[17*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), RED),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
        ]))
        story.append(tbl)
        _body(story, S, body)

    # ── 8. Key Observations ──
    story.append(PageBreak())
    _section(story, S, "8. Major Key Observations")
    _body(story, S,
        "The following observations synthesise the quantitative results into "
        "actionable research conclusions:")

    observations = [
        ("Retrieval Quality is the Dominant Failure Mode",
         "Context Precision (0.322) and Context Recall (0.334) are both below 0.35, "
         "meaning the retrieval step fails for the majority of queries. Generation "
         "quality (Answer Relevancy = 0.508) is comparatively stronger, suggesting "
         "the LLM can generate reasonable answers when given good context — the "
         "system's ceiling is retrieval, not generation."),

        ("Query Complexity Stratification is Sharp",
         "Easy factual queries (asking for a single metric about a named category or "
         "delivery status) achieve Context Precision of 0.8–1.0, while hard comparison "
         "and analytical queries that require multi-document reasoning achieve "
         "0.0–0.33. The performance distribution is bimodal — not a gradual degradation. "
         "This suggests that the KB structure (6 document-type layers) is well-designed "
         "for simple lookups but insufficient for cross-layer reasoning."),

        ("KB Document Imbalance Creates Systematic Retrieval Bias",
         "Order-level documents (99K+) vastly outnumber aggregate-level documents "
         "(73 category docs, 27 state docs, 25 month docs). With cosine-similarity "
         "retrieval, a query about 'March 2017' retrieves individual March 2017 orders "
         "rather than the month_2017_03 aggregate summary, because the aggregate "
         "document is statistically rare in the embedding space. A metadata-filtered "
         "retrieval strategy (filter by document_type before ranking) would directly "
         "address this."),

        ("LLM Adds Cross-Document Reasoning at the Cost of Faithfulness",
         "For analytical queries, the model attempts to synthesise information across "
         "retrieved documents — an appropriate behaviour — but does so by drawing on "
         "its parametric knowledge rather than strictly the context. This explains "
         "the high Hallucination score (0.732): the model is not making random errors "
         "but is deliberately going beyond the context to provide richer analysis. A "
         "stricter system prompt ('Do not provide any information beyond what is "
         "explicitly stated in the documents') would reduce hallucination but might "
         "also reduce answer quality for analytical queries."),

        ("RAGAS and DeepEval Metrics Converge — Framework Choice is Secondary",
         "Because RAGAS and DeepEval metrics are computed with identical formulas, "
         "their results are perfectly correlated (r=1.0 for matching metric pairs). "
         "This validates the reference-based approach: the metric scores are stable "
         "and framework-agnostic. The key methodological contribution is the use of "
         "exact ChromaDB document ID matching for context precision/recall rather "
         "than token-overlap thresholds, which eliminates the false-positive inflation "
         "caused by shared structural KB vocabulary."),

        ("Baseline Establishes Concrete Improvement Targets",
         "Naive RAG scores define the baseline: Context Precision 0.322, "
         "Faithfulness 0.268, Factual Correctness 0.116. Any enhanced retrieval "
         "approach (Hybrid RAG: BM25 + dense; HyDE: hypothetical document expansion) "
         "should be evaluated against these baselines. Based on the failure analysis, "
         "Hybrid RAG is expected to improve seller/order ID queries (+BM25 exact match) "
         "and temporal queries (+BM25 date token overlap), while HyDE may improve "
         "analytical and comparison queries by better representing complex information "
         "needs in the embedding space."),
    ]

    for i, (title, body) in enumerate(observations, 1):
        story.append(Spacer(1, 6))
        banner = [[Paragraph(f'<font color="white"><b>◆ Observation #{i}: {title}</b></font>', S["Body"])]]
        tbl = Table(banner, colWidths=[17*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), AMBER),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
        ]))
        story.append(tbl)
        _body(story, S, body)

    # ── 9. Recommendations ──
    _section(story, S, "9. Recommendations for Future Work")
    recs = [
        ["Priority", "Recommendation", "Expected Impact"],
        ["High",   "Metadata-filtered retrieval: pre-filter by document_type based on query classification before ranking", "Context Precision +0.15–0.25 for temporal/state queries"],
        ["High",   "Hybrid retrieval (BM25 + dense): add sparse matching to handle exact entity IDs (seller/order UUIDs)", "Context Precision +0.10–0.20 for seller/order queries"],
        ["Medium", "Output length control: add 'Answer in one sentence' instruction for factual queries", "Factual Correctness (ROUGE-L) +0.10–0.20"],
        ["Medium", "HyDE (Hypothetical Document Embeddings): expand query to hypothetical answer, embed that instead", "Context Precision +0.05–0.15 for analytical queries"],
        ["Low",    "Stricter system prompt: 'Answer only from the provided documents'", "Hallucination −0.10–0.20 (with possible Answer Relevancy cost)"],
    ]
    _table(story, recs, col_widths=[2*cm, 9*cm, 6*cm])

    # ── 10. Conclusion ──
    _section(story, S, "10. Conclusion")
    _body(story, S,
        "The Naive RAG evaluation reveals a system that performs well on simple "
        "entity-lookup queries but struggles systematically with temporal aggregates, "
        "exact-ID retrieval, and multi-document reasoning. The reference-based "
        "evaluation framework — using golden dataset source IDs for exact ID matching "
        "and deterministic metric computation — provides a reliable, reproducible, "
        "and LLM-free assessment methodology that can be extended to Hybrid RAG "
        "and HyDE evaluations with the same infrastructure.")
    _body(story, S,
        "The key finding is that <b>retrieval quality, not generation quality, is the "
        "primary bottleneck</b>. With Context Precision and Recall both at ~0.33, the "
        "LLM is answering from wrong or incomplete context 67% of the time. Addressing "
        "retrieval failures — through metadata filtering, hybrid search, or query "
        "expansion — should be the immediate priority for improving Naive RAG "
        "performance on this e-commerce dataset.")

    _hr(story, DARK_BLUE, 2)
    story.append(Spacer(1, 6))
    _body(story, S,
        "<i>Evaluation conducted: 1 May 2026 | Model: llama-3.3-70b-versatile | "
        "Dataset: 100 Olist e-commerce Q&A pairs | Evaluation: reference-based, "
        "0 LLM judge calls | Frameworks: RAGAS + DeepEval</i>")

    doc.build(story)
    print(f"[PDF 2] Saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base, exist_ok=True)

    pdf1 = os.path.join(base, "Implementation_of_Naive_RAG.pdf")
    pdf2 = os.path.join(base, "Evaluation_of_Naive_RAG.pdf")

    # Delete old file if present
    old = os.path.join(base, "Naive_RAG_Documentation.pdf")
    if os.path.exists(old):
        os.remove(old)
        print(f"[DEL] Removed old file: {old}")

    build_pdf1(pdf1)
    build_pdf2(pdf2)
    print("\nDone. Both PDFs are in the docs/ folder.")

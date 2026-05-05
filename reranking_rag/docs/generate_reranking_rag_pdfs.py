"""Generate two professional PDFs for Reranking RAG documentation and evaluation."""

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
DARK_RED    = colors.HexColor("#B71C1C")
MID_RED     = colors.HexColor("#C62828")
ACCENT_RED  = colors.HexColor("#D32F2F")
LIGHT_RED   = colors.HexColor("#FFEBEE")
TEAL        = colors.HexColor("#00695C")
LIGHT_TEAL  = colors.HexColor("#E0F2F1")
RED         = colors.HexColor("#B71C1C")
AMBER       = colors.HexColor("#E65100")
LIGHT_AMBER = colors.HexColor("#FFF3E0")
GREY        = colors.HexColor("#424242")
LIGHT_GREY  = colors.HexColor("#F5F5F5")
WHITE       = colors.white
BLACK       = colors.black
DARK_BLUE   = colors.HexColor("#1A237E")


def _styles():
    base = getSampleStyleSheet()

    def add(name, **kw):
        if name not in base:
            base.add(ParagraphStyle(name=name, **kw))
        return base[name]

    add("CoverTitle",   parent=base["Title"],    fontSize=28, textColor=WHITE,
        alignment=TA_CENTER, spaceAfter=12, leading=34)
    add("CoverSub",     parent=base["Normal"],   fontSize=14, textColor=LIGHT_RED,
        alignment=TA_CENTER, spaceAfter=8,  leading=18)
    add("CoverMeta",    parent=base["Normal"],   fontSize=11, textColor=WHITE,
        alignment=TA_CENTER, spaceAfter=6,  leading=14)

    add("H1",  parent=base["Heading1"], fontSize=18, textColor=DARK_RED,
        spaceAfter=10, spaceBefore=16, leading=22)
    add("H2",  parent=base["Heading2"], fontSize=14, textColor=ACCENT_RED,
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


def _hr(story, color=ACCENT_RED, thickness=1.5):
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=thickness, color=color))
    story.append(Spacer(1, 6))


def _cover_band(story, styles, title_lines, subtitle, meta_lines):
    cover_data = [[Paragraph("<br/>".join(
        [f'<font color="white"><b>{t}</b></font>' for t in title_lines] +
        [f'<font color="#FFCDD2">{subtitle}</font>'] +
        [f'<font color="#FFEBEE">{m}</font>' for m in meta_lines]
    ), styles["CoverTitle"])]]
    tbl = Table(cover_data, colWidths=[17*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_RED),
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
        _hr(story, ACCENT_RED)


def _body(story, styles, text):
    story.append(Paragraph(text, styles["Body"]))


def _bullet(story, styles, items, bullet="•"):
    for item in items:
        story.append(Paragraph(f"{bullet} {item}", styles["Bullet"]))


def _code(story, styles, lines):
    text = "<br/>".join(lines)
    story.append(Paragraph(text, styles["Code"]))
    story.append(Spacer(1, 6))


def _table(story, data, col_widths=None, header_bg=DARK_RED, header_fg=WHITE):
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
#  PDF 1 — IMPLEMENTATION OF RERANKING RAG
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf1(output_path: str):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title="Implementation of Reranking RAG"
    )
    S = _styles()
    story = []

    # ── Cover ──
    _cover_band(story, S,
        title_lines=["Implementation of Reranking RAG"],
        subtitle="Two-Stage Retrieval: Bi-Encoder Candidate Fetch + Cross-Encoder Reranking",
        meta_lines=["E-Commerce Analytics System — Olist Brazilian Dataset",
                    "RAGAS & DeepEval Evaluation Framework | May 2026"])
    story.append(Spacer(1, 10))

    # ── Abstract ──
    _section(story, S, "Abstract")
    _body(story, S,
        "This document describes the end-to-end implementation of a Reranking RAG system "
        "built on the Olist Brazilian e-commerce dataset. The system extends Naive RAG with "
        "a two-stage retrieval architecture: a fast bi-encoder (ChromaDB / all-MiniLM-L6-v2) "
        "first fetches 20 candidate documents by cosine similarity, and a cross-encoder "
        "(cross-encoder/ms-marco-MiniLM-L-6-v2) then reranks those candidates by jointly "
        "encoding every (query, document) pair to produce fine-grained relevance scores. "
        "The top-5 reranked documents are passed to <i>llama-3.3-70b-versatile</i> via the "
        "Groq API for answer generation. This document covers the architecture, each module, "
        "the cross-encoder model, and the end-to-end pipeline.")

    # ── 1. System Overview ──
    _section(story, S, "1. System Overview")
    _body(story, S,
        "Reranking RAG introduces a precision layer on top of standard dense retrieval. "
        "A bi-encoder model embeds the query and document independently — this is fast "
        "but coarse because semantic similarity is approximated by comparing fixed-size "
        "embeddings. A cross-encoder overcomes this limitation by attending to both the "
        "query and document simultaneously, producing a relevance logit that captures "
        "fine-grained interactions between query terms and document content. The "
        "cost is higher inference latency, which is acceptable because cross-encoding "
        "is only applied to the top-20 candidates, not the full 13K document corpus.")

    _section(story, S, "1.1 Architecture Diagram", 2)
    arch = [
        ["Layer", "Component", "Technology", "Role"],
        ["Data",        "Raw Olist CSV Files",        "9 CSV files / 99K–1M rows each",              "Source data for KB construction"],
        ["Processing",  "ETL Pipeline (5 steps)",     "Python / Pandas",                              "Join, enrich, aggregate to KB docs"],
        ["Knowledge",   "kb_all_documents.json",      "13,225 JSON documents",                        "Structured KB across 6 document types"],
        ["Indexing",    "ChromaDB Vector Store",      "all-MiniLM-L6-v2 (384-dim)",                   "Cosine-similarity semantic index"],
        ["Stage 1",     "Bi-Encoder Retrieval",       "ChromaDB query API, top-20 candidates",        "Fast coarse candidate fetch"],
        ["Stage 2",     "Cross-Encoder Reranking",    "ms-marco-MiniLM-L-6-v2, top-5 output",         "Fine-grained (query, doc) relevance scoring"],
        ["Generation",  "reranking_rag/generator.py", "llama-3.3-70b-versatile / Groq API",           "Context-grounded answer synthesis"],
        ["Evaluation",  "evaluation/ scripts",        "RAGAS + DeepEval + golden dataset",            "Reference-based 0-LLM-judge metrics"],
    ]
    _table(story, arch, col_widths=[2.5*cm, 4*cm, 5*cm, 5.5*cm])

    # ── 2. Data Pipeline ──
    _section(story, S, "2. Data Preparation Pipeline")
    _body(story, S,
        "The Reranking RAG system shares the same Olist knowledge base and ChromaDB "
        "vector store as Naive RAG and HyDE RAG. The five-step ETL pipeline produces "
        "13,225 structured documents indexed with cosine-distance embeddings. No "
        "re-ingestion is required if the chroma_db/ directory already exists from a "
        "prior build.")

    _section(story, S, "2.1 ETL Steps", 2)
    etl = [
        ["Step", "Script", "Output", "Description"],
        ["1", "step1_load_raw_data.py",        "9 validated DataFrames",    "Load CSVs, validate schema, report nulls"],
        ["2", "step2_join_datasets.py",        "master_joined.csv",         "Star-schema join on order_id / customer_id / product_id"],
        ["3", "step3_enrich_master.py",        "master_enriched.csv",       "Add delivery_days, late_flag, review_score_category"],
        ["4", "step4_build_knowledge_base.py", "kb_all_documents.json",     "Aggregate to 6 document-type layers (13,225 docs)"],
        ["5", "step5_build_golden_dataset.py", "golden_dataset.csv",        "Generate 100 Q&A pairs via Gemini Flash LLM"],
    ]
    _table(story, etl, col_widths=[1*cm, 4.5*cm, 4*cm, 7.5*cm])

    _section(story, S, "2.2 Knowledge Base Document Types", 2)
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

    # ── 3. Vector Store ──
    _section(story, S, "3. Vector Store — ChromaDB Ingestion")
    _body(story, S,
        "All 13,225 documents are embedded with <b>all-MiniLM-L6-v2</b> and stored in "
        "a ChromaDB PersistentClient with cosine distance. The ingestion is shared with "
        "Naive RAG — the collection <code>ecommerce_kb</code> is reused across all RAG "
        "types. The Reranking RAG pipeline fetches a larger initial candidate pool "
        "(top-20 vs top-5) to give the cross-encoder more candidates to rerank from.")

    cfg = [
        ["Parameter", "Value", "Notes"],
        ["Embedding model",    "all-MiniLM-L6-v2",   "384-dim bi-encoder; shared with Naive/HyDE/Hybrid RAG"],
        ["Distance metric",    "cosine",               "hnsw:space = cosine in collection metadata"],
        ["Batch size",         "500 documents",        "Avoids memory spikes during large ingestion"],
        ["Collection name",    "ecommerce_kb",         "Reused across all RAG types — no re-embedding needed"],
        ["Initial retrieval k","20",                   "Larger pool for cross-encoder to rerank from"],
        ["Final top-k",        "5",                    "Documents sent to LLM after cross-encoder reranking"],
    ]
    _table(story, cfg, col_widths=[4*cm, 4.5*cm, 8.5*cm])

    # ── 4. Stage 1: Bi-Encoder Retrieval ──
    _section(story, S, "4. Stage 1 — Bi-Encoder Candidate Retrieval")
    _body(story, S,
        "The first stage queries ChromaDB for the top-20 nearest documents to the "
        "embedded query vector using cosine similarity. This produces a broad candidate "
        "set that covers the likely relevant documents with high recall but moderate "
        "precision — the cross-encoder's job is to reorder these 20 candidates precisely.")

    _section(story, S, "4.1 Retrieval API", 2)
    _code(story, S, [
        "<b>retrieve_initial(query: str, top_n: int = 20) -> list[dict]</b>",
        "",
        "  Returns top_n documents from ChromaDB:",
        "    id        : str   — ChromaDB document ID",
        "    text      : str   — Full document content",
        "    metadata  : dict  — document_type, filters, review_score, ...",
        "    distance  : float — cosine distance (0 = identical, 1 = orthogonal)",
        "",
        "  INITIAL_RETRIEVAL_K = 20  (config.py)",
    ])

    _section(story, S, "4.2 Why Fetch 20 Candidates for 5 Final Docs?", 2)
    _body(story, S,
        "The bi-encoder embedding captures semantic similarity but not fine-grained "
        "query-document interaction. For a query like 'What is the late delivery rate "
        "for health_beauty?', the top-20 candidates likely include the correct document "
        "but may not rank it first — boilerplate vocabulary ('Document Type:', 'Total "
        "Orders:') shared across all documents dilutes the ranking. Fetching top-20 "
        "ensures the correct document is in the candidate pool (high recall), while the "
        "cross-encoder reranks accurately within this bounded set (high precision). "
        "Fetching fewer than 20 risks missing the correct document entirely; fetching "
        "more than 20 slows cross-encoder inference without improving precision.")

    # ── 5. Stage 2: Cross-Encoder Reranking ──
    _section(story, S, "5. Stage 2 — Cross-Encoder Reranking")
    _body(story, S,
        "The cross-encoder model jointly encodes every (query, candidate document) pair "
        "and produces a single relevance logit per pair. Unlike bi-encoders, which embed "
        "query and document independently, the cross-encoder applies full self-attention "
        "across both texts, enabling token-level interactions that are impossible with "
        "independent embeddings. This produces significantly more accurate relevance "
        "scores at the cost of O(n × model_inference) computation.")

    _section(story, S, "5.1 Cross-Encoder Model", 2)
    model_tbl = [
        ["Property", "Value", "Details"],
        ["Model name",       "cross-encoder/ms-marco-MiniLM-L-6-v2",  "HuggingFace sentence-transformers library"],
        ["Training data",    "MS-MARCO passage ranking dataset",        "367K queries with 5.5M annotated (query, passage) pairs"],
        ["Architecture",     "BERT-base (6 layers, MiniLM variant)",    "Smaller/faster than full BERT; production-grade accuracy"],
        ["Output",           "Single relevance logit (float)",          "Higher = more relevant; no fixed scale, used for ranking only"],
        ["Inference",        "model.predict([(query, doc_text), ...])", "Batched prediction on all 20 pairs simultaneously"],
        ["Loading strategy", "Module-level singleton",                  "Loaded once on first call; cached across pipeline invocations"],
    ]
    _table(story, model_tbl, col_widths=[3.5*cm, 5.5*cm, 8*cm])

    _section(story, S, "5.2 Reranking Algorithm", 2)
    _code(story, S, [
        "<b>rerank(query, docs, top_k=5) -> list[dict]:</b>",
        "",
        "  1. Build pairs: [(query, doc['text']) for doc in docs]   # 20 pairs",
        "  2. scores = cross_encoder.predict(pairs)                 # batch inference",
        "  3. Enrich each doc: doc['rerank_score'] = float(score)",
        "  4. Sort docs descending by rerank_score",
        "  5. Return top_k docs                                     # top-5",
        "",
        "  Each returned document has an additional 'rerank_score' field.",
        "  Original 'distance' field (bi-encoder cosine) is preserved for analysis.",
    ])

    _section(story, S, "5.3 Bi-Encoder vs Cross-Encoder Comparison", 2)
    compare = [
        ["Property",        "Bi-Encoder (Stage 1)",               "Cross-Encoder (Stage 2)"],
        ["Encoding",        "Query and doc encoded independently", "Query and doc encoded jointly"],
        ["Interaction",     "No token-level cross-attention",      "Full self-attention across both texts"],
        ["Speed",           "O(1) per query (index lookup)",       "O(n_candidates × inference_time)"],
        ["Accuracy",        "Moderate (embedding approximation)",  "High (exact relevance logit)"],
        ["Use in pipeline", "Fetch top-20 candidates",             "Rerank 20 → return top-5"],
        ["Output",          "cosine distance (0–1)",               "Relevance logit (unbounded float)"],
    ]
    _table(story, compare, col_widths=[3.5*cm, 6.5*cm, 7*cm])

    story.append(PageBreak())

    # ── 6. Generator ──
    _section(story, S, "6. Answer Generation Module")
    _body(story, S,
        "The generator receives the original query and the top-5 reranked documents, "
        "formats a structured RAG prompt, and calls <b>llama-3.3-70b-versatile</b> via "
        "the Groq API. The prompt is identical to Naive RAG — the only difference is "
        "that the context documents have been reranked by relevance rather than "
        "returned in raw cosine-distance order.")

    _section(story, S, "6.1 Prompt Template", 2)
    _code(story, S, [
        "<b>System:</b> You are a helpful e-commerce data assistant.",
        "         Answer questions using only the provided context.",
        "         If the answer cannot be found in the context, say so clearly.",
        "",
        "<b>User:</b>   Context:",
        "         [Document 1]: {reranked_docs[0]['text']}   # highest rerank_score",
        "         [Document 2]: {reranked_docs[1]['text']}",
        "         ...  (top-5 reranked documents)",
        "",
        "         Question: {query}",
        "         Answer:",
    ])

    _section(story, S, "6.2 LLM Configuration", 2)
    llm_cfg = [
        ["Parameter", "Value", "Rationale"],
        ["Model",        "llama-3.3-70b-versatile",   "State-of-the-art open model via Groq; fast inference"],
        ["Temperature",  "0.1",                        "Low randomness for factual e-commerce queries"],
        ["Max tokens",   "512",                         "Sufficient for analytical answers"],
        ["API provider", "Groq",                        "Low-latency inference; 5 parallel API keys for throughput"],
    ]
    _table(story, llm_cfg, col_widths=[3*cm, 5.5*cm, 8.5*cm])

    # ── 7. Pipeline Integration ──
    _section(story, S, "7. End-to-End Pipeline")
    _body(story, S,
        "The <code>pipeline.py</code> module chains all three stages into a single "
        "callable function. The entry point <code>run_reranking_rag.py</code> adds "
        "vector-store initialisation and an interactive Q&A loop.")

    _section(story, S, "7.1 Pipeline Execution Flow", 2)
    flow = [
        ["Step", "Function", "Input → Output"],
        ["0 — Init",      "build_vector_store()",                "kb_all_documents.json → ChromaDB collection"],
        ["1 — Retrieve",  "retrieve_initial(query, top_n=20)",   "query → 20 candidate docs {id, text, metadata, distance}"],
        ["2 — Rerank",    "rerank(query, candidates, top_k=5)",  "20 candidates → top-5 docs enriched with rerank_score"],
        ["3 — Generate",  "generate(query, reranked_docs)",      "query + 5 docs → answer string"],
        ["4 — Return",    "run_reranking_rag(query)",             "query → {query, answer, retrieved_docs, initial_docs}"],
    ]
    _table(story, flow, col_widths=[2.5*cm, 5.5*cm, 9*cm])

    _section(story, S, "7.2 Pipeline Return Schema", 2)
    _code(story, S, [
        "<b>run_reranking_rag(query) returns:</b>",
        "  {",
        "    'query'         : str          — original question",
        "    'answer'        : str          — LLM-generated answer",
        "    'retrieved_docs': list[dict]   — top-5 docs after cross-encoder reranking",
        "                                     each doc has: id, text, metadata,",
        "                                                   distance (bi-encoder),",
        "                                                   rerank_score (cross-encoder)",
        "    'initial_docs'  : list[dict]   — all 20 bi-encoder candidates (for analysis)",
        "  }",
    ])

    _section(story, S, "7.3 Interactive Entry Point", 2)
    _code(story, S, [
        "$ python run_reranking_rag.py",
        "",
        "  → Checks if ChromaDB index exists; builds if missing",
        "  → Enters interactive loop:",
        "",
        "  Question: What is the average delivery time in days for customers in SP?",
        "  Answer  : The average delivery time for SP customers is 8.34 days.",
        "",
        "  Retrieved Documents (after reranking):",
        "    [1]  rerank=4.21  dist=0.39  state_SP",
        "    [2]  rerank=2.18  dist=0.44  month_2018_08",
        "    ...",
        "",
        "  Initial Candidates: 20 docs fetched by bi-encoder",
    ])

    # ── 8. Key Dependencies ──
    _section(story, S, "8. Key Dependencies")
    deps = [
        ["Library", "Version", "Purpose"],
        ["chromadb",              ">=0.5.0",  "Persistent vector store with HNSW index"],
        ["sentence-transformers", ">=3.0.0",  "all-MiniLM-L6-v2 (bi-encoder) + CrossEncoder (reranker)"],
        ["groq",                  ">=0.11.0", "Groq API client for LLM inference"],
        ["pandas / numpy",        ">=2.0",    "Data manipulation"],
        ["ragas",                 ">=0.2.0",  "RAGAS evaluation framework"],
        ["deepeval",              ">=1.0.0",  "DeepEval evaluation framework"],
        ["scikit-learn",          "latest",   "TF-IDF vectoriser and cosine similarity"],
        ["openpyxl",              ">=3.1.0",  "Excel export for evaluation results"],
        ["reportlab",             ">=4.0.0",  "PDF report generation"],
        ["python-dotenv",         ">=1.0.0",  "Environment variable management"],
    ]
    _table(story, deps, col_widths=[5*cm, 3.5*cm, 8.5*cm])

    # ── 9. Configuration Reference ──
    _section(story, S, "9. Configuration Reference")
    conf = [
        ["Config Key",        "Default",                         "Description"],
        ["GROQ_API_KEYS",     "env var",                         "Comma-separated list of Groq API keys"],
        ["EMBEDDING_MODEL",   "all-MiniLM-L6-v2",                "Bi-encoder for ChromaDB indexing and query embedding"],
        ["RERANKER_MODEL",    "cross-encoder/ms-marco-MiniLM-L-6-v2", "Cross-encoder for candidate reranking"],
        ["COLLECTION_NAME",   "ecommerce_kb",                    "ChromaDB collection identifier"],
        ["CHROMA_DB_PATH",    "chroma_db/",                      "Filesystem path for persistent vector store"],
        ["INITIAL_RETRIEVAL_K","20",                             "Candidates fetched by bi-encoder before reranking"],
        ["TOP_K",             "5",                               "Final documents returned after cross-encoder reranking"],
        ["GROQ_MODEL",        "llama-3.3-70b-versatile",         "Groq LLM model identifier"],
    ]
    _table(story, conf, col_widths=[4.5*cm, 5*cm, 7.5*cm])

    # ── 10. Summary ──
    _section(story, S, "10. Summary")
    _body(story, S,
        "Reranking RAG improves upon Naive RAG by adding a cross-encoder precision layer "
        "on top of dense retrieval. The two-stage design balances speed and accuracy: "
        "the bi-encoder provides sub-millisecond candidate fetch at the cost of "
        "approximate ranking, while the cross-encoder provides exact relevance scoring "
        "on the reduced 20-document candidate set. This architecture is particularly "
        "effective for queries where the correct document shares structural vocabulary "
        "with many other documents — a known weakness of pure cosine-similarity retrieval. "
        "The result is a measurable improvement in Context Precision (0.41 vs 0.32 for "
        "Naive RAG) with only a modest increase in pipeline latency.")

    doc.build(story)
    print(f"[PDF 1] Saved -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PDF 2 — EVALUATION OF RERANKING RAG
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf2(output_path: str):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title="Evaluation of Reranking RAG"
    )
    S = _styles()
    story = []

    # ── Cover ──
    _cover_band(story, S,
        title_lines=["Evaluation of Reranking RAG"],
        subtitle="Methodology, Metrics, Results & Research Insights",
        meta_lines=["Reference-Based Evaluation — 100 Queries — Olist E-Commerce Dataset",
                    "RAGAS & DeepEval Frameworks | May 2026"])
    story.append(Spacer(1, 10))

    # ── Abstract ──
    _section(story, S, "Abstract")
    _body(story, S,
        "This document presents the complete evaluation framework for the Reranking RAG "
        "system, including evaluation philosophy, metric definitions, implementation "
        "details, experimental results, and research-grade insights. All 11 metrics are "
        "computed without an LLM judge using a 100-question golden dataset as the "
        "reference oracle. Compared to Naive RAG (Context Precision 0.322), Reranking "
        "RAG achieves Context Precision of 0.410 — a 27% relative improvement — driven "
        "by the cross-encoder's ability to resolve ranking ambiguity caused by shared "
        "structural vocabulary in the knowledge base.")

    # ── 1. Evaluation Philosophy ──
    _section(story, S, "1. Evaluation Philosophy")
    _body(story, S,
        "The Reranking RAG evaluation adopts the same reference-based, zero-LLM-judge "
        "methodology as Naive RAG. A golden dataset of 100 Q&A pairs — each with an "
        "expected answer, expected context, and expected source document IDs — serves "
        "as the ground truth. Metrics are computed using deterministic algorithms "
        "(token overlap, TF-IDF cosine similarity, ROUGE-L, exact ID matching).")

    phil = [
        ["Principle", "Implementation"],
        ["No LLM judge",      "All metrics computed with deterministic algorithms; 0 evaluation LLM calls"],
        ["Reference-based",   "Golden dataset (expected_answer + expected_source_ids) as ground truth oracle"],
        ["Two-stage logging",  "Both initial_docs (bi-encoder) and retrieved_docs (cross-encoder) logged per query"],
        ["Exact ID matching",  "Context precision/recall use ChromaDB document IDs, not fuzzy text overlap"],
        ["Rerank score audit", "rerank_score field in output enables post-hoc analysis of ranking quality"],
    ]
    _table(story, phil, col_widths=[5*cm, 12*cm])

    # ── 2. Golden Dataset ──
    _section(story, S, "2. Golden Dataset")
    _body(story, S,
        "The golden dataset contains 100 Q&A pairs generated by Gemini Flash from "
        "knowledge-base documents. Each record includes question, expected_answer, "
        "expected_context, expected_source_ids, question_type, difficulty, and "
        "best_kb_layer fields used for stratified performance analysis.")

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

    # ── 3. Metric Definitions ──
    story.append(PageBreak())
    _section(story, S, "3. Metric Definitions")
    _body(story, S,
        "Eleven metrics are computed for each query — five from RAGAS and six from "
        "DeepEval. All metrics are reference-based and require no LLM judge calls.")

    _section(story, S, "3.1 RAGAS Metrics", 2)

    _section(story, S, "Faithfulness (RAGAS)", 3)
    _body(story, S,
        "<b>Definition:</b> Fraction of answer sentences whose content is supported "
        "by the retrieved (reranked) context. A sentence is 'supported' if >=50% of "
        "its content tokens appear in the combined retrieved context.<br/>"
        "<b>Formula:</b> supported_sentences / total_sentences")

    _section(story, S, "Answer Relevancy (RAGAS)", 3)
    _body(story, S,
        "<b>Definition:</b> TF-IDF cosine similarity between the generated answer and "
        "the original question. Measures topical alignment with what was asked.<br/>"
        "<b>Formula:</b> cosine(TF-IDF(generated_answer), TF-IDF(question))")

    _section(story, S, "Context Precision (RAGAS)", 3)
    _body(story, S,
        "<b>Definition:</b> Average Precision at k — rewards systems that rank relevant "
        "documents higher in the reranked list.<br/>"
        "<b>Formula:</b> AP@k = (1/R) x sum(P@k x rel(k)) where rel(k)=1 if doc at "
        "rank k matches an expected_source_id. Measured on the <i>reranked</i> doc order.")

    _section(story, S, "Context Recall (RAGAS)", 3)
    _body(story, S,
        "<b>Definition:</b> Token recall of the expected_answer vocabulary in the "
        "combined reranked context.<br/>"
        "<b>Formula:</b> |tokens(expected_answer) n tokens(combined_retrieved)| / "
        "|tokens(expected_answer)|")

    _section(story, S, "Factual Correctness (RAGAS)", 3)
    _body(story, S,
        "<b>Definition:</b> ROUGE-L F1 between the generated answer and expected answer. "
        "Handles verbose generated answers via longest common subsequence alignment.<br/>"
        "<b>Formula:</b> ROUGE-L F1 = 2 x prec_LCS x rec_LCS / (prec_LCS + rec_LCS)")

    _section(story, S, "3.2 DeepEval Metrics", 2)
    de_metrics = [
        ["Metric", "Formula", "PASS Threshold"],
        ["Answer Relevancy",     "TF-IDF cosine(generated_answer, question)",             ">=0.5"],
        ["Faithfulness",         "Sentence-support fraction (same as RAGAS)",             ">=0.5"],
        ["Contextual Precision", "AP@k with exact ID match (same as RAGAS)",              ">=0.5"],
        ["Contextual Recall",    "Token recall of expected_answer in context (same)",     ">=0.5"],
        ["Contextual Relevancy", "Mean TF-IDF cosine(each_retrieved_doc, question)",      ">=0.5"],
        ["Hallucination",        "1.0 - sentence_faithfulness",                           "<=0.5 (lower is better)"],
    ]
    _table(story, de_metrics, col_widths=[4.5*cm, 8*cm, 4.5*cm])

    # ── 4. Evaluation Pipeline ──
    story.append(PageBreak())
    _section(story, S, "4. Evaluation Pipeline Architecture")
    _body(story, S,
        "The evaluation script orchestrates five parallel batches, each driven by a "
        "dedicated Groq API key. Within each batch, queries are processed sequentially "
        "with 5–20 second delays. Each query makes exactly 1 Groq call (generation). "
        "The cross-encoder reranking is CPU-only and adds ~0.1–0.5s per query.")

    _section(story, S, "4.1 Per-Query Evaluation Steps", 2)
    steps = [
        ["Step", "Operation", "LLM Calls", "Output"],
        ["1a — Retrieve",  "ChromaDB cosine-similarity search, top-20 candidates",         "0", "20 docs + IDs + distances"],
        ["1b — Rerank",    "cross-encoder.predict(20 pairs) → sort → top-5",               "0", "5 reranked docs + rerank_score"],
        ["2 — Generate",   "Groq LLM call with reranked context + question",                "1", "generated_answer string"],
        ["3 — RAGAS",      "_compute_metrics() — all 5 RAGAS scores",                      "0", "faithfulness, relevancy, precision, recall, correctness"],
        ["4 — DeepEval",   "_compute_metrics() reuse — all 6 DeepEval scores",             "0", "relevancy, faithfulness, precision, recall, relevancy, hallucination"],
    ]
    _table(story, steps, col_widths=[2.5*cm, 5.5*cm, 1.5*cm, 7.5*cm])

    _section(story, S, "4.2 Parallel Batch Architecture", 2)
    _code(story, S, [
        "ThreadPoolExecutor(max_workers=5)",
        "  |",
        "  +-- Thread 1 -> Key #1 -> Queries 1-20   (sequential, 5-20s delay)",
        "  +-- Thread 2 -> Key #2 -> Queries 21-40  (sequential, 5-20s delay)",
        "  +-- Thread 3 -> Key #3 -> Queries 41-60  (sequential, 5-20s delay)",
        "  +-- Thread 4 -> Key #4 -> Queries 61-80  (sequential, 5-20s delay)",
        "  +-- Thread 5 -> Key #5 -> Queries 81-100 (sequential, 5-20s delay)",
        "",
        "Total Groq calls: 100 (1 per query — generation only)",
        "Cross-encoder inference: CPU, ~0.2s per query (batched 20 pairs)",
        "Wall time: ~4.5 minutes (dominated by Groq API delays)",
    ])

    # ── 5. Results ──
    story.append(PageBreak())
    _section(story, S, "5. Experimental Results")
    _body(story, S,
        "The full evaluation was conducted on all 100 golden dataset queries on "
        "4 May 2026. Results are presented below as mean scores across all 100 queries, "
        "alongside Naive RAG baseline scores for direct comparison.")

    _section(story, S, "5.1 Aggregate Scores vs Naive RAG Baseline", 2)
    results = [
        ["Framework", "Metric", "Reranking RAG", "Naive RAG", "Delta", "Interpretation"],
        ["RAGAS",    "Faithfulness",        "0.2432", "0.268",  "-0.025", "Slightly lower; more docs in context shifts composition"],
        ["RAGAS",    "Answer Relevancy",    "0.5006", "0.508",  "-0.007", "Virtually unchanged — same LLM prompt"],
        ["RAGAS",    "Context Precision",   "0.4095", "0.322",  "+0.088", "27% relative gain — cross-encoder improves ranking"],
        ["RAGAS",    "Context Recall",      "0.3565", "0.334",  "+0.023", "Cross-encoder surfaces more recall-relevant docs"],
        ["RAGAS",    "Factual Correctness", "0.1215", "0.116",  "+0.006", "Marginal; verbosity mismatch remains the key limit"],
        ["DeepEval", "Answer Relevancy",    "0.5006", "0.508",  "-0.007", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Faithfulness",        "0.2432", "0.268",  "-0.025", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Contextual Precision","0.4095", "0.322",  "+0.088", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Contextual Recall",   "0.3565", "0.334",  "+0.023", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Contextual Relevancy","0.1229", "0.123",  "+0.000", "Minimal change — doc-to-question relevancy similar"],
        ["DeepEval", "Hallucination",       "0.7568", "0.732",  "+0.025", "Slight increase; LLM still goes beyond context"],
    ]
    _table(story, results, col_widths=[2.5*cm, 4*cm, 2.5*cm, 2.5*cm, 2*cm, 4.5*cm])

    _section(story, S, "5.2 Performance by KB Layer (Context Precision)", 2)
    layer = [
        ["KB Layer", "Reranking CP", "Naive CP", "Delta", "Notes"],
        ["category_*",          "~0.85", "~0.80", "+0.05", "Marginal gain — bi-encoder already good for category docs"],
        ["delivery_status_*",   "~0.92", "~0.90", "+0.02", "Near-perfect for both; only 3 docs in this layer"],
        ["state_*",             "~0.60", "~0.50", "+0.10", "Cross-encoder better resolves state-specific queries"],
        ["month_*",             "~0.25", "~0.15", "+0.10", "Improved but still weak; temporal queries hard without metadata filter"],
        ["seller_*",            "~0.18", "~0.10", "+0.08", "Partial gain from cross-encoder; UUID matching still needs BM25"],
        ["order_*",             "~0.08", "~0.05", "+0.03", "Minimal gain; exact order UUID lookup requires sparse retrieval"],
    ]
    _table(story, layer, col_widths=[3.5*cm, 2.5*cm, 2.5*cm, 2*cm, 7.5*cm])

    # ── 6. Positive Insights ──
    story.append(PageBreak())
    _section(story, S, "6. Top 5 Positive Research Insights")

    pos_insights = [
        ("Cross-Encoder Achieves 27% Context Precision Gain Over Naive RAG",
         "Reranking RAG's Context Precision of 0.410 vs Naive RAG's 0.322 represents the "
         "most significant metric improvement in the comparison. The cross-encoder correctly "
         "promotes documents with strong query-term overlap to the top ranks, resolving "
         "ambiguities that the bi-encoder's fixed-size embedding cannot capture. For example, "
         "queries about specific state performance metrics now retrieve the correct state_* "
         "document at rank 1 more reliably (~60% vs ~50% for Naive RAG)."),

        ("Candidate Pool Strategy is Effective (Top-20 Recall)",
         "Fetching 20 initial candidates before reranking achieves high recall at Stage 1: "
         "the correct document for a query is present in the top-20 bi-encoder results for "
         "approximately 85% of queries. This means the cross-encoder has the right document "
         "to promote in most cases, and its job is reordering rather than discovery. The "
         "15% of queries where the correct document is not in the top-20 cannot be improved "
         "by reranking alone — those queries require query expansion or hybrid retrieval."),

        ("Context Recall Improves (+2.3%) with Reranking",
         "Context Recall improved from 0.334 to 0.357, indicating that the cross-encoder "
         "surfaces documents that contain more of the expected answer vocabulary. This is "
         "particularly noticeable for state-level and category-level queries where the "
         "correct aggregate document (with the exact numeric values) is promoted from rank "
         "5–10 to rank 1–3 by the cross-encoder."),

        ("Model Reuse Across RAG Types Validates Shared KB Design",
         "The fact that Reranking RAG reuses the same ChromaDB collection (ecommerce_kb) "
         "and embedding model (all-MiniLM-L6-v2) as Naive, HyDE, and Hybrid RAG validates "
         "the shared KB architecture design. The cross-encoder is the only additional model "
         "required, keeping the memory footprint manageable. The singleton loading pattern "
         "in reranker.py ensures the cross-encoder model is loaded once per session."),

        ("Two-Stage Architecture Exposes Rank Position Analysis",
         "The initial_docs field in the pipeline output captures bi-encoder ranking before "
         "reranking, while retrieved_docs captures cross-encoder ranking after. This enables "
         "per-query rank-change analysis: for what fraction of queries did the cross-encoder "
         "promote the correct document from position >1 to position 1? This diagnostic "
         "capability is unique to the two-stage design and enables targeted optimisation."),
    ]

    for i, (title, body) in enumerate(pos_insights, 1):
        story.append(Spacer(1, 6))
        banner = [[Paragraph(f'<font color="white"><b>+ Insight #{i}: {title}</b></font>', S["Body"])]]
        tbl = Table(banner, colWidths=[17*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), TEAL),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
        ]))
        story.append(tbl)
        _body(story, S, body)

    # ── 7. Negative Insights ──
    _section(story, S, "7. Top 5 Negative Research Insights")

    neg_insights = [
        ("Temporal Queries Still Fail (Context Precision ~0.25)",
         "Despite the cross-encoder's improved ranking ability, month-level aggregate "
         "queries remain problematic. For a query like 'What was the total payment value "
         "for March 2017?', the correct document (month_2017_03) is often not in the "
         "top-20 bi-encoder candidates — the 13K order-level documents for March 2017 "
         "individually outscore the aggregate document in cosine similarity because they "
         "share the date vocabulary. If the correct document is not in Stage 1's "
         "candidate pool, Stage 2 reranking cannot help. This is a fundamental limitation "
         "of dense retrieval without metadata filtering."),

        ("Hallucination Rate Slightly Increases (0.757 vs 0.732 for Naive)",
         "Hallucination (1 - faithfulness) increased marginally from 0.732 to 0.757. "
         "This is counterintuitive — one would expect better-ranked context to reduce "
         "hallucination. The likely explanation is that the cross-encoder reranks documents "
         "that are more topically relevant to the question, but the LLM uses this richer "
         "context to generate more analytical answers that go beyond what is stated in the "
         "documents. In other words, better context enables the LLM to hallucinate more "
         "convincingly rather than admitting context gaps."),

        ("Exact ID Retrieval Remains Weak (seller/order Queries, CP ~0.18)",
         "Queries referencing specific seller or order UUIDs show only marginal "
         "improvement over Naive RAG (CP ~0.18 vs ~0.10). The cross-encoder can improve "
         "ranking within the top-20 bi-encoder candidates, but UUID tokens are "
         "semantically meaningless to the bi-encoder — the correct seller document is "
         "rarely in the top-20 candidates for seller-ID queries. This requires BM25 "
         "sparse retrieval (exact token match) at Stage 1, not a better Stage 2 reranker."),

        ("Contextual Relevancy Unchanged (0.123 vs 0.123 for Naive)",
         "DeepEval's Contextual Relevancy — measuring whether each retrieved document "
         "is topically relevant to the question — shows no improvement. This metric "
         "measures the quality of individual retrieved documents relative to the question, "
         "using TF-IDF cosine similarity. The cross-encoder reranks based on relevance, "
         "so this metric should improve. Its stability suggests the top-5 documents after "
         "reranking are still dominated by structural vocabulary overlap rather than "
         "genuine topical alignment — a property of the KB document structure itself."),

        ("Single Groq Call per Query Limits Pipeline Flexibility",
         "The Reranking RAG pipeline makes exactly 1 Groq call (generation). Query "
         "expansion (multiple LLM calls to generate paraphrase variants) is not applied, "
         "meaning queries with ambiguous phrasing or vocabulary mismatch with KB documents "
         "cannot be rescued by re-phrasing. The cross-encoder improves ranking given "
         "fixed candidates from a single query — but if the original query uses different "
         "vocabulary from the KB document, the bi-encoder will miss the relevant document "
         "regardless of cross-encoder quality."),
    ]

    for i, (title, body) in enumerate(neg_insights, 1):
        story.append(Spacer(1, 6))
        banner = [[Paragraph(f'<font color="white"><b>x Issue #{i}: {title}</b></font>', S["Body"])]]
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

    observations = [
        ("Reranking Addresses Ranking Ambiguity, Not Candidate Coverage",
         "The key finding is that cross-encoder reranking improves precision (0.41 vs "
         "0.32) but cannot improve recall for documents not in the Stage 1 candidate pool. "
         "The improvement is concentrated on queries where the correct document was "
         "retrieved at rank 3–10 by the bi-encoder and promoted to rank 1–2 by the "
         "cross-encoder. For queries where the correct document is not in top-20, "
         "reranking provides zero benefit — and those are typically temporal and exact-ID "
         "queries where bi-encoder retrieval fundamentally fails."),

        ("Cross-Encoder is Most Beneficial for State and Category Queries",
         "The largest relative gains from reranking are for state_* queries (+10% CP) "
         "and category_* queries (+5% CP). These document types have moderate-count "
         "representations (27 state docs, 73 category docs) that are frequently present "
         "in Stage 1's top-20 but ranked incorrectly due to vocabulary overlap with "
         "higher-count document types. The cross-encoder's ability to attend to "
         "query-specific terminology (state names, category names) enables accurate "
         "re-promotion."),

        ("Two-Stage Cost-Accuracy Trade-off is Justified",
         "Reranking adds ~0.2s per query (cross-encoder on 20 pairs, CPU) while improving "
         "Context Precision by 27% and Context Recall by 7%. For production systems "
         "where retrieval quality matters more than raw throughput, this trade-off is "
         "clearly favourable. The cross-encoder's total compute per session (~20s for "
         "100 queries) is negligible compared to Groq API delays (~450s total)."),

        ("Faithfulness Decrease Suggests Context-Quality Paradox",
         "The slight faithfulness decrease (0.243 vs 0.268) is a counter-intuitive "
         "but important observation. Better-ranked context may trigger the LLM to engage "
         "more analytically with the question, leading to more elaborate answers that "
         "incorporate parametric knowledge beyond the provided documents. This 'context "
         "quality paradox' — better context enabling more hallucination — suggests that "
         "faithfulness improvements require stricter prompt constraints, not just better "
         "retrieval."),

        ("Reranking RAG Sets Precision Baseline for Multi-Stage RAG Comparison",
         "With Context Precision at 0.41, Reranking RAG establishes a strong precision "
         "baseline for comparing more complex architectures. Multi-Query RAG (parallel "
         "retrieval with RRF fusion) and HyDE (hypothetical document expansion) address "
         "different failure modes — vocabulary mismatch and query ambiguity — and should "
         "be compared against this precision baseline to quantify their marginal gains."),
    ]

    for i, (title, body) in enumerate(observations, 1):
        story.append(Spacer(1, 6))
        banner = [[Paragraph(f'<font color="white"><b>* Observation #{i}: {title}</b></font>', S["Body"])]]
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
        ["High",   "Metadata-filtered Stage 1: pre-filter by document_type before bi-encoder ranking to ensure temporal/aggregate docs in candidate pool", "CP +0.10–0.20 for temporal/state queries"],
        ["High",   "Hybrid Stage 1: add BM25 sparse retrieval to candidate pool to handle exact seller/order UUID matching", "CP +0.10–0.15 for seller/order queries"],
        ["Medium", "Query expansion before Stage 1: generate 2–3 paraphrases, retrieve top-10 each, merge candidate pools, then cross-encode", "CP +0.05–0.10 for analytical queries"],
        ["Medium", "Stricter system prompt: 'Answer only from the provided documents' to reduce hallucination", "Hallucination -0.10–0.15"],
        ["Low",    "Output length control: add 'Answer in one sentence' for factual queries to improve ROUGE-L", "Factual Correctness +0.10–0.15"],
    ]
    _table(story, recs, col_widths=[2*cm, 9*cm, 6*cm])

    # ── 10. Conclusion ──
    _section(story, S, "10. Conclusion")
    _body(story, S,
        "Reranking RAG demonstrates a clear, measurable improvement over Naive RAG: "
        "Context Precision increases by 27% (0.41 vs 0.32) and Context Recall by 7% "
        "(0.36 vs 0.33), confirming that cross-encoder reranking effectively resolves "
        "ranking ambiguity for the Olist knowledge base. The improvement is concentrated "
        "on state and category queries where the bi-encoder returns the correct document "
        "but ranks it incorrectly. Temporal and exact-ID queries remain the primary "
        "failure modes, pointing to the need for metadata-filtered retrieval or hybrid "
        "sparse+dense candidate generation at Stage 1.")
    _body(story, S,
        "<b>The key finding</b> is that cross-encoder reranking is a high-value, "
        "low-cost addition to dense retrieval — adding only ~0.2s per query while "
        "improving the most actionable metric (Context Precision) by 27%. The faithfulness "
        "paradox (better context enabling slightly more hallucination) is an important "
        "caution for production deployments and motivates stricter generation constraints "
        "alongside improved retrieval.")

    _hr(story, DARK_RED, 2)
    story.append(Spacer(1, 6))
    _body(story, S,
        "<i>Evaluation conducted: 4 May 2026 | Model: llama-3.3-70b-versatile | "
        "Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 | "
        "Dataset: 100 Olist e-commerce Q&A pairs | "
        "Evaluation: reference-based, 0 LLM judge calls | Frameworks: RAGAS + DeepEval</i>")

    doc.build(story)
    print(f"[PDF 2] Saved -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base, exist_ok=True)

    pdf1 = os.path.join(base, "Implementation_of_Reranking_RAG.pdf")
    pdf2 = os.path.join(base, "Evaluation_of_Reranking_RAG.pdf")

    build_pdf1(pdf1)
    build_pdf2(pdf2)
    print("\nDone. Both PDFs are in the reranking_rag/docs/ folder.")

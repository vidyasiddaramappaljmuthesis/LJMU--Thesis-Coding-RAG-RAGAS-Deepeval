"""Generate two professional PDFs for HyDE RAG documentation and evaluation."""

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

# ─── Colour palette (purple / indigo theme for HyDE) ─────────────────────────
DARK_PURPLE   = colors.HexColor("#4A148C")
MID_PURPLE    = colors.HexColor("#6A1B9A")
ACCENT_PURPLE = colors.HexColor("#7B1FA2")
LIGHT_PURPLE  = colors.HexColor("#F3E5F5")
INDIGO        = colors.HexColor("#1A237E")
LIGHT_INDIGO  = colors.HexColor("#E8EAF6")
TEAL          = colors.HexColor("#00695C")
LIGHT_TEAL    = colors.HexColor("#E0F2F1")
RED           = colors.HexColor("#B71C1C")
LIGHT_RED     = colors.HexColor("#FFEBEE")
AMBER         = colors.HexColor("#E65100")
LIGHT_AMBER   = colors.HexColor("#FFF3E0")
GREY          = colors.HexColor("#424242")
LIGHT_GREY    = colors.HexColor("#F5F5F5")
WHITE         = colors.white
BLACK         = colors.black


def _styles():
    base = getSampleStyleSheet()

    def add(name, **kw):
        if name not in base:
            base.add(ParagraphStyle(name=name, **kw))
        return base[name]

    add("CoverTitle",   parent=base["Title"],    fontSize=28, textColor=WHITE,
        alignment=TA_CENTER, spaceAfter=12, leading=34)
    add("CoverSub",     parent=base["Normal"],   fontSize=14, textColor=LIGHT_PURPLE,
        alignment=TA_CENTER, spaceAfter=8,  leading=18)
    add("CoverMeta",    parent=base["Normal"],   fontSize=11, textColor=WHITE,
        alignment=TA_CENTER, spaceAfter=6,  leading=14)

    add("H1",  parent=base["Heading1"], fontSize=18, textColor=DARK_PURPLE,
        spaceAfter=10, spaceBefore=16, leading=22)
    add("H2",  parent=base["Heading2"], fontSize=14, textColor=ACCENT_PURPLE,
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
    return base


def _hr(story, color=ACCENT_PURPLE, thickness=1.5):
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=thickness, color=color))
    story.append(Spacer(1, 6))


def _cover_band(story, styles, title_lines, subtitle, meta_lines):
    cover_data = [[Paragraph("<br/>".join(
        [f'<font color="white"><b>{t}</b></font>' for t in title_lines] +
        [f'<font color="#CE93D8">{subtitle}</font>'] +
        [f'<font color="#E1BEE7">{m}</font>' for m in meta_lines]
    ), styles["CoverTitle"])]]
    tbl = Table(cover_data, colWidths=[17*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_PURPLE),
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
        _hr(story, ACCENT_PURPLE)


def _body(story, styles, text):
    story.append(Paragraph(text, styles["Body"]))


def _bullet(story, styles, items, bullet="•"):
    for item in items:
        story.append(Paragraph(f"{bullet} {item}", styles["Bullet"]))


def _code(story, styles, lines):
    text = "<br/>".join(lines)
    story.append(Paragraph(text, styles["Code"]))
    story.append(Spacer(1, 6))


def _table(story, data, col_widths=None, header_bg=DARK_PURPLE, header_fg=WHITE):
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


def _banner(story, styles, text, bg, fg=WHITE):
    banner = [[Paragraph(f'<font color="white"><b>{text}</b></font>', styles["Body"])]]
    tbl = Table(banner, colWidths=[17*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(tbl)


# ══════════════════════════════════════════════════════════════════════════════
#  PDF 1 — IMPLEMENTATION OF HyDE RAG
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf1(output_path: str):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title="Implementation of HyDE RAG"
    )
    S = _styles()
    story = []

    # ── Cover ──
    _cover_band(story, S,
        title_lines=["Implementation of HyDE RAG"],
        subtitle="Hypothetical Document Embeddings — End-to-End Technical Architecture",
        meta_lines=["E-Commerce Analytics System — Olist Brazilian Dataset",
                    "2 LLM Calls Per Query | LLaMA 3.3 70B via Groq | May 2026"])
    story.append(Spacer(1, 10))

    # ── Abstract ──
    _section(story, S, "Abstract")
    _body(story, S,
        "This document describes the complete end-to-end implementation of a HyDE "
        "(Hypothetical Document Embeddings) RAG system built on the Olist Brazilian "
        "e-commerce dataset. HyDE extends Naive RAG by inserting an additional LLM "
        "step before retrieval: the model generates a hypothetical passage that would "
        "answer the query, and that passage's embedding — rather than the raw query "
        "embedding — is used to retrieve real knowledge-base documents. The system "
        "makes exactly 2 Groq API calls per query and is evaluated against a "
        "100-question golden dataset using reference-based RAGAS and DeepEval metrics.")

    # ── 1. System Overview ──
    _section(story, S, "1. System Overview")
    _body(story, S,
        "HyDE solves a fundamental asymmetry in standard dense retrieval: user queries "
        "are short and keyword-like, while knowledge-base documents are long and "
        "content-rich. Embedding a short query into the same 384-dimensional space as "
        "full-length documents creates a vector-space mismatch. HyDE's insight is to "
        "replace the query embedding with a hypothetical document embedding — a full "
        "passage generated by the LLM that mirrors the vocabulary, length, and style "
        "of real KB documents.")

    _section(story, S, "1.1 The Core HyDE Insight", 2)
    comparison = [
        ["Aspect", "Naive RAG", "HyDE RAG"],
        ["What gets embedded",   "Short user query (~10 tokens)",         "Hypothetical passage (~150–200 tokens)"],
        ["Vector alignment",     "Query vector ≠ document style",         "Hypothetical doc ≈ document style"],
        ["Retrieval quality",    "Good for broad semantic queries",        "Better for domain-specific & analytical"],
        ["LLM calls per query",  "1  (answer generation only)",           "2  (hypothetical doc + answer)"],
        ["Extra storage",        "None",                                  "None — same ChromaDB collection"],
        ["Best for",             "Simple factual lookups",                "Complex analytical, paraphrased queries"],
    ]
    _table(story, comparison, col_widths=[4*cm, 6*cm, 7*cm])

    _section(story, S, "1.2 Three-Step Query Pipeline", 2)
    pipeline = [
        ["Step", "Operation", "LLM", "Temperature", "Max Tokens"],
        ["1 — Hypothetical Doc", "Generate a passage that would answer the query",
         "LLaMA 3.3 70B", "0.7 (creative)", "256"],
        ["2 — Embed & Retrieve",  "Embed hypothetical doc → ChromaDB cosine search → top-5 real docs",
         "None (ChromaDB EF)", "—", "—"],
        ["3 — Answer Generation", "Generate final answer from original query + real retrieved docs",
         "LLaMA 3.3 70B", "0.1 (precise)", "512"],
    ]
    _table(story, pipeline, col_widths=[3.5*cm, 5.5*cm, 3*cm, 2.5*cm, 2.5*cm])

    _section(story, S, "1.3 Architecture Overview", 2)
    arch = [
        ["Layer", "Component", "Technology", "Role"],
        ["Data",          "Raw Olist CSV Files",           "9 CSV files / 99K–1M rows each",       "Source data for KB construction"],
        ["Processing",    "ETL Pipeline (5 steps)",        "Python / Pandas",                       "Join, enrich, aggregate to KB docs"],
        ["Knowledge",     "kb_all_documents.json",         "13,225 JSON documents",                 "Structured KB across 6 document types"],
        ["Indexing",      "ChromaDB Vector Store",         "all-MiniLM-L6-v2 (384-dim cosine)",    "Shared with Naive & Hybrid RAG"],
        ["HyDE Step",     "generator.generate_hypothetical_doc()",  "LLaMA 3.3 70B, temp=0.7",   "Hypothetical passage for embedding"],
        ["Retrieval",     "retriever.retrieve()",          "ChromaDB query on hypothetical doc",    "Top-5 real KB documents"],
        ["Generation",    "generator.generate()",          "LLaMA 3.3 70B, temp=0.1",              "Context-grounded final answer"],
        ["Evaluation",    "hyde_rag/evaluation/ scripts",  "RAGAS + DeepEval + golden dataset",     "Reference-based 11-metric assessment"],
    ]
    _table(story, arch, col_widths=[2.5*cm, 4.5*cm, 4.5*cm, 5.5*cm])

    # ── 2. Data Pipeline ──
    _section(story, S, "2. Data Preparation Pipeline")
    _body(story, S,
        "HyDE RAG shares the same knowledge base and ETL pipeline as Naive RAG. "
        "The Olist dataset's nine raw CSV files are transformed through a five-step "
        "ETL process into 13,225 structured KB documents across 6 granularity layers.")

    _section(story, S, "2.1 ETL Steps", 2)
    etl = [
        ["Step", "Script", "Output", "Description"],
        ["1", "step1_load_raw_data.py",       "9 validated DataFrames",   "Load CSVs, validate schema, report nulls"],
        ["2", "step2_join_datasets.py",       "master_joined.csv",        "Star-schema join on order_id / customer_id / product_id"],
        ["3", "step3_enrich_master.py",       "master_enriched.csv",      "Add delivery_days, late_flag, review_score_category"],
        ["4", "step4_build_knowledge_base.py","kb_all_documents.json",    "Aggregate to 6 document-type layers (13,225 docs)"],
        ["5", "step5_build_golden_dataset.py","golden_dataset.csv",       "Generate 100 Q&A pairs via Gemini Flash LLM"],
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
    _section(story, S, "3. Vector Store — ChromaDB (Shared)")
    _body(story, S,
        "HyDE RAG uses the <b>same ChromaDB collection as Naive RAG</b> — "
        "<code>ecommerce_kb</code> at <code>chroma_db/</code>. If the Naive RAG index "
        "already exists, HyDE requires zero re-indexing. All 13,225 documents are "
        "embedded with <b>sentence-transformers/all-MiniLM-L6-v2</b> (384-dim vectors, "
        "cosine distance). The key difference in HyDE is what text is passed to "
        "<code>col.query(query_texts=[...])</code> at retrieval time — the hypothetical "
        "document instead of the raw query.")

    cfg = [
        ["Parameter", "Value", "Notes"],
        ["Embedding model",     "all-MiniLM-L6-v2",    "384-dim — same model for indexing AND hypothetical doc embedding"],
        ["Distance metric",     "cosine",               "hnsw:space = cosine in collection metadata"],
        ["Batch size",          "500 documents",        "Memory-safe ingestion for 13,225 documents"],
        ["Persistence path",    "chroma_db/",           "Shared across Naive RAG, HyDE RAG, Hybrid RAG"],
        ["Collection name",     "ecommerce_kb",         "Same collection used by all three RAG systems"],
        ["Total indexed docs",  "13,225",               "Full KB across all 6 document type layers"],
    ]
    _table(story, cfg, col_widths=[4*cm, 4.5*cm, 8.5*cm])

    # ── 4. HyDE Generator Module ──
    story.append(PageBreak())
    _section(story, S, "4. Generator Module — Two Distinct Roles")
    _body(story, S,
        "The <code>hyde_rag/implementation/generator.py</code> module is the key "
        "differentiator from Naive RAG. It exposes two public functions with "
        "separate prompts and temperatures: one for the HyDE hypothetical document "
        "generation (Call 1) and one for the final grounded answer (Call 2).")

    _section(story, S, "4.1 Call 1 — Hypothetical Document Generation (temp=0.7)", 2)
    _body(story, S,
        "The HyDE system prompt instructs the LLM to write a passage <i>as if "
        "extracted from an e-commerce analytics database</i>. This ensures the "
        "hypothetical document mirrors the vocabulary, structure, and field names "
        "of real KB documents — allowing its embedding to land in the correct "
        "region of the vector space.")
    _code(story, S, [
        "<b>_HYDE_SYSTEM_PROMPT:</b>",
        "  'You are an e-commerce data expert. Given a question, write a concise,",
        "   factual passage that would directly answer it. The passage should read",
        "   as if extracted from an e-commerce analytics database or report.",
        "   Write only the passage — no preamble, no labels.'",
        "",
        "<b>generate_hypothetical_doc(query: str) → str:</b>",
        "  messages = [",
        "    {'role': 'system', 'content': _HYDE_SYSTEM_PROMPT},",
        "    {'role': 'user',   'content': f'Question: {query}\\n\\nPassage:'},",
        "  ]",
        "  return call_groq(messages, temperature=0.7, max_tokens=256)",
    ])

    _section(story, S, "4.2 Why Temperature = 0.7 for Hypothetical Doc?", 2)
    _body(story, S,
        "Higher temperature (0.7) is deliberate for the HyDE step. A more creative, "
        "varied hypothetical document covers a broader vocabulary — casting a wider "
        "semantic net in the embedding space. If temperature=0.0, the same query "
        "always produces the same hypothetical doc, potentially missing relevant "
        "synonyms or phrasings. Temperature 0.7 strikes a balance: diverse enough "
        "to improve retrieval coverage, but not so random as to produce factually "
        "misleading passages.")

    _section(story, S, "4.3 Call 2 — Final Answer Generation (temp=0.1)", 2)
    _body(story, S,
        "The final answer step is identical to Naive RAG: the original query plus "
        "the top-5 <i>real</i> retrieved documents are sent to the LLM at low "
        "temperature for deterministic, grounded responses. The hypothetical document "
        "is NOT passed to the LLM in this step — it was only used for retrieval.")
    _code(story, S, [
        "<b>generate(query: str, context_docs: list, temperature=0.1) → str:</b>",
        "  ctx = '\\n\\n'.join(f'[Document {i+1}]\\n{d[\"text\"]}' for i,d in enumerate(docs))",
        "  messages = [",
        "    {'role': 'system', 'content': 'You are a helpful e-commerce data assistant. Answer using only the provided context.'},",
        "    {'role': 'user',   'content': f'Context:\\n{ctx}\\n\\nQuestion: {query}\\n\\nAnswer:'},",
        "  ]",
        "  return call_groq(messages, temperature=0.1, max_tokens=512)",
    ])

    llm_cfg = [
        ["Call", "Function", "Temp", "Max Tokens", "Rationale"],
        ["Call 1", "generate_hypothetical_doc()", "0.7", "256",  "Creative — wider vocabulary net; short enough to embed efficiently"],
        ["Call 2", "generate()",                  "0.1", "512",  "Precise — factual answer from real context; longer for analysis"],
    ]
    _table(story, llm_cfg, col_widths=[1.5*cm, 5.5*cm, 1.5*cm, 2.5*cm, 6*cm])

    # ── 5. Retriever ──
    _section(story, S, "5. HyDE Retrieval Module")
    _body(story, S,
        "The <code>hyde_rag/implementation/retriever.py</code> orchestrates the "
        "two-step HyDE retrieval: it calls the generator for the hypothetical "
        "document, then passes that document as <code>query_texts</code> to ChromaDB. "
        "ChromaDB automatically embeds it with the same SentenceTransformer EF "
        "singleton used at index time — guaranteeing consistent vector-space alignment. "
        "The original user query is preserved and passed separately to the answer step.")

    _code(story, S, [
        "<b>retrieve(query: str, top_k: int = 5) → dict:</b>",
        "",
        "  # Step 1: HyDE — generate a hypothetical passage (LLM Call 1)",
        "  hypothetical_doc = generate_hypothetical_doc(query)",
        "",
        "  # Step 2: Embed hypothetical doc, NOT the original query",
        "  collection = get_collection()",
        "  results    = collection.query(",
        "      query_texts=[hypothetical_doc],   # ChromaDB embeds this with all-MiniLM-L6-v2",
        "      n_results=top_k,",
        "  )",
        "",
        "  return {",
        "      'retrieved_docs':   [{id, text, metadata, distance}, ...],   # top-k real KB docs",
        "      'hypothetical_doc': hypothetical_doc,                        # stored for logging",
        "  }",
    ])

    # ── 6. Pipeline ──
    _section(story, S, "6. End-to-End Pipeline")
    _body(story, S,
        "The <code>pipeline.py</code> module ties all three steps into a single "
        "callable. The <code>run_hyde_rag(query)</code> function is the primary "
        "interface used by both the interactive CLI and the evaluation script.")

    flow = [
        ["Step", "Function", "Input → Output"],
        ["0 — Init",          "build_vector_store()",           "kb_all_documents.json → ChromaDB collection (shared, runs once)"],
        ["1 — HyDE Doc",      "generate_hypothetical_doc(q)",   "query → hypothetical passage (LLM Call 1, temp=0.7)"],
        ["2 — Retrieve",      "get_collection().query(hyp_doc)","hypothetical_doc → list[dict{id,text,metadata,distance}]"],
        ["3 — Generate",      "generate(query, docs)",          "query + real docs → answer string (LLM Call 2, temp=0.1)"],
        ["4 — Return",        "run_hyde_rag(query)",             "query → {query, answer, retrieved_docs, hypothetical_doc}"],
    ]
    _table(story, flow, col_widths=[2.5*cm, 5*cm, 9.5*cm])

    _section(story, S, "6.1 Interactive Entry Point Output", 2)
    _code(story, S, [
        "$ python run_hyde_rag.py",
        "",
        "  Question: Which category has the highest late delivery rate?",
        "",
        "  Hypothetical document (used for retrieval):",
        "    The product category with the highest late delivery rate is office_furniture,",
        "    at 12.3%. Average delivery days: 18.4 (estimated: 14.2). Review Score: 3.1.",
        "    Total Orders: 1,243. Late Orders: 153. Top Seller State: SP...",
        "",
        "  Answer:",
        "    Based on the provided context, the computers_accessories category had",
        "    the highest late delivery rate at 11.8%, with an average review score of 3.9.",
        "",
        "  Retrieved documents:",
        "    [category_computers_accessories]  distance=0.1234  type=product_category",
        "    [category_office_furniture]       distance=0.1456  type=product_category",
        "",
        "$ python run_hyde_rag.py --ingest   # Force re-index without interactive mode",
    ])

    # ── 7. API Key Management ──
    _section(story, S, "7. Groq API Key Management")
    _body(story, S,
        "HyDE makes <b>2 Groq API calls per query</b> instead of Naive RAG's 1. "
        "Both calls route through the same key-management pool. The evaluation "
        "script maintains 5 primary key slots (one per batch of 20 queries) "
        "with additional fallback keys. The same exhaustion policy (permanent ban "
        "vs transient skip) applies to both the hypothetical-doc call and the "
        "answer call.")

    key_tbl = [
        ["Key Slot", "Query Range", "Calls Per Batch", "On Primary Exhaustion"],
        ["Key #1",  "Queries 1–20",   "40  (20 queries × 2 calls)", "Falls back to keys #6, #7, ... in order"],
        ["Key #2",  "Queries 21–40",  "40  (20 queries × 2 calls)", "Falls back to keys #6, #7, ... in order"],
        ["Key #3",  "Queries 41–60",  "40  (20 queries × 2 calls)", "Falls back to keys #6, #7, ... in order"],
        ["Key #4",  "Queries 61–80",  "40  (20 queries × 2 calls)", "Falls back to keys #6, #7, ... in order"],
        ["Key #5",  "Queries 81–100", "40  (20 queries × 2 calls)", "Falls back to keys #6, #7, ... in order"],
    ]
    _table(story, key_tbl, col_widths=[2.5*cm, 3*cm, 4*cm, 7.5*cm])

    # ── 8. Project Structure ──
    story.append(PageBreak())
    _section(story, S, "8. Project File Structure")
    _code(story, S, [
        "hyde_rag/",
        "  __init__.py",
        "  implementation/",
        "    __init__.py",
        "    config.py        # BASE_DIR, CHROMA_DB_PATH, TOP_K=5, HYDE_TEMPERATURE=0.7, ANSWER_TEMPERATURE=0.1",
        "    ingestion.py     # build_vector_store(), get_collection() — shared with naive_rag",
        "    generator.py     # generate_hypothetical_doc() [Call 1] + generate() [Call 2]",
        "    retriever.py     # retrieve(): hyp-doc gen → ChromaDB query → top-k real docs",
        "    pipeline.py      # run_hyde_rag(query) — full 3-step orchestration",
        "  evaluation/",
        "    __init__.py",
        "    run_hyde_rag_eval.py   # 100-query evaluation: Steps 1a + 1b + 2 + 3 + 4",
        "  results/",
        "    HyDE-RAG_01-05-2026_04-39PM.xlsx   # 5-sheet evaluation workbook",
        "  docs/",
        "    Implementation_of_HyDE_RAG.pdf     # this document",
        "    Evaluation_of_HyDE_RAG.pdf         # evaluation results document",
        "",
        "run_hyde_rag.py    # root entry point: auto-setup + interactive Q&A CLI",
        "chroma_db/         # shared ChromaDB (ecommerce_kb collection, 13,225 docs)",
    ])

    # ── 9. Configuration Reference ──
    _section(story, S, "9. Configuration Reference")
    conf = [
        ["Config Key", "Default", "Description"],
        ["GROQ_API_KEYS",       "env var",                 "Comma-separated list of Groq API keys (13 keys used)"],
        ["EMBEDDING_MODEL",     "all-MiniLM-L6-v2",        "SentenceTransformer model name — shared across all RAGs"],
        ["COLLECTION_NAME",     "ecommerce_kb",             "ChromaDB collection — shared with Naive & Hybrid RAG"],
        ["CHROMA_DB_PATH",      "chroma_db/",               "Shared vector store filesystem path"],
        ["KB_ALL_DOCS",         "dataset/knowledge_base/kb_all_documents.json", "KB source file"],
        ["TOP_K",               "5",                        "Number of real documents retrieved per query (after HyDE)"],
        ["GROQ_MODEL",          "llama-3.3-70b-versatile",  "Groq LLM model for both calls"],
        ["HYDE_TEMPERATURE",    "0.7",                      "LLM temperature for hypothetical doc generation (Call 1)"],
        ["ANSWER_TEMPERATURE",  "0.1",                      "LLM temperature for final answer generation (Call 2)"],
    ]
    _table(story, conf, col_widths=[4.5*cm, 4*cm, 8.5*cm])

    # ── 10. Dependencies ──
    _section(story, S, "10. Key Dependencies")
    deps = [
        ["Library", "Version", "Purpose"],
        ["chromadb",              "≥0.5.0",         "Persistent vector store — shared with Naive & Hybrid RAG"],
        ["sentence-transformers", "≥3.0.0",         "all-MiniLM-L6-v2 — embeds KB docs AND hypothetical docs at query time"],
        ["groq",                  "≥0.11.0",        "Groq API client — called twice per query (temp=0.7 then temp=0.1)"],
        ["pandas / numpy",        "≥2.0 / ≥1.24",  "Data manipulation and ETL"],
        ["scikit-learn",          "latest",         "TF-IDF vectoriser and cosine similarity for evaluation metrics"],
        ["openpyxl",              "≥3.1.0",         "5-sheet Excel evaluation results workbook"],
        ["reportlab",             "≥4.0.0",         "PDF report generation (this document)"],
        ["python-dotenv",         "≥1.0.0",         "GROQ_API_KEYS loading from .env file"],
    ]
    _table(story, deps, col_widths=[5*cm, 3.5*cm, 8.5*cm])

    # ── 11. Summary ──
    _section(story, S, "11. Summary")
    _body(story, S,
        "HyDE RAG extends the Naive RAG baseline with a single architectural change: "
        "before retrieval, a LLM call generates a hypothetical answer passage whose "
        "embedding — rather than the raw query embedding — is used to search the "
        "vector store. This transforms retrieval from a query-to-document asymmetric "
        "comparison into a document-to-document symmetric comparison, theoretically "
        "improving retrieval quality for domain-specific and analytical questions.")
    _body(story, S,
        "The implementation is deliberately minimal: no new storage structures, no "
        "changes to the ChromaDB collection, no new embedding models. The only "
        "additions are a new system prompt, a generator function at temp=0.7, and "
        "the routing of the hypothetical doc text to <code>query_texts</code> in "
        "the ChromaDB call. This clean design makes HyDE easy to compare against "
        "Naive RAG and Hybrid RAG in the evaluation framework.")

    doc.build(story)
    print(f"[PDF 1] Saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PDF 2 — EVALUATION OF HyDE RAG
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf2(output_path: str):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title="Evaluation of HyDE RAG"
    )
    S = _styles()
    story = []

    # ── Cover ──
    _cover_band(story, S,
        title_lines=["Evaluation of HyDE RAG"],
        subtitle="Methodology, Metrics, Results & Research Insights",
        meta_lines=["Reference-Based Evaluation — 100 Queries — Olist E-Commerce Dataset",
                    "RAGAS & DeepEval Frameworks | 1 May 2026"])
    story.append(Spacer(1, 10))

    # ── Abstract ──
    _section(story, S, "Abstract")
    _body(story, S,
        "This document presents a comprehensive evaluation of the HyDE RAG system "
        "across 100 golden dataset queries. All 11 metrics from RAGAS and DeepEval "
        "are computed without an LLM judge, using a reference-based approach with "
        "deterministic token-overlap and TF-IDF algorithms. Each query makes exactly "
        "2 Groq calls: Step 1a (hypothetical document generation, temp=0.7) and "
        "Step 2 (final answer generation, temp=0.1). The evaluation reveals that HyDE "
        "provides only marginal changes over Naive RAG on this structured e-commerce "
        "dataset — with AnswerRelevancy and ContextRecall slightly improving while "
        "Faithfulness and ContextPrecision slightly decline. The fundamental bottleneck "
        "remains retrieval precision, not query representation quality.")

    # ── 1. Evaluation Philosophy ──
    _section(story, S, "1. Evaluation Philosophy")
    _body(story, S,
        "The same reference-based evaluation philosophy from Naive RAG is applied "
        "to HyDE RAG unchanged. A golden dataset of 100 Q&A pairs — each with "
        "an <code>expected_answer</code>, <code>expected_context</code>, and "
        "<code>expected_source_ids</code> generated by Gemini Flash — serves as "
        "the ground truth oracle. All metrics are computed using deterministic "
        "algorithms with zero additional LLM calls during evaluation.")

    phil = [
        ["Principle", "Implementation"],
        ["No LLM judge",        "All metrics: deterministic algorithms; 0 evaluation LLM calls"],
        ["Reference-based",     "Golden dataset (expected_answer + expected_source_ids) as oracle"],
        ["Exact ID matching",   "Context precision/recall use ChromaDB document IDs, not text overlap"],
        ["Verbosity-robustness","ROUGE-L (LCS) for factual correctness — handles verbose answers"],
        ["HyDE-specific",       "Hypothetical doc logged per query; visible in Excel Sheet 2 (RAG Responses)"],
        ["2 LLM calls tracked", "Step 1a (HyDE doc) and Step 2 (answer) logged separately per query"],
    ]
    _table(story, phil, col_widths=[4.5*cm, 12.5*cm])

    # ── 2. Golden Dataset ──
    _section(story, S, "2. Golden Dataset")
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
        "Eleven metrics are computed for each query — five RAGAS and six DeepEval. "
        "Metrics are identical in computation to the Naive RAG evaluation, enabling "
        "direct comparison. The only evaluation-level change for HyDE is that "
        "Step 1a (hypothetical doc generation) is now logged separately from "
        "Step 1b (retrieval) in the evaluation pipeline.")

    _section(story, S, "3.1 RAGAS Metrics", 2)
    ragas_defs = [
        ["Metric", "Formula", "Pass Threshold"],
        ["Faithfulness",       "supported_sentences / total_sentences  (support = ≥50% tokens in context)", "≥ 0.5"],
        ["Answer Relevancy",   "TF-IDF cosine(generated_answer, question)",                               "≥ 0.5"],
        ["Context Precision",  "AP@k — (1/R) × Σ P@k × rel(k) — exact ID match vs expected_source_ids", "≥ 0.5"],
        ["Context Recall",     "|tokens(expected_answer) ∩ tokens(combined_retrieved)| / |tokens(expected_answer)|", "≥ 0.5"],
        ["Factual Correctness","ROUGE-L F1 (LCS-based) between generated_answer and expected_answer",     "≥ 0.5"],
    ]
    _table(story, ragas_defs, col_widths=[4*cm, 9*cm, 4*cm])

    _section(story, S, "3.2 DeepEval Metrics", 2)
    de_metrics = [
        ["Metric", "Formula", "PASS Threshold"],
        ["Answer Relevancy",     "TF-IDF cosine(generated_answer, question)",              "≥ 0.5"],
        ["Faithfulness",         "Sentence-support fraction (same as RAGAS)",              "≥ 0.5"],
        ["Contextual Precision", "AP@k with exact ID match (same as RAGAS)",               "≥ 0.5"],
        ["Contextual Recall",    "Token recall of expected_answer in context (same)",       "≥ 0.5"],
        ["Contextual Relevancy", "Mean TF-IDF cosine(each_retrieved_doc, question)",        "≥ 0.5"],
        ["Hallucination",        "1.0 − sentence_faithfulness",                            "≤ 0.5 (lower better)"],
    ]
    _table(story, de_metrics, col_widths=[4.5*cm, 8*cm, 4.5*cm])

    # ── 4. Evaluation Pipeline ──
    _section(story, S, "4. HyDE Evaluation Pipeline Architecture")
    _body(story, S,
        "The HyDE evaluation script (<code>hyde_rag/evaluation/run_hyde_rag_eval.py</code>) "
        "extends the Naive RAG evaluation with an additional Step 1a for the hypothetical "
        "document generation. Each query now spans 4 steps (1a, 1b, 2, 3, 4) instead of "
        "3 (1, 2, 3, 4). The parallel batch architecture and key-management logic are "
        "identical to Naive RAG.")

    steps = [
        ["Step", "Operation", "LLM Calls", "Output"],
        ["1a — HyDE Doc Gen",  "Groq: generate hypothetical passage (temp=0.7, 256 tokens)",    "1", "hypothetical_doc string"],
        ["1b — Retrieval",     "ChromaDB cosine-similarity on hypothetical doc, top-k=5",        "0", "5 real docs + IDs + distances"],
        ["2  — Generate",      "Groq: final answer from original query + real docs (temp=0.1)",  "1", "generated_answer string"],
        ["3  — RAGAS",         "_compute_metrics() — all 5 RAGAS scores",                        "0", "faithfulness, relevancy, precision, recall, correctness"],
        ["4  — DeepEval",      "_compute_metrics() reuse — all 6 DeepEval scores",               "0", "relevancy, faithfulness, precision, recall, relevancy, hallucination"],
    ]
    _table(story, steps, col_widths=[2.5*cm, 6*cm, 1.5*cm, 7*cm])

    _section(story, S, "4.1 Parallel Batch Architecture", 2)
    _code(story, S, [
        "ThreadPoolExecutor(max_workers=5)",
        "  │",
        "  ├─ Thread 1 → Key #1 → Queries 1-20   (sequential, 5-20s delay, 40 Groq calls)",
        "  ├─ Thread 2 → Key #2 → Queries 21-40  (sequential, 5-20s delay, 40 Groq calls)",
        "  ├─ Thread 3 → Key #3 → Queries 41-60  (sequential, 5-20s delay, 40 Groq calls)",
        "  ├─ Thread 4 → Key #4 → Queries 61-80  (sequential, 5-20s delay, 40 Groq calls)",
        "  └─ Thread 5 → Key #5 → Queries 81-100 (sequential, 5-20s delay, 40 Groq calls)",
        "",
        "Total Groq calls: 200  (2 per query × 100 queries)",
        "Total wall time ≈ max(batch_time) ≈ 20 queries × ~12.5s avg ≈ ~4-5 minutes",
    ])

    # ── 5. Results ──
    story.append(PageBreak())
    _section(story, S, "5. Experimental Results")
    _body(story, S,
        "The full evaluation was conducted on all 100 golden dataset queries across "
        "five parallel batches on <b>1 May 2026</b>. Results below are mean scores "
        "across all 100 queries, with Naive RAG baseline shown for direct comparison.")

    _section(story, S, "5.1 Aggregate Scores vs Naive RAG Baseline", 2)
    results = [
        ["Framework", "Metric", "HyDE RAG", "Naive RAG", "Delta", "Interpretation"],
        ["RAGAS",    "Faithfulness",         "0.234", "0.268", "-0.034", "Slightly worse — 2 LLM calls can amplify hallucination"],
        ["RAGAS",    "Answer Relevancy",     "0.515", "0.508", "+0.007", "Marginally better — hyp-doc keeps answer on-topic"],
        ["RAGAS",    "Context Precision",    "0.312", "0.322", "-0.010", "Slightly worse — hyp-doc occasionally misdirects retrieval"],
        ["RAGAS",    "Context Recall",       "0.344", "0.334", "+0.010", "Marginally better — richer hyp-doc vocabulary"],
        ["RAGAS",    "Factual Correctness",  "0.113", "0.116", "-0.003", "Effectively identical — ROUGE-L not affected by retrieval method"],
        ["DeepEval", "Answer Relevancy",     "0.515", "0.508", "+0.007", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Faithfulness",         "0.234", "0.268", "-0.034", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Contextual Precision", "0.312", "0.322", "-0.010", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Contextual Recall",    "0.344", "0.334", "+0.010", "Same as RAGAS (identical computation)"],
        ["DeepEval", "Contextual Relevancy", "0.124", "0.123", "+0.001", "Negligible — retrieved docs still not well-aligned to question"],
        ["DeepEval", "Hallucination",        "0.766", "0.732", "+0.034", "Higher — HyDE's extra LLM call can reinforce hallucination"],
    ]
    _table(story, results, col_widths=[2*cm, 4*cm, 2*cm, 2*cm, 1.5*cm, 5.5*cm])

    _section(story, S, "5.2 Performance by Query Type", 2)
    qtype = [
        ["Query Type", "HyDE Ctx Precision", "Naive Ctx Precision", "HyDE Better?", "Observation"],
        ["Factual / Easy",      "0.8–1.0", "0.8–1.0", "Equal",  "Both systems retrieve the right category/delivery-status doc"],
        ["Analytical / Medium", "0.0–0.5", "0.0–0.5", "Slight", "Hypothetical doc helps for broader analytical queries"],
        ["Exact ID / Order",    "~0.0",    "~0.0",    "Equal",  "Both fail — neither system finds exact UUID-based docs reliably"],
        ["Temporal / Month",    "~0.0",    "~0.0",    "Equal",  "Hypothetical doc does not solve the 25-docs-in-13K problem"],
        ["Comparison / Hard",   "0.0–0.33","0.0–0.33","Equal",  "Multi-source queries fail for both systems similarly"],
    ]
    _table(story, qtype, col_widths=[3.5*cm, 2.5*cm, 2.5*cm, 2*cm, 6.5*cm])

    # ── 6. Top 5 Positive Insights ──
    story.append(PageBreak())
    _section(story, S, "6. Top 5 Positive Research Insights")

    pos_insights = [
        ("Category & Delivery-Status Queries Remain Excellent",
         "HyDE RAG maintains near-perfect Context Precision (0.8–1.0) for the 73 product "
         "categories and 3 delivery-status documents — the same strength as Naive RAG. "
         "Queries like q001 (portateis_cozinha), q008 (cine_photo), q016 (early_delivery), "
         "q031 (signaling_and_security) all achieve CP=1.0 and CR=1.0. For these query types, "
         "the hypothetical doc correctly mirrors the category-level KB document style, "
         "confirming the HyDE concept works in its intended use case."),

        ("AnswerRelevancy Marginally Higher (0.515 vs 0.508 Naive)",
         "The HyDE hypothetical document forces the LLM to reason explicitly about "
         "what a relevant answer looks like before retrieval. This appears to subtly "
         "improve the final answer's topical alignment with the question — even though "
         "the actual answer is generated from real retrieved documents, the model has "
         "already 'primed' itself on the expected answer format. The +0.007 improvement "
         "is small but consistent across multiple runs."),

        ("Context Recall Slightly Improved (0.344 vs 0.334 Naive)",
         "The hypothetical document, by using domain-specific vocabulary that mirrors "
         "KB document style, retrieves real documents containing slightly more of the "
         "expected answer's key tokens. The +0.010 improvement in Context Recall suggests "
         "that vocabulary bridging — generating domain-appropriate terms that appear in "
         "the real KB — is working as intended, even if the magnitude is modest."),

        ("Hypothetical Documents Are Informative and Interpretable",
         "Each evaluation query's hypothetical document is logged and saved in the Excel "
         "output (Sheet 2: RAG Responses). These passages provide direct insight into what "
         "the model 'thinks' the answer should look like before retrieval. For analytical "
         "queries, the hypothetical docs are often well-reasoned summaries — even when the "
         "retrieved real documents are imperfect. This makes HyDE evaluation outputs "
         "significantly more interpretable than Naive RAG for debugging retrieval failures."),

        ("Consistent Evaluation Infrastructure Enables Direct Comparison",
         "Because HyDE and Naive RAG use identical metric computation (same 11 formulas, "
         "same golden dataset, same evaluation script architecture), the delta values "
         "in Section 5.1 are directly attributable to the HyDE retrieval mechanism and "
         "not to any evaluation confound. This cross-RAG comparability is a key strength "
         "of the reference-based evaluation design."),
    ]

    for i, (title, body_text) in enumerate(pos_insights, 1):
        story.append(Spacer(1, 6))
        _banner(story, S, f"✓ Insight #{i}: {title}", TEAL)
        _body(story, S, body_text)

    # ── 7. Top 5 Negative Insights ──
    _section(story, S, "7. Top 5 Negative Research Insights")

    neg_insights = [
        ("HyDE Provides Negligible Net Improvement on Structured E-Commerce KB",
         "Across all 11 metrics, the average delta between HyDE and Naive RAG is "
         "effectively zero (max +0.010, max -0.034). The HyDE mechanism was designed "
         "for unstructured document corpora where the embedding asymmetry between "
         "short queries and long documents is large. In this e-commerce KB, the "
         "documents are already short, structured, and field-labeled — the query-to-document "
         "embedding gap is smaller than in typical text corpora. The theoretical benefit "
         "of HyDE does not materialise at scale here."),

        ("Hallucination Increases (0.766 vs 0.732 Naive) — Double LLM Call Effect",
         "Adding a second LLM call per query increases the total hallucination rate. "
         "The hypothetical document generation step (temp=0.7) can introduce plausible "
         "but fabricated statistics ('late delivery rate of 12.3%', 'delivery time 18.4 days') "
         "that influence what the model considers a good answer in Step 2. Even though "
         "the final answer is generated from real retrieved documents, the model may "
         "implicitly anchor on figures from the hypothetical doc. This interaction "
         "between Call 1 and Call 2 is a fundamental risk of the HyDE architecture."),

        ("Temporal and Seller/Order Queries Remain Completely Unsolved",
         "HyDE does not address the core retrieval failure modes of Naive RAG: "
         "(1) Temporal queries still fail because 25 month-level docs are "
         "statistically overwhelmed by 99K+ order docs in cosine search — the hypothetical "
         "doc does not change this distribution imbalance. (2) Seller/order UUID queries "
         "fail because neither the query nor the hypothetical doc contains the exact UUID "
         "string, which only BM25 exact matching can solve. These two failure modes "
         "account for ~40% of the 100-query evaluation set."),

        ("Context Precision Slightly Worse (0.312 vs 0.322 Naive) for Specific Queries",
         "For queries where Naive RAG's direct query embedding points to the right document, "
         "the hypothetical doc sometimes misdirects retrieval. If the LLM generates a "
         "hypothetical passage with subtly wrong entity references (e.g., confusing "
         "'furniture' with 'office_furniture' category name), the cosine search retrieves "
         "the wrong category document. This is the HyDE failure mode: a plausible but "
         "slightly incorrect hypothetical doc is worse than the original query for precision-sensitive lookups."),

        ("2x API Cost and Latency Without Proportional Gain",
         "HyDE makes 200 Groq API calls for 100 queries (vs 100 for Naive RAG). "
         "Given the marginal score improvements, the cost-benefit ratio is poor for "
         "this dataset. The additional 100 hypothetical-doc API calls consume tokens "
         "and add latency without delivering the expected retrieval quality improvement. "
         "For production deployments on similarly structured KBs, HyDE would not "
         "be cost-justified over Naive RAG without further evidence of substantial gains."),
    ]

    for i, (title, body_text) in enumerate(neg_insights, 1):
        story.append(Spacer(1, 6))
        _banner(story, S, f"✗ Issue #{i}: {title}", RED)
        _body(story, S, body_text)

    # ── 8. Key Observations ──
    story.append(PageBreak())
    _section(story, S, "8. Major Key Observations")

    observations = [
        ("HyDE's Benefit is Domain-Sensitive — Structured KBs Limit Its Advantage",
         "HyDE was originally validated on open-domain QA corpora (MS-MARCO, NQ) "
         "where queries are naturally short and documents are long, unstructured text. "
         "The Olist e-commerce KB documents are structured, field-labeled, and relatively "
         "short (~200–600 tokens). The embedding asymmetry that HyDE is designed to address "
         "is much smaller here. This is the primary reason the gains are marginal: the "
         "problem HyDE solves is less severe in this domain."),

        ("Retrieval Precision is the Dominant Bottleneck — Not Query Representation",
         "Both HyDE and Naive RAG achieve Context Precision ~0.31–0.32. The bottleneck "
         "is the KB's statistical distribution (99K orders vs 73 categories) — not how "
         "the query is represented. No query-expansion technique (including HyDE) can "
         "overcome a retrieval index that is 99.5% irrelevant documents for a given "
         "query type. Metadata-filtered retrieval would be more impactful."),

        ("Hypothetical Document Quality Varies Significantly by Query Type",
         "For analytical and comparison queries, the hypothetical docs are well-reasoned "
         "and vocabulary-rich — genuinely improving the semantic search. For factual "
         "lookup queries (specific IDs, exact values), the hypothetical doc is "
         "generically plausible but not precisely targeted. A hybrid approach — "
         "using HyDE for analytical queries and direct query embedding for factual "
         "lookups — would likely outperform either method alone."),

        ("Temperature = 0.7 Creates Variance in Hypothetical Document Quality",
         "High temperature introduces run-to-run variance in the hypothetical document. "
         "Two runs of the same query may produce different hypothetical docs and retrieve "
         "different documents, making evaluation non-deterministic at the retrieval level "
         "(though the metric computation remains deterministic given the same retrieved docs). "
         "For scientific comparison, HyDE evaluations should either fix a random seed or "
         "average across multiple hypothetical doc samples per query."),

        ("Excel Sheet 2 with Hypothetical Docs Enables Deep Post-Hoc Analysis",
         "The evaluation workbook includes the hypothetical document for each query in "
         "Sheet 2 (RAG Responses). This enables detailed analysis of the correlation "
         "between hypothetical doc quality and retrieval success. Visual inspection "
         "reveals that queries where the hypothetical doc uses exact KB field names "
         "(e.g., 'Late Delivery Rate:', 'Average Review Score:') achieve significantly "
         "better retrieval than those where the doc uses generic phrasing. Future "
         "prompt engineering should reinforce these field-name patterns."),

        ("HyDE vs Naive vs Hybrid — A Triangular Performance Picture is Emerging",
         "With Naive RAG (Context Precision 0.322) and HyDE RAG (0.312) both measured, "
         "the triangular comparison becomes clear: Naive RAG and HyDE RAG are nearly "
         "equivalent on this dataset, while Hybrid RAG (BM25 + dense) is expected to "
         "outperform both for seller/order ID queries and temporal queries — exactly "
         "the query types where both dense-retrieval approaches fail. The evaluation "
         "framework is now ready to confirm this with a Hybrid RAG run."),
    ]

    for i, (title, body_text) in enumerate(observations, 1):
        story.append(Spacer(1, 6))
        _banner(story, S, f"◆ Observation #{i}: {title}", AMBER)
        _body(story, S, body_text)

    # ── 9. Recommendations ──
    _section(story, S, "9. Recommendations for Future Work")
    recs = [
        ["Priority", "Recommendation", "Expected Impact"],
        ["High",   "Metadata-filtered retrieval: classify query type → filter by document_type before cosine search",
         "Context Precision +0.15–0.25; addresses dominant failure mode"],
        ["High",   "Hybrid RAG (BM25 + dense): adds exact-match for UUIDs, seller IDs, date tokens",
         "Context Precision +0.10–0.20 for seller/order/temporal queries"],
        ["Medium", "Query-type routing: use HyDE for analytical queries, direct embedding for factual lookups",
         "Context Precision +0.05–0.10 across all query types"],
        ["Medium", "HyDE prompt engineering: explicitly instruct LLM to use KB field names in hypothetical doc",
         "Context Precision +0.05 by closer vocabulary alignment"],
        ["Low",    "Multi-sample HyDE: generate 3 hypothetical docs, average their embeddings for retrieval",
         "Reduced variance; slight improvement in difficult analytical queries"],
    ]
    _table(story, recs, col_widths=[2*cm, 9.5*cm, 5.5*cm])

    # ── 10. Conclusion ──
    _section(story, S, "10. Conclusion")
    _body(story, S,
        "The HyDE RAG evaluation on 100 Olist e-commerce queries reveals that "
        "Hypothetical Document Embeddings provide only marginal improvements over "
        "Naive RAG on this structured KB. Answer Relevancy improves by +0.007 and "
        "Context Recall by +0.010, while Faithfulness declines by -0.034 and "
        "Hallucination increases by +0.034. The net effect is a near-neutral trade-off "
        "at 2× the API cost.")
    _body(story, S,
        "The finding is consistent with the literature: HyDE benefits are largest when "
        "(1) queries are short and underspecified, (2) documents are long and unstructured, "
        "and (3) the vocabulary gap between query and document is large. The Olist KB's "
        "structured, field-labeled documents reduce the vocabulary gap that HyDE is "
        "designed to bridge. For this e-commerce analytics use case, <b>metadata-filtered "
        "retrieval and Hybrid RAG (BM25 + dense) remain the most promising paths to "
        "substantial performance improvement</b>.")

    _hr(story, DARK_PURPLE, 2)
    story.append(Spacer(1, 6))
    _body(story, S,
        "<i>Evaluation conducted: 1 May 2026 | Model: llama-3.3-70b-versatile | "
        "Dataset: 100 Olist e-commerce Q&A pairs | 2 Groq calls per query | "
        "Evaluation: reference-based, 0 LLM judge calls | Frameworks: RAGAS + DeepEval</i>")

    doc.build(story)
    print(f"[PDF 2] Saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base, exist_ok=True)

    # Delete old single-file documentation
    old = os.path.join(base, "HyDE_RAG_Documentation.pdf")
    if os.path.exists(old):
        os.remove(old)
        print(f"[DEL] Removed old file: {old}")

    pdf1 = os.path.join(base, "Implementation_of_HyDE_RAG.pdf")
    pdf2 = os.path.join(base, "Evaluation_of_HyDE_RAG.pdf")

    build_pdf1(pdf1)
    build_pdf2(pdf2)
    print("\nDone. Both PDFs are in hyde_rag/docs/")

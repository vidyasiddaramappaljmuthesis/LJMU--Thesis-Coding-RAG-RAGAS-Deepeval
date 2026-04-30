"""
Documentation generator — Golden Dataset Creation Guide.
Produces:  docs/golden_dataset_creation_guide.pdf
Run:       python docs/generate_golden_dataset_docs.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from pathlib import Path
import datetime

# ── Output path ───────────────────────────────────────────────────────────────
OUT_DIR  = Path(__file__).parent
PDF_PATH = OUT_DIR / "golden_dataset_creation_guide.pdf"

# ── Colour palette ────────────────────────────────────────────────────────────
C_DARK_BLUE   = colors.HexColor("#1A237E")
C_BLUE        = colors.HexColor("#283593")
C_ACCENT      = colors.HexColor("#1565C0")
C_LIGHT_BLUE  = colors.HexColor("#E3F2FD")
C_TEAL        = colors.HexColor("#00695C")
C_ORANGE      = colors.HexColor("#E65100")
C_GREEN       = colors.HexColor("#2E7D32")
C_PURPLE      = colors.HexColor("#6A1B9A")
C_HEADER_BG   = colors.HexColor("#1A237E")
C_ROW_EVEN    = colors.HexColor("#F5F5F5")
C_ROW_ODD     = colors.white
C_CODE_BG     = colors.HexColor("#F8F8F8")
C_CODE_BORDER = colors.HexColor("#CCCCCC")
C_SECTION_BG  = colors.HexColor("#EEF2FF")
C_WARN_BG     = colors.HexColor("#FFF8E1")
C_WARN_BORDER = colors.HexColor("#F9A825")
C_SUCCESS_BG  = colors.HexColor("#E8F5E9")
C_SUCCESS_BDR = colors.HexColor("#2E7D32")

W, H = A4

# ── Styles ────────────────────────────────────────────────────────────────────

def _make_styles():
    s = {}
    s["title"] = ParagraphStyle(
        "DocTitle", fontSize=26, leading=34, alignment=TA_CENTER,
        textColor=colors.white, spaceAfter=6, fontName="Helvetica-Bold",
    )
    s["subtitle"] = ParagraphStyle(
        "DocSubtitle", fontSize=13, leading=19, alignment=TA_CENTER,
        textColor=colors.HexColor("#BBDEFB"), spaceAfter=4, fontName="Helvetica",
    )
    s["cover_meta"] = ParagraphStyle(
        "CoverMeta", fontSize=11, leading=16, alignment=TA_CENTER,
        textColor=colors.HexColor("#CFD8DC"), spaceAfter=2, fontName="Helvetica",
    )
    s["h1"] = ParagraphStyle(
        "H1", fontSize=18, leading=24, spaceBefore=18, spaceAfter=8,
        textColor=C_DARK_BLUE, fontName="Helvetica-Bold", borderPad=4,
    )
    s["h2"] = ParagraphStyle(
        "H2", fontSize=14, leading=20, spaceBefore=14, spaceAfter=6,
        textColor=C_ACCENT, fontName="Helvetica-Bold",
    )
    s["h3"] = ParagraphStyle(
        "H3", fontSize=12, leading=17, spaceBefore=10, spaceAfter=4,
        textColor=C_TEAL, fontName="Helvetica-Bold",
    )
    s["h4"] = ParagraphStyle(
        "H4", fontSize=11, leading=15, spaceBefore=8, spaceAfter=3,
        textColor=C_ORANGE, fontName="Helvetica-Bold",
    )
    s["body"] = ParagraphStyle(
        "Body", fontSize=10, leading=15, spaceAfter=6,
        textColor=colors.HexColor("#212121"), fontName="Helvetica",
        alignment=TA_JUSTIFY,
    )
    s["body_left"] = ParagraphStyle(
        "BodyLeft", fontSize=10, leading=15, spaceAfter=4,
        textColor=colors.HexColor("#212121"), fontName="Helvetica",
    )
    s["note"] = ParagraphStyle(
        "Note", fontSize=9.5, leading=14, spaceAfter=4,
        textColor=colors.HexColor("#37474F"), fontName="Helvetica-Oblique",
        leftIndent=10,
    )
    s["bullet"] = ParagraphStyle(
        "Bullet", fontSize=10, leading=15, spaceAfter=3,
        textColor=colors.HexColor("#212121"), fontName="Helvetica",
        leftIndent=16, bulletIndent=6,
    )
    s["code"] = ParagraphStyle(
        "Code", fontSize=8.5, leading=13, spaceAfter=0,
        textColor=colors.HexColor("#212121"), fontName="Courier", leftIndent=8,
    )
    s["code_comment"] = ParagraphStyle(
        "CodeComment", fontSize=8.5, leading=13, spaceAfter=0,
        textColor=colors.HexColor("#5D6D7E"), fontName="Courier-Oblique", leftIndent=8,
    )
    s["caption"] = ParagraphStyle(
        "Caption", fontSize=8.5, leading=12, spaceAfter=6, spaceBefore=2,
        textColor=colors.HexColor("#757575"), fontName="Helvetica-Oblique",
        alignment=TA_CENTER,
    )
    return s


ST = _make_styles()


# ── Helper flowables ──────────────────────────────────────────────────────────

def sp(h=0.3):
    return Spacer(1, h * cm)

def hr(color=C_ACCENT, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceAfter=6, spaceBefore=4)

def h1(text):
    return [sp(0.3), hr(C_DARK_BLUE, 1.5), Paragraph(text, ST["h1"]), hr(C_ACCENT, 0.5)]

def h2(text):
    return [sp(0.2), Paragraph(text, ST["h2"])]

def h3(text):
    return [sp(0.1), Paragraph(text, ST["h3"])]

def h4(text):
    return [Paragraph(text, ST["h4"])]

def body(text):
    return Paragraph(text, ST["body"])

def body_l(text):
    return Paragraph(text, ST["body_left"])

def note(text):
    return Paragraph(f"<i>{text}</i>", ST["note"])

def bullet(text, level=1):
    indent = 16 * level
    style = ParagraphStyle(
        f"Bul{level}", parent=ST["bullet"],
        leftIndent=indent, bulletIndent=indent - 10,
    )
    return Paragraph(f"• {text}", style)

def code_block(lines):
    rows = []
    for line, is_cmt in lines:
        st = ST["code_comment"] if is_cmt else ST["code"]
        rows.append([Paragraph(line, st)])
    tbl = Table(rows, colWidths=[16.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_CODE_BG),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_CODE_BORDER),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl

def info_box(title, lines, bg=C_SECTION_BG, border=C_ACCENT):
    content = [[Paragraph(f"<b>{title}</b>", ST["body_left"])]]
    for l in lines:
        content.append([Paragraph(f"  {l}", ST["body_left"])])
    tbl = Table(content, colWidths=[16.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("BOX",           (0, 0), (-1, -1), 1.0, border),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND",    (0, 0), (-1, 0), border),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
    ]))
    return tbl

def warn_box(title, lines):
    return info_box(title, lines, bg=C_WARN_BG, border=C_WARN_BORDER)

def success_box(title, lines):
    return info_box(title, lines, bg=C_SUCCESS_BG, border=C_SUCCESS_BDR)

def data_table(headers, rows, col_widths=None):
    data = [headers] + rows
    n_cols = len(headers)
    if col_widths is None:
        col_widths = [16.5 * cm / n_cols] * n_cols
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  C_HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]
    for i in range(1, len(data)):
        bg = C_ROW_EVEN if i % 2 == 0 else C_ROW_ODD
        style.append(("BACKGROUND", (0, i), (-1, i), bg))
    tbl.setStyle(TableStyle(style))
    return tbl


# ── Cover page ────────────────────────────────────────────────────────────────

def cover_page():
    e = []
    e.append(sp(2.5))

    banner_data = [
        [Paragraph("Golden Dataset Creation Guide", ST["title"])],
        [Paragraph("End-to-End Documentation", ST["subtitle"])],
        [Paragraph("Olist E-Commerce RAG Evaluation Pipeline", ST["subtitle"])],
    ]
    banner = Table(banner_data, colWidths=[17 * cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_DARK_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
        ("BOX",           (0, 0), (-1, -1), 2, C_ACCENT),
    ]))
    e.append(banner)
    e.append(sp(1.2))

    date_str = datetime.datetime.now().strftime("%B %Y")
    meta_rows = [
        [Paragraph("<b>Project:</b>   E-Commerce RAG with RAGAS and DeepEval", ST["body_left"])],
        [Paragraph(f"<b>Date:</b>     {date_str}", ST["body_left"])],
        [Paragraph("<b>Dataset:</b>   Brazilian Olist E-Commerce (Kaggle)", ST["body_left"])],
        [Paragraph("<b>Model:</b>     Google Gemini 3 Flash (gemini-3-flash-preview)", ST["body_left"])],
        [Paragraph("<b>Purpose:</b>   100-question golden Q&amp;A dataset for RAGAS + DeepEval evaluation", ST["body_left"])],
    ]
    meta_tbl = Table(meta_rows, colWidths=[17 * cm])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_LIGHT_BLUE),
        ("BOX",           (0, 0), (-1, -1), 1, C_ACCENT),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    e.append(meta_tbl)
    e.append(sp(1.5))

    stats = [
        ("100",  "Total\nQuestions"),
        ("5",    "Gemini API\nKeys"),
        ("7",    "KB Layers\nCovered"),
        ("3",    "Difficulty\nLevels"),
    ]
    stat_cells = []
    for val, lbl in stats:
        cell = Table(
            [[Paragraph(f"<b>{val}</b>", ParagraphStyle(
                "SV", fontSize=22, textColor=colors.white,
                alignment=TA_CENTER, fontName="Helvetica-Bold"))],
             [Paragraph(lbl, ParagraphStyle(
                "SL", fontSize=8, textColor=colors.HexColor("#BBDEFB"),
                alignment=TA_CENTER, fontName="Helvetica", leading=11))]],
            colWidths=[3.8 * cm],
        )
        cell.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_DARK_BLUE),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("BOX",           (0, 0), (-1, -1), 1, C_ACCENT),
        ]))
        stat_cells.append(cell)

    stats_row = Table([stat_cells], colWidths=[4.0 * cm] * 4)
    stats_row.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    e.append(stats_row)
    e.append(sp(1.2))

    e.append(body_l(
        "This guide explains every step involved in generating the golden evaluation "
        "dataset: from selecting source KB documents, building prompts, calling the "
        "Gemini 3 Flash model across five API keys, saving checkpoints, and producing "
        "the final <b>golden_dataset.csv</b> used by RAGAS and DeepEval."
    ))
    e.append(PageBreak())
    return e


# ── Section 1 — Overview ─────────────────────────────────────────────────────

def section_overview():
    e = []
    e += h1("1. What is a Golden Dataset?")
    e.append(body(
        "A golden dataset (also called a ground-truth evaluation set) is a curated collection "
        "of question–answer pairs where the correct answer is known in advance. In a RAG "
        "evaluation pipeline, it serves as the benchmark: each question is sent to the RAG "
        "system, and the returned answer is compared against the pre-known ground-truth answer "
        "using automated metrics."
    ))
    e.append(sp(0.3))

    e += h2("1.1  Why a Golden Dataset is Essential for RAG Evaluation")
    e.append(body(
        "Without a golden dataset you cannot measure whether your RAG pipeline is actually "
        "correct — only whether it produces fluent-sounding text. The two evaluation "
        "frameworks used in this project depend on it in different ways:"
    ))
    e.append(data_table(
        ["Framework", "Uses Golden Dataset For", "Key Metrics"],
        [
            ["RAGAS",
             "Compares generated answer to ground_truth,\nand retrieved contexts to expected_context",
             "Answer Correctness, Context Recall,\nFaithfulness, Context Precision"],
            ["DeepEval",
             "Compares actual_output to expected_output,\nand retrieval to context list",
             "G-Eval Correctness, Contextual Recall,\nContextual Precision, Hallucination"],
        ],
        col_widths=[3 * cm, 8 * cm, 5.5 * cm],
    ))
    e.append(sp(0.3))

    e += h2("1.2  Why Use an LLM to Generate the Golden Dataset?")
    e.append(body(
        "There are three main approaches to generating a golden dataset. This project uses "
        "<b>Approach 3</b> — LLM generation from knowledge base documents — as it is the "
        "most suitable for free-text RAG evaluation:"
    ))
    e.append(data_table(
        ["Approach", "How", "Best For", "Limitation"],
        [
            ["1. Manual curation",
             "Humans write Q&A pairs",
             "High-stakes production systems",
             "Very slow; 100 questions takes days"],
            ["2. Pandas computation",
             "Python calculates exact stats\nfrom the enriched CSV",
             "Factual numeric questions\n(counts, averages)",
             "Cannot generate natural-language\nor analytical questions"],
            ["3. LLM from KB docs\n(THIS PROJECT)",
             "LLM reads KB documents\nand generates Q&A",
             "Natural-language RAG eval;\ncovering all question types",
             "Needs quota management;\nanswers may be slightly imprecise"],
        ],
        col_widths=[3.2 * cm, 4 * cm, 4.3 * cm, 5 * cm],
    ))
    e.append(sp(0.3))

    e += h2("1.3  Why Gemini 3 Flash?")
    for item in [
        "<b>Free tier available</b> — 20 requests/day per API key, enabling 5 keys × 20 = 100 questions at zero cost.",
        "<b>JSON mode</b> — supports response_mime_type='application/json', eliminating regex parsing of LLM output.",
        "<b>Quality</b> — produces coherent, KB-grounded questions with minimal hallucination at temperature=0.4.",
        "<b>Speed</b> — Flash variant is ~3× faster than Pro, important given the 5–20 second inter-call delay strategy.",
    ]:
        e.append(bullet(item))
    e.append(sp(0.3))
    e.append(warn_box(
        "Free Tier Constraint",
        [
            "Gemini free tier allows 20 requests per day per API key per model.",
            "To generate 100 questions, 5 separate API keys are required.",
            "Each key processes exactly 20 questions; keys must be fresh (unused that day).",
            "A checkpoint system saves progress per key so a failed run resumes the next day.",
        ],
    ))
    return e


# ── Section 2 — Architecture ──────────────────────────────────────────────────

def section_architecture():
    e = []
    e += h1("2. End-to-End Architecture")
    e.append(body(
        "The golden dataset generation is Step 5 of the full preprocessing pipeline. "
        "It depends on two outputs from earlier steps: the enriched master CSV (Step 3) "
        "and the knowledge base JSON documents (Step 4)."
    ))
    e.append(sp(0.3))

    e += h2("2.1  Pipeline Position")
    e.append(data_table(
        ["Step", "Module", "Output", "Required By Step 5?"],
        [
            ["1 — Load",    "step1_load_raw_data.py",      "9 raw DataFrames",                   "No (indirect)"],
            ["2 — Join",    "step2_join_datasets.py",      "final_olist_master.csv (113k rows)",  "No (indirect)"],
            ["3 — Enrich",  "step3_enrich_master.py",      "final_olist_master_enriched.csv",     "Optional (df arg)"],
            ["4 — KB Build","step4_build_knowledge_base.py","kb_all_documents.json (13,225 docs)","YES — source documents"],
            ["5 — Golden",  "step5_build_golden_dataset.py","golden_dataset.csv (100 rows)",      "—  (this step)"],
        ],
        col_widths=[2.5*cm, 5*cm, 5*cm, 4*cm],
    ))
    e.append(sp(0.3))

    e += h2("2.2  High-Level Flow")
    e.append(code_block([
        ("generate_golden_dataset(df, kb_docs)", False),
        ("  │", False),
        ("  ├─ _read_api_keys()             # read GOOGLE_API_KEY_1 .. GOOGLE_API_KEY_5", True),
        ("  │", False),
        ("  ├─ _load_kb_from_disk()         # fallback: load kb_all_documents.json", True),
        ("  │", False),
        ("  ├─ _group_and_sample(kb_docs)   # bucket docs by layer, cap each layer", True),
        ("  │", False),
        ("  ├─ _build_job_list(docs_by_layer)  # build 100 _Job objects, shuffle them", True),
        ("  │", False),
        ("  └─ for key_idx, (api_key, batch) in enumerate(zip(keys, batches)):", False),
        ("       │", False),
        ("       ├─ if checkpoint exists → load rows, skip API calls", True),
        ("       │", False),
        ("       └─ else → _process_key_batch(api_key, key_idx, batch)", False),
        ("                    │", False),
        ("                    ├─ for each job:", False),
        ("                    │     ├─ build prompt (_prompt_single / _prompt_cross)", True),
        ("                    │     ├─ _call_gemini(client, prompt, key_idx)", True),
        ("                    │     └─ _wait_between_calls()  # 5-20s random delay", True),
        ("                    └─ _save_checkpoint(rows, ckpt_path)", True),
        ("", False),
        ("  → Merge all_rows → assign question_id → save golden_dataset.csv", True),
    ]))
    e.append(sp(0.3))

    e += h2("2.3  Output File")
    e.append(info_box(
        "Final Output",
        [
            "Path:    dataset/golden/golden_dataset.csv",
            "Rows:    100 (one per question)",
            "Columns: question_id, question, expected_answer, expected_context,",
            "         expected_source_ids, question_type, difficulty, best_kb_layer",
            "",
            "Intermediate checkpoints:",
            "  dataset/golden/golden_checkpoint_key1.json  (20 rows)",
            "  dataset/golden/golden_checkpoint_key2.json  (20 rows)",
            "  ...  up to golden_checkpoint_key5.json",
        ],
    ))
    return e


# ── Section 3 — KB Layers and Mapping ────────────────────────────────────────

def section_kb_layers():
    e = []
    e += h1("3. Knowledge Base Layers Used as Source")
    e.append(body(
        "Step 4 produced six KB layers totalling 13,225 JSON documents. "
        "Step 5 uses documents from all six layers as the factual grounding "
        "for Gemini-generated questions. Each layer maps to a distinct "
        "analytical grain and drives a different question style."
    ))
    e.append(sp(0.3))

    e += h2("3.1  Layer → Question Style Mapping")
    e.append(data_table(
        ["Layer", "document_type", "Docs", "Grain", "Question Style"],
        [
            ["order",           "order_level",             "10,000 (sampled)", "order_id",        "Factual lookups about one order"],
            ["category",        "category_level",          "74",               "product category","Revenue, delivery, and review analytics"],
            ["seller",          "seller_level",            "3,095",            "seller_id",       "Seller performance and fulfilment stats"],
            ["state",           "customer_state_level",    "27",               "customer state",  "Geographic order and delivery analysis"],
            ["month",           "month_level",             "25",               "YYYY-MM",         "Temporal trends over time"],
            ["delivery_status", "delivery_status_insight", "4",                "status label",    "Outcome group comparisons"],
        ],
        col_widths=[2.5*cm, 4*cm, 3.5*cm, 2.5*cm, 4*cm],
    ))
    e.append(sp(0.3))

    e += h2("3.2  Document Sampling per Layer")
    e.append(body(
        "Not all 13,225 documents are fed to Gemini. A random sample is drawn "
        "from each layer to keep prompts manageable and to ensure variety across "
        "the 100 questions. Sampling is seeded (RANDOM_SEED=42) for reproducibility."
    ))
    e.append(data_table(
        ["Layer", "Total Docs Available", "Sampled for Jobs (_LAYER_MAX_DOCS)"],
        [
            ["order",           "10,000", "60"],
            ["category",        "74",     "74  (all)"],
            ["seller",          "3,095",  "50"],
            ["state",           "27",     "27  (all)"],
            ["month",           "25",     "25  (all)"],
            ["delivery_status", "4",      "4   (all)"],
        ],
        col_widths=[4*cm, 5.5*cm, 7*cm],
    ))
    e.append(sp(0.3))

    e += h2("3.3  Cross-Layer (Multi-Hop) Questions")
    e.append(body(
        "Ten of the 100 questions are cross-layer: the Gemini prompt receives two "
        "documents from different layers and must generate a question that requires "
        "both documents to answer. This tests the RAG system's ability to retrieve "
        "and synthesise information from multiple sources."
    ))
    e.append(data_table(
        ["Cross-Layer Pair", "Example Question Type"],
        [
            ["category + state",           "Compare a category's avg payment vs a state's avg payment"],
            ["month + delivery_status",    "How did late deliveries change compared to the overall month trend?"],
            ["category + month",           "Was the top category's performance better in a given month?"],
            ["seller + state",             "Does a seller's delivery rate differ from their home state average?"],
            ["state + month",              "How did a state's order volume change month-over-month?"],
            ["category + delivery_status", "Which category is most affected by late deliveries?"],
        ],
        col_widths=[5.5*cm, 11*cm],
    ))
    return e


# ── Section 4 — Target Distribution ──────────────────────────────────────────

def section_distribution():
    e = []
    e += h1("4. Question Distribution — 100 Questions Across Layers and Difficulties")
    e.append(body(
        "The 100 questions are not uniformly distributed. The distribution was designed "
        "to weight higher-volume, analytically richer layers and to include a "
        "mix of easy (factual), medium (analytical), and hard (comparison/synthesis) questions."
    ))
    e.append(sp(0.3))

    e += h2("4.1  Target Distribution Table (_LAYER_TARGETS)")
    e.append(data_table(
        ["Layer", "Easy", "Medium", "Hard", "Total", "Question Type by Difficulty"],
        [
            ["order",           "14", "3",  "0",  "17", "factual / analytical"],
            ["category",        "10", "7",  "3",  "20", "factual / analytical / comparison"],
            ["seller",          "10", "3",  "0",  "13", "factual / analytical"],
            ["state",           "7",  "7",  "3",  "17", "factual / analytical / comparison"],
            ["month",           "7",  "3",  "3",  "13", "factual / analytical / comparison"],
            ["delivery_status", "3",  "4",  "3",  "10", "factual / analytical / comparison"],
            ["cross_layer",     "0",  "3",  "7",  "10", "analytical / comparison"],
            ["<b>TOTAL</b>",   "<b>51</b>","<b>30</b>","<b>19</b>","<b>100</b>",""],
        ],
        col_widths=[3.5*cm, 1.5*cm, 1.8*cm, 1.5*cm, 1.8*cm, 6.4*cm],
    ))
    e.append(sp(0.3))

    e += h2("4.2  Difficulty to Question Type Mapping")
    e.append(data_table(
        ["difficulty", "question_type", "Description", "Prompt Style"],
        [
            ["easy",   "factual",    "Single fact readable directly from one field",
             "What is the...  /  How many...  /  What was the..."],
            ["medium",  "analytical","Interpretation of multiple fields, not just a lookup",
             "Based on the data...  /  What does X suggest about..."],
            ["hard",    "comparison","Synthesis of multiple metrics to draw a conclusion",
             "Given the performance...  /  What strategic insight..."],
        ],
        col_widths=[2.5*cm, 2.8*cm, 5*cm, 6.2*cm],
    ))
    e.append(sp(0.3))

    e += h2("4.3  Actual Distribution Achieved (100 rows)")
    e.append(success_box(
        "Final golden_dataset.csv — achieved distribution",
        [
            "Easy (factual):      51 questions",
            "Medium (analytical): 30 questions",
            "Hard (comparison):   19 questions",
            "",
            "Layer breakdown:",
            "  category: 20  |  order: 17  |  state: 17  |  seller: 13",
            "  month: 13     |  delivery_status: 10  |  cross_layer: 10",
        ],
    ))
    return e


# ── Section 5 — Five-Key Rotation Strategy ───────────────────────────────────

def section_key_rotation():
    e = []
    e += h1("5. Five-Key Rotation Strategy")
    e.append(body(
        "Google Gemini's free tier limits each API key to 20 requests per day per model. "
        "Since the golden dataset requires 100 questions, five separate Google accounts "
        "and API keys are used — each responsible for exactly 20 questions. "
        "This section explains the design decisions behind the rotation strategy."
    ))
    e.append(sp(0.3))

    e += h2("5.1  Why Not Retry on Quota Errors?")
    e.append(body(
        "A common pattern is to catch a 429 RESOURCE_EXHAUSTED error and retry after "
        "a backoff delay. This is the wrong strategy for daily quota exhaustion:"
    ))
    e.append(warn_box(
        "Daily Quota vs Rate Limit — Critical Difference",
        [
            "Rate limit (RPM — requests per minute): recovers in 60 seconds. Retry is safe.",
            "Daily quota (PerDay): does NOT recover until midnight. Every retry wastes time.",
            "Symptom: retryDelay > 120s or error message contains 'PerDay'.",
            "Correct action: skip immediately, move to the next API key.",
        ],
    ))
    e.append(sp(0.3))
    e.append(body(
        "The pipeline distinguishes the two by parsing the retryDelay hint from the "
        "error string. If the delay exceeds 120 seconds, it raises a <b>_DailyQuotaError</b>, "
        "which causes the batch processor to abort the current key's remaining jobs and "
        "move on to the next key (or finish if all keys are exhausted)."
    ))
    e.append(sp(0.3))

    e += h2("5.2  Key Assignment — Pre-Built Job Batches")
    e.append(body(
        "All 100 job objects are built before any API call is made. They are shuffled "
        "randomly so each key gets a representative mix of layers and difficulties — "
        "not, for example, key 1 doing all easy order questions."
    ))
    e.append(code_block([
        ("# Build 100 jobs upfront, shuffle, then split into 5 batches of 20", True),
        ("all_jobs = _build_job_list(docs_by_layer)   # list of 100 _Job objects", False),
        ("random.shuffle(all_jobs)                    # mix layers/difficulties", False),
        ("", False),
        ("batches = []", False),
        ("for i in range(len(keys)):                  # len(keys) == 5", False),
        ("    start = i * QUERIES_PER_KEY             # 0, 20, 40, 60, 80", False),
        ("    end   = start + QUERIES_PER_KEY         # 20, 40, 60, 80, 100", False),
        ("    batches.append(all_jobs[start:end])", False),
    ]))
    e.append(sp(0.3))

    e += h2("5.3  Per-Key Batch Statistics (Actual Run)")
    e.append(data_table(
        ["Key", "Rows Saved", "Easy", "Medium", "Hard", "Top Layers"],
        [
            ["Key 1 (GOOGLE_API_KEY_1)", "20", "13", "4", "3",
             "delivery_status: 7, category: 3, seller: 4"],
            ["Key 2 (GOOGLE_API_KEY_2)", "20", "9",  "8", "3",
             "category: 9, state: 5, delivery_status: 2"],
            ["Key 3 (GOOGLE_API_KEY_3)", "20", "11", "5", "4",
             "order: 6, month: 5, seller: 3"],
            ["Key 4 (GOOGLE_API_KEY_4)", "20", "8",  "7", "5",
             "month: 5, state: 4, seller: 3"],
            ["Key 5 (GOOGLE_API_KEY_5)", "20", "10", "6", "4",
             "order: 6, state: 5, category: 3"],
        ],
        col_widths=[4.5*cm, 2*cm, 1.5*cm, 2*cm, 1.5*cm, 5*cm],
    ))
    e.append(sp(0.3))

    e += h2("5.4  5–20 Second Random Delay Between Calls")
    e.append(body(
        "After each Gemini API call (successful or failed), the pipeline waits a "
        "random delay between 5 and 20 seconds before the next call. This serves two purposes:"
    ))
    for item in [
        "<b>Rate limit avoidance</b> — spreading calls over time keeps the request rate well below the per-minute limit.",
        "<b>Politeness</b> — prevents hammering the API endpoint and triggering adaptive throttling.",
    ]:
        e.append(bullet(item))
    e.append(sp(0.2))
    e.append(code_block([
        ("def _wait_between_calls() -> None:", False),
        ("    delay = random.uniform(_DELAY_MIN_SEC, _DELAY_MAX_SEC)  # 5 to 20 seconds", False),
        ("    logger.info(f'  Waiting {delay:.1f}s before next call ...')", False),
        ("    time.sleep(delay)", False),
        ("", False),
        ("# Called after every job (except the last job in a batch)", True),
        ("if job_num < len(jobs):", False),
        ("    _wait_between_calls()", False),
    ]))
    e.append(sp(0.3))
    e.append(note(
        "With QUERIES_PER_KEY=20 and an average delay of 12.5s, each key batch takes "
        "approximately 20 × (API call ~2s + delay ~12.5s) ≈ 290 seconds (~5 minutes). "
        "All 5 keys together take roughly 25–30 minutes of active API time."
    ))
    return e


# ── Section 6 — Job Dataclass ─────────────────────────────────────────────────

def section_job_dataclass():
    e = []
    e += h1("6. The _Job Dataclass — Pre-Building All 100 Jobs")
    e.append(body(
        "Each of the 100 Gemini calls is represented by a <b>_Job</b> dataclass "
        "instance. Jobs are created before any API call is made. This design ensures "
        "the 20-per-key split is deterministic, reproducible, and free of side effects "
        "from API failures."
    ))
    e.append(sp(0.3))

    e += h2("6.1  _Job Dataclass Definition")
    e.append(code_block([
        ("@dataclass", False),
        ("class _Job:", False),
        ("    layer:       str         # 'order' | 'category' | ... | 'cross_layer'", True),
        ("    difficulty:  str         # 'easy' | 'medium' | 'hard'", True),
        ("    doc:         dict        # primary KB document (always present)", True),
        ("    doc2:        Optional[dict] = None   # second doc for cross-layer jobs", True),
        ("    layer1_name: Optional[str]  = None   # e.g. 'category'", True),
        ("    layer2_name: Optional[str]  = None   # e.g. 'state'", True),
        ("", False),
        ("    @property", False),
        ("    def best_kb_layer(self) -> str:", False),
        ("        if self.layer == 'cross_layer':", False),
        ("            return f'{self.layer1_name}+{self.layer2_name}'  # e.g. 'category+state'", False),
        ("        return self.layer", False),
        ("", False),
        ("    @property", False),
        ("    def question_type(self) -> str:", False),
        ("        return _DIFFICULTY_TO_QTYPE[self.difficulty]  # easy→factual, etc.", False),
    ]))
    e.append(sp(0.3))

    e += h2("6.2  _build_job_list Logic")
    e.append(body(
        "The job list is assembled in two passes. Single-layer layers iterate "
        "through their target counts for each difficulty, sampling documents with "
        "replacement if needed. Cross-layer jobs cycle through predefined pairs."
    ))
    e.append(code_block([
        ("for layer, targets in _LAYER_TARGETS.items():", False),
        ("    if layer == 'cross_layer':", False),
        ("        for difficulty, count in targets.items():", False),
        ("            for _ in range(count):", False),
        ("                l1, l2 = pairs[pair_idx % len(pairs)]", False),
        ("                jobs.append(_Job(layer='cross_layer', difficulty=difficulty,", False),
        ("                                 doc=random.choice(d1), doc2=random.choice(d2),", False),
        ("                                 layer1_name=l1, layer2_name=l2))", False),
        ("    else:", False),
        ("        for difficulty, count in targets.items():", False),
        ("            for i in range(count):", False),
        ("                jobs.append(_Job(layer=layer, difficulty=difficulty,", False),
        ("                                 doc=shuffled[i % len(shuffled)]))", False),
        ("", False),
        ("random.shuffle(jobs)   # mix layers so each key batch gets variety", True),
    ]))
    e.append(sp(0.3))

    e += h2("6.3  Cross-Layer Pairs")
    e.append(body(
        "Eight predefined layer pairs are used for cross-layer jobs. "
        "They are shuffled each run and cycled through using pair_idx % 8:"
    ))
    e.append(data_table(
        ["Pair Index", "Layer 1", "Layer 2", "Analytical Connection"],
        [
            ["0", "category",        "state",           "Category performance by customer geography"],
            ["1", "month",           "delivery_status", "Seasonal delivery performance patterns"],
            ["2", "category",        "month",           "Category sales and trends over time"],
            ["3", "seller",          "state",           "Seller performance vs state average"],
            ["4", "state",           "month",           "State order volumes month-over-month"],
            ["5", "category",        "delivery_status", "Category exposure to late deliveries"],
            ["6", "month",           "state",           "Month trends within a specific state"],
            ["7", "seller",          "month",           "Seller monthly performance trends"],
        ],
        col_widths=[2*cm, 3.5*cm, 3.5*cm, 7.5*cm],
    ))
    return e


# ── Section 7 — Prompt Engineering ───────────────────────────────────────────

def section_prompts():
    e = []
    e += h1("7. Prompt Engineering")
    e.append(body(
        "Two prompt templates are used: one for single-layer jobs and one for "
        "cross-layer (multi-hop) jobs. Both templates instruct Gemini to return "
        "only a valid JSON object with 'question' and 'expected_answer' keys."
    ))
    e.append(sp(0.3))

    e += h2("7.1  Single-Layer Prompt (_prompt_single)")
    e.append(body("The difficulty level controls the task instruction and answer note:"))
    e.append(data_table(
        ["difficulty", "Task Instruction", "Answer Note", "Starter Examples"],
        [
            ["easy",
             "Generate 1 FACTUAL question answerable\nby reading a specific value directly\nfrom the document.",
             "Provide the specific value or short\nfact stated in the document.",
             "What is the...  /  How many...  /  What was the..."],
            ["medium",
             "Generate 1 ANALYTICAL question requiring\ninterpretation of multiple fields —\nnot just reading one number.",
             "1-2 sentences interpreting the data,\nnot just a raw value.",
             "Based on the data...  /  What does X suggest about..."],
            ["hard",
             "Generate 1 CHALLENGING question requiring\nsynthesis of multiple metrics to draw\na conclusion.",
             "2-3 sentences synthesising data points\nand reaching a conclusion.",
             "Given the performance metrics...  /  What strategic insight..."],
        ],
        col_widths=[2*cm, 5*cm, 4.5*cm, 5*cm],
    ))
    e.append(sp(0.3))
    e.append(body("<b>Prompt structure (single-layer):</b>"))
    e.append(code_block([
        ("You are building an evaluation dataset for a Brazilian e-commerce RAG system.", False),
        ("The knowledge base document below describes {layer_desc}.", False),
        ("", False),
        ("TASK", False),
        ("{style}  # difficulty-specific instruction", True),
        ("", False),
        ("RULES", False),
        ("- Answerable using ONLY this document.", False),
        ("- No yes/no questions.", False),
        ("- Do NOT say 'according to the document' — phrase as a natural business query.", False),
        ("", False),
        ("DOCUMENT", False),
        ("{doc['text']}  # full KB document text", True),
        ("", False),
        ("ANSWER NOTE", False),
        ("{answer_note}  # difficulty-specific answer guidance", True),
        ("", False),
        ("Return ONLY a valid JSON object (no markdown):", False),
        ('{\"question\": \"...\", \"expected_answer\": \"...\"}', False),
    ]))
    e.append(sp(0.3))

    e += h2("7.2  Cross-Layer Prompt (_prompt_cross)")
    e.append(body(
        "The cross-layer prompt receives two complete KB documents and instructs "
        "Gemini to generate a question that <b>requires both documents</b> to answer. "
        "The key constraint is that the question must be impossible to answer "
        "with just one document."
    ))
    e.append(code_block([
        ("You are building a multi-hop evaluation dataset for a Brazilian e-commerce RAG system.", False),
        ("", False),
        ("TASK", False),
        ("{task}  # medium: analytical linking | hard: synthesis requiring both docs", True),
        ("", False),
        ("RULES", False),
        ("- MUST require information from BOTH documents.", False),
        ("- No yes/no questions.", False),
        ("- Do NOT say 'Document 1' or 'Document 2' — phrase naturally.", False),
        ("", False),
        ("DOCUMENT 1 (layer: {layer1})", False),
        ("{doc1['text']}", False),
        ("", False),
        ("DOCUMENT 2 (layer: {layer2})", False),
        ("{doc2['text']}", False),
        ("", False),
        ("Return ONLY a valid JSON object (no markdown):", False),
        ('{\"question\": \"...\", \"expected_answer\": \"...\"}', False),
    ]))
    e.append(sp(0.3))

    e += h2("7.3  Response Parsing")
    e.append(body(
        "Gemini is called with response_mime_type='application/json' and "
        "temperature=0.4. Despite JSON mode, occasional markdown fences "
        "(```json ... ```) are stripped before parsing:"
    ))
    e.append(code_block([
        ("text = response.text.strip()", False),
        ("if '```' in text:", False),
        ("    parts = text.split('```')", False),
        ("    text  = parts[1] if len(parts) > 1 else text", False),
        ("    if text.lower().startswith('json'):", False),
        ("        text = text[4:].strip()", False),
        ("", False),
        ("parsed = json.loads(text)", False),
        ("if isinstance(parsed, list) and parsed:  # model sometimes returns [{...}]", True),
        ("    parsed = parsed[0]", False),
        ("", False),
        ("if 'question' in parsed and 'expected_answer' in parsed:", False),
        ("    return parsed  # valid response", False),
    ]))
    return e


# ── Section 8 — Checkpoint System ────────────────────────────────────────────

def section_checkpoints():
    e = []
    e += h1("8. Checkpoint System — Resumable Runs")
    e.append(body(
        "Generating 100 questions across 5 API keys takes ~30 minutes and spans "
        "at least one day (due to daily quota limits). The checkpoint system ensures "
        "that if the process is interrupted — or if a key's quota runs out — the "
        "completed key batches are not re-processed on the next run."
    ))
    e.append(sp(0.3))

    e += h2("8.1  Checkpoint File Format")
    e.append(body("One JSON file is saved per API key after its 20 jobs complete:"))
    e.append(code_block([
        ("# dataset/golden/golden_checkpoint_key1.json", True),
        ("[", False),
        ("  {", False),
        ('    "question":            "What is the late delivery rate for health_beauty?",', False),
        ('    "expected_answer":     "4.56%",', False),
        ('    "expected_context":    "[\"Document Type: Product Category Summary\\n...\"]",', False),
        ('    "expected_source_ids": "[\"category_health_beauty\"]",', False),
        ('    "question_type":       "factual",', False),
        ('    "difficulty":          "easy",', False),
        ('    "best_kb_layer":       "category"', False),
        ("  },", False),
        ("  { ... },  # 19 more rows", True),
        ("]", False),
    ]))
    e.append(sp(0.3))

    e += h2("8.2  Checkpoint Load Logic")
    e.append(code_block([
        ("for key_idx, (api_key, batch) in enumerate(zip(keys, batches), start=1):", False),
        ("    ckpt_path = _CHECKPOINT_DIR / f'golden_checkpoint_key{key_idx}.json'", False),
        ("", False),
        ("    if ckpt_path.exists():", False),
        ("        # Load saved rows — zero API calls made for this key", True),
        ("        with open(ckpt_path, encoding='utf-8') as fh:", False),
        ("            rows = json.load(fh)", False),
        ("        logger.info(f'[Key {key_idx}] Checkpoint found — loading, skipping API calls')", False),
        ("    else:", False),
        ("        # No checkpoint → call Gemini for all 20 jobs", True),
        ("        rows = _process_key_batch(api_key, key_idx, batch)", False),
        ("        _save_checkpoint(rows, ckpt_path)   # save immediately", False),
    ]))
    e.append(sp(0.3))

    e += h2("8.3  Day-by-Day Resumption Plan")
    e.append(data_table(
        ["Day", "Keys Used", "Checkpoints Written", "Remaining"],
        [
            ["Day 1", "Key 1 + Key 2 (40 questions)", "key1.json, key2.json", "60 questions"],
            ["Day 2", "Key 3 + Key 4 (40 questions)", "key3.json, key4.json", "20 questions"],
            ["Day 3", "Key 5 (20 questions)",          "key5.json",           "0 — complete"],
        ],
        col_widths=[2*cm, 5*cm, 5*cm, 4.5*cm],
    ))
    e.append(sp(0.2))
    e.append(note(
        "On Day 2, re-running the pipeline finds key1.json and key2.json already "
        "present and skips them entirely. Only keys 3 and 4 make API calls."
    ))
    e.append(sp(0.3))

    e += h2("8.4  Deleting Checkpoints to Regenerate")
    e.append(warn_box(
        "To regenerate the golden dataset from scratch",
        [
            "Delete all checkpoint files:",
            "  del dataset\\golden\\golden_checkpoint_key*.json",
            "",
            "Then re-run the pipeline. All 5 keys will be used again.",
            "WARNING: This will consume all 5 keys' daily quotas.",
            "Make sure the keys have not been used today before running.",
        ],
    ))
    return e


# ── Section 9 — Output Schema ─────────────────────────────────────────────────

def section_schema():
    e = []
    e += h1("9. Output Schema — golden_dataset.csv")
    e.append(body(
        "The final CSV contains exactly 100 rows and 8 columns. "
        "The schema is designed to be directly consumable by both RAGAS and DeepEval "
        "with minimal transformation."
    ))
    e.append(sp(0.3))

    e += h2("9.1  Column Definitions")
    e.append(data_table(
        ["Column", "Type", "Example Value", "Description"],
        [
            ["question_id",         "str",  "q001",
             "Sequential unique ID. q001 through q100."],
            ["question",            "str",  "What is the late delivery rate for health_beauty?",
             "Natural-language question generated by Gemini."],
            ["expected_answer",     "str",  "4.56%",
             "Gemini-generated answer grounded in the KB document."],
            ["expected_context",    "str",  '["Document Type: Product Category..."]',
             "JSON array of full KB document text(s) used as context."],
            ["expected_source_ids", "str",  '["category_health_beauty"]',
             "JSON array of KB document IDs (matches doc['id'])."],
            ["question_type",       "str",  "factual",
             "Derived from difficulty: easy→factual, medium→analytical, hard→comparison."],
            ["difficulty",          "str",  "easy",
             "Gemini prompt difficulty level: easy | medium | hard."],
            ["best_kb_layer",       "str",  "category",
             "Source layer. Cross-layer uses 'layer1+layer2' format."],
        ],
        col_widths=[3.8*cm, 1.5*cm, 4*cm, 7.2*cm],
    ))
    e.append(sp(0.3))

    e += h2("9.2  RAGAS Field Mapping")
    e.append(body(
        "RAGAS requires a specific column naming convention. Map the golden dataset "
        "columns before passing to RAGAS evaluation functions:"
    ))
    e.append(code_block([
        ("import pandas as pd", False),
        ("from ragas import evaluate", False),
        ("from ragas.metrics import answer_correctness, context_recall, faithfulness", False),
        ("", False),
        ("golden = pd.read_csv('dataset/golden/golden_dataset.csv')", False),
        ("", False),
        ("# RAGAS expects: 'question', 'ground_truth', 'contexts'", True),
        ("ragas_df = pd.DataFrame({", False),
        ("    'question':    golden['question'],", False),
        ("    'ground_truth': golden['expected_answer'],", False),
        ("    'contexts':    golden['expected_context'].apply(json.loads),", False),
        ("    # 'answer' and 'context' will be added by the RAG pipeline at eval time", True),
        ("})", False),
    ]))
    e.append(sp(0.3))

    e += h2("9.3  DeepEval Field Mapping")
    e.append(code_block([
        ("from deepeval.test_case import LLMTestCase", False),
        ("import json", False),
        ("", False),
        ("golden = pd.read_csv('dataset/golden/golden_dataset.csv')", False),
        ("", False),
        ("# DeepEval expects: input, expected_output, context (list)", True),
        ("test_cases = [", False),
        ("    LLMTestCase(", False),
        ("        input            = row['question'],", False),
        ("        expected_output  = row['expected_answer'],", False),
        ("        context          = json.loads(row['expected_context']),", False),
        ("        # actual_output will be filled by the RAG pipeline at eval time", True),
        ("    )", False),
        ("    for _, row in golden.iterrows()", False),
        ("]", False),
    ]))
    return e


# ── Section 10 — Real Examples ────────────────────────────────────────────────

def section_examples():
    e = []
    e += h1("10. Real Examples from the Golden Dataset")
    e.append(body(
        "The following are actual rows from golden_dataset.csv generated during "
        "the production run. They illustrate the variety of question styles "
        "produced across difficulty levels and KB layers."
    ))
    e.append(sp(0.3))

    # Easy example
    e += h2("10.1  Easy (Factual) — Layer: category")
    e.append(data_table(
        ["Field", "Value"],
        [
            ["question_id",         "q001"],
            ["question",            "What is the late delivery rate for the\nportateis_cozinha_e_preparadores_de_alimentos category?"],
            ["expected_answer",     "7.69%"],
            ["question_type",       "factual"],
            ["difficulty",          "easy"],
            ["best_kb_layer",       "category"],
            ["expected_source_ids", '["category_portateis_cozinha_e_preparadores_de_alimentos"]'],
        ],
        col_widths=[4.5*cm, 12*cm],
    ))
    e.append(sp(0.2))
    e.append(note("Source document snippet: 'Late Delivery Rate: 7.69%' — directly readable from the Category KB document."))
    e.append(sp(0.3))

    # Medium example
    e += h2("10.2  Medium (Analytical) — Layer: delivery_status")
    e.append(data_table(
        ["Field", "Value"],
        [
            ["question",        "How does the relationship between delivery precision and average\ndelivery time impact the customer experience for the on_time status group?"],
            ["expected_answer",
             "Even though the average delivery time is nearly 20 days (19.22),\nthe fact that orders arrive exactly when estimated (0.00 days late)\nresults in a positive customer experience, reflected in a high average\nreview score of 4.03."],
            ["question_type",   "analytical"],
            ["difficulty",      "medium"],
            ["best_kb_layer",   "delivery_status"],
        ],
        col_widths=[4.5*cm, 12*cm],
    ))
    e.append(sp(0.2))
    e.append(note("This answer synthesises three fields: Average Delivery Days (19.22), Difference vs Estimated (0.00), and Average Review Score (4.03)."))
    e.append(sp(0.3))

    # Hard example
    e += h2("10.3  Hard (Comparison) — Layer: delivery_status")
    e.append(data_table(
        ["Field", "Value"],
        [
            ["question",        "What strategic insight can be drawn from the relationship between\ndelivery delays, customer satisfaction, and the primary geographic\ncorridor for this specific delivery group?"],
            ["expected_answer",
             "Late deliveries (avg 33.95 days, 10.62 days past estimate) result\nin a significantly low satisfaction score of 2.27. Despite the top\ncustomer and seller states both being SP, the intra-state corridor\ncannot mitigate the negative impact of late fulfilment."],
            ["question_type",   "comparison"],
            ["difficulty",      "hard"],
            ["best_kb_layer",   "delivery_status"],
        ],
        col_widths=[4.5*cm, 12*cm],
    ))
    e.append(sp(0.2))
    e.append(note("Hard questions require synthesising 4+ fields and reaching a business-level conclusion beyond the raw numbers."))
    e.append(sp(0.3))

    # Cross-layer example
    e += h2("10.4  Cross-Layer (Analytical) — Layer: category+state")
    e.append(data_table(
        ["Field", "Value"],
        [
            ["question",        "How does the average payment value for the furniture_living_room\ncategory compare to the average payment value in the state of AM,\nand which of these two has a higher average review score?"],
            ["expected_answer",
             "The furniture_living_room category has a higher average payment\nvalue of 211.90 compared to 188.97 in AM. However, AM has a higher\naverage review score of 4.21 compared to the category's 4.03."],
            ["question_type",   "analytical"],
            ["difficulty",      "medium"],
            ["best_kb_layer",   "category+state"],
            ["expected_source_ids",
             '["category_furniture_living_room", "state_AM"]'],
        ],
        col_widths=[4.5*cm, 12*cm],
    ))
    e.append(sp(0.2))
    e.append(note(
        "This question requires two documents: one category-level doc and one state-level doc. "
        "A RAG system must retrieve both to answer correctly — this tests multi-hop retrieval."
    ))
    return e


# ── Section 11 — How to Run ───────────────────────────────────────────────────

def section_how_to_run():
    e = []
    e += h1("11. How to Run the Golden Dataset Generation")
    e.append(sp(0.3))

    e += h2("11.1  Prerequisites")
    e.append(data_table(
        ["Requirement", "Details"],
        [
            ["Python packages",      "google-genai>=1.0.0  |  pandas>=2.0.0  |  numpy>=1.24.0"],
            ["KB documents",         "Run --steps kb first to produce dataset/golden/ folder and kb_all_documents.json"],
            ["API keys",             "5 Google Cloud / AI Studio API keys with Gemini access"],
            ["Daily quota status",   "All 5 keys must not have used their 20 Gemini requests today"],
            ["Environment variables","GOOGLE_API_KEY_1 through GOOGLE_API_KEY_5 set in shell"],
        ],
        col_widths=[4*cm, 12.5*cm],
    ))
    e.append(sp(0.3))

    e += h2("11.2  Setting API Keys (Windows PowerShell)")
    e.append(code_block([
        ("# Set all 5 keys in PowerShell before running the pipeline", True),
        ("$env:GOOGLE_API_KEY_1 = 'AIzaSy...'", False),
        ("$env:GOOGLE_API_KEY_2 = 'AIzaSy...'", False),
        ("$env:GOOGLE_API_KEY_3 = 'AIzaSy...'", False),
        ("$env:GOOGLE_API_KEY_4 = 'AIzaSy...'", False),
        ("$env:GOOGLE_API_KEY_5 = 'AIzaSy...'", False),
        ("", False),
        ("# Run golden dataset generation only", True),
        ("python data_preparation.py --steps golden", False),
        ("", False),
        ("# Or run the full pipeline (steps 1-5)", True),
        ("python data_preparation.py", False),
    ]))
    e.append(sp(0.3))

    e += h2("11.3  Setting API Keys (Linux / macOS Bash)")
    e.append(code_block([
        ("export GOOGLE_API_KEY_1='AIzaSy...'", False),
        ("export GOOGLE_API_KEY_2='AIzaSy...'", False),
        ("export GOOGLE_API_KEY_3='AIzaSy...'", False),
        ("export GOOGLE_API_KEY_4='AIzaSy...'", False),
        ("export GOOGLE_API_KEY_5='AIzaSy...'", False),
        ("python data_preparation.py --steps golden", False),
    ]))
    e.append(sp(0.3))

    e += h2("11.4  Expected Log Output (Successful Run)")
    e.append(code_block([
        ("INFO  Loaded 13,225 KB documents", False),
        ("INFO  Using 5 API key(s) x 20 queries = 100 planned queries", False),
        ("INFO  Built 100 jobs total", False),
        ("INFO    Sampled  60/10000 docs for 'order'", False),
        ("INFO    Sampled  74/   74 docs for 'category'", False),
        ("INFO  [Key 1] Processing 20 jobs ...", False),
        ("INFO  [Key 1] Job 01/20  layer=category  difficulty=easy", False),
        ("INFO    Waiting 12.3s before next call ...", False),
        ("INFO  [Key 1] Job 02/20  layer=state  difficulty=medium", False),
        ("  ...", True),
        ("INFO  [Key 1] Checkpoint saved: golden_checkpoint_key1.json  (20 rows)", False),
        ("INFO  [Key 1] 20 rows collected (0 skipped/empty)", False),
        ("  ... (keys 2-5 follow same pattern) ...", True),
        ("INFO  Golden dataset complete:", False),
        ("INFO    Total     : 100", False),
        ("INFO    Easy      : 51", False),
        ("INFO    Medium    : 30", False),
        ("INFO    Hard      : 19", False),
        ("INFO    Saved -> golden_dataset.csv  (100 questions)", False),
        ("INFO    Pipeline complete  (1703.3s)", False),
    ]))
    e.append(sp(0.3))

    e += h2("11.5  Run Time")
    e.append(data_table(
        ["Phase", "Approximate Time"],
        [
            ["Load KB docs from disk",             "< 5 seconds"],
            ["Build 100 job objects",               "< 1 second"],
            ["Per key batch (20 calls + delays)",   "~5-8 minutes per key"],
            ["All 5 keys (if no checkpoints exist)","~25-40 minutes total"],
            ["Re-run with all checkpoints present", "< 10 seconds (no API calls)"],
        ],
        col_widths=[8*cm, 8.5*cm],
    ))
    return e


# ── Section 12 — Pitfalls and Best Practices ──────────────────────────────────

def section_pitfalls():
    e = []
    e += h1("12. Pitfalls, Design Decisions and Best Practices")
    e.append(sp(0.3))

    e += h2("12.1  Known Pitfalls")

    pitfalls = [
        (
            "Retrying on 429 RESOURCE_EXHAUSTED (daily quota)",
            "If you retry when the daily quota is exhausted, every retry attempt "
            "wastes time (waiting 5-20s) and eventually times out. The pipeline "
            "detects 'PerDay' in the error string or retryDelay > 120s to skip "
            "the key immediately and raise _DailyQuotaError."
        ),
        (
            "Using a key that was already used today",
            "If GOOGLE_API_KEY_1 was used for 20 Gemini requests earlier today, "
            "its quota is gone. Re-running the pipeline with no checkpoint for key 1 "
            "will cause every call to fail. Always verify key quota status before "
            "running without existing checkpoints."
        ),
        (
            "Gemini returning markdown-fenced JSON (```json ... ```)",
            "Despite using response_mime_type='application/json', Gemini occasionally "
            "wraps the response in markdown code fences. The parser strips the fences "
            "before calling json.loads(). Always test JSON parsing with a stub before "
            "live API calls."
        ),
        (
            "Cross-layer questions answered from a single document",
            "The cross-layer prompt instructs Gemini not to say 'Document 1' or "
            "'Document 2'. However, Gemini may occasionally generate a question "
            "answerable from only one document. Review cross-layer rows manually "
            "if RAGAS context recall is low for those entries."
        ),
        (
            "Checkpoint from a partial (failed) run",
            "If _save_checkpoint is called with fewer than 20 rows (because some "
            "Gemini calls returned empty), the checkpoint is still saved. On re-run, "
            "the short checkpoint is loaded as-is. If you want a full 20-row batch, "
            "delete the checkpoint and re-run that key."
        ),
    ]
    for title, desc in pitfalls:
        e.append(KeepTogether([
            Paragraph(f"<b>{title}:</b>", ST["h4"]),
            Paragraph(desc, ST["body_left"]),
            sp(0.2),
        ]))

    e += h2("12.2  Design Decisions Explained")
    e.append(data_table(
        ["Decision", "Why"],
        [
            ["Pre-build all 100 jobs before API calls",
             "Ensures the 20-per-key split is deterministic. If the split were decided\n"
             "during API calls, a failure mid-batch would leave the split in an unknown state."],
            ["Shuffle jobs before batching",
             "Ensures each key gets a variety of layers and difficulties.\n"
             "Without shuffle, key 1 would get all 'order' easy questions."],
            ["temperature=0.4",
             "Low enough to keep answers grounded in the document; high enough to\n"
             "produce varied phrasings rather than templated output."],
            ["No retry on 429 daily quota",
             "Every failed retry still counts toward the daily quota in some API\n"
             "implementations. Skip immediately to preserve quota for the next key."],
            ["One checkpoint per key (not per question)",
             "Saving per-question would be 100 file writes. Per-key is 5 writes\n"
             "while still guaranteeing progress is saved before moving to the next key."],
            ["RANDOM_SEED=42 for all sampling",
             "Ensures that _group_and_sample picks the same documents on every run,\n"
             "making the golden dataset reproducible even if re-generated from scratch."],
        ],
        col_widths=[5*cm, 11.5*cm],
    ))
    e.append(sp(0.3))

    e += h2("12.3  Quality Checks After Generation")
    for item in [
        "Verify all 8 GOLDEN_COLUMNS are present in the CSV.",
        "Check question_id is sequential (q001 to q100) and unique.",
        "Confirm expected_context parses as a valid JSON list with at least one string.",
        "Confirm expected_source_ids parses as a valid JSON list.",
        "Spot-check 5-10 hard questions — answers should reference multiple metrics from the source document.",
        "For cross-layer rows, verify expected_source_ids contains two different document IDs.",
        "Run: df['difficulty'].value_counts() — expect roughly 50/30/20 split.",
    ]:
        e.append(bullet(item))
    return e


# ── Page footer ───────────────────────────────────────────────────────────────

def on_first_page(canvas, doc):
    pass

def on_later_pages(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#757575"))
    canvas.drawString(2 * cm, 1.2 * cm,
                      "Golden Dataset Creation Guide — Olist E-Commerce RAG Pipeline")
    canvas.drawRightString(W - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.5 * cm, W - 2 * cm, 1.5 * cm)
    canvas.restoreState()


# ── Build PDF ─────────────────────────────────────────────────────────────────

def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.2 * cm, bottomMargin=2.2 * cm,
        title="Golden Dataset Creation Guide — Olist RAG Evaluation Pipeline",
        author="RAG Pipeline",
    )

    story = []

    # Cover
    story += cover_page()

    # Table of Contents
    story += h1("Table of Contents")
    toc_items = [
        ("1.  What is a Golden Dataset?",                     False),
        ("    1.1  Why a Golden Dataset is Essential",        True),
        ("    1.2  Why Use an LLM to Generate It?",           True),
        ("    1.3  Why Gemini 3 Flash?",                      True),
        ("2.  End-to-End Architecture",                       False),
        ("    2.1  Pipeline Position",                        True),
        ("    2.2  High-Level Flow",                          True),
        ("3.  Knowledge Base Layers Used as Source",          False),
        ("    3.1  Layer → Question Style Mapping",           True),
        ("    3.2  Document Sampling per Layer",              True),
        ("    3.3  Cross-Layer (Multi-Hop) Questions",        True),
        ("4.  Question Distribution — 100 Questions",         False),
        ("    4.1  Target Distribution Table",                True),
        ("    4.2  Difficulty to Question Type Mapping",      True),
        ("    4.3  Actual Distribution Achieved",             True),
        ("5.  Five-Key Rotation Strategy",                    False),
        ("    5.1  Why Not Retry on Quota Errors?",           True),
        ("    5.2  Key Assignment — Pre-Built Job Batches",   True),
        ("    5.3  Per-Key Batch Statistics (Actual Run)",    True),
        ("    5.4  5–20 Second Random Delay Between Calls",   True),
        ("6.  The _Job Dataclass — Pre-Building 100 Jobs",    False),
        ("7.  Prompt Engineering",                            False),
        ("    7.1  Single-Layer Prompt",                      True),
        ("    7.2  Cross-Layer Prompt",                       True),
        ("    7.3  Response Parsing",                         True),
        ("8.  Checkpoint System — Resumable Runs",            False),
        ("9.  Output Schema — golden_dataset.csv",            False),
        ("    9.1  Column Definitions",                       True),
        ("    9.2  RAGAS Field Mapping",                      True),
        ("    9.3  DeepEval Field Mapping",                   True),
        ("10. Real Examples from the Golden Dataset",         False),
        ("11. How to Run the Golden Dataset Generation",      False),
        ("12. Pitfalls, Design Decisions and Best Practices", False),
    ]
    for title, is_sub in toc_items:
        st = ParagraphStyle(
            "TOCItem",
            fontSize=10 if is_sub else 11,
            leading=16,
            leftIndent=20 if is_sub else 0,
            textColor=C_ACCENT if is_sub else C_DARK_BLUE,
            fontName="Helvetica" if is_sub else "Helvetica-Bold",
            spaceAfter=2,
        )
        story.append(Paragraph(title, st))

    story.append(PageBreak())

    # All sections
    for sec in [
        section_overview,
        section_architecture,
        section_kb_layers,
        section_distribution,
        section_key_rotation,
        section_job_dataclass,
        section_prompts,
        section_checkpoints,
        section_schema,
        section_examples,
        section_how_to_run,
        section_pitfalls,
    ]:
        story += sec()
        story.append(PageBreak())

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    print(f"PDF written to: {PDF_PATH}")
    print(f"File size:      {PDF_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build_pdf()

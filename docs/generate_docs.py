"""
Documentation generator for the Olist RAG Knowledge Base pipeline.
Produces:  docs/Olist_RAG_Knowledge_Base_Documentation.pdf
Run:       python docs/generate_docs.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
import datetime

# ── Output path ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent
PDF_PATH = OUT_DIR / "Olist_RAG_Knowledge_Base_Documentation.pdf"

# ── Colour palette ────────────────────────────────────────────────────────────
C_DARK_BLUE   = colors.HexColor("#1A237E")
C_BLUE        = colors.HexColor("#283593")
C_ACCENT      = colors.HexColor("#1565C0")
C_LIGHT_BLUE  = colors.HexColor("#E3F2FD")
C_TEAL        = colors.HexColor("#00695C")
C_ORANGE      = colors.HexColor("#E65100")
C_HEADER_BG   = colors.HexColor("#1A237E")
C_ROW_EVEN    = colors.HexColor("#F5F5F5")
C_ROW_ODD     = colors.white
C_CODE_BG     = colors.HexColor("#F8F8F8")
C_CODE_BORDER = colors.HexColor("#CCCCCC")
C_SECTION_BG  = colors.HexColor("#EEF2FF")
C_GREEN       = colors.HexColor("#2E7D32")
C_RED         = colors.HexColor("#B71C1C")

W, H = A4  # 595.27 x 841.89 pts

# ── Styles ────────────────────────────────────────────────────────────────────
base_styles = getSampleStyleSheet()

def make_styles():
    s = {}

    s["title"] = ParagraphStyle(
        "DocTitle",
        fontSize=28, leading=36, alignment=TA_CENTER,
        textColor=colors.white, spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    s["subtitle"] = ParagraphStyle(
        "DocSubtitle",
        fontSize=14, leading=20, alignment=TA_CENTER,
        textColor=colors.HexColor("#BBDEFB"), spaceAfter=4,
        fontName="Helvetica",
    )
    s["cover_meta"] = ParagraphStyle(
        "CoverMeta",
        fontSize=11, leading=16, alignment=TA_CENTER,
        textColor=colors.HexColor("#CFD8DC"), spaceAfter=2,
        fontName="Helvetica",
    )

    s["h1"] = ParagraphStyle(
        "H1",
        fontSize=18, leading=24, spaceBefore=18, spaceAfter=8,
        textColor=C_DARK_BLUE, fontName="Helvetica-Bold",
        borderPad=4,
    )
    s["h2"] = ParagraphStyle(
        "H2",
        fontSize=14, leading=20, spaceBefore=14, spaceAfter=6,
        textColor=C_ACCENT, fontName="Helvetica-Bold",
    )
    s["h3"] = ParagraphStyle(
        "H3",
        fontSize=12, leading=17, spaceBefore=10, spaceAfter=4,
        textColor=C_TEAL, fontName="Helvetica-Bold",
    )
    s["h4"] = ParagraphStyle(
        "H4",
        fontSize=11, leading=15, spaceBefore=8, spaceAfter=3,
        textColor=C_ORANGE, fontName="Helvetica-Bold",
    )

    s["body"] = ParagraphStyle(
        "Body",
        fontSize=10, leading=15, spaceAfter=6,
        textColor=colors.HexColor("#212121"), fontName="Helvetica",
        alignment=TA_JUSTIFY,
    )
    s["body_left"] = ParagraphStyle(
        "BodyLeft",
        fontSize=10, leading=15, spaceAfter=4,
        textColor=colors.HexColor("#212121"), fontName="Helvetica",
    )
    s["note"] = ParagraphStyle(
        "Note",
        fontSize=9.5, leading=14, spaceAfter=4,
        textColor=colors.HexColor("#37474F"), fontName="Helvetica-Oblique",
        leftIndent=10,
    )
    s["bullet"] = ParagraphStyle(
        "Bullet",
        fontSize=10, leading=15, spaceAfter=3,
        textColor=colors.HexColor("#212121"), fontName="Helvetica",
        leftIndent=16, bulletIndent=6,
    )
    s["code"] = ParagraphStyle(
        "Code",
        fontSize=8.5, leading=13, spaceAfter=0,
        textColor=colors.HexColor("#212121"), fontName="Courier",
        leftIndent=8,
    )
    s["code_comment"] = ParagraphStyle(
        "CodeComment",
        fontSize=8.5, leading=13, spaceAfter=0,
        textColor=colors.HexColor("#5D6D7E"), fontName="Courier-Oblique",
        leftIndent=8,
    )
    s["caption"] = ParagraphStyle(
        "Caption",
        fontSize=8.5, leading=12, spaceAfter=6, spaceBefore=2,
        textColor=colors.HexColor("#757575"), fontName="Helvetica-Oblique",
        alignment=TA_CENTER,
    )
    s["toc_h1"] = ParagraphStyle(
        "TOCH1",
        fontSize=11, leading=16, spaceAfter=2,
        textColor=C_DARK_BLUE, fontName="Helvetica-Bold",
    )
    s["toc_h2"] = ParagraphStyle(
        "TOCH2",
        fontSize=10, leading=14, spaceAfter=1, leftIndent=14,
        textColor=C_ACCENT, fontName="Helvetica",
    )
    return s

ST = make_styles()

# ── Helper flowables ──────────────────────────────────────────────────────────

def sp(h=0.3):
    return Spacer(1, h * cm)

def hr(color=C_ACCENT, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=4)

def h1(text):
    elems = [sp(0.3), hr(C_DARK_BLUE, 1.5), Paragraph(text, ST["h1"]), hr(C_ACCENT, 0.5)]
    return elems

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
    """Render a list of (line, is_comment) tuples as a shaded code box."""
    rows = []
    for line, is_cmt in lines:
        st = ST["code_comment"] if is_cmt else ST["code"]
        rows.append([Paragraph(line, st)])

    tbl = Table(rows, colWidths=[16.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_CODE_BG),
        ("BOX",        (0, 0), (-1, -1), 0.5, C_CODE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",(0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [C_CODE_BG]),
    ]))
    return tbl

def info_box(title, lines, bg=C_SECTION_BG, border=C_ACCENT):
    """Shaded information callout box."""
    content = [[Paragraph(f"<b>{title}</b>", ST["body_left"])]]
    for l in lines:
        content.append([Paragraph(f"  {l}", ST["body_left"])])
    tbl = Table(content, colWidths=[16.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX",        (0, 0), (-1, -1), 1.0, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("BACKGROUND", (0, 0), (-1, 0), border),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
    ]))
    return tbl

def data_table(headers, rows, col_widths=None):
    """Styled data table with alternating row colours."""
    data = [headers] + rows
    n_cols = len(headers)
    if col_widths is None:
        col_widths = [16.5 * cm / n_cols] * n_cols

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",   (0, 0), (-1, 0),  C_HEADER_BG),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]
    for i in range(1, len(data)):
        bg = C_ROW_EVEN if i % 2 == 0 else C_ROW_ODD
        style.append(("BACKGROUND", (0, i), (-1, i), bg))

    tbl.setStyle(TableStyle(style))
    return tbl

# ── Cover page ────────────────────────────────────────────────────────────────

def cover_page():
    elems = []
    elems.append(sp(3))

    # Dark banner
    banner_data = [[Paragraph("Olist E-Commerce RAG Pipeline", ST["title"])],
                   [Paragraph("End-to-End Knowledge Base Construction", ST["subtitle"])],
                   [Paragraph("From Raw CSV Data to ChromaDB-Ready Documents", ST["subtitle"])]]
    banner = Table(banner_data, colWidths=[17 * cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), C_DARK_BLUE),
        ("TOPPADDING",   (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 14),
        ("LEFTPADDING",  (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("BOX",          (0, 0), (-1, -1), 2, C_ACCENT),
    ]))
    elems.append(banner)
    elems.append(sp(1.2))

    date_str = datetime.datetime.now().strftime("%B %Y")
    meta_rows = [
        [Paragraph("<b>Project:</b>  E-Commerce RAG with RAGAS and DeepEval", ST["body_left"])],
        [Paragraph(f"<b>Date:</b>    {date_str}", ST["body_left"])],
        [Paragraph("<b>Dataset:</b>  Brazilian Olist E-Commerce (Kaggle)", ST["body_left"])],
        [Paragraph("<b>Purpose:</b>  Build a structured knowledge base for RAG evaluation", ST["body_left"])],
    ]
    meta_tbl = Table(meta_rows, colWidths=[17 * cm])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), C_LIGHT_BLUE),
        ("BOX",          (0, 0), (-1, -1), 1, C_ACCENT),
        ("LEFTPADDING",  (0, 0), (-1, -1), 16),
        ("TOPPADDING",   (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
    ]))
    elems.append(meta_tbl)
    elems.append(sp(1.5))

    # Summary stat boxes
    stats = [
        ("9", "Raw CSV\nFiles"),
        ("113,425", "Master\nRows"),
        ("55", "Enriched\nColumns"),
        ("13,225", "KB\nDocuments"),
    ]
    stat_cells = []
    for val, lbl in stats:
        cell = Table(
            [[Paragraph(f"<b>{val}</b>", ParagraphStyle("SV", fontSize=20,
              textColor=colors.white, alignment=TA_CENTER, fontName="Helvetica-Bold"))],
             [Paragraph(lbl, ParagraphStyle("SL", fontSize=8,
              textColor=colors.HexColor("#BBDEFB"), alignment=TA_CENTER,
              fontName="Helvetica", leading=11))]],
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
    elems.append(stats_row)
    elems.append(PageBreak())
    return elems

# ── Section builders ──────────────────────────────────────────────────────────

def section_overview():
    e = []
    e += h1("1. Project Overview")
    e.append(body(
        "This document explains how to build a structured knowledge base (KB) "
        "from the Brazilian Olist e-commerce dataset for use in a Retrieval-Augmented "
        "Generation (RAG) system. The pipeline reads nine raw CSV files, joins and "
        "enriches them into a master dataset, then produces six layers of human-readable "
        "JSON documents that can be loaded directly into ChromaDB."
    ))
    e.append(sp())

    e += h2("1.1 What is a RAG Knowledge Base?")
    e.append(body(
        "A knowledge base in a RAG system is a collection of text documents that the "
        "retriever searches to find relevant context before the LLM generates an answer. "
        "Each document should be:"
    ))
    for item in [
        "Self-contained — the text alone must answer a likely question.",
        "Human-readable — not raw rows, but a natural-language summary.",
        "Consistently structured — same fields in every document of the same type.",
        "Enriched — include derived metrics (e.g. late delivery rate) that raw data lacks.",
    ]:
        e.append(bullet(item))
    e.append(sp())

    e += h2("1.2 Why Not Use Raw CSV Rows as Documents?")
    e.append(body(
        "The master table has 113,425 rows at order-item grain (one row per item in each "
        "order). Using raw rows directly causes three problems:"
    ))
    tbl = data_table(
        ["Problem", "Impact", "Our Solution"],
        [
            ["One order split across\nmultiple rows", "Retriever sees incomplete facts", "Aggregate to order grain (1 doc per order)"],
            ["Numbers without context", "LLM cannot interpret\n'delivery_days: 8.44'", "Template-based natural language text"],
            ["100 k+ documents", "Retrieval is slow and noisy", "Summary layers reduce to 13,225 docs"],
        ],
        col_widths=[5 * cm, 5 * cm, 6.5 * cm],
    )
    e.append(tbl)
    e.append(sp())

    e += h2("1.3 Pipeline Architecture")
    e.append(body("The pipeline has five sequential steps:"))
    pipeline_data = [
        ["Step", "Module", "Input", "Output"],
        ["1. Load",    "loader.py",       "9 raw CSV files",               "Dict of DataFrames"],
        ["2. Join",    "joiner.py",       "Dict of DataFrames",            "final_olist_master.csv\n(113,425 rows × 40 cols)"],
        ["3. Enrich",  "enricher.py",     "Master CSV",                    "final_olist_master_enriched.csv\n(113,425 rows × 55 cols)"],
        ["4. KB Build","kb_builder.py",   "Enriched CSV",                  "6 × kb_*.json + kb_all_documents.json\n(13,225 documents total)"],
        ["5. Golden",  "golden_dataset.py","Enriched CSV + KB docs",       "golden_dataset.csv\n(~50 Q&A pairs)"],
    ]
    tbl2 = Table(pipeline_data, colWidths=[2.5*cm, 3.5*cm, 4.5*cm, 5.5*cm], repeatRows=1)
    tbl2.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  C_HEADER_BG),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#BBBBBB")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS",(0, 1),(-1, -1), [C_ROW_ODD, C_ROW_EVEN]),
    ]))
    e.append(tbl2)
    return e


def section_dataset():
    e = []
    e += h1("2. Raw Dataset Description")
    e.append(body(
        "The Olist dataset is a publicly available Brazilian e-commerce dataset "
        "from Kaggle. It contains real commercial orders from 2016 to 2018. "
        "Nine CSV files are loaded and joined to form one master table."
    ))
    e.append(sp(0.4))

    e += h2("2.1 File Inventory")
    rows = [
        ["olist_orders_dataset.csv",           "99,441",  "order_id, customer_id, order_status,\npurchase/approval/delivery timestamps"],
        ["olist_order_items_dataset.csv",       "112,650", "order_id, product_id, seller_id,\nprice, freight_value"],
        ["olist_order_payments_dataset.csv",    "103,886", "order_id, payment_type,\npayment_installments, payment_value"],
        ["olist_order_reviews_dataset.csv",     "99,224",  "order_id, review_score,\nreview_comment_title, review_comment_message"],
        ["olist_customers_dataset.csv",         "99,441",  "customer_id, customer_city,\ncustomer_state"],
        ["olist_products_dataset.csv",          "32,951",  "product_id, product_category_name,\nweight, dimensions"],
        ["olist_sellers_dataset.csv",           "3,095",   "seller_id, seller_city, seller_state"],
        ["olist_geolocation_dataset.csv",       "1,000,163","zip_code, lat, lng, city, state\n(not used in joins)"],
        ["product_category_name_translation.csv","71",     "product_category_name (Portuguese),\nproduct_category_name_english"],
    ]
    e.append(data_table(
        ["File", "Rows", "Key Columns"],
        rows,
        col_widths=[6.5 * cm, 2 * cm, 8 * cm],
    ))
    e.append(sp())

    e += h2("2.2 Entity Relationship")
    e.append(body("The files share these foreign-key relationships:"))
    er_lines = [
        ("orders.order_id          -->  order_items.order_id", False),
        ("orders.order_id          -->  payments.order_id", False),
        ("orders.order_id          -->  reviews.order_id", False),
        ("orders.customer_id       -->  customers.customer_id", False),
        ("order_items.product_id   -->  products.product_id", False),
        ("order_items.seller_id    -->  sellers.seller_id", False),
        ("products.product_category_name  -->  category_translation.product_category_name", False),
    ]
    e.append(code_block(er_lines))
    e.append(sp())

    e += h2("2.3 Important Data Notes")
    notes = [
        ("BOM character", "product_category_name_translation.csv contains a UTF-8 BOM. Read with encoding='utf-8-sig'."),
        ("Multi-item orders", "112,650 item rows but only 99,441 unique orders. Always use nunique(order_id) for order counts."),
        ("Multi-payment rows", "Some orders have credit_card + voucher. Aggregate payments before joining."),
        ("Duplicate reviews", "Some orders have multiple review submissions. Keep only the latest (sort desc by review_answer_timestamp)."),
        ("Missing delivery dates", "~2,965 orders have no delivery date (canceled or unavailable). These get delivery_status='not_delivered'."),
    ]
    for title, desc in notes:
        e.append(KeepTogether([
            Paragraph(f"<b>{title}:</b> {desc}", ST["body_left"]),
            sp(0.15),
        ]))
    return e


def section_step1_load():
    e = []
    e += h1("3. Step 1 — Loading Raw Data (loader.py)")
    e.append(body(
        "All nine CSV files are loaded using pandas.read_csv(). "
        "Date columns are parsed immediately. The UTF-8 BOM is stripped "
        "from the category translation file by using encoding='utf-8-sig' for every file."
    ))
    e.append(sp(0.3))

    e += h2("3.1 Key Design Decisions")
    for item in [
        "encoding='utf-8-sig' — strips the BOM that appears in the category translation file.",
        "low_memory=False — prevents dtype inference warnings on large files.",
        "pd.to_datetime(errors='coerce') — silently converts unparseable dates to NaT instead of raising.",
        "All files share a single registry (RAW_FILES dict in config.py) for easy path management.",
    ]:
        e.append(bullet(item))
    e.append(sp(0.3))

    e += h2("3.2 Code")
    e.append(code_block([
        ("# pipeline/loader.py  (simplified)", True),
        ("import pandas as pd", False),
        ("from .config import RAW_FILES", False),
        ("", False),
        ("DATETIME_COLS = {", False),
        ("    'orders': ['order_purchase_timestamp', 'order_approved_at',", False),
        ("               'order_delivered_carrier_date',", False),
        ("               'order_delivered_customer_date',", False),
        ("               'order_estimated_delivery_date'],", False),
        ("    'reviews': ['review_creation_date', 'review_answer_timestamp'],", False),
        ("}", False),
        ("", False),
        ("def load_dataset(name: str) -> pd.DataFrame:", False),
        ("    df = pd.read_csv(RAW_FILES[name], encoding='utf-8-sig',", False),
        ("                     low_memory=False)", False),
        ("    for col in DATETIME_COLS.get(name, []):", False),
        ("        df[col] = pd.to_datetime(df[col], errors='coerce')", False),
        ("    return df", False),
    ]))
    e.append(sp(0.3))

    e += h2("3.3 Output")
    e.append(data_table(
        ["Variable", "Type", "Contents"],
        [
            ["datasets['orders']",      "DataFrame", "99,441 rows, 8 datetime-parsed columns"],
            ["datasets['order_items']", "DataFrame", "112,650 rows, price and freight as float"],
            ["datasets['payments']",    "DataFrame", "103,886 rows — raw, pre-aggregation"],
            ["datasets['reviews']",     "DataFrame", "99,224 rows — raw, pre-aggregation"],
            ["datasets['customers']",   "DataFrame", "99,441 rows"],
            ["datasets['products']",    "DataFrame", "32,951 rows"],
            ["datasets['sellers']",     "DataFrame", "3,095 rows"],
            ["datasets['category_translation']","DataFrame","71 rows, BOM stripped"],
        ],
        col_widths=[5 * cm, 3 * cm, 8 * cm],
    ))
    return e


def section_step2_join():
    e = []
    e += h1("4. Step 2 — Joining Datasets (joiner.py)")
    e.append(body(
        "Before joining, payments and reviews are aggregated to one row per order_id. "
        "This prevents duplicate rows when an order has multiple payment methods or "
        "re-submitted reviews. The join chain starts with orders as the base and "
        "progressively adds every dimension table using LEFT JOIN semantics so that "
        "orders without products or sellers are kept."
    ))
    e.append(sp(0.3))

    e += h2("4.1 Payment Aggregation")
    e.append(body(
        "An order may use credit card + gift voucher together. "
        "Both rows exist in the payments table. We collapse them to one row:"
    ))
    e.append(code_block([
        ("payments_agg = payments.groupby('order_id').agg(", False),
        ("    total_payment_value   = ('payment_value',       'sum'),", False),
        ("    # 'credit_card|voucher' — sorted and pipe-joined", True),
        ("    payment_types         = ('payment_type',        lambda x: '|'.join(sorted(set(x)))),", False),
        ("    max_installments      = ('payment_installments','max'),", False),
        ("    payment_methods_count = ('payment_type',        'nunique'),", False),
        (").reset_index()", False),
    ]))
    e.append(sp(0.3))

    e += h2("4.2 Review Aggregation")
    e.append(body(
        "When a customer re-reviews an order, multiple rows exist. "
        "We keep only the latest review (sorted descending by review_answer_timestamp):"
    ))
    e.append(code_block([
        ("reviews_agg = (", False),
        ("    reviews", False),
        ("    .sort_values('review_answer_timestamp', ascending=False)", False),
        ("    .groupby('order_id')", False),
        ("    .first()   # picks the latest review per order", True),
        ("    .reset_index()", False),
        (")", False),
    ]))
    e.append(sp(0.3))

    e += h2("4.3 Full Join Chain")
    e.append(code_block([
        ("df = orders                                         # base: 99,441 rows", False),
        ("df = df.merge(order_items,    on='order_id',    how='left')  # -> 113,425 rows", False),
        ("df = df.merge(payments_agg,   on='order_id',    how='left')", False),
        ("df = df.merge(reviews_agg,    on='order_id',    how='left')", False),
        ("df = df.merge(customers,      on='customer_id', how='left')", False),
        ("df = df.merge(products,       on='product_id',  how='left')", False),
        ("df = df.merge(sellers,        on='seller_id',   how='left')", False),
        ("df = df.merge(category_translation,", False),
        ("               on='product_category_name', how='left')", False),
        ("# Result: 113,425 rows x 40 columns", True),
    ]))
    e.append(sp(0.3))

    e += h2("4.4 Why the Row Count Increases")
    e.append(info_box(
        "Row Count Explanation",
        [
            "orders has 99,441 unique orders.",
            "order_items has 112,650 item rows (some orders have 2-3 items).",
            "After LEFT JOIN on order_id, each item gets its own row.",
            "Result: 113,425 rows = one row per (order, item).",
            "Order-level fields (delivery dates, payment, review) repeat on each item row.",
            "This is the correct 'denormalised master' grain for analysis.",
        ],
    ))
    e.append(sp(0.3))

    e += h2("4.5 Output: final_olist_master.csv")
    e.append(data_table(
        ["Property", "Value"],
        [
            ["Rows",    "113,425"],
            ["Columns", "40"],
            ["Grain",   "One row per (order_id × order_item_id)"],
            ["File size","~59 MB"],
        ],
        col_widths=[5 * cm, 11.5 * cm],
    ))
    return e


def section_step3_enrich():
    e = []
    e += h1("5. Step 3 — Enriching the Master Dataset (enricher.py)")
    e.append(body(
        "Enrichment adds 15 derived columns that make the master dataset analysis-ready. "
        "Raw timestamps are decomposed into calendar parts; time differences between "
        "delivery milestones are computed; delivery outcomes are bucketed into human-readable "
        "categories; and review scores are mapped to sentiment labels."
    ))
    e.append(sp(0.3))

    e += h2("5.1 All 15 Enriched Columns")
    rows = [
        ["purchase_month",          "int",     "Calendar month number (1–12) from order_purchase_timestamp"],
        ["purchase_month_name",     "str",     "Month name, e.g. 'October'"],
        ["purchase_year",           "int",     "Calendar year, e.g. 2017"],
        ["purchase_day_name",       "str",     "Day of week, e.g. 'Monday'"],
        ["purchase_hour",           "int",     "Hour of day (0–23)"],
        ["approval_hours",          "float",   "Hours from purchase to order approval"],
        ["carrier_handover_days",   "float",   "Days from approval to carrier pickup"],
        ["delivery_days",           "float",   "Days from purchase to customer delivery"],
        ["estimated_delivery_days", "float",   "Days from purchase to estimated delivery date"],
        ["delivery_difference_days","int",     "actual − estimated (positive = late, negative = early)"],
        ["delivery_status",         "str",     "'early' | 'on_time' | 'late' | 'not_delivered'"],
        ["delivery_bucket",         "str",     "8-band bucket: very_early → very_late"],
        ["product_category_final",  "str",     "English category name (fallback to Portuguese if missing)"],
        ["item_total_value",        "float",   "price + freight_value per item"],
        ["review_bucket",           "str",     "'positive'(≥4) | 'neutral'(3) | 'negative'(≤2) | 'no_review'"],
    ]
    e.append(data_table(
        ["Column", "Dtype", "Description"],
        rows,
        col_widths=[4.5 * cm, 1.5 * cm, 10.5 * cm],
    ))
    e.append(sp(0.3))

    e += h2("5.2 Delivery Status Logic")
    e.append(body(
        "The delivery_difference_days column is the integer number of days between "
        "the actual delivery date and the estimated delivery date. "
        "Positive means the order arrived late; negative means early."
    ))
    e.append(code_block([
        ("# Compute integer-day difference", True),
        ("df['delivery_difference_days'] = (", False),
        ("    df['order_delivered_customer_date']", False),
        ("    - df['order_estimated_delivery_date']", False),
        (").dt.days   # positive = late, negative = early", True),
        ("", False),
        ("# Assign status using np.select (conditions evaluated in order)", True),
        ("has_delivery = df['order_delivered_customer_date'].notna()", False),
        ("d = df['delivery_difference_days']", False),
        ("", False),
        ("conditions = [~has_delivery, has_delivery & (d > 0),", False),
        ("              has_delivery & (d < 0), has_delivery & (d == 0)]", False),
        ("choices    = ['not_delivered', 'late', 'early', 'on_time']", False),
        ("df['delivery_status'] = np.select(conditions, choices)", False),
    ]))
    e.append(sp(0.3))

    e += h2("5.3 Delivery Bucket Bands")
    e.append(data_table(
        ["delivery_bucket", "delivery_difference_days range", "Meaning"],
        [
            ["very_early",    "< −14 days",    "Delivered 2+ weeks before estimate"],
            ["early",         "−14 to −8 days","Delivered 1–2 weeks early"],
            ["slightly_early","−7 to −1 days", "Delivered within a week early"],
            ["on_time",       "= 0 days",       "Delivered exactly on estimated date"],
            ["slightly_late", "+1 to +7 days",  "Delivered up to a week late"],
            ["late",          "+8 to +14 days", "Delivered 1–2 weeks late"],
            ["very_late",     "> +14 days",     "Delivered more than 2 weeks late"],
            ["not_delivered", "N/A",            "No delivery date recorded"],
        ],
        col_widths=[3.5 * cm, 4.5 * cm, 8.5 * cm],
    ))
    e.append(sp(0.3))

    e += h2("5.4 Dataset Statistics After Enrichment")
    e.append(data_table(
        ["Metric", "Value"],
        [
            ["Total rows",               "113,425"],
            ["Total columns",            "55 (40 original + 15 enriched)"],
            ["Unique orders",            "99,441"],
            ["Orders delivered early",   "88,649 (89.1%)"],
            ["Orders delivered late",    "6,535 (6.6%)"],
            ["Orders not delivered",     "2,965 (3.0%)"],
            ["Positive reviews",         "76,046 (76.5%)"],
            ["Distinct product categories", "74"],
            ["File size",                "~70 MB"],
        ],
        col_widths=[7 * cm, 9.5 * cm],
    ))
    return e


def section_step4_kb():
    e = []
    e += h1("6. Step 4 — Building the Knowledge Base (kb_builder.py)")
    e.append(body(
        "The knowledge base is built from final_olist_master_enriched.csv. "
        "Six document layers are created, each targeting a different analytical "
        "grain. All documents share the same JSON schema. "
        "The final kb_all_documents.json file (13,225 documents) is loaded "
        "into ChromaDB for retrieval."
    ))
    e.append(sp(0.3))

    e += h2("6.1 Document Schema")
    e.append(body("Every document, regardless of layer, follows this JSON structure:"))
    e.append(code_block([
        ("{", False),
        ('  "id":    "<layer>_<source_id>",            // unique document ID', True),
        ('  "text":  "Document Type: ...\\nKey: Value\\n...", // embeddable text', True),
        ('  "metadata": {', False),
        ('    "document_type": "order_level",          // layer identifier', True),
        ('    "source_id":     "<order_id>",           // raw key for tracing', True),
        ('    "order_id":      "<order_id>",           // filterable field', True),
        ('    "delivery_status": "early",              // filterable field', True),
        ('    "purchase_month":  "2017-10",            // filterable field', True),
        ('    ...                                      // other layer fields', True),
        ('  }', False),
        ('}', False),
    ]))
    e.append(sp(0.3))

    e += h2("6.2 Why Structured Text (Not Pipe-Separated)?")
    e.append(body(
        "The text field uses 'Key: Value' lines with blank lines between sections "
        "rather than a single pipe-separated string. This gives two advantages:"
    ))
    for item in [
        "Human readable — anyone can understand the document without decoding it.",
        "Chunk-friendly — each section is a natural semantic unit for text splitters.",
        "LLM-friendly — the LLM can read 'Delivery Status: early' exactly like a structured fact.",
        "Grep-friendly — engineers can search 'Late Delivery Rate' across all documents easily.",
    ]:
        e.append(bullet(item))
    e.append(sp(0.3))

    e += h2("6.3 Critical Aggregation Rule: Item-Level vs Order-Level")
    e.append(body(
        "The master dataset is at order-item grain (one row per item). "
        "Using it directly for order-level statistics like delivery rate or "
        "review score would double- or triple-count orders that contain multiple items. "
        "The fix is to deduplicate by (entity, order_id) before aggregating:"
    ))
    e.append(code_block([
        ("# WRONG — counts each item row, not each order", True),
        ("late_count = df[df['delivery_status']=='late'].shape[0]", False),
        ("", False),
        ("# CORRECT — count unique late orders", True),
        ("late_count = df[df['delivery_status']=='late']['order_id'].nunique()", False),
        ("", False),
        ("# CORRECT — for per-seller stats, deduplicate to (seller, order) grain", True),
        ("sell_ord = df.groupby(['seller_id','order_id']).agg(", False),
        ("    review_score    = ('review_score',    'first'),", False),
        ("    delivery_status = ('delivery_status', 'first'),", False),
        (").reset_index()", False),
        ("avg_review_per_seller = sell_ord.groupby('seller_id')['review_score'].mean()", False),
    ]))
    return e


def section_kb_layers():
    e = []
    e += h1("7. Knowledge Base Layers — Detail and Examples")

    # ── Layer 1 ─────────────────────────────────────────────────────────────
    e += h2("7.1  Layer 1 — Order-Level Documents  (10,000 sampled)")
    e.append(body(
        "One document per order_id. Because 99,441 order documents would make "
        "ChromaDB ingestion slow during initial prototyping, we sample 10,000 orders "
        "using a fixed random seed (seed=42) for reproducibility. "
        "All aggregations are done at order grain first."
    ))
    e.append(sp(0.2))
    e.append(body("<b>Aggregations per order:</b>"))
    agg_rows = [
        ["total_items",         "count(order_item_id)",                "Number of items in the order"],
        ["unique_products",     "nunique(product_id)",                  "Number of distinct products"],
        ["unique_sellers",      "nunique(seller_id)",                   "Number of distinct sellers"],
        ["product_categories",  "sorted unique values joined by '|'",  "All categories in the order"],
        ["seller_states",       "sorted unique values joined by '|'",  "States of all sellers"],
        ["total_price",         "sum(price)",                           "Total item price (excl. freight)"],
        ["total_freight",       "sum(freight_value)",                   "Total freight cost"],
        ["total_item_value",    "total_price + total_freight",          "Combined item + freight total"],
        ["payment_types",       "first (from payment aggregation)",     "e.g. 'credit_card|voucher'"],
        ["total_payment_value", "first (from payment aggregation)",     "What the customer actually paid"],
        ["max_installments",    "first (from payment aggregation)",     "Largest instalment count used"],
        ["review_score",        "first (latest review)",                "1–5 star score"],
        ["delivery_days",       "first",                                "Days from purchase to delivery"],
        ["delivery_difference_days","first",                            "Actual − estimated (days)"],
    ]
    e.append(data_table(
        ["Field", "Aggregation", "Meaning"],
        agg_rows,
        col_widths=[4 * cm, 5 * cm, 7.5 * cm],
    ))
    e.append(sp(0.3))
    e.append(body("<b>Example document:</b>"))
    e.append(code_block([
        ("Document Type: Order-Level Summary", False),
        ("Order ID: 8704f37bae751578cdec362f48c61232", False),
        ("Customer ID: 262d7e8f0907b9d31c145b04e46d6a5e", False),
        ("Customer City: aruja", False),
        ("Customer State: SP", False),
        ("", False),
        ("Order Status: delivered", False),
        ("Purchase Month: December 2017", False),
        ("Purchase Timestamp: 2017-12-13 21:19:54", False),
        ("Approved At: 2017-12-14 07:12:37", False),
        ("Delivered to Carrier Date: 2017-12-14 23:42:43", False),
        ("Delivered to Customer Date: 2017-12-22 18:24:57", False),
        ("Estimated Delivery Date: 2018-01-08 00:00:00", False),
        ("", False),
        ("Delivery Days: 8.88", False),
        ("Estimated Delivery Days: 25.11", False),
        ("Delivery Difference Days: -16.23", False),
        ("Delivery Status: early", False),
        ("Delivery Bucket: very_early", False),
        ("", False),
        ("Total Items: 1", False),
        ("Product Categories: fashion_underwear_beach", False),
        ("Seller States: SP", False),
        ("", False),
        ("Total Product Price: 59.90", False),
        ("Total Freight Value: 11.92", False),
        ("Total Item Value: 71.82", False),
        ("Payment Types: credit_card", False),
        ("Total Payment Value: 71.82", False),
        ("Maximum Payment Installments: 2", False),
        ("", False),
        ("Review Score: 4", False),
        ("Review Bucket: positive", False),
        ("Review Comment Title: No comment title", False),
        ("Review Comment Message: No comment message", False),
        ("", False),
        ("This document summarises one customer order...", False),
    ]))
    e.append(sp(0.3))

    # ── Layer 2 ─────────────────────────────────────────────────────────────
    e += h2("7.2  Layer 2 — Product Category Documents  (74 documents)")
    e.append(body(
        "One document per product_category_final (English name). "
        "Answers questions like 'Which category has the highest revenue?' "
        "or 'What is the late delivery rate for health_beauty?'"
    ))
    e.append(sp(0.2))
    e.append(code_block([
        ("Document Type: Product Category Summary", False),
        ("Product Category: health_beauty", False),
        ("", False),
        ("Total Orders: 9,670", False),
        ("Total Items Sold: 9,922", False),
        ("Total Product Revenue: 1,147,245.30", False),
        ("Total Freight Value: 176,411.55", False),
        ("Average Product Price: 115.63", False),
        ("Average Freight Value: 17.78", False),
        ("Average Payment Value: 142.87", False),
        ("", False),
        ("Average Review Score: 4.17", False),
        ("Positive Review Count: 7,538", False),
        ("Negative Review Count: 910", False),
        ("", False),
        ("Late Delivery Count: 419", False),
        ("Early or On-Time Delivery Count: 8,776", False),
        ("Late Delivery Rate: 4.56%", False),
        ("Average Delivery Days: 11.19", False),
        ("", False),
        ("Top Customer State: SP", False),
        ("Top Seller State: SP", False),
        ("", False),
        ("This document summarises sales, delivery, freight, and review", False),
        ("performance for the health_beauty product category.", False),
    ]))
    e.append(sp(0.3))

    # ── Layer 3 ─────────────────────────────────────────────────────────────
    e += h2("7.3  Layer 3 — Seller Documents  (3,095 documents)")
    e.append(body(
        "One document per seller_id. Enables questions like "
        "'Which seller in SP state has the most late deliveries?' or "
        "'What is the average review score for seller X?'"
    ))
    e.append(sp(0.2))
    e.append(code_block([
        ("Document Type: Seller-Level Summary", False),
        ("Seller ID: 4a3ca9315b744ce9f8e9374361493884", False),
        ("Seller City: sao paulo", False),
        ("Seller State: SP", False),
        ("", False),
        ("Total Orders: 831", False),
        ("Total Items Sold: 1,008", False),
        ("Total Product Revenue: 154,320.45", False),
        ("Total Freight Value: 13,988.20", False),
        ("Average Freight Value: 13.88", False),
        ("", False),
        ("Average Review Score: 3.92", False),
        ("Positive Review Count: 568", False),
        ("Negative Review Count: 148", False),
        ("", False),
        ("Late Delivery Count: 47", False),
        ("Late Delivery Rate: 5.66%", False),
        ("Average Delivery Days: 9.48", False),
        ("", False),
        ("Top Product Category: bed_bath_table", False),
        ("Top Customer State: SP", False),
        ("", False),
        ("This document summarises seller fulfilment, revenue, freight,", False),
        ("delivery, and review performance.", False),
    ]))
    e.append(sp(0.3))

    # ── Layer 4 ─────────────────────────────────────────────────────────────
    e += h2("7.4  Layer 4 — Customer-State Documents  (27 documents)")
    e.append(body(
        "One document per Brazilian state. Only 27 documents but very "
        "useful for geographic comparisons: 'Which state has the most orders?' "
        "or 'What is the average delivery time for customers in RJ?'"
    ))
    e.append(sp(0.2))
    e.append(code_block([
        ("Document Type: Customer-State Summary", False),
        ("Customer State: SP", False),
        ("", False),
        ("Total Orders: 41,746", False),
        ("Unique Customers: 40,291", False),
        ("Total Payment Value: 6,504,313.73", False),
        ("Average Payment Value: 155.83", False),
        ("", False),
        ("Average Review Score: 4.12", False),
        ("Positive Review Count: 32,051", False),
        ("Negative Review Count: 5,407", False),
        ("", False),
        ("Late Delivery Count: 2,413", False),
        ("Late Delivery Rate: 5.81%", False),
        ("Average Delivery Days: 9.49", False),
        ("", False),
        ("Top Product Category: bed_bath_table", False),
        ("Top Payment Type: credit_card", False),
        ("", False),
        ("This document summarises order volume, payment value, delivery", False),
        ("performance, and review behaviour for customers in SP.", False),
    ]))
    e.append(sp(0.3))

    # ── Layer 5 ─────────────────────────────────────────────────────────────
    e += h2("7.5  Layer 5 — Month-Level Temporal Documents  (25 documents)")
    e.append(body(
        "One document per calendar month in the dataset (September 2016 to "
        "September 2018). Temporal questions: 'Which month had the highest revenue?' "
        "or 'How did late delivery rate change over 2018?'"
    ))
    e.append(sp(0.2))
    e.append(code_block([
        ("Document Type: Month-Level Temporal Summary", False),
        ("Purchase Month: 2017-11", False),
        ("Month Name: November 2017", False),
        ("", False),
        ("Total Orders: 7,544", False),
        ("Delivered Orders: 6,924", False),
        ("Canceled Orders: 391", False),
        ("", False),
        ("Late Delivery Count: 464", False),
        ("Early or On-Time Delivery Count: 6,246", False),
        ("Late Delivery Rate: 6.70%", False),
        ("Average Delivery Days: 11.62", False),
        ("", False),
        ("Total Payment Value: 1,118,342.50", False),
        ("Average Payment Value: 148.23", False),
        ("Average Freight Value: 20.37", False),
        ("", False),
        ("Average Review Score: 4.08", False),
        ("Negative Review Count: 618", False),
        ("", False),
        ("Most Popular Product Category: bed_bath_table", False),
        ("Top Customer State: SP", False),
        ("Top Seller State: SP", False),
        ("", False),
        ("This document summarises monthly e-commerce fulfilment,", False),
        ("revenue, delivery, and review performance.", False),
    ]))
    e.append(sp(0.3))

    # ── Layer 6 ─────────────────────────────────────────────────────────────
    e += h2("7.6  Layer 6 — Delivery-Status Insight Documents  (4 documents)")
    e.append(body(
        "One document per delivery_status value: early, on_time, late, not_delivered. "
        "These high-level insight documents answer 'What are the characteristics of late orders?' "
        "and show how review scores differ by delivery outcome."
    ))
    e.append(sp(0.2))
    e.append(code_block([
        ("Document Type: Delivery-Status Insight", False),
        ("Delivery Status: late", False),
        ("", False),
        ("Total Orders in Group: 6,535", False),
        ("Share of Delivered Orders: 6.77%", False),
        ("Average Delivery Days: 26.04", False),
        ("Average Difference vs Estimated: 9.16 days late", False),
        ("", False),
        ("Average Review Score: 2.73", False),
        ("Average Payment Value: 163.04", False),
        ("", False),
        ("Top Product Category: bed_bath_table", False),
        ("Top Customer State: SP", False),
        ("Top Seller State: SP", False),
        ("", False),
        ("This document provides aggregate insight into orders with", False),
        ("delivery status 'late'.", False),
    ]))
    e.append(sp(0.3))

    # ── Summary table ────────────────────────────────────────────────────────
    e += h2("7.7  Knowledge Base Summary")
    e.append(data_table(
        ["Layer", "File", "Docs", "Grain", "Use Case"],
        [
            ["1 Order",    "kb_order_documents.json",         "10,000","order_id",               "Specific order lookup"],
            ["2 Category", "kb_category_documents.json",      "74",    "product_category_final", "Category analytics"],
            ["3 Seller",   "kb_seller_documents.json",        "3,095", "seller_id",              "Seller performance"],
            ["4 State",    "kb_customer_state_documents.json","27",    "customer_state",          "Geographic analysis"],
            ["5 Month",    "kb_month_documents.json",         "25",    "YYYY-MM",                "Temporal trends"],
            ["6 Delivery", "kb_delivery_status_documents.json","4",   "delivery_status",         "Outcome comparison"],
            ["ALL",        "kb_all_documents.json",           "13,225","—",                      "ChromaDB ingestion"],
        ],
        col_widths=[2 * cm, 5.5 * cm, 1.5 * cm, 3.5 * cm, 4 * cm],
    ))
    return e


def section_how_to_run():
    e = []
    e += h1("8. How to Run the Pipeline")

    e += h2("8.1 Prerequisites")
    e.append(code_block([
        ("# Python 3.10+", True),
        ("pip install pandas numpy", False),
        ("", False),
        ("# Place raw CSV files in:", True),
        ("dataset/raw/olist_customers_dataset.csv", False),
        ("dataset/raw/olist_order_items_dataset.csv", False),
        ("dataset/raw/olist_order_payments_dataset.csv", False),
        ("dataset/raw/olist_order_reviews_dataset.csv", False),
        ("dataset/raw/olist_orders_dataset.csv", False),
        ("dataset/raw/olist_products_dataset.csv", False),
        ("dataset/raw/olist_sellers_dataset.csv", False),
        ("dataset/raw/olist_geolocation_dataset.csv", False),
        ("dataset/raw/product_category_name_translation.csv", False),
    ]))
    e.append(sp(0.3))

    e += h2("8.2 Run Commands")
    e.append(data_table(
        ["Command", "What it does", "Time"],
        [
            ["python main.py --steps enrich",
             "Load CSVs → join → enrich\nProduces: dataset/processed/\nfinal_olist_master.csv\nfinal_olist_master_enriched.csv",
             "~2 min"],
            ["python main.py --steps kb",
             "Load enriched CSV → build 6 KB layers\nProduces: dataset/kb/kb_*.json\nand kb_all_documents.json",
             "~35 sec"],
            ["python main.py --steps golden",
             "Load enriched CSV + KB → build golden Q&A\nProduces: dataset/golden/\ngolden_dataset.csv",
             "~10 sec"],
            ["python main.py",
             "Full pipeline (all steps)\nProduces all outputs above",
             "~3 min"],
        ],
        col_widths=[6 * cm, 7 * cm, 2 * cm],
    ))
    e.append(sp(0.3))

    e += h2("8.3 Project File Structure")
    e.append(code_block([
        ("project/", False),
        ("|-- main.py                        # pipeline entry point", False),
        ("|-- requirements.txt               # pandas, numpy", False),
        ("|-- pipeline/", False),
        ("|   |-- __init__.py", False),
        ("|   |-- config.py                  # paths and constants", False),
        ("|   |-- loader.py                  # Step 1: load CSVs", False),
        ("|   |-- joiner.py                  # Step 2: aggregate + join", False),
        ("|   |-- enricher.py                # Step 3: derived columns", False),
        ("|   |-- kb_builder.py              # Step 4: KB documents", False),
        ("|   |-- golden_dataset.py          # Step 5: evaluation Q&A", False),
        ("|-- dataset/", False),
        ("|   |-- raw/                       # 9 input CSV files (unchanged)", False),
        ("|   |-- processed/                 # master.csv + enriched.csv", False),
        ("|   |-- kb/                        # 7 KB JSON files", False),
        ("|   |-- golden/                    # golden_dataset.csv", False),
        ("|-- docs/                          # this PDF", False),
    ]))
    e.append(sp(0.3))

    e += h2("8.4 Key Configuration (pipeline/config.py)")
    e.append(data_table(
        ["Constant", "Default", "Effect"],
        [
            ["ORDER_KB_SAMPLE",    "10,000", "Orders sampled into Layer 1. Set to None for all 99,441."],
            ["TOP_SELLERS_FOR_KB", "None",   "Cap seller documents. None = all 3,095 sellers."],
            ["RANDOM_SEED",        "42",     "Ensures the same 10,000 orders are sampled every run."],
        ],
        col_widths=[4.5 * cm, 2.5 * cm, 9.5 * cm],
    ))
    return e


def section_chromadb():
    e = []
    e += h1("9. Loading the KB into ChromaDB")
    e.append(body(
        "The kb_all_documents.json file is ready for direct ingestion into ChromaDB. "
        "Each document's 'text' field is embedded; each document's 'metadata' dict "
        "is stored as ChromaDB metadata for filtering."
    ))
    e.append(sp(0.3))

    e += h2("9.1 Ingestion Code")
    e.append(code_block([
        ("import chromadb, json", False),
        ("from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction", False),
        ("", False),
        ("# Load documents", True),
        ("with open('dataset/kb/kb_all_documents.json', encoding='utf-8') as f:", False),
        ("    docs = json.load(f)", False),
        ("", False),
        ("# Create collection", True),
        ("client     = chromadb.PersistentClient(path='./chroma_store')", False),
        ("embed_fn   = SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2')", False),
        ("collection = client.get_or_create_collection('olist_kb',", False),
        ("                embedding_function=embed_fn)", False),
        ("", False),
        ("# Batch upsert (ChromaDB max batch = 5,461)", True),
        ("BATCH = 500", False),
        ("for i in range(0, len(docs), BATCH):", False),
        ("    batch = docs[i:i+BATCH]", False),
        ("    collection.upsert(", False),
        ("        ids        = [d['id']       for d in batch],", False),
        ("        documents  = [d['text']     for d in batch],", False),
        ("        metadatas  = [d['metadata'] for d in batch],", False),
        ("    )", False),
        ("print(f'Upserted {collection.count()} documents')", False),
    ]))
    e.append(sp(0.3))

    e += h2("9.2 Metadata Filtering Examples")
    e.append(body(
        "Because each document has structured metadata, ChromaDB can filter before "
        "embedding search — making retrieval faster and more precise:"
    ))
    e.append(code_block([
        ("# Retrieve only late-delivery orders from SP state", True),
        ("results = collection.query(", False),
        ("    query_texts = ['Which orders were delivered late?'],", False),
        ("    where       = {'$and': [{'delivery_status': 'late'},", False),
        ("                            {'customer_state':  'SP'}]},", False),
        ("    n_results   = 5,", False),
        (")", False),
        ("", False),
        ("# Retrieve only category-level documents", True),
        ("results = collection.query(", False),
        ("    query_texts = ['highest revenue category'],", False),
        ("    where       = {'document_type': 'category_level'},", False),
        ("    n_results   = 3,", False),
        (")", False),
    ]))
    e.append(sp(0.3))

    e += h2("9.3 Available Metadata Fields per Layer")
    e.append(data_table(
        ["Layer", "Key Metadata Fields"],
        [
            ["Order",          "document_type, order_id, order_status, customer_state,\ndelivery_status, delivery_bucket, purchase_month, review_bucket"],
            ["Category",       "document_type, source_id (category name),\ntotal_orders, avg_review, late_rate_pct, total_revenue"],
            ["Seller",         "document_type, source_id (seller_id), seller_city, seller_state,\ntotal_orders, avg_review, total_revenue, late_rate_pct"],
            ["Customer State", "document_type, source_id (state code),\ntotal_orders, order_pct, avg_review, late_rate_pct, avg_delivery_days"],
            ["Month",          "document_type, source_id ('YYYY-MM'), year, month,\nmonth_name, total_orders, late_rate_pct, avg_review, total_payment"],
            ["Delivery Status","document_type, source_id (status string),\ntotal_orders, avg_review, avg_delivery_days"],
        ],
        col_widths=[3 * cm, 13.5 * cm],
    ))
    return e


def section_golden():
    e = []
    e += h1("10. Golden Dataset for RAG Evaluation")
    e.append(body(
        "The golden dataset is a set of ~50 question-answer pairs computed entirely "
        "from the enriched dataset using pandas — never LLM. This ensures answers "
        "are deterministic and correct, which is required for RAGAS and DeepEval metrics."
    ))
    e.append(sp(0.3))

    e += h2("10.1 Golden Dataset Schema")
    e.append(data_table(
        ["Column", "Type", "Description"],
        [
            ["question_id",       "str", "Unique ID, e.g. 'q001'"],
            ["question",          "str", "The natural-language question"],
            ["expected_answer",   "str", "Answer computed by pandas from enriched data"],
            ["expected_context",  "str", "Full text of the KB document(s) that contain the answer"],
            ["expected_source_ids","str", "JSON array of KB document IDs, e.g. '[\"order_abc\"]'"],
            ["question_type",     "str", "'factual' | 'analytical' | 'comparative' | 'aggregation'"],
            ["difficulty",        "str", "'easy' | 'medium' | 'hard'"],
            ["best_kb_layer",     "str", "'order' | 'category' | 'seller' | 'state' | 'month' | 'delivery_status'"],
        ],
        col_widths=[4 * cm, 2 * cm, 10.5 * cm],
    ))
    e.append(sp(0.3))

    e += h2("10.2 Question Distribution")
    e.append(data_table(
        ["Group", "Count", "best_kb_layer", "Example Question"],
        [
            ["Order factual",       "10", "order",           "Was order X delivered on time?"],
            ["Delivery analytics",  "10", "delivery_status", "What % of orders were delivered late?"],
            ["Category analytics",  "10", "category",        "Which category has the highest revenue?"],
            ["Seller performance",  "10", "seller",          "Which seller has the most late deliveries?"],
            ["Month/State analytics","10","month / state",   "Which month had the most orders?"],
        ],
        col_widths=[4 * cm, 1.8 * cm, 3.5 * cm, 7.2 * cm],
    ))
    e.append(sp(0.3))

    e += h2("10.3 Why Pandas, Not LLM, for Golden Answers?")
    e.append(data_table(
        ["Criterion", "Pandas Calculation", "LLM Generation"],
        [
            ["Accuracy",        "Exact — sum, mean, count are deterministic", "May hallucinate numbers"],
            ["Reproducibility", "Same answer every run",                      "Varies between calls"],
            ["Cost",            "Free",                                       "API cost per question"],
            ["Thesis validity", "Scientifically justifiable",                 "Hard to defend in evaluation"],
            ["Speed",           "Milliseconds",                               "Seconds per question"],
        ],
        col_widths=[3.5 * cm, 6.5 * cm, 6.5 * cm],
    ))
    return e


def section_tips():
    e = []
    e += h1("11. Tips, Pitfalls and Best Practices")

    e += h2("11.1 Common Pitfalls")
    pitfalls = [
        ("Double-counting multi-item orders",
         "Always use .nunique('order_id') for order counts, not .count(). "
         "For per-entity delivery/review stats, deduplicate to (entity, order_id) grain first."),
        ("BOM in category translation file",
         "Use encoding='utf-8-sig' for every CSV read. Without it, the first column name "
         "becomes '\\ufeffproduct_category_name' and the join silently produces all NaN values."),
        ("NaT in date arithmetic",
         "pd.to_datetime(errors='coerce') converts bad dates to NaT. "
         "Arithmetic with NaT returns NaT, not an error. Always check .isna() before computing rates."),
        ("Delivery status on undelivered orders",
         "delivery_difference_days is NaN when order_delivered_customer_date is NaT. "
         "Check for the delivered date first in np.select, otherwise undelivered orders "
         "get an incorrect status."),
        ("Unicode in review comments",
         "Brazilian reviews contain accented characters (ã, ç, etc.). "
         "Always open JSON files with encoding='utf-8'. The pipeline writes with ensure_ascii=False."),
    ]
    for title, desc in pitfalls:
        e.append(KeepTogether([
            Paragraph(f"<b>{title}:</b>", ST["h4"]),
            Paragraph(desc, ST["body_left"]),
            sp(0.2),
        ]))

    e += h2("11.2 Performance Tips")
    tips = [
        "Use low_memory=False in pd.read_csv() to avoid mixed-type warnings on large files.",
        "Build the KB from the pre-saved enriched CSV (--steps kb) rather than re-running the full join — saves ~2 minutes.",
        "For ChromaDB ingestion, use batches of 500 documents. The library has an internal batch size limit.",
        "Use a fixed RANDOM_SEED=42 so the 10,000 sampled orders are identical across runs — vital for evaluation consistency.",
        "For V2 (full 99,441 order documents), switch ORDER_KB_SAMPLE=None and expect the KB JSON to be ~160 MB.",
    ]
    for t in tips:
        e.append(bullet(t))
    e.append(sp(0.3))

    e += h2("11.3 Extending the Pipeline")
    e.append(data_table(
        ["Extension", "How to implement"],
        [
            ["Add geolocation distance",
             "Merge geolocation on zip code; compute haversine distance\nbetween seller and customer; add distance_km to order document"],
            ["Full 99k order KB",
             "Set ORDER_KB_SAMPLE = None in config.py. Run --steps kb.\nExpect ~160 MB JSON and ~30 min ChromaDB ingestion."],
            ["Hybrid search (BM25 + vector)",
             "Use ChromaDB's query() with both embedding and metadata filters.\nOr use LlamaIndex BM25Retriever over the same documents."],
            ["LLM-polished summaries",
             "After computing stats with pandas, pass the structured\ntext through a Claude/GPT prompt to make it more natural.\nKeep numbers unchanged — only rewrite the prose."],
            ["Golden dataset expansion",
             "Add multi-hop questions that require combining category +\nmonth documents, or seller + state documents."],
        ],
        col_widths=[4.5 * cm, 12 * cm],
    ))
    return e


# ── Page number footer ────────────────────────────────────────────────────────

class NumberedCanvas:
    """Add page numbers to every page via the onPage callback."""
    def __init__(self, filename, **kwargs):
        from reportlab.pdfgen.canvas import Canvas
        self._canvas = Canvas(filename, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self._canvas.__dict__))
        self._canvas.showPage()

    def save(self):
        total = len(self._saved)
        for i, state in enumerate(self._saved, 1):
            self._canvas.__dict__.update(state)
            self._draw_footer(i, total)
            self._canvas.showPage()
        self._canvas.save()

    def _draw_footer(self, page, total):
        c = self._canvas
        c.saveState()
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#757575"))
        c.drawString(2 * cm, 1.2 * cm,
                     "Olist RAG Knowledge Base Pipeline — End-to-End Documentation")
        c.drawRightString(W - 2 * cm, 1.2 * cm, f"Page {page} of {total}")
        c.setStrokeColor(colors.HexColor("#CCCCCC"))
        c.setLineWidth(0.5)
        c.line(2 * cm, 1.5 * cm, W - 2 * cm, 1.5 * cm)
        c.restoreState()


def on_first_page(canvas, doc):
    pass  # no footer on cover


def on_later_pages(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#757575"))
    canvas.drawString(2 * cm, 1.2 * cm,
                      "Olist RAG Knowledge Base Pipeline — End-to-End Documentation")
    canvas.drawRightString(W - 2 * cm, 1.2 * cm,
                           f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.5 * cm, W - 2 * cm, 1.5 * cm)
    canvas.restoreState()


# ── Build PDF ─────────────────────────────────────────────────────────────────

def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title="Olist RAG Knowledge Base — End-to-End Documentation",
        author="RAG Pipeline",
    )

    story = []

    # Cover
    story += cover_page()

    # Table of Contents header
    story += h1("Table of Contents")
    toc_items = [
        ("1. Project Overview",                    ""),
        ("   1.1  What is a RAG Knowledge Base?",  ""),
        ("   1.2  Why Not Use Raw Rows as Docs?",  ""),
        ("   1.3  Pipeline Architecture",           ""),
        ("2. Raw Dataset Description",              ""),
        ("3. Step 1 — Loading Raw Data",            ""),
        ("4. Step 2 — Joining Datasets",            ""),
        ("5. Step 3 — Enriching the Master Dataset",""),
        ("6. Step 4 — Building the Knowledge Base", ""),
        ("7. Knowledge Base Layers — Detail & Examples",""),
        ("   7.1  Layer 1: Order-Level (10,000)",   ""),
        ("   7.2  Layer 2: Product Category (74)",  ""),
        ("   7.3  Layer 3: Seller (3,095)",          ""),
        ("   7.4  Layer 4: Customer State (27)",     ""),
        ("   7.5  Layer 5: Month Temporal (25)",     ""),
        ("   7.6  Layer 6: Delivery Status (4)",     ""),
        ("8. How to Run the Pipeline",               ""),
        ("9. Loading into ChromaDB",                 ""),
        ("10. Golden Dataset for RAG Evaluation",    ""),
        ("11. Tips, Pitfalls & Best Practices",      ""),
    ]
    for title, _ in toc_items:
        indent = 20 if title.startswith("   ") else 0
        st = ParagraphStyle(
            "TOCItem",
            fontSize=10 if indent else 11,
            leading=16,
            leftIndent=indent,
            textColor=C_ACCENT if indent else C_DARK_BLUE,
            fontName="Helvetica" if indent else "Helvetica-Bold",
            spaceAfter=2,
        )
        story.append(Paragraph(title, st))

    story.append(PageBreak())

    # All sections
    for sec in [
        section_overview,
        section_dataset,
        section_step1_load,
        section_step2_join,
        section_step3_enrich,
        section_step4_kb,
        section_kb_layers,
        section_how_to_run,
        section_chromadb,
        section_golden,
        section_tips,
    ]:
        story += sec()
        story.append(PageBreak())

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    print(f"PDF written to: {PDF_PATH}")
    print(f"File size: {PDF_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build_pdf()

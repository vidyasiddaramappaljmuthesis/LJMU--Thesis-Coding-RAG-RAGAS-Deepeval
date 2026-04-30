# E-Commerce RAG Evaluation Pipeline with RAGAS and DeepEval

> An end-to-end Retrieval-Augmented Generation (RAG) pipeline built on the Brazilian Olist e-commerce dataset, featuring a multi-layer knowledge base, automated golden dataset generation via Google Gemini, and comprehensive evaluation using RAGAS and DeepEval.

---

## Table of Contents

1. [Overview](#overview)
2. [Research Motivation](#research-motivation)
3. [Dataset](#dataset)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Knowledge Base Design](#knowledge-base-design)
6. [Evaluation Framework](#evaluation-framework)
7. [Golden Dataset Specification](#golden-dataset-specification)
8. [Project Structure](#project-structure)
9. [Installation](#installation)
10. [Configuration](#configuration)
11. [Usage](#usage)
12. [Key Statistics](#key-statistics)
13. [Dependencies](#dependencies)

---

## Overview

This project constructs a production-grade RAG evaluation framework for the domain of e-commerce analytics. The system transforms nine raw relational CSV files from the [Kaggle Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) into a structured, multi-layer knowledge base composed of 13,225 natural-language documents. These documents are designed for ingestion into a ChromaDB vector store and subsequent evaluation with the RAGAS and DeepEval frameworks.

The pipeline addresses a core challenge in applied RAG research: the absence of domain-specific, ground-truth evaluation sets. By automating golden dataset generation — distributing 100 question-answer pairs across three difficulty tiers and six knowledge layers — it enables rigorous, reproducible benchmarking of retrieval precision, answer faithfulness, and contextual relevance.

---

## Research Motivation

Evaluating RAG systems in production environments is non-trivial. Standard benchmarks rarely reflect the complexity of real-world retrieval scenarios, which involve:

- **Multi-grain queries** — questions targeting single records vs. aggregate statistics vs. temporal trends
- **Cross-document synthesis** — answers that require combining evidence from multiple document layers
- **Domain-specific reasoning** — e-commerce logic such as delivery SLA analysis, review sentiment, and seller performance

This pipeline operationalises a systematic methodology for constructing evaluation datasets that cover all three dimensions, using an automated Gemini-assisted generation step constrained by human-designed prompts and verifiable expected answers derived from pandas aggregations (not from the LLM itself).

---

## Dataset

**Source:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
**Coverage:** 2016–2018, Brazilian marketplace transactions  
**Grain:** Order-item level (one row per item purchased within an order)

| Raw File | Rows | Description |
|---|---|---|
| `olist_orders_dataset.csv` | 99,441 | Order lifecycle and timestamps |
| `olist_order_items_dataset.csv` | 112,650 | Items per order with seller and price |
| `olist_order_payments_dataset.csv` | 103,886 | Payment type and installments |
| `olist_order_reviews_dataset.csv` | 99,224 | Customer review scores and comments |
| `olist_customers_dataset.csv` | 99,441 | Customer location (city, state) |
| `olist_products_dataset.csv` | 32,951 | Product dimensions and category |
| `olist_sellers_dataset.csv` | 3,095 | Seller location |
| `olist_geolocation_dataset.csv` | 1,000,163 | Zip code → lat/lon mappings |
| `product_category_name_translation.csv` | 71 | Portuguese → English category names |

After joining and enrichment, the master analytical dataset contains **113,425 rows × 55 columns**.

---

## Pipeline Architecture

The pipeline executes in five sequential steps, all orchestrated by `data_preparation.py`.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION PIPELINE                       │
│                                                                     │
│  Step 1          Step 2          Step 3          Step 4     Step 5  │
│  ──────          ──────          ──────          ──────     ──────  │
│  Load Raw   ──>  Join       ──>  Enrich     ──>  Build  ──> Golden  │
│  CSVs (9)        Datasets        Master          KB          Dataset │
│                                                                     │
│  99K–1M          113,425         113,425          13,225     100     │
│  rows each       × 40 cols       × 55 cols        docs        Q&As  │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 1 — Load Raw Data (`step1_load_raw_data.py`)

Loads all nine CSV files into an in-memory dictionary of DataFrames. Handles UTF-8 BOM stripping for the category translation file and auto-parses datetime columns (`order_purchase_timestamp`, `order_delivered_customer_date`, etc.).

### Step 2 — Join Datasets (`step2_join_datasets.py`)

Performs multi-table aggregation and joining:

- **Payments** aggregated to order level (sum of payment values, comma-separated payment types)
- **Reviews** aggregated to order level (latest review score and comment per order)
- **Join chain:** orders → order_items → payments → reviews → customers → products → sellers → category_translation

Output: `final_olist_master.csv` — 113,425 rows × 40 columns at order-item grain.

### Step 3 — Enrich Master (`step3_enrich_master.py`)

Derives 15 analytical columns grouped into four families:

**Temporal Features**
| Column | Description |
|---|---|
| `purchase_month` | Integer month of purchase (1–12) |
| `purchase_year` | Year of purchase |
| `purchase_month_name` | Month name (e.g., "August") |
| `purchase_day_name` | Day of week (e.g., "Monday") |
| `purchase_hour` | Hour of day (0–23) |

**Delivery Features**
| Column | Description |
|---|---|
| `approval_hours` | Hours from purchase to payment approval |
| `carrier_handover_days` | Days from approval to carrier pickup |
| `delivery_days` | Actual delivery duration in days |
| `estimated_delivery_days` | Promised delivery duration |
| `delivery_difference_days` | Actual minus estimated (negative = early) |
| `delivery_status` | `early`, `on_time`, `late`, `not_delivered` |
| `delivery_bucket` | 8-band bucket (very_early to very_late) |

**Product Features**
| Column | Description |
|---|---|
| `product_category_final` | English category name |
| `item_total_value` | Price + freight per item |

**Review Feature**
| Column | Description |
|---|---|
| `review_bucket` | `positive` (≥4), `neutral` (3), `negative` (≤2), `no_review` |

Output: `final_olist_master_enriched.csv` — 113,425 rows × 55 columns.

### Step 4 — Build Knowledge Base (`step4_build_knowledge_base.py`)

Converts the enriched master dataset into six hierarchical layers of natural-language documents. Each document has three components:

```json
{
  "id": "order_ORD-abc123",
  "text": "Order ID: ORD-abc123\nProduct Category: electronics\nDelivery Status: early\n...",
  "metadata": {
    "document_type": "order",
    "delivery_status": "early",
    "customer_state": "SP",
    "purchase_month": 8,
    "review_score": 5
  }
}
```

All documents are combined into `kb_all_documents.json` (13,225 total) for ChromaDB ingestion.

### Step 5 — Build Golden Dataset (`step5_build_golden_dataset.py`)

Generates 100 ground-truth Q&A pairs using five Google Gemini API keys (20 queries per key). Each Q&A record includes:
- The question text
- The expected answer (verified against pandas aggregations)
- The expected context (exact KB document texts)
- Source document IDs
- Question type, difficulty level, and target KB layer

---

## Knowledge Base Design

The six-layer architecture enables retrieval at every analytical grain relevant to e-commerce operations:

| Layer | Documents | Grain | Example Query |
|---|---|---|---|
| 1 — Order-Level | 10,000 (sampled) | One per order | "What was the delivery status of order X?" |
| 2 — Category-Level | 74 | One per product category | "Which category has the highest average review score?" |
| 3 — Seller-Level | 3,095 | One per seller | "How many orders did seller Y fulfil in 2018?" |
| 4 — Customer State | 27 | One per Brazilian state | "What is the average delivery time to São Paulo?" |
| 5 — Month-Level | 25 | One per calendar month | "Which month had the highest order volume in 2017?" |
| 6 — Delivery Status | 4 | One per delivery outcome | "What proportion of orders were delivered late?" |

**Design rationale:** Human-readable natural-language text (rather than raw CSV rows or pipe-delimited strings) ensures compatibility with sentence-transformer embeddings and improves LLM comprehension during generation. Fixed `RANDOM_SEED = 42` guarantees reproducible sampling of the 10,000-order Layer 1 subset across all pipeline runs.

---

## Evaluation Framework

### RAGAS

[RAGAS](https://docs.ragas.io) evaluates RAG pipelines across four primary dimensions:

| Metric | Measures |
|---|---|
| **Answer Faithfulness** | Whether the generated answer is grounded in the retrieved context |
| **Answer Relevance** | Whether the answer addresses the question |
| **Context Precision** | Whether retrieved documents are relevant to the question |
| **Context Recall** | Whether the retrieved set covers all necessary ground-truth contexts |

### DeepEval

[DeepEval](https://docs.confident-ai.com) provides additional LLM-as-judge metrics:

| Metric | Measures |
|---|---|
| **G-Eval** | Task-specific quality assessment using a custom rubric |
| **Hallucination** | Factual accuracy against reference documents |
| **Contextual Relevancy** | Alignment of retrieved context with the question |
| **Answer Correctness** | Semantic match between generated and expected answers |

### Evaluation Workflow

```
golden_dataset.csv
       │
       ▼
  ChromaDB Retrieval  ──>  Retrieved Contexts
       │                          │
       ▼                          ▼
  LLM Generation      ──>  Generated Answer
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
                  RAGAS                    DeepEval
              (automated metrics)      (LLM-as-judge metrics)
```

---

## Golden Dataset Specification

The golden dataset (`dataset/golden/golden_dataset.csv`) provides 100 evaluation records distributed across three difficulty tiers:

| Difficulty | Count | Question Type | Example |
|---|---|---|---|
| **Easy** | 51 | Factual lookup | "What payment method was used in order X?" |
| **Medium** | 30 | Analytical aggregation | "Which category has the highest late delivery rate?" |
| **Hard** | 20 | Cross-layer synthesis | "What combination of factors correlates with late deliveries in SP in 2018?" |
| **Cross-layer** | 11 | Multi-document | "How does seller location correlate with delivery performance?" |

**Schema:**

| Column | Type | Description |
|---|---|---|
| `question_id` | string | Unique identifier (e.g., `Q001`) |
| `question` | string | Natural-language question |
| `expected_answer` | string | Ground-truth answer |
| `expected_context` | JSON array | List of KB document texts used |
| `expected_source_ids` | JSON array | List of KB document IDs |
| `question_type` | string | `factual`, `analytical`, `comparison`, `temporal` |
| `difficulty` | string | `easy`, `medium`, `hard` |
| `best_kb_layer` | string | Primary KB layer (e.g., `order`, `category`) |

**Generation parameters:** Gemini model `gemini-2.0-flash-preview`, temperature 0.4, JSON response format. Five API keys are rotated to distribute quota across 100 questions.

---

## Project Structure

```
.
├── data_preparation.py                   # Pipeline entry point (CLI orchestrator)
├── requirements.txt                      # Python dependencies
│
├── preprocessing/
│   ├── config.py                         # Paths, constants, RANDOM_SEED
│   ├── step1_load_raw_data.py            # Load raw CSVs
│   ├── step2_join_datasets.py            # Multi-table join to master
│   ├── step3_enrich_master.py            # Derive 15 analytical features
│   ├── step4_build_knowledge_base.py     # Build 6 KB layers as JSON
│   └── step5_build_golden_dataset.py     # Generate Q&A pairs via Gemini
│
├── dataset/
│   ├── raw/                              # 9 Kaggle Olist CSV files
│   ├── processed/
│   │   ├── final_olist_master.csv        # Joined master (113,425 × 40)
│   │   └── final_olist_master_enriched.csv  # Enriched master (113,425 × 55)
│   ├── knowledge_base/
│   │   ├── kb_order_documents.json       # 10,000 order-level docs
│   │   ├── kb_category_documents.json    # 74 category-level docs
│   │   ├── kb_seller_documents.json      # 3,095 seller-level docs
│   │   ├── kb_customer_state_documents.json  # 27 state-level docs
│   │   ├── kb_month_documents.json       # 25 month-level docs
│   │   ├── kb_delivery_status_documents.json # 4 delivery outcome docs
│   │   └── kb_all_documents.json         # Combined (13,225 total)
│   └── golden/
│       └── golden_dataset.csv            # 100 evaluation Q&A pairs
│
├── docs/
│   ├── generate_docs.py                  # PDF documentation generator
│   └── Olist_RAG_Knowledge_Base_Documentation.pdf
│
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_step1_load_raw_data.py
│   ├── test_step2_join_datasets.py
│   ├── test_step3_enrich_master.py
│   ├── test_step4_build_knowledge_base.py
│   └── test_step5_build_golden_dataset.py
│
└── logs/
    └── pipeline.log                      # Execution log with timestamps
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- [Kaggle Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (download and place CSVs in `dataset/raw/`)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd "Ecommerece RAG with RAGAS and Deepeval"

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

### API Keys (for golden dataset generation)

The golden dataset step requires up to five Google Gemini API keys to distribute quota across 100 generation requests. Set them as environment variables before running Step 5:

```bash
export GOOGLE_API_KEY_1="your_key_1"
export GOOGLE_API_KEY_2="your_key_2"
export GOOGLE_API_KEY_3="your_key_3"
export GOOGLE_API_KEY_4="your_key_4"
export GOOGLE_API_KEY_5="your_key_5"
```

At minimum, `GOOGLE_API_KEY_1` is required. Keys 2–5 are optional; the pipeline falls back gracefully if they are absent.

### Pipeline Configuration

Core settings are centralised in `preprocessing/config.py`:

| Constant | Default | Description |
|---|---|---|
| `RANDOM_SEED` | `42` | Seed for reproducible sampling |
| `ORDER_KB_SAMPLE` | `10000` | Number of orders sampled into Layer 1 |
| `DATA_RAW` | `dataset/raw/` | Raw CSV input directory |
| `DATA_PROCESSED` | `dataset/processed/` | Processed output directory |
| `DATA_KB` | `dataset/knowledge_base/` | Knowledge base JSON output |
| `DATA_GOLDEN` | `dataset/golden/` | Golden dataset output |

---

## Usage

### Run the full pipeline

```bash
python data_preparation.py --steps all
```

### Run individual steps

```bash
# Steps 1–3: data loading, joining, and enrichment
python data_preparation.py --steps enrich

# Step 4: build knowledge base (requires enriched CSV)
python data_preparation.py --steps kb

# Step 5: generate golden dataset (requires KB + API keys)
python data_preparation.py --steps golden
```

### Run tests

```bash
pytest tests/ -v
```

### Regenerate PDF documentation

```bash
python docs/generate_docs.py
```

---

## Key Statistics

### Dataset

| Metric | Value |
|---|---|
| Total order-item rows | 113,425 |
| Unique orders | 99,441 |
| Product categories | 74 |
| Sellers | 3,095 |
| Brazilian states covered | 27 |
| Temporal coverage | Aug 2016 – Oct 2018 (25 months) |

### Delivery Performance

| Status | Count | Share |
|---|---|---|
| Delivered early | 88,649 | 89.1% |
| Delivered late | 6,535 | 6.6% |
| Not delivered | 2,965 | 3.0% |
| On time | ~1,276 | 1.3% |

### Review Sentiment

| Bucket | Stars | Share |
|---|---|---|
| Positive | 4–5 | ~76.5% |
| Neutral | 3 | ~8.6% |
| Negative | 1–2 | ~11.4% |
| No review | — | ~3.5% |

### Knowledge Base

| Layer | Documents |
|---|---|
| Order-level (sampled) | 10,000 |
| Category-level | 74 |
| Seller-level | 3,095 |
| Customer state-level | 27 |
| Month-level | 25 |
| Delivery status-level | 4 |
| **Total** | **13,225** |

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `pandas` | ≥2.0.0 | Data manipulation and aggregation |
| `numpy` | ≥1.24.0 | Numerical operations |
| `google-genai` | ≥1.0.0 | Gemini API for Q&A generation |
| `ragas` | ≥0.2.0 | RAG evaluation metrics |
| `deepeval` | ≥1.0.0 | LLM evaluation framework |
| `chromadb` | ≥0.5.0 | Vector database for document retrieval |
| `langchain` | ≥0.3.0 | LLM orchestration |
| `langchain-google-genai` | ≥2.0.0 | LangChain + Gemini integration |
| `reportlab` | ≥4.0.0 | PDF documentation generation |

---

## Acknowledgements

- **Dataset:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), released under CC BY-NC-SA 4.0.
- **Evaluation frameworks:** [RAGAS](https://github.com/explodinggradients/ragas) and [DeepEval](https://github.com/confident-ai/deepeval).
- **Vector store:** [ChromaDB](https://www.trychroma.com/).
- **Generation model:** Google Gemini via the `google-genai` SDK.

# E-Commerce RAG Evaluation Pipeline with RAGAS and DeepEval

> An end-to-end benchmark comparing five Retrieval-Augmented Generation (RAG) strategies on the Brazilian Olist e-commerce dataset. The project covers the full lifecycle: raw CSV ingestion → multi-layer knowledge base → automated golden dataset generation → five distinct RAG pipelines → rigorous evaluation with RAGAS and DeepEval.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Dataset](#dataset)
4. [Data Preparation Pipeline](#data-preparation-pipeline)
5. [Knowledge Base Design](#knowledge-base-design)
6. [RAG Pipelines](#rag-pipelines)
   - [Naive RAG](#1-naive-rag)
   - [HyDE RAG](#2-hyde-rag)
   - [Multi-Query RAG](#3-multi-query-rag)
   - [Reranking RAG](#4-reranking-rag)
   - [Hybrid RAG](#5-hybrid-rag)
7. [Evaluation Framework](#evaluation-framework)
8. [Golden Dataset Specification](#golden-dataset-specification)
9. [Project Structure](#project-structure)
10. [Installation](#installation)
11. [Configuration](#configuration)
12. [Usage](#usage)
13. [Key Statistics](#key-statistics)
14. [Dependencies](#dependencies)
15. [Acknowledgements](#acknowledgements)

---

## Project Overview

This project constructs a production-grade RAG evaluation framework for e-commerce analytics. It transforms nine raw relational CSV files from the [Kaggle Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) into a structured, multi-layer knowledge base of 13,225 natural-language documents ingested into a ChromaDB vector store.

Five RAG strategies are implemented and benchmarked side-by-side:

| Strategy | Core Technique |
|---|---|
| **Naive RAG** | Direct semantic retrieval (bi-encoder + cosine similarity) |
| **HyDE RAG** | Hypothetical Document Embeddings — LLM imagines a document first |
| **Multi-Query RAG** | Query expansion (4 variants) + Reciprocal Rank Fusion |
| **Reranking RAG** | Two-stage retrieval — semantic shortlist → cross-encoder reranker |
| **Hybrid RAG** | Dual-path fusion — BM25 keyword search + semantic search + RRF |

Each pipeline is evaluated against a 100-query golden dataset using 11 reference-based metrics from RAGAS (5) and DeepEval (6) — with no LLM judge, keeping evaluation costs near zero.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          END-TO-END PIPELINE                                   │
│                                                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │  RAW DATA    │    │   PROCESSED  │    │  KNOWLEDGE   │    │   GOLDEN    │  │
│  │  9 Olist     │───▶│   MASTER     │───▶│   BASE       │───▶│   DATASET   │  │
│  │  CSV files   │    │  113K × 55   │    │  13,225 docs │    │  100 Q&As   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └─────────────┘  │
│         │                  │                    │                    │         │
│      Step 1-3           Step 3              Step 4               Step 5        │
│   (preprocessing)     (enrichment)       (KB build)          (Gemini gen)      │
│                                                │                    │         │
│                             ┌──────────────────┘                    │         │
│                             ▼                                        │         │
│              ┌──────────────────────────────┐                       │         │
│              │       ChromaDB + BM25        │                       │         │
│              │     Vector Store / Index     │◀──────────────────────┘         │
│              └──────────────┬───────────────┘                                 │
│                             │                                                  │
│          ┌──────────────────┼──────────────────────────┐                      │
│          ▼                  ▼                ▼          ▼          ▼           │
│     Naive RAG         HyDE RAG      Multi-Query    Reranking   Hybrid RAG      │
│     (baseline)      (hyp. docs)       RAG + RRF    RAG + CE    BM25+Sem+RRF   │
│          │                  │                │          │          │           │
│          └──────────────────┴────────────────┴──────────┴──────────┘          │
│                                         │                                      │
│                                         ▼                                      │
│                          ┌──────────────────────────┐                         │
│                          │  Groq LLaMA 3.3 70B      │                         │
│                          │  (Answer Generation)     │                         │
│                          └──────────────────────────┘                         │
│                                         │                                      │
│                          ┌──────────────┴──────────────┐                      │
│                          ▼                             ▼                       │
│                       RAGAS                       DeepEval                     │
│                  (5 metrics, no LLM)          (6 metrics, no LLM)              │
│                          │                             │                       │
│                          └──────────────┬──────────────┘                      │
│                                         ▼                                      │
│                            Excel Workbook (5 sheets)                           │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset

**Source:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
**Coverage:** 2016–2018, Brazilian marketplace transactions
**Grain:** Order-item level (one row per item within an order)

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

After joining and enrichment the master analytical dataset contains **113,425 rows × 55 columns**.

---

## Data Preparation Pipeline

All five steps are orchestrated by `data_preparation.py`.

```
Step 1          Step 2          Step 3          Step 4          Step 5
──────          ──────          ──────          ──────          ──────
Load Raw   ──▶  Join       ──▶  Enrich     ──▶  Build KB  ──▶  Golden
CSVs (9)        Datasets        Master          (13,225 docs)   Dataset
                                                                (100 Q&As)
```

### Step 1 — Load Raw Data (`step1_load_raw_data.py`)

Loads all nine CSV files into an in-memory dictionary of DataFrames. Handles UTF-8 BOM stripping for the category translation file and auto-parses all datetime columns.

### Step 2 — Join Datasets (`step2_join_datasets.py`)

- **Payments** aggregated to order level (sum of payment values, comma-separated payment types)
- **Reviews** aggregated to order level (latest review score and comment per order)
- **Join chain:** orders → order_items → payments → reviews → customers → products → sellers → category_translation

Output: `dataset/processed/final_olist_master.csv` — 113,425 rows × 40 columns.

### Step 3 — Enrich Master (`step3_enrich_master.py`)

Derives 15 analytical columns in four families:

**Temporal**

| Column | Description |
|---|---|
| `purchase_month` | Integer month (1–12) |
| `purchase_year` | Year of purchase |
| `purchase_month_name` | Month name (e.g., "August") |
| `purchase_day_name` | Day of week |
| `purchase_hour` | Hour of day (0–23) |

**Delivery**

| Column | Description |
|---|---|
| `approval_hours` | Hours from purchase to payment approval |
| `carrier_handover_days` | Days from approval to carrier pickup |
| `delivery_days` | Actual delivery duration in days |
| `estimated_delivery_days` | Promised delivery duration |
| `delivery_difference_days` | Actual minus estimated (negative = early) |
| `delivery_status` | `early`, `on_time`, `late`, `not_delivered` |
| `delivery_bucket` | 8-band bucket (very_early → very_late) |

**Product**

| Column | Description |
|---|---|
| `product_category_final` | English category name |
| `item_total_value` | Price + freight per item |

**Review**

| Column | Description |
|---|---|
| `review_bucket` | `positive` (≥4), `neutral` (3), `negative` (≤2), `no_review` |

Output: `dataset/processed/final_olist_master_enriched.csv` — 113,425 rows × 55 columns.

### Step 4 — Build Knowledge Base (`step4_build_knowledge_base.py`)

Converts the enriched master into six layers of natural-language JSON documents:

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

All six files are merged into `kb_all_documents.json` (13,225 total) for ChromaDB ingestion.

### Step 5 — Build Golden Dataset (`step5_build_golden_dataset.py`)

Generates 100 ground-truth Q&A pairs using five Google Gemini API keys (20 per key). Each record includes the question, expected answer (verified via pandas aggregations), expected context documents, source IDs, question type, difficulty tier, and target KB layer. Checkpoint files allow resumption after failures.

---

## Knowledge Base Design

| Layer | Documents | Grain | Example Query |
|---|---|---|---|
| 1 — Order-Level | 10,000 (sampled) | One per order | "What was the delivery status of order X?" |
| 2 — Category-Level | 74 | One per product category | "Which category has the highest average review score?" |
| 3 — Seller-Level | 3,095 | One per seller | "How many orders did seller Y fulfil in 2018?" |
| 4 — Customer State | 27 | One per Brazilian state | "What is the average delivery time to São Paulo?" |
| 5 — Month-Level | 25 | One per calendar month | "Which month had the highest order volume in 2017?" |
| 6 — Delivery Status | 4 | One per delivery outcome | "What proportion of orders were delivered late?" |

Human-readable natural-language text (rather than raw CSV rows) ensures compatibility with sentence-transformer embeddings and improves LLM comprehension. `RANDOM_SEED = 42` guarantees reproducible sampling of the 10,000-order Layer 1 subset.

---

## RAG Pipelines

All five pipelines share the same module layout:

```
<rag_type>/
├── run_<rag_type>_rag.py          # Interactive entry point
├── implementation/
│   ├── config.py                  # API keys, TOP_K, model names
│   ├── ingestion.py               # ChromaDB (+ BM25 for hybrid)
│   ├── retriever.py               # Retrieval strategy
│   ├── generator.py               # Groq LLaMA 3.3 70B answer generation
│   └── pipeline.py                # End-to-end orchestrator
└── evaluation/
    └── run_<rag_type>_rag_eval.py # RAGAS + DeepEval → Excel
```

---

### 1. Naive RAG

**Strategy:** Query → ChromaDB cosine similarity → top-5 docs → Groq answer

```
Query
  │
  ▼
all-MiniLM-L6-v2 embedding
  │
  ▼
ChromaDB cosine search  ──▶  top-5 documents
  │
  ▼
Groq LLaMA 3.3 70B  ──▶  Answer
```

**Key config:** `TOP_K = 5`

**When to use:** Baseline reference. Fastest and simplest; sets the performance floor.

**Run:**
```bash
python naive_rag/run_naive_rag.py
```

---

### 2. HyDE RAG

**Strategy:** Query → generate hypothetical document (high temp) → embed hypothetical doc → retrieve → generate answer (low temp)

```
Query
  │
  ▼
Groq (temp=0.7)  ──▶  Hypothetical document
  │
  ▼
all-MiniLM-L6-v2 embedding of hypothetical doc
  │
  ▼
ChromaDB cosine search  ──▶  top-5 real documents
  │
  ▼
Groq (temp=0.1)  ──▶  Answer
```

**Key config:** `HYDE_TEMPERATURE = 0.7`, `GENERATOR_TEMPERATURE = 0.1`, `TOP_K = 5`

**Why it works:** Embedding a plausible hypothetical answer bridges the semantic gap between a short query and long document passages, improving recall in dense retrieval.

**Run:**
```bash
python hyde_rag/run_hyde_rag.py
```

---

### 3. Multi-Query RAG

**Strategy:** Query → expand to 4 variants → retrieve 10 docs per variant → Reciprocal Rank Fusion → top-5 fused docs → generate

```
Query
  │
  ▼
Groq (temp=0.7)  ──▶  [variant_1, variant_2, variant_3]
  │                           + original query = 4 variants
  ▼
ChromaDB ×4 (10 docs each = 40 candidates)
  │
  ▼
Reciprocal Rank Fusion (k=60)  ──▶  deduplicated + re-ranked
  │
  ▼
top-5 fused documents
  │
  ▼
Groq LLaMA 3.3 70B  ──▶  Answer
```

**Key config:** `NUM_QUERY_VARIANTS = 4`, `RRF_K = 60`, `TOP_K = 5`

**Why it works:** Query paraphrasing diversifies the retrieval surface, reducing vocabulary-mismatch failures. RRF deduplicates candidates and rewards documents that rank highly across multiple query phrasings.

**Run:**
```bash
python multiquery_rag/run_multiquery_rag.py
```

---

### 4. Reranking RAG

**Strategy:** Query → initial semantic retrieval (20 candidates) → cross-encoder reranking → top-5 → generate

```
Query
  │
  ▼
all-MiniLM-L6-v2 embedding
  │
  ▼
ChromaDB cosine search  ──▶  top-20 candidates
  │
  ▼
cross-encoder/ms-marco-MiniLM-L-6-v2
  (jointly encodes query+doc pairs, produces relevance score)
  │
  ▼
top-5 reranked documents
  │
  ▼
Groq LLaMA 3.3 70B  ──▶  Answer
```

**Key config:** `INITIAL_K = 20`, `TOP_K = 5`, `RERANKER_MODEL = cross-encoder/ms-marco-MiniLM-L-6-v2`

**Why it works:** Bi-encoders encode query and document independently (fast but imprecise); cross-encoders attend to both jointly (precise but slow). Using the cross-encoder only on a shortlist of 20 gets accuracy without paying full reranking cost over 13K docs.

**Run:**
```bash
python reranking_rag/run_reranking_rag.py
```

---

### 5. Hybrid RAG

**Strategy:** Query → parallel {BM25 keyword search, ChromaDB semantic search} → RRF fusion → top-5 → generate

```
Query
  ├──▶  BM25 keyword search  ──▶  top-10 by token overlap
  │
  └──▶  ChromaDB semantic    ──▶  top-10 by cosine similarity
  │
  ▼
Reciprocal Rank Fusion (k=60)  ──▶  merged + re-ranked list
  │
  ▼
top-5 fused documents
  │
  ▼
Groq LLaMA 3.3 70B  ──▶  Answer
```

**Key config:** `SEMANTIC_TOP_K = 10`, `KEYWORD_TOP_K = 10`, `RRF_K = 60`, `TOP_K = 5`

**Why it works:** BM25 excels at exact keyword matches (product IDs, seller names, specific terms); semantic search handles paraphrasing and concept-level queries. Their combination is complementary — RRF integrates both signal types without tuning blend weights.

**Run:**
```bash
python hybrid_rag/run_hybrid_rag.py
```

---

## Evaluation Framework

All five RAG pipelines share an identical evaluation harness. For each of the 100 golden queries:

1. Retrieve documents using the RAG's own retriever
2. Generate an answer with Groq LLaMA 3.3 70B (1 LLM call per query)
3. Compute 11 reference-based metrics — no LLM judge, no additional API costs
4. Export results to an Excel workbook (5 sheets)

Evaluation runs are distributed across Groq API keys via `ThreadPoolExecutor` (20 queries per key).

### RAGAS Metrics

| Metric | Measures | Computation |
|---|---|---|
| **Faithfulness** | Answer grounded in retrieved context | Sentence-level token overlap ≥ 50% |
| **Answer Relevancy** | Answer addresses the question | TF-IDF cosine(answer, question) |
| **Context Precision** | Retrieved docs are relevant | Average-Precision@k vs. expected source IDs |
| **Context Recall** | Retrieved set covers the answer | Token-recall(expected answer tokens in retrieved contexts) |
| **Factual Correctness** | Answer matches ground truth | ROUGE-L F1(generated answer, expected answer) |

### DeepEval Metrics

| Metric | Measures | Computation |
|---|---|---|
| **AnswerRelevancy** | Answer addresses question | TF-IDF cosine(answer, question) |
| **Faithfulness** | Answer supported by context | Sentence-level token overlap ≥ 50% |
| **ContextualPrecision** | Retrieved docs are relevant | AP@k vs. expected source IDs |
| **ContextualRecall** | Contexts cover expected answer | Token-recall(expected answer in contexts) |
| **ContextualRelevancy** | Context relevance to question | Mean TF-IDF cosine(each doc, question) |
| **Hallucination** | Absence of fabrication | 1.0 − faithfulness score |

**Passing threshold:** Score ≥ 0.5

### Excel Output Format

Each evaluation script produces a workbook with five sheets:

| Sheet | Contents |
|---|---|
| Combined Results | All 11 metrics per query + final mean row |
| RAG Responses | Retrieved context texts per query |
| RAGAS Metrics | Faithfulness, Relevancy, Precision, Recall, Factual Correctness |
| DeepEval Metrics | 6 DeepEval scores + computation method column |
| Summary | Mean / min / max / std across all 100 queries |

**Run evaluation for any RAG type:**
```bash
python naive_rag/evaluation/run_naive_rag_eval.py
python hyde_rag/evaluation/run_hyde_rag_eval.py
python multiquery_rag/evaluation/run_multiquery_rag_eval.py
python reranking_rag/evaluation/run_reranking_rag_eval.py
python hybrid_rag/evaluation/run_hybrid_rag_eval.py
```

---

## Golden Dataset Specification

`dataset/golden/golden_dataset.csv` — 100 evaluation records across three difficulty tiers:

| Difficulty | Count | Question Type |
|---|---|---|
| Easy | 51 | Factual lookup ("What payment method was used in order X?") |
| Medium | 30 | Analytical aggregation ("Which category has the highest late delivery rate?") |
| Hard | 19 | Cross-layer synthesis ("How does seller location correlate with delivery performance?") |

**Schema:**

| Column | Type | Description |
|---|---|---|
| `question_id` | string | Unique identifier (`Q001`–`Q100`) |
| `question` | string | Natural-language question |
| `expected_answer` | string | Ground-truth answer (pandas-verified) |
| `expected_context` | JSON array | KB document texts used |
| `expected_source_ids` | JSON array | KB document IDs |
| `question_type` | string | `factual`, `analytical`, `comparison`, `temporal` |
| `difficulty` | string | `easy`, `medium`, `hard` |
| `best_kb_layer` | string | Primary KB layer (`order`, `category`, `seller`, …) |

**Generation:** Gemini `gemini-2.0-flash-preview`, temperature 0.4, JSON response format. Five API keys rotated across 100 questions; checkpoint files enable resumption.

---

## Project Structure

```
.
├── data_preparation.py                        # Root entry point — delegates to preprocessing/
├── pyproject.toml                             # Package metadata and dependencies
├── requirements.txt                           # Pip-installable dependency list
│
├── preprocessing/                             # Data preparation (Steps 1–5)
│   ├── config.py                              # Central config (paths, RANDOM_SEED=42)
│   ├── data_preparation.py                    # CLI orchestrator (called via root wrapper)
│   ├── step1_load_raw_data.py
│   ├── step2_join_datasets.py
│   ├── step3_enrich_master.py
│   ├── step4_build_knowledge_base.py
│   └── step5_build_golden_dataset.py
│
├── dataset/
│   ├── raw/                                   # 9 Kaggle Olist CSV files (user-supplied)
│   ├── processed/
│   │   ├── final_olist_master.csv             # 113,425 × 40
│   │   └── final_olist_master_enriched.csv    # 113,425 × 55
│   ├── knowledge_base/
│   │   ├── kb_order_documents.json            # 10,000 order-level docs
│   │   ├── kb_category_documents.json         # 74 category docs
│   │   ├── kb_seller_documents.json           # 3,095 seller docs
│   │   ├── kb_customer_state_documents.json   # 27 state docs
│   │   ├── kb_month_documents.json            # 25 month docs
│   │   ├── kb_delivery_status_documents.json  # 4 delivery-status docs
│   │   └── kb_all_documents.json              # Combined (13,225 total)
│   └── golden/
│       ├── golden_dataset.csv                 # 100 Q&A pairs
│       └── golden_checkpoint_key*.json        # Per-key generation checkpoints
│
├── naive_rag/
│   ├── run_naive_rag.py
│   ├── docs/
│   │   ├── Implementation_of_Naive_RAG.pdf
│   │   ├── Evaluation_of_Naive_RAG.pdf
│   │   └── generate_naive_rag_pdfs.py
│   ├── implementation/
│   │   ├── config.py
│   │   ├── ingestion.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── pipeline.py
│   ├── evaluation/
│   │   └── run_naive_rag_eval.py
│   └── results/
│       └── Naive-RAG_<timestamp>.xlsx
│
├── hyde_rag/
│   ├── run_hyde_rag.py
│   ├── docs/
│   │   ├── Implementation_of_HyDE_RAG.pdf
│   │   ├── Evaluation_of_HyDE_RAG.pdf
│   │   └── generate_hyde_rag_pdfs.py
│   ├── implementation/
│   │   ├── config.py
│   │   ├── hypothetical.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── pipeline.py
│   ├── evaluation/
│   │   └── run_hyde_rag_eval.py
│   └── results/
│       └── HyDE-RAG_<timestamp>.xlsx
│
├── multiquery_rag/
│   ├── run_multiquery_rag.py
│   ├── docs/
│   │   ├── Implementation_of_MultiQuery_RAG.pdf
│   │   ├── Evaluation_of_MultiQuery_RAG.pdf
│   │   └── generate_multiquery_rag_pdfs.py
│   ├── implementation/
│   │   ├── config.py
│   │   ├── query_expander.py
│   │   ├── retriever.py
│   │   ├── fusion.py
│   │   ├── generator.py
│   │   └── pipeline.py
│   ├── evaluation/
│   │   └── run_multiquery_rag_eval.py
│   └── results/
│       └── MultiQuery-RAG_<timestamp>.xlsx
│
├── reranking_rag/
│   ├── run_reranking_rag.py
│   ├── implementation/
│   │   ├── config.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── pipeline.py
│   └── evaluation/
│       └── run_reranking_rag_eval.py
│
├── reranking_rag/
│   ├── run_reranking_rag.py
│   ├── docs/
│   │   ├── Implementation_of_Reranking_RAG.pdf
│   │   ├── Evaluation_of_Reranking_RAG.pdf
│   │   └── generate_reranking_rag_pdfs.py
│   ├── implementation/
│   │   ├── config.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── pipeline.py
│   ├── evaluation/
│   │   └── run_reranking_rag_eval.py
│   └── results/
│       └── Reranking-RAG_<timestamp>.xlsx
│
├── hybrid_rag/
│   ├── run_hybrid_rag.py
│   ├── docs/
│   │   ├── Hybrid_RAG_Documentation.pdf
│   │   └── generate_hybrid_rag_pdfs.py
│   ├── implementation/
│   │   ├── config.py
│   │   ├── ingestion.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   ├── utils.py
│   │   └── pipeline.py
│   ├── evaluation/
│   │   └── run_hybrid_rag_eval.py
│   └── results/
│       └── Hybrid-RAG_<timestamp>.xlsx
│
├── shared/
│   └── groq_client.py                         # Groq key rotation + retry logic
│
├── scripts/
│   ├── check_metrics.py                       # Dev utility: verify RAGAS metric imports
│   └── smoke_test_reranking.py                # Manual end-to-end smoke test (not pytest)
│
├── tests/
│   ├── conftest.py
│   ├── run_rag_tests.py                       # Excel-reporting test orchestrator
│   ├── test_config.py
│   ├── test_step1_load_raw_data.py
│   ├── test_step2_join_datasets.py
│   ├── test_step3_enrich_master.py
│   ├── test_step4_build_knowledge_base.py
│   ├── test_step5_build_golden_dataset.py
│   ├── naive_rag/
│   ├── hyde_rag/
│   ├── multiquery_rag/
│   ├── reranking_rag/
│   ├── hybrid_rag/
│   └── completed/                             # Historical test-run Excel reports
│       ├── naive_rag/
│       ├── hyde_rag/
│       ├── multiquery_rag/
│       ├── reranking_rag/
│       └── hybrid_rag/
│
├── docs/
│   ├── generate_docs.py
│   ├── generate_golden_dataset_docs.py
│   ├── Olist_RAG_Knowledge_Base_Documentation.pdf
│   └── golden_dataset_creation_guide.pdf
│
├── results/
│   └── RAG_Master_Comparison_<timestamp>.xlsx # Cross-RAG comparison workbook
│
└── logs/
    └── pipeline.log
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- [Kaggle Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — download and place all nine CSV files in `dataset/raw/`

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd "Ecommerece RAG with RAGAS and Deepeval"

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install the project and all dependencies
pip install -e .

# Or install dependencies only (without editable install)
pip install -r requirements.txt
```

Verify evaluation dependencies after install:

```bash
python scripts/check_metrics.py
```

---

## Configuration

### `.env` file

Copy `.env.example` to `.env` and fill in your API keys:

```env
# Groq — used for answer generation in all five RAG pipelines
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...
GROQ_API_KEY_4=gsk_...
GROQ_API_KEY_5=gsk_...

# Google Gemini — used only for golden dataset generation (Step 5)
GOOGLE_API_KEY_1=AIza...
GOOGLE_API_KEY_2=AIza...
GOOGLE_API_KEY_3=AIza...
GOOGLE_API_KEY_4=AIza...
GOOGLE_API_KEY_5=AIza...
```

At minimum, `GROQ_API_KEY_1` and `GOOGLE_API_KEY_1` are required. Additional keys enable parallel execution and higher throughput.

### Pipeline Configuration (`preprocessing/config.py`)

| Constant | Default | Description |
|---|---|---|
| `RANDOM_SEED` | `42` | Seed for reproducible KB sampling |
| `ORDER_KB_SAMPLE` | `10000` | Orders sampled into Layer 1 |
| `DATA_RAW` | `dataset/raw/` | Raw CSV input directory |
| `DATA_PROCESSED` | `dataset/processed/` | Processed output directory |
| `DATA_KB` | `dataset/knowledge_base/` | KB JSON output |
| `DATA_GOLDEN` | `dataset/golden/` | Golden dataset output |

### RAG Configuration (per-pipeline `config.py`)

| Pipeline | Key Parameters |
|---|---|
| Naive RAG | `TOP_K = 5` |
| HyDE RAG | `TOP_K = 5`, `HYDE_TEMPERATURE = 0.7`, `GENERATOR_TEMPERATURE = 0.1` |
| Multi-Query RAG | `NUM_QUERY_VARIANTS = 4`, `RRF_K = 60`, `TOP_K = 5` |
| Reranking RAG | `INITIAL_K = 20`, `TOP_K = 5`, `RERANKER_MODEL = cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Hybrid RAG | `SEMANTIC_TOP_K = 10`, `KEYWORD_TOP_K = 10`, `RRF_K = 60`, `TOP_K = 5` |

All pipelines use `sentence-transformers/all-MiniLM-L6-v2` for embeddings and `llama-3.3-70b-versatile` via Groq for generation.

---

## Usage

### 1. Run the data preparation pipeline

```bash
# Run all five steps (root wrapper)
python data_preparation.py --steps all

# Run only data loading, joining, and enrichment (Steps 1–3)
python data_preparation.py --steps enrich

# Build the knowledge base — Step 4 (requires enriched CSV)
python data_preparation.py --steps kb

# Generate the golden dataset — Step 5 (requires KB + Gemini API keys)
python data_preparation.py --steps golden

# Alternative: run as a module (equivalent to the above)
python -m preprocessing.data_preparation --steps all
```

### 2. Run a RAG pipeline interactively

```bash
python naive_rag/run_naive_rag.py
python hyde_rag/run_hyde_rag.py
python multiquery_rag/run_multiquery_rag.py
python reranking_rag/run_reranking_rag.py
python hybrid_rag/run_hybrid_rag.py
```

### 3. Run evaluation for a RAG pipeline

Each script evaluates all 100 golden queries and exports an Excel workbook to `<rag_type>/results/`.

```bash
python naive_rag/evaluation/run_naive_rag_eval.py
python hyde_rag/evaluation/run_hyde_rag_eval.py
python multiquery_rag/evaluation/run_multiquery_rag_eval.py
python reranking_rag/evaluation/run_reranking_rag_eval.py
python hybrid_rag/evaluation/run_hybrid_rag_eval.py
```

### 4. Run the test suite

```bash
# All tests
pytest tests/ -v

# Only preprocessing tests
pytest tests/test_step*.py -v

# Tests for a specific RAG type
pytest tests/naive_rag/ -v

# All RAG pipeline tests via the Excel-reporting orchestrator
python tests/run_rag_tests.py
```

### 5. Run the Reranking RAG smoke test (manual validation)

```bash
python scripts/smoke_test_reranking.py
```

### 6. Regenerate documentation PDFs

```bash
# Root-level knowledge base and golden dataset docs
python docs/generate_docs.py
python docs/generate_golden_dataset_docs.py

# Per-RAG implementation and evaluation PDFs
python naive_rag/docs/generate_naive_rag_pdfs.py
python hyde_rag/docs/generate_hyde_rag_pdfs.py
python multiquery_rag/docs/generate_multiquery_rag_pdfs.py
python reranking_rag/docs/generate_reranking_rag_pdfs.py
python hybrid_rag/docs/generate_hybrid_rag_pdfs.py
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

### Golden Dataset

| Difficulty | Count |
|---|---|
| Easy (factual) | 51 |
| Medium (analytical) | 30 |
| Hard (cross-layer) | 19 |
| **Total** | **100** |

### Delivery Performance

| Status | Count | Share |
|---|---|---|
| Early | 88,649 | 89.1% |
| Late | 6,535 | 6.6% |
| Not delivered | 2,965 | 3.0% |
| On time | ~1,276 | 1.3% |

### Review Sentiment

| Bucket | Stars | Share |
|---|---|---|
| Positive | 4–5 | ~76.5% |
| Neutral | 3 | ~8.6% |
| Negative | 1–2 | ~11.4% |
| No review | — | ~3.5% |

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `pandas` | ≥2.0.0 | Data manipulation and aggregation |
| `numpy` | ≥1.24.0 | Numerical operations |
| `python-dotenv` | ≥1.0.0 | API key loading from `.env` |
| `google-genai` | ≥1.0.0 | Gemini API for golden dataset generation |
| `ragas` | ≥0.2.0 | Reference-based RAG evaluation (5 metrics) |
| `deepeval` | ≥1.0.0 | Reference-based RAG evaluation (6 metrics) |
| `chromadb` | ≥0.5.0 | Vector database for semantic retrieval |
| `sentence-transformers` | ≥3.0.0 | all-MiniLM-L6-v2 embeddings + cross-encoder reranker |
| `groq` | ≥0.11.0 | LLaMA 3.3 70B inference |
| `rank-bm25` | ≥0.3.1 | BM25 keyword search (Hybrid RAG) |
| `langchain` | ≥0.3.0 | LLM orchestration helpers |
| `langchain-google-genai` | ≥2.0.0 | LangChain + Gemini integration |
| `langchain-groq` | ≥0.2.0 | LangChain + Groq integration |
| `langchain-core` | ≥0.3.0,<0.3.70 | Core LangChain utilities (pinned to avoid broken torch import) |
| `openpyxl` | ≥3.1.0 | Excel workbook export |
| `reportlab` | ≥4.0.0 | PDF documentation generation |

---

## Acknowledgements

- **Dataset:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), released under CC BY-NC-SA 4.0
- **Evaluation frameworks:** [RAGAS](https://github.com/explodinggradients/ragas) and [DeepEval](https://github.com/confident-ai/deepeval)
- **Vector store:** [ChromaDB](https://www.trychroma.com/)
- **Generation model:** [Groq](https://groq.com/) — LLaMA 3.3 70B Versatile
- **Golden dataset generation:** Google Gemini via the `google-genai` SDK
- **Cross-encoder reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` via [sentence-transformers](https://www.sbert.net/)

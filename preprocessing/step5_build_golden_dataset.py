"""
Step 5 – Golden dataset generation using Gemini 3 Flash.

Produces 150 Q&A pairs distributed across 6 KB layers + cross-layer questions.
Each row is compatible with both RAGAS and DeepEval evaluation frameworks.

RAGAS fields   : question  | ground_truth (expected_answer) | contexts (expected_context)
DeepEval fields: input     | expected_output                | context

Environment
-----------
    GOOGLE_API_KEY  — required, set before running
    Windows  : $env:GOOGLE_API_KEY="your-key-here"
    Mac/Linux: export GOOGLE_API_KEY="your-key-here"

Distribution (150 total)
------------------------
    Layer              Easy  Medium  Hard  Total
    order               20      5      0     25
    category            15     10      5     30
    seller              15      5      0     20
    state               10     10      5     25
    month               10      5      5     20
    delivery_status      5      5      5     15
    cross_layer          0      5     10     15
    ─────────────────────────────────────────────
    Total               75     45     30    150
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Dict, List, Tuple

import pandas as pd
from google import genai
from google.genai import types

from .config import DATA_KB, RANDOM_SEED

logger = logging.getLogger(__name__)

# ── Output schema ─────────────────────────────────────────────────────────────

GOLDEN_COLUMNS = [
    "question_id",          # q001 … q150
    "question",             # natural language question
    "expected_answer",      # ground-truth answer (string)
    "expected_context",     # JSON list of KB document texts used as context
    "expected_source_ids",  # JSON list of KB document IDs
    "question_type",        # factual | analytical | comparison
    "difficulty",           # easy | medium | hard
    "best_kb_layer",        # which KB layer(s) answer this
]

# ── Target distribution ────────────────────────────────────────────────────────

_LAYER_TARGETS: Dict[str, Dict[str, int]] = {
    "order":           {"easy": 20, "medium": 5,  "hard": 0},
    "category":        {"easy": 15, "medium": 10, "hard": 5},
    "seller":          {"easy": 15, "medium": 5,  "hard": 0},
    "state":           {"easy": 10, "medium": 10, "hard": 5},
    "month":           {"easy": 10, "medium": 5,  "hard": 5},
    "delivery_status": {"easy": 5,  "medium": 5,  "hard": 5},
    "cross_layer":     {"easy": 0,  "medium": 5,  "hard": 10},
}

# ── Mappings ──────────────────────────────────────────────────────────────────

_DOC_TYPE_TO_LAYER: Dict[str, str] = {
    "order_level":             "order",
    "category_level":          "category",
    "seller_level":            "seller",
    "customer_state_level":    "state",
    "month_level":             "month",
    "delivery_status_insight": "delivery_status",
}

_DIFFICULTY_TO_QTYPE: Dict[str, str] = {
    "easy":   "factual",
    "medium": "analytical",
    "hard":   "comparison",
}

# Cross-layer pairs tried in rotation for cross-layer question generation
_CROSS_LAYER_PAIRS: List[Tuple[str, str]] = [
    ("category",        "state"),
    ("month",           "delivery_status"),
    ("category",        "month"),
    ("seller",          "state"),
    ("state",           "month"),
    ("category",        "delivery_status"),
    ("month",           "state"),
    ("seller",          "month"),
]

# Max docs sampled per layer (keeps prompts short, controls API usage)
_LAYER_MAX_DOCS: Dict[str, int] = {
    "order":           60,
    "category":        74,   # use all — small set, high value
    "seller":          50,
    "state":           27,   # use all
    "month":           25,   # use all
    "delivery_status": 4,    # use all
}

_GEMINI_MODEL      = "gemini-3-flash-preview"
_DELAY_MIN_SEC     = 5    # minimum wait between API calls
_DELAY_MAX_SEC     = 20   # maximum wait between API calls
_MAX_RETRIES       = 3


def _wait_between_calls() -> None:
    """Sleep a random 5-20 s after each completed API call."""
    delay = random.uniform(_DELAY_MIN_SEC, _DELAY_MAX_SEC)
    logger.info(f"  Waiting {delay:.1f}s before next API call...")
    time.sleep(delay)


# ── Public entry point ────────────────────────────────────────────────────────

def generate_golden_dataset(df: pd.DataFrame, kb_docs: List[dict]) -> pd.DataFrame:
    """
    Generate the 150-question golden evaluation dataset using Gemini 3 Flash.

    Parameters
    ----------
    df       : enriched master DataFrame (available for future answer validation)
    kb_docs  : list of KB documents from build_knowledge_base(); loaded from
               disk automatically if empty

    Returns
    -------
    pd.DataFrame with GOLDEN_COLUMNS — 150 rows, RAGAS + DeepEval compatible.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error(
            "GOOGLE_API_KEY environment variable is not set.\n"
            "  Windows  : $env:GOOGLE_API_KEY='your-key'\n"
            "  Mac/Linux: export GOOGLE_API_KEY='your-key'"
        )
        return pd.DataFrame(columns=GOLDEN_COLUMNS)

    if not kb_docs:
        kb_docs = _load_kb_from_disk()
    if not kb_docs:
        logger.error("No KB documents found. Run '--steps kb' first.")
        return pd.DataFrame(columns=GOLDEN_COLUMNS)

    logger.info(f"Loaded {len(kb_docs):,} KB documents for golden dataset generation")
    random.seed(RANDOM_SEED)

    client = genai.Client(api_key=api_key)
    docs_by_layer = _group_and_sample(kb_docs)

    all_rows: List[dict] = []

    for layer, targets in _LAYER_TARGETS.items():
        total_target = sum(targets.values())
        logger.info(
            f"  [Layer: {layer:<16}]  "
            f"easy={targets['easy']:2d}  medium={targets['medium']:2d}  "
            f"hard={targets['hard']:2d}  total={total_target:2d}"
        )
        try:
            if layer == "cross_layer":
                rows = _gen_cross_layer(client, docs_by_layer, targets)
            else:
                layer_docs = docs_by_layer.get(layer, [])
                if not layer_docs:
                    logger.warning(f"  No docs available for layer '{layer}', skipping")
                    continue
                rows = _gen_layer(client, layer, layer_docs, targets)
        except Exception as exc:
            logger.error(f"  Layer '{layer}' failed: {exc}")
            rows = []

        logger.info(f"  -> {len(rows):3d} questions generated for {layer}")
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No questions generated. Check GOOGLE_API_KEY and KB documents.")
        return pd.DataFrame(columns=GOLDEN_COLUMNS)

    df_out = pd.DataFrame(all_rows)
    df_out.insert(0, "question_id", [f"q{i + 1:03d}" for i in range(len(df_out))])

    for col in GOLDEN_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = ""

    df_out = df_out[GOLDEN_COLUMNS].reset_index(drop=True)

    logger.info(
        f"\n  Golden dataset summary:\n"
        f"    Total questions : {len(df_out)}\n"
        f"    Easy            : {(df_out['difficulty'] == 'easy').sum()}\n"
        f"    Medium          : {(df_out['difficulty'] == 'medium').sum()}\n"
        f"    Hard            : {(df_out['difficulty'] == 'hard').sum()}\n"
        f"    Factual         : {(df_out['question_type'] == 'factual').sum()}\n"
        f"    Analytical      : {(df_out['question_type'] == 'analytical').sum()}\n"
        f"    Comparison      : {(df_out['question_type'] == 'comparison').sum()}"
    )
    return df_out


# ── Document grouping & sampling ──────────────────────────────────────────────

def _group_and_sample(kb_docs: List[dict]) -> Dict[str, List[dict]]:
    """Group all KB docs by layer, then sample to manageable size per layer."""
    grouped: Dict[str, List[dict]] = {}
    for doc in kb_docs:
        doc_type = doc.get("metadata", {}).get("document_type", "")
        layer = _DOC_TYPE_TO_LAYER.get(doc_type)
        if layer:
            grouped.setdefault(layer, []).append(doc)

    sampled: Dict[str, List[dict]] = {}
    for layer, docs in grouped.items():
        max_n = _LAYER_MAX_DOCS.get(layer, 50)
        sampled[layer] = random.sample(docs, min(max_n, len(docs)))
        logger.info(
            f"  Sampled {len(sampled[layer]):>3} / {len(docs):>5} docs "
            f"for layer '{layer}'"
        )

    return sampled


# ── Single-layer question generation ─────────────────────────────────────────

def _gen_layer(
    client,
    layer: str,
    docs: List[dict],
    targets: Dict[str, int],
) -> List[dict]:
    """Generate questions for one KB layer across all difficulty levels."""
    rows: List[dict] = []

    for difficulty, target in targets.items():
        if target == 0:
            continue

        # Ask for 2 questions per call for easy/medium; 1 for hard
        batch_size = 1 if difficulty == "hard" else 2

        shuffled = docs[:]
        random.shuffle(shuffled)
        doc_idx   = 0
        collected = 0

        while collected < target:
            doc = shuffled[doc_idx % len(shuffled)]
            doc_idx += 1

            n      = min(batch_size, target - collected)
            prompt = _prompt_single(doc, difficulty, n, layer)
            items  = _call_gemini(client, prompt)

            for item in items[:n]:
                q = (item.get("question") or "").strip()
                a = (item.get("expected_answer") or "").strip()
                if not q or not a:
                    continue
                rows.append({
                    "question":            q,
                    "expected_answer":     a,
                    "expected_context":    json.dumps([doc["text"]]),
                    "expected_source_ids": json.dumps([doc["id"]]),
                    "question_type":       _DIFFICULTY_TO_QTYPE[difficulty],
                    "difficulty":          difficulty,
                    "best_kb_layer":       layer,
                })
                collected += 1
                if collected >= target:
                    break

            _wait_between_calls()

            # Safety: stop if we've cycled through docs too many times without progress
            if doc_idx > len(shuffled) * 4 and collected < target:
                logger.warning(
                    f"  Doc-cycle limit reached for {layer}/{difficulty}: "
                    f"got {collected}/{target}"
                )
                break

    return rows


# ── Cross-layer question generation ──────────────────────────────────────────

def _gen_cross_layer(
    client,
    docs_by_layer: Dict[str, List[dict]],
    targets: Dict[str, int],
) -> List[dict]:
    """Generate questions that require two documents from different layers."""
    rows: List[dict] = []
    pairs = _CROSS_LAYER_PAIRS[:]
    random.shuffle(pairs)
    pair_idx = 0

    for difficulty, target in targets.items():
        if target == 0:
            continue

        collected    = 0
        attempts     = 0
        max_attempts = target * 5

        while collected < target and attempts < max_attempts:
            attempts += 1
            layer1, layer2 = pairs[pair_idx % len(pairs)]
            pair_idx += 1

            docs1 = docs_by_layer.get(layer1, [])
            docs2 = docs_by_layer.get(layer2, [])
            if not docs1 or not docs2:
                continue

            doc1   = random.choice(docs1)
            doc2   = random.choice(docs2)
            prompt = _prompt_cross(doc1, layer1, doc2, layer2, difficulty)
            items  = _call_gemini(client, prompt)

            for item in items[:1]:
                q = (item.get("question") or "").strip()
                a = (item.get("expected_answer") or "").strip()
                if not q or not a:
                    continue
                rows.append({
                    "question":            q,
                    "expected_answer":     a,
                    "expected_context":    json.dumps([doc1["text"], doc2["text"]]),
                    "expected_source_ids": json.dumps([doc1["id"], doc2["id"]]),
                    "question_type":       _DIFFICULTY_TO_QTYPE[difficulty],
                    "difficulty":          difficulty,
                    "best_kb_layer":       f"{layer1}+{layer2}",
                })
                collected += 1
                break

            _wait_between_calls()

        if collected < target:
            logger.warning(
                f"  Cross-layer {difficulty}: partial result "
                f"({collected}/{target})"
            )

    return rows


# ── Prompt builders ───────────────────────────────────────────────────────────

def _prompt_single(doc: dict, difficulty: str, n: int, layer: str) -> str:
    """Prompt for generating questions from a single KB document."""
    layer_desc = {
        "order":           "an individual customer order (delivery, payment, and review details)",
        "category":        "a product category (aggregated sales, delivery, and review statistics)",
        "seller":          "an individual seller (fulfilment performance and revenue metrics)",
        "state":           "a Brazilian customer state (order volumes and delivery performance)",
        "month":           "a month of e-commerce activity (orders, revenue, and delivery trends)",
        "delivery_status": "a delivery outcome group (early / late / on_time / not_delivered)",
    }.get(layer, "e-commerce data")

    if difficulty == "easy":
        style = (
            f"Generate {n} FACTUAL question(s) answerable by reading a specific "
            "value directly from the document.\n"
            "Good starters: 'What is the...', 'How many...', 'What was the...'"
        )
        answer_note = (
            "Provide the specific value or short fact stated in the document."
        )
    elif difficulty == "medium":
        style = (
            f"Generate {n} ANALYTICAL question(s) that require interpreting or "
            "connecting multiple fields in the document — not just reading one number.\n"
            "Good starters: 'Based on the data...', 'What does ... suggest about...', "
            "'How does ... compare to...'"
        )
        answer_note = (
            "Provide a 1-2 sentence answer that interprets the data and explains "
            "what the numbers mean, not just a raw value."
        )
    else:  # hard
        style = (
            f"Generate {n} CHALLENGING question(s) requiring the reader to synthesise "
            "multiple metrics and draw a conclusion or recommendation.\n"
            "Good starters: 'Given the performance metrics...', "
            "'What combination of factors...', 'What strategic insight...'"
        )
        answer_note = (
            "Provide a 2-3 sentence answer that synthesises data points, "
            "identifies patterns, and reaches a conclusion."
        )

    return f"""You are building an evaluation dataset for a Brazilian e-commerce RAG system.
The knowledge base document below describes {layer_desc}.

TASK
{style}

RULES
- Questions must be answerable using ONLY the information in this document.
- Do NOT ask yes/no questions.
- Do NOT use phrases like "according to the document" or "in this text" in the question.
- Phrase questions as natural business queries an analyst would ask.
- Each question must cover different data points.

DOCUMENT
{doc["text"]}

ANSWER NOTE
{answer_note}

Return ONLY a valid JSON array (no markdown fences, no explanation):
[
  {{"question": "...", "expected_answer": "..."}},
  ...
]
Generate exactly {n} item(s)."""


def _prompt_cross(
    doc1: dict, layer1: str,
    doc2: dict, layer2: str,
    difficulty: str,
) -> str:
    """Prompt for generating a question that requires two KB documents."""
    if difficulty == "medium":
        task = (
            "Generate 1 ANALYTICAL question that connects data from BOTH documents. "
            "The question should link a fact from one layer to a fact from the other."
        )
        answer_note = (
            "A 1-2 sentence answer using specific values from both documents "
            "to draw a connection."
        )
    else:  # hard
        task = (
            "Generate 1 CHALLENGING question that can ONLY be answered by combining "
            "insights from BOTH documents. It should be impossible to answer with just one."
        )
        answer_note = (
            "A 2-3 sentence answer that synthesises data from both documents, "
            "identifies a pattern or contrast, and draws a conclusion."
        )

    return f"""You are building a multi-hop evaluation dataset for a Brazilian e-commerce RAG system.

TASK
{task}

RULES
- The question MUST require information from BOTH documents to answer completely.
- Do NOT ask yes/no questions.
- Do NOT reference "Document 1", "Document 2", or "the document" in the question.
- Phrase the question as a natural business query.

DOCUMENT 1 (layer: {layer1})
{doc1["text"]}

DOCUMENT 2 (layer: {layer2})
{doc2["text"]}

ANSWER NOTE
{answer_note}

Return ONLY a valid JSON array (no markdown fences, no explanation):
[
  {{"question": "...", "expected_answer": "..."}}
]
Generate exactly 1 item."""


# ── Gemini API call with retry ────────────────────────────────────────────────

def _call_gemini(client, prompt: str) -> List[dict]:
    """Call Gemini 3 Flash and return a parsed list of {question, expected_answer}."""
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    response_mime_type="application/json",
                ),
            )

            if not response or not response.text:
                logger.warning(f"  Empty response (attempt {attempt + 1})")
                time.sleep(2 ** attempt)
                continue

            text = response.text.strip()

            # Strip markdown fences if model adds them despite JSON mode
            if "```" in text:
                parts = text.split("```")
                text  = parts[1] if len(parts) > 1 else text
                if text.lower().startswith("json"):
                    text = text[4:].strip()

            parsed = json.loads(text)

            if isinstance(parsed, list):
                return parsed

            # Handle dict wrappers like {"questions": [...]}
            if isinstance(parsed, dict):
                for key in ("questions", "items", "data", "results"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]

            return []

        except json.JSONDecodeError as exc:
            logger.warning(f"  JSON parse error (attempt {attempt + 1}): {exc}")
        except Exception as exc:
            logger.warning(f"  API error (attempt {attempt + 1}): {exc}")

        if attempt < _MAX_RETRIES - 1:
            time.sleep(2 ** attempt)  # 1 s then 2 s

    return []


# ── Disk fallback ─────────────────────────────────────────────────────────────

def _load_kb_from_disk() -> List[dict]:
    """Load kb_all_documents.json when kb_docs not passed in-memory."""
    path = DATA_KB / "kb_all_documents.json"
    if not path.exists():
        logger.warning(f"KB file not found: {path}")
        return []
    logger.info(f"Loading KB docs from disk: {path.name}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

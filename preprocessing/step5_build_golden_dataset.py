"""
Step 5 – Golden dataset generation using 5 Gemini 3 Flash API keys.

Each of the 5 keys generates exactly 20 questions → 100 total.
A checkpoint JSON is saved after every key batch so a partial run can be
resumed the next day without re-spending already-used quota.

API keys are read from environment variables GOOGLE_API_KEY_1 … GOOGLE_API_KEY_5.

Distribution (100 total)
------------------------
    Layer              Easy  Medium  Hard  Total
    order               14      3      0     17
    category            10      7      3     20
    seller              10      3      0     13
    state                7      7      3     17
    month                7      3      3     13
    delivery_status      3      4      3     10
    cross_layer          0      3      7     10
    ─────────────────────────────────────────────
    Total               51     30     19    100

RAGAS fields   : question | ground_truth | contexts
DeepEval fields: input    | expected_output | context
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from google import genai
from google.genai import types

from .config import DATA_KB, DATA_GOLDEN, RANDOM_SEED

logger = logging.getLogger(__name__)

# ── Output schema ─────────────────────────────────────────────────────────────

GOLDEN_COLUMNS = [
    "question_id",
    "question",
    "expected_answer",
    "expected_context",
    "expected_source_ids",
    "question_type",
    "difficulty",
    "best_kb_layer",
]

# ── Key / batch configuration ─────────────────────────────────────────────────

QUERIES_PER_KEY  = 20      # each API key generates exactly this many queries
NUM_KEYS         = 5       # GOOGLE_API_KEY_1 … GOOGLE_API_KEY_5
_GEMINI_MODEL    = "gemini-3-flash-preview"
_DELAY_MIN_SEC   = 5
_DELAY_MAX_SEC   = 20
_CHECKPOINT_DIR  = DATA_GOLDEN   # checkpoints saved alongside the final CSV

# ── Target distribution ────────────────────────────────────────────────────────

_LAYER_TARGETS: Dict[str, Dict[str, int]] = {
    "order":           {"easy": 14, "medium": 3,  "hard": 0},
    "category":        {"easy": 10, "medium": 7,  "hard": 3},
    "seller":          {"easy": 10, "medium": 3,  "hard": 0},
    "state":           {"easy":  7, "medium": 7,  "hard": 3},
    "month":           {"easy":  7, "medium": 3,  "hard": 3},
    "delivery_status": {"easy":  3, "medium": 4,  "hard": 3},
    "cross_layer":     {"easy":  0, "medium": 3,  "hard": 7},
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

_LAYER_MAX_DOCS: Dict[str, int] = {
    "order":           60,
    "category":        74,
    "seller":          50,
    "state":           27,
    "month":           25,
    "delivery_status": 4,
}


# ── Job dataclass ─────────────────────────────────────────────────────────────

@dataclass
class _Job:
    layer:         str
    difficulty:    str
    doc:           dict
    doc2:          Optional[dict] = field(default=None)
    layer1_name:   Optional[str]  = field(default=None)
    layer2_name:   Optional[str]  = field(default=None)

    @property
    def best_kb_layer(self) -> str:
        if self.layer == "cross_layer":
            return f"{self.layer1_name}+{self.layer2_name}"
        return self.layer

    @property
    def question_type(self) -> str:
        return _DIFFICULTY_TO_QTYPE[self.difficulty]


# ── Public entry point ────────────────────────────────────────────────────────

def generate_golden_dataset(df: pd.DataFrame, kb_docs: List[dict]) -> pd.DataFrame:
    """
    Generate 100-question golden dataset using 5 Gemini API keys (20 each).

    Keys are read from GOOGLE_API_KEY_1 … GOOGLE_API_KEY_5 env vars.
    Checkpoints are saved after each key batch to allow next-day resumption.
    """
    keys = _read_api_keys()
    if not keys:
        logger.error(
            "No API keys found. Set environment variables:\n"
            "  $env:GOOGLE_API_KEY_1='key1'\n"
            "  $env:GOOGLE_API_KEY_2='key2'  … up to GOOGLE_API_KEY_5"
        )
        return pd.DataFrame(columns=GOLDEN_COLUMNS)

    if not kb_docs:
        kb_docs = _load_kb_from_disk()
    if not kb_docs:
        logger.error("No KB documents found. Run '--steps kb' first.")
        return pd.DataFrame(columns=GOLDEN_COLUMNS)

    logger.info(f"Loaded {len(kb_docs):,} KB documents")
    logger.info(f"Using {len(keys)} API key(s) × {QUERIES_PER_KEY} queries = "
                f"{len(keys) * QUERIES_PER_KEY} planned queries")

    random.seed(RANDOM_SEED)
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    docs_by_layer = _group_and_sample(kb_docs)
    all_jobs      = _build_job_list(docs_by_layer)

    logger.info(f"Built {len(all_jobs)} jobs total")

    # Split into key-sized batches
    batches: List[List[_Job]] = []
    for i in range(len(keys)):
        start = i * QUERIES_PER_KEY
        end   = start + QUERIES_PER_KEY
        batches.append(all_jobs[start:end])

    # Process each key batch
    all_rows: List[dict] = []
    for key_idx, (api_key, batch) in enumerate(zip(keys, batches), start=1):
        ckpt_path = _CHECKPOINT_DIR / f"golden_checkpoint_key{key_idx}.json"

        if ckpt_path.exists():
            logger.info(f"  [Key {key_idx}] Checkpoint found — loading, skipping API calls")
            with open(ckpt_path, encoding="utf-8") as fh:
                rows = json.load(fh)
        else:
            logger.info(f"  [Key {key_idx}] Processing {len(batch)} jobs ...")
            rows = _process_key_batch(api_key, key_idx, batch)
            _save_checkpoint(rows, ckpt_path)

        logger.info(f"  [Key {key_idx}] {len(rows)} rows collected "
                    f"({QUERIES_PER_KEY - len(rows)} skipped/empty)")
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No questions generated across all keys.")
        return pd.DataFrame(columns=GOLDEN_COLUMNS)

    df_out = pd.DataFrame(all_rows)
    df_out.insert(0, "question_id", [f"q{i + 1:03d}" for i in range(len(df_out))])
    for col in GOLDEN_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = ""

    df_out = df_out[GOLDEN_COLUMNS].reset_index(drop=True)

    easy   = (df_out["difficulty"] == "easy").sum()
    medium = (df_out["difficulty"] == "medium").sum()
    hard   = (df_out["difficulty"] == "hard").sum()

    logger.info(
        f"\n  Golden dataset complete:\n"
        f"    Total     : {len(df_out)}\n"
        f"    Easy      : {easy}\n"
        f"    Medium    : {medium}\n"
        f"    Hard      : {hard}"
    )
    return df_out


# ── API key reader ─────────────────────────────────────────────────────────────

def _read_api_keys() -> List[str]:
    """Read GOOGLE_API_KEY_1 … GOOGLE_API_KEY_5 from environment."""
    keys = []
    for i in range(1, NUM_KEYS + 1):
        k = os.environ.get(f"GOOGLE_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
        else:
            logger.warning(f"  GOOGLE_API_KEY_{i} not set — skipping key {i}")
    return keys


# ── Document grouping ─────────────────────────────────────────────────────────

def _group_and_sample(kb_docs: List[dict]) -> Dict[str, List[dict]]:
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
        logger.info(f"  Sampled {len(sampled[layer]):>3}/{len(docs):>5} docs for '{layer}'")

    return sampled


# ── Job list builder ──────────────────────────────────────────────────────────

def _build_job_list(docs_by_layer: Dict[str, List[dict]]) -> List[_Job]:
    """Build the full flat list of jobs, then shuffle so batches are mixed."""
    jobs: List[_Job] = []
    pairs = _CROSS_LAYER_PAIRS[:]
    pair_idx = 0

    for layer, targets in _LAYER_TARGETS.items():
        if layer == "cross_layer":
            random.shuffle(pairs)
            for difficulty, count in targets.items():
                if count == 0:
                    continue
                for _ in range(count):
                    l1, l2 = pairs[pair_idx % len(pairs)]
                    pair_idx += 1
                    d1 = docs_by_layer.get(l1, [])
                    d2 = docs_by_layer.get(l2, [])
                    if not d1 or not d2:
                        continue
                    jobs.append(_Job(
                        layer      = "cross_layer",
                        difficulty = difficulty,
                        doc        = random.choice(d1),
                        doc2       = random.choice(d2),
                        layer1_name= l1,
                        layer2_name= l2,
                    ))
        else:
            layer_docs = docs_by_layer.get(layer, [])
            if not layer_docs:
                continue
            for difficulty, count in targets.items():
                if count == 0:
                    continue
                shuffled = layer_docs[:]
                random.shuffle(shuffled)
                for i in range(count):
                    jobs.append(_Job(
                        layer      = layer,
                        difficulty = difficulty,
                        doc        = shuffled[i % len(shuffled)],
                    ))

    random.shuffle(jobs)   # mix layers across key batches
    return jobs


# ── Key batch processor ───────────────────────────────────────────────────────

def _process_key_batch(api_key: str, key_idx: int, jobs: List[_Job]) -> List[dict]:
    """Process exactly len(jobs) jobs with one Gemini API key."""
    client = genai.Client(api_key=api_key)
    rows: List[dict] = []

    for job_num, job in enumerate(jobs, start=1):
        logger.info(f"  [Key {key_idx}] Job {job_num:02d}/{len(jobs)}  "
                    f"layer={job.best_kb_layer}  difficulty={job.difficulty}")

        if job.layer == "cross_layer":
            prompt = _prompt_cross(
                job.doc,  job.layer1_name,
                job.doc2, job.layer2_name,
                job.difficulty,
            )
        else:
            prompt = _prompt_single(job.doc, job.difficulty, job.layer)

        item = _call_gemini(client, prompt, key_idx)

        if item:
            ctx_texts = [job.doc["text"]]
            ctx_ids   = [job.doc["id"]]
            if job.doc2:
                ctx_texts.append(job.doc2["text"])
                ctx_ids.append(job.doc2["id"])

            rows.append({
                "question":            item["question"],
                "expected_answer":     item["expected_answer"],
                "expected_context":    json.dumps(ctx_texts),
                "expected_source_ids": json.dumps(ctx_ids),
                "question_type":       job.question_type,
                "difficulty":          job.difficulty,
                "best_kb_layer":       job.best_kb_layer,
            })
        else:
            logger.warning(f"  [Key {key_idx}] Job {job_num:02d} returned empty — skipping")

        # Random delay AFTER each call (even failed ones preserve spacing)
        if job_num < len(jobs):
            _wait_between_calls()

    return rows


# ── Prompt builders ───────────────────────────────────────────────────────────

def _prompt_single(doc: dict, difficulty: str, layer: str) -> str:
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
            "Generate 1 FACTUAL question answerable by reading a specific value "
            "directly from the document.\n"
            "Good starters: 'What is the...', 'How many...', 'What was the...'"
        )
        answer_note = "Provide the specific value or short fact stated in the document."
    elif difficulty == "medium":
        style = (
            "Generate 1 ANALYTICAL question requiring interpretation of multiple "
            "fields — not just reading one number.\n"
            "Good starters: 'Based on the data...', "
            "'What does ... suggest about...', 'How does ... compare to...'"
        )
        answer_note = (
            "1-2 sentences interpreting the data, not just a raw value."
        )
    else:
        style = (
            "Generate 1 CHALLENGING question requiring synthesis of multiple "
            "metrics to draw a conclusion.\n"
            "Good starters: 'Given the performance metrics...', "
            "'What combination of factors...', 'What strategic insight...'"
        )
        answer_note = (
            "2-3 sentences synthesising data points and reaching a conclusion."
        )

    return f"""You are building an evaluation dataset for a Brazilian e-commerce RAG system.
The knowledge base document below describes {layer_desc}.

TASK
{style}

RULES
- Answerable using ONLY this document.
- No yes/no questions.
- Do NOT say "according to the document" — phrase as a natural business query.

DOCUMENT
{doc["text"]}

ANSWER NOTE
{answer_note}

Return ONLY a valid JSON object (no markdown):
{{"question": "...", "expected_answer": "..."}}"""


def _prompt_cross(
    doc1: dict, layer1: str,
    doc2: dict, layer2: str,
    difficulty: str,
) -> str:
    if difficulty == "medium":
        task = (
            "Generate 1 ANALYTICAL question connecting data from BOTH documents. "
            "It should link a fact from one layer to a fact from the other."
        )
        answer_note = "1-2 sentences using specific values from both documents."
    else:
        task = (
            "Generate 1 CHALLENGING question answerable ONLY by combining "
            "insights from BOTH documents — impossible with just one."
        )
        answer_note = (
            "2-3 sentences synthesising both documents, identifying a pattern "
            "or contrast, and drawing a conclusion."
        )

    return f"""You are building a multi-hop evaluation dataset for a Brazilian e-commerce RAG system.

TASK
{task}

RULES
- MUST require information from BOTH documents.
- No yes/no questions.
- Do NOT say "Document 1" or "Document 2" — phrase naturally.

DOCUMENT 1 (layer: {layer1})
{doc1["text"]}

DOCUMENT 2 (layer: {layer2})
{doc2["text"]}

ANSWER NOTE
{answer_note}

Return ONLY a valid JSON object (no markdown):
{{"question": "...", "expected_answer": "..."}}"""


# ── Gemini API call ───────────────────────────────────────────────────────────

def _call_gemini(client, prompt: str, key_idx: int) -> Optional[dict]:
    """
    Call Gemini once.  On 429 (quota exhausted) → return None immediately
    (don't retry — every retry burns precious daily quota).
    On other transient errors → retry once after a short wait.
    """
    for attempt in range(2):
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
                logger.warning(f"  [Key {key_idx}] Empty response (attempt {attempt + 1})")
                time.sleep(3)
                continue

            text = response.text.strip()
            if "```" in text:
                parts = text.split("```")
                text  = parts[1] if len(parts) > 1 else text
                if text.lower().startswith("json"):
                    text = text[4:].strip()

            parsed = json.loads(text)

            # Model sometimes returns a list with one item
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]

            if isinstance(parsed, dict) and "question" in parsed and "expected_answer" in parsed:
                return parsed

            logger.warning(f"  [Key {key_idx}] Unexpected JSON shape: {str(parsed)[:100]}")
            return None

        except json.JSONDecodeError as exc:
            logger.warning(f"  [Key {key_idx}] JSON parse error: {exc}")
            return None

        except Exception as exc:
            err_str = str(exc)

            # 429 daily quota — skip immediately, do NOT retry
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                retry_hint = _parse_retry_delay(err_str)
                if "PerDay" in err_str or "per_day" in err_str.lower() or retry_hint > 120:
                    logger.error(
                        f"  [Key {key_idx}] DAILY QUOTA EXHAUSTED — "
                        f"this key has no remaining calls today. "
                        f"Remaining jobs for this key will be skipped."
                    )
                    # Raise so the batch processor can break early
                    raise _DailyQuotaError(f"Key {key_idx} daily quota exhausted")
                else:
                    # Rate-limit (per-minute) — wait the suggested delay then retry
                    wait = min(retry_hint, 60)
                    logger.warning(
                        f"  [Key {key_idx}] Rate limit (RPM) — "
                        f"waiting {wait}s then retrying once ..."
                    )
                    time.sleep(wait)
                    continue

            logger.warning(f"  [Key {key_idx}] API error attempt {attempt + 1}: {exc}")
            if attempt == 0:
                time.sleep(5)

    return None


def _parse_retry_delay(error_str: str) -> int:
    """Extract retryDelay seconds from a 429 error string."""
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s", error_str)
    return int(match.group(1)) if match else 60


class _DailyQuotaError(Exception):
    """Raised when a key's daily API quota is exhausted."""


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _save_checkpoint(rows: List[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    logger.info(f"  Checkpoint saved: {path.name}  ({len(rows)} rows)")


# ── Delay helper ──────────────────────────────────────────────────────────────

def _wait_between_calls() -> None:
    delay = random.uniform(_DELAY_MIN_SEC, _DELAY_MAX_SEC)
    logger.info(f"  Waiting {delay:.1f}s before next call ...")
    time.sleep(delay)


# ── Disk fallback ─────────────────────────────────────────────────────────────

def _load_kb_from_disk() -> List[dict]:
    path = DATA_KB / "kb_all_documents.json"
    if not path.exists():
        logger.warning(f"KB file not found: {path}")
        return []
    logger.info(f"Loading KB docs from disk: {path.name}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

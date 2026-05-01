"""
End-to-end HyDE RAG evaluation: RAGAS + DeepEval -> Excel

All metrics are REFERENCE-BASED (zero LLM calls during evaluation).
Golden dataset provides expected_answer + expected_context as ground truth.
Each query makes exactly 2 Groq calls:
  Step 1a : Hypothetical document generation  (temp=0.7, 256 tokens)
  Step 2  : Final answer generation            (temp=0.1, 512 tokens)

HyDE flow per query:
  question
    └─► LLM (generate_hypothetical_doc, temp=0.7)  →  hypothetical_doc
              └─► ChromaDB.query(query_texts=[hypothetical_doc])
                        └─► top-k real docs
                                  └─► LLM (generate_answer, temp=0.1)  →  answer

RAGAS metrics  : Faithfulness, AnswerRelevancy, ContextPrecision,
                 ContextRecall, FactualCorrectness
DeepEval metrics: AnswerRelevancy, Faithfulness, ContextualPrecision,
                  ContextualRecall, ContextualRelevancy, Hallucination

Metric computation (no LLM judge, aligned with standard definitions):
  Faithfulness        : sentence-level support — fraction of answer sentences whose
                        meaningful tokens are >= 50% covered by retrieved context
  AnswerRelevancy     : TF-IDF cosine(generated_answer, question)
  ContextPrecision    : Average-Precision@k  (relevant = retrieved doc ID in expected_source_ids)
  ContextRecall       : token-recall(expected_answer tokens in combined_retrieved_contexts)
  FactualCorrectness  : ROUGE-L F1(generated_answer, expected_answer)
  DE AnswerRelevancy  : TF-IDF cosine(generated_answer, question)
  DE Faithfulness     : same sentence-level formula as RAGAS Faithfulness
  DE CtxPrecision     : AP@k — same formula as RAGAS ContextPrecision
  DE CtxRecall        : same token-recall formula as RAGAS ContextRecall
  DE CtxRelevancy     : mean TF-IDF cosine(each_retrieved_doc, question)
  DE Hallucination    : 1.0 - sentence_faithfulness

Excel output (5 sheets):
  1. Combined Results  -- every query as a row, every metric as a column, FINAL = mean
  2. RAG Responses     -- hypothetical_doc + retrieved contexts per query
  3. RAGAS Metrics     -- per-query RAGAS scores
  4. DeepEval Metrics  -- per-query DeepEval scores + computation method
  5. Summary           -- mean / min / max / std for all metrics

Usage:
    conda run --no-capture-output -n rag_eval python hyde_rag/evaluation/run_hyde_rag_eval.py
    conda run --no-capture-output -n rag_eval python hyde_rag/evaluation/run_hyde_rag_eval.py --limit 5
    conda run --no-capture-output -n rag_eval python hyde_rag/evaluation/run_hyde_rag_eval.py --limit 10 --batch-size 2
"""

import sys, json, re, argparse, logging, textwrap, threading, random, time
from pathlib import Path
from typing import Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from hyde_rag.implementation.ingestion import build_vector_store, get_collection
from hyde_rag.implementation.config import GROQ_API_KEYS, GROQ_MODEL, TOP_K, HYDE_TEMPERATURE, ANSWER_TEMPERATURE

GOLDEN_PATH = PROJECT_ROOT / "dataset" / "golden" / "golden_dataset.csv"
RAG_NAME    = "HyDE-RAG"


def _default_output() -> Path:
    from datetime import datetime
    ts = datetime.now().strftime("%d-%m-%Y_%I-%M%p")
    return PROJECT_ROOT / "hyde_rag" / "results" / f"{RAG_NAME}_{ts}.xlsx"


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── RAGAS metric column names ─────────────────────────────────────────────────
RAGAS_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_correctness(mode=f1)",
]

# ── DeepEval metric column names ──────────────────────────────────────────────
DE_COLS = [
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "ContextualRelevancyMetric",
    "HallucinationMetric",
]

RAGAS_DISPLAY = {
    "faithfulness":                   "RAGAS_Faithfulness",
    "answer_relevancy":               "RAGAS_AnswerRelevancy",
    "context_precision":              "RAGAS_ContextPrecision",
    "context_recall":                 "RAGAS_ContextRecall",
    "factual_correctness(mode=f1)":   "RAGAS_FactualCorrectness",
}
DE_DISPLAY = {
    "AnswerRelevancyMetric":     "DE_AnswerRelevancy",
    "FaithfulnessMetric":        "DE_Faithfulness",
    "ContextualPrecisionMetric": "DE_ContextualPrecision",
    "ContextualRecallMetric":    "DE_ContextualRecall",
    "ContextualRelevancyMetric": "DE_ContextualRelevancy",
    "HallucinationMetric":       "DE_Hallucination",
}

DE_REASONS = {
    "AnswerRelevancyMetric":     "TF-IDF cosine(generated_answer, question)",
    "FaithfulnessMetric":        "sentence-support(ans sentences in retrieved_contexts)",
    "ContextualPrecisionMetric": "AP@k (ID match vs expected_source_ids)",
    "ContextualRecallMetric":    "token-recall(expected_answer tokens in retrieved_contexts)",
    "ContextualRelevancyMetric": "mean TF-IDF cosine(each_retrieved_doc, question)",
    "HallucinationMetric":       "1.0 - sentence_faithfulness",
}

RAGAS_METHODS = {
    "faithfulness":                  "sentence-support(ans sentences in ctx)",
    "answer_relevancy":              "TF-IDF cosine(ans, question)",
    "context_precision":             "AP@k (ID match vs expected_source_ids)",
    "context_recall":                "token-recall(exp_ans tokens in ctx)",
    "factual_correctness(mode=f1)":  "ROUGE-L F1(ans, exp_ans)",
}


# ══════════════════════════════════════════════════════════════════════════════
# Parallel key routing  (Groq used for hypothetical doc + answer generation)
# ══════════════════════════════════════════════════════════════════════════════

_QUERIES_PER_KEY   = 20
_PRIMARY_KEY_COUNT = 5

_exhausted_keys: set = set()
_exhausted_lock      = threading.Lock()
_print_lock          = threading.Lock()


def _primary_key_idx(zero_based_pos: int, batch_size: int = None) -> int:
    sz = batch_size if batch_size else _QUERIES_PER_KEY
    return min(zero_based_pos // sz, _PRIMARY_KEY_COUNT - 1)


def _ordered_keys(primary_idx: int) -> List[str]:
    result: List[str] = []
    if primary_idx < len(GROQ_API_KEYS):
        k = GROQ_API_KEYS[primary_idx]
        with _exhausted_lock:
            if k not in _exhausted_keys:
                result.append(k)
    for i in range(_PRIMARY_KEY_COUNT, len(GROQ_API_KEYS)):
        k = GROQ_API_KEYS[i]
        with _exhausted_lock:
            if k not in _exhausted_keys:
                result.append(k)
    return result


def _groq_chat(primary_idx: int, messages: list, json_mode: bool = False,
               temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """
    Call Groq for RAG GENERATION only -- never called for evaluation.
    Supports configurable temperature and max_tokens for HyDE's two-call pattern:
      - Hypothetical doc : temperature=0.7, max_tokens=256
      - Final answer     : temperature=0.1, max_tokens=512
    Exhaustion policy:
      PERMANENT ban  : organization_restricted | tokens per day | HTTP 400/org
      TRANSIENT skip : per-minute/per-hour 429 (key stays in pool)
    """
    import groq as _g
    attempts = _ordered_keys(primary_idx)
    if not attempts:
        raise RuntimeError("No available Groq keys left")
    for key in attempts:
        try:
            kw = dict(model=GROQ_MODEL, messages=messages,
                      temperature=temperature, max_tokens=max_tokens)
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
            r = _g.Groq(api_key=key).chat.completions.create(**kw)
            return r.choices[0].message.content
        except Exception as e:
            err       = str(e)
            err_lower = err.lower()
            if "organization_restricted" in err_lower:
                with _exhausted_lock:
                    _exhausted_keys.add(key)
                with _print_lock:
                    log.warning("Key ...%s org-restricted -> permanently banned", key[-6:])
            elif "tokens per day" in err_lower or ("tpd" in err_lower and "limit" in err_lower):
                with _exhausted_lock:
                    _exhausted_keys.add(key)
                with _print_lock:
                    log.warning("Key ...%s daily TPD exhausted -> banned for session", key[-6:])
            elif "400" in err and "organization" in err_lower:
                with _exhausted_lock:
                    _exhausted_keys.add(key)
                with _print_lock:
                    log.warning("Key ...%s 400/org error -> permanently banned", key[-6:])
            else:
                with _print_lock:
                    log.warning("Key ...%s transient limit -> skipping (key stays in pool)", key[-6:])
    raise RuntimeError("All available Groq keys exhausted")


_HYDE_SYSTEM_PROMPT = (
    "You are an e-commerce data expert. Given a question, write a concise, "
    "factual passage that would directly answer it. The passage should read "
    "as if extracted from an e-commerce analytics database or report. "
    "Write only the passage — no preamble, no labels."
)

_E_COMMERCE_SYSTEM_PROMPT = (
    "You are a helpful e-commerce data assistant. "
    "Answer questions using only the provided context. "
    "If the answer cannot be found in the context, say so clearly."
)


def _generate_hypothetical_doc(question: str, primary_idx: int) -> str:
    """HyDE Step 1a — generate a hypothetical passage that would answer the question."""
    return _groq_chat(
        primary_idx,
        [
            {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {question}\n\nPassage:"},
        ],
        temperature=HYDE_TEMPERATURE,
        max_tokens=256,
    )


def _hyde_retrieve(question: str, primary_idx: int, top_k: int = TOP_K) -> tuple:
    """
    HyDE Step 1b — embed the hypothetical doc and retrieve real top-k docs.
    Returns (docs_list, hypothetical_doc_str).
    """
    hypothetical_doc = _generate_hypothetical_doc(question, primary_idx)
    col     = get_collection()
    results = col.query(query_texts=[hypothetical_doc], n_results=top_k)
    docs = [
        {
            "id":       results["ids"][0][i],
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]
    return docs, hypothetical_doc


def _generate_answer(question: str, docs: list, primary_idx: int) -> str:
    ctx = "\n\n".join(f"[Document {i+1}]\n{d['text']}" for i, d in enumerate(docs))
    return _groq_chat(
        primary_idx,
        [
            {"role": "system", "content": _E_COMMERCE_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"},
        ],
        temperature=ANSWER_TEMPERATURE,
        max_tokens=512,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Reference-based metric helpers  (zero LLM calls)
# ══════════════════════════════════════════════════════════════════════════════

def _token_list(text: str) -> list:
    return re.sub(r'[^\w\s]', '', (text or "").lower()).split()


def _token_set(text: str) -> set:
    return set(_token_list(text))


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    try:
        m = TfidfVectorizer(min_df=1).fit_transform([text_a or " ", text_b or " "])
        return float(_cosine_sim(m[0], m[1])[0][0])
    except Exception:
        return 0.0


def _split_sentences(text: str) -> list:
    return [s.strip() for s in re.split(r'[.!?]+', (text or "")) if s.strip()]


def _sentence_token_recall(sentence: str, context_tokens: set) -> float:
    s_toks = {t for t in _token_set(sentence) if len(t) > 1}
    if not s_toks:
        return 0.0
    return len(s_toks & context_tokens) / len(s_toks)


def _sentence_faithfulness(generated_answer: str, combined_context: str,
                            support_threshold: float = 0.5) -> float:
    sentences  = _split_sentences(generated_answer)
    if not sentences:
        return 0.0
    ctx_tokens = _token_set(combined_context)
    supported  = sum(
        1 for s in sentences
        if _sentence_token_recall(s, ctx_tokens) >= support_threshold
    )
    return supported / len(sentences)


def _lcs_length(a: list, b: list) -> int:
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if tok_a == tok_b else max(curr[j - 1], prev[j])
        prev = curr
    return prev[len(b)]


def _rouge_l(pred: str, ref: str) -> float:
    p_toks = _token_list(pred)
    r_toks = _token_list(ref)
    if not p_toks or not r_toks:
        return 0.0
    lcs = _lcs_length(p_toks, r_toks)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p_toks)
    rec  = lcs / len(r_toks)
    return 2 * prec * rec / (prec + rec)


# ══════════════════════════════════════════════════════════════════════════════
# Core metric computation
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(
    generated_answer: str,
    retrieved_contexts: list,
    question: str,
    expected_answer: str,
    retrieved_context_ids: list,
    expected_source_ids: list,
) -> dict:
    combined_retrieved = " ".join(retrieved_contexts)
    expected_id_set    = set(expected_source_ids)

    rel_flags = [doc_id in expected_id_set for doc_id in retrieved_context_ids]
    num_rel   = sum(rel_flags)

    faithfulness     = _sentence_faithfulness(generated_answer, combined_retrieved)
    answer_relevancy = _tfidf_cosine(generated_answer, question)

    if not retrieved_context_ids or num_rel == 0:
        context_precision = 0.0
    else:
        ap = sum(
            (sum(rel_flags[:k + 1]) / (k + 1)) * rel_flags[k]
            for k in range(len(rel_flags))
        ) / num_rel
        context_precision = float(ap)

    exp_tokens     = {t for t in _token_set(expected_answer) if len(t) > 1}
    ctx_tokens     = _token_set(combined_retrieved)
    context_recall = (len(exp_tokens & ctx_tokens) / len(exp_tokens)
                      if exp_tokens else 1.0)

    factual_correctness = _rouge_l(generated_answer, expected_answer)

    de_answer_relevancy     = _tfidf_cosine(generated_answer, question)
    de_faithfulness         = faithfulness
    de_contextual_precision = context_precision
    de_contextual_recall    = context_recall

    if retrieved_contexts:
        de_contextual_relevancy = sum(
            _tfidf_cosine(doc, question) for doc in retrieved_contexts
        ) / len(retrieved_contexts)
    else:
        de_contextual_relevancy = 0.0

    de_hallucination = max(0.0, 1.0 - faithfulness)

    return {
        "faithfulness":                  round(faithfulness,             4),
        "answer_relevancy":              round(answer_relevancy,          4),
        "context_precision":             round(context_precision,         4),
        "context_recall":                round(context_recall,            4),
        "factual_correctness(mode=f1)":  round(factual_correctness,       4),
        "AnswerRelevancyMetric":         round(de_answer_relevancy,       4),
        "FaithfulnessMetric":            round(de_faithfulness,           4),
        "ContextualPrecisionMetric":     round(de_contextual_precision,   4),
        "ContextualRecallMetric":        round(de_contextual_recall,      4),
        "ContextualRelevancyMetric":     round(de_contextual_relevancy,   4),
        "HallucinationMetric":           round(de_hallucination,          4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Terminal helpers  (thread-safe, buffered per query)
# ══════════════════════════════════════════════════════════════════════════════

_W = 90


def _flush(lines: List[str]):
    with _print_lock:
        for ln in lines:
            print(ln)
        sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_index():
    try:
        get_collection()
        log.info("ChromaDB index found.")
    except Exception:
        log.info("Building ChromaDB index (~2 min)...")
        build_vector_store()


def _parse_json_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return val
    try:
        r = json.loads(val)
        return r if isinstance(r, list) else [str(r)]
    except Exception:
        return [str(val)] if val else []


# ══════════════════════════════════════════════════════════════════════════════
# Single-query evaluator  (runs inside a worker thread)
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_one(row, global_idx: int, total: int, primary_idx: int) -> tuple:
    """
    Steps 1a-4 for one query. All terminal output buffered and flushed atomically.

    Step 1a : Groq hypothetical doc generation  (1 Groq call, temp=0.7)
    Step 1b : ChromaDB retrieval via hyp-doc embedding  (no LLM)
    Step 2  : Groq final answer generation      (1 Groq call, temp=0.1)
    Step 3  : RAGAS metrics                     (reference-based, no LLM)
    Step 4  : DeepEval metrics                  (reference-based, no LLM)

    Returns (rag_result_dict, ragas_row_dict, deepeval_row_dict).
    """
    qid         = row.question_id
    difficulty  = getattr(row, "difficulty", "-")
    qtype       = getattr(row, "question_type", "-")
    kb_layer    = getattr(row, "best_kb_layer", "-")
    question    = row.question
    expected    = row.expected_answer
    exp_ctx     = _parse_json_list(row.expected_context)
    exp_src_ids = _parse_json_list(getattr(row, "expected_source_ids", "[]"))

    buf: List[str] = [
        "",
        "=" * _W,
        f"  QUERY {global_idx}/{total}  |  {qid}  |  difficulty: {difficulty}"
        f"  |  type: {qtype}  |  Groq key: #{primary_idx + 1}",
        "=" * _W,
        f"  Question : {question}",
        "",
    ]

    # ── STEP 1a: HyDE Hypothetical Document Generation ────────────────────────
    buf.append(f"  +- STEP 1a . HyDE Hypothetical Doc Generation  (temp={HYDE_TEMPERATURE}) {'-' * 15}")
    hypothetical_doc = ""
    try:
        hypothetical_doc = _generate_hypothetical_doc(question, primary_idx)
        buf.append(f"  |  Hypothetical document generated ({len(hypothetical_doc)} chars):")
        for part in textwrap.wrap(hypothetical_doc, width=_W - 10):
            buf.append(f"  |      {part}")
        buf.append(f"  +{'-' * (_W - 2)}")
    except Exception as exc:
        buf += [f"  |  ERROR: {exc}", f"  +{'-' * (_W - 2)}",
                f"\n  [SKIP]  Query {global_idx}/{total} [{qid}] -- hypothetical doc generation error\n"]
        _flush(buf)
        empty = {
            "question_id": qid, "difficulty": difficulty, "question_type": qtype,
            "best_kb_layer": kb_layer, "question": question, "expected_answer": expected,
            "generated_answer": f"HYDE DOC GEN ERROR: {exc}", "hypothetical_doc": "",
            "retrieved_contexts": [], "expected_context": exp_ctx,
        }
        return empty, {"question_id": qid}, {"question_id": qid, "question": question}

    # ── STEP 1b: ChromaDB Retrieval via Hypothetical Doc Embedding ────────────
    buf.append(f"  +- STEP 1b . ChromaDB Retrieval (via hyp-doc embedding) {'-' * max(0, _W - 57)}")
    docs = []
    try:
        col     = get_collection()
        results = col.query(query_texts=[hypothetical_doc], n_results=TOP_K)
        docs = [
            {
                "id":       results["ids"][0][i],
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]
        buf.append(f"  |  Retrieved {len(docs)} documents  (cosine similarity, top-k={len(docs)})")
        for i, d in enumerate(docs, 1):
            dist = d.get("distance", 0)
            meta = d.get("metadata", {})
            src  = meta.get("source", meta.get("file", meta.get("id", "")))
            snip = d["text"].replace("\n", " ")[:110]
            buf.append(f"  |    [{i}] dist={dist:.4f}  src={src}")
            buf.append(f"  |         +- {snip}...")
        buf.append(f"  +{'-' * (_W - 2)}")
    except Exception as exc:
        buf += [f"  |  ERROR: {exc}", f"  +{'-' * (_W - 2)}",
                f"\n  [SKIP]  Query {global_idx}/{total} [{qid}] -- retrieval error\n"]
        _flush(buf)
        empty = {
            "question_id": qid, "difficulty": difficulty, "question_type": qtype,
            "best_kb_layer": kb_layer, "question": question, "expected_answer": expected,
            "generated_answer": f"RETRIEVAL ERROR: {exc}",
            "hypothetical_doc": hypothetical_doc,
            "retrieved_contexts": [], "expected_context": exp_ctx,
        }
        return empty, {"question_id": qid}, {"question_id": qid, "question": question}

    # ── STEP 2: Groq Final Answer Generation ─────────────────────────────────
    title2 = f"Groq Final Answer Generation  ({GROQ_MODEL}, temp={ANSWER_TEMPERATURE})  [key #{primary_idx + 1}]"
    buf.append(f"  +- STEP 2 . {title2} {'-' * max(0, _W - 12 - len(title2))}")
    answer = ""
    try:
        answer = _generate_answer(question, docs, primary_idx)
        buf.append(f"  |  Generated answer:")
        for part in textwrap.wrap(answer, width=_W - 10):
            buf.append(f"  |      {part}")
        buf.append(f"  +{'-' * (_W - 2)}")
    except Exception as exc:
        buf += [f"  |  ERROR: {exc}", f"  +{'-' * (_W - 2)}"]
        answer = f"GENERATION ERROR: {exc}"

    rag_r = {
        "question_id": qid, "difficulty": difficulty, "question_type": qtype,
        "best_kb_layer": kb_layer, "question": question, "expected_answer": expected,
        "hypothetical_doc":      hypothetical_doc,
        "generated_answer":      answer,
        "retrieved_contexts":    [d["text"] for d in docs],
        "retrieved_context_ids": [d.get("id", "") for d in docs],
        "expected_context":      exp_ctx,
        "expected_source_ids":   exp_src_ids,
    }

    # ── STEP 3: RAGAS Metrics (reference-based, no LLM) ──────────────────────
    buf.append(f"  +- STEP 3 . RAGAS Metrics  [reference-based, 0 LLM calls] {'-' * 20}")
    buf.append(f"  |  Computing RAGAS metrics vs golden dataset for [{qid}]...")
    buf.append(f"  |  Expected source IDs : {exp_src_ids}")
    buf.append(f"  |  Retrieved doc IDs   : {rag_r['retrieved_context_ids']}")
    buf.append(f"  |  Expected answer     : {str(expected)[:70]}")

    ragas_row = {"question_id": qid}
    m = {}
    try:
        m = _compute_metrics(
            answer,
            rag_r["retrieved_contexts"],
            question,
            str(expected),
            rag_r["retrieved_context_ids"],
            exp_src_ids,
        )

        buf += ["  |",
                f"  |  {'Metric':<38}  {'Score':>8}  Status  Method",
                f"  |  {'-'*38}  {'-'*8}  {'-'*6}  {'-'*30}"]
        for col in RAGAS_COLS:
            v   = m[col]
            st  = "PASS" if v >= 0.5 else "FAIL"
            mth = RAGAS_METHODS[col]
            buf.append(f"  |  {RAGAS_DISPLAY[col]:<38}  {v:>8.4f}  [{st}]  {mth}")
            ragas_row[col] = v

        buf += ["  |",
                f"  |  [OK]  RAGAS evaluation COMPLETED for query {global_idx}/{total} [{qid}]"]
    except Exception as exc:
        buf.append(f"  |  ERROR: {exc}")
        for col in RAGAS_COLS:
            ragas_row[col] = None
        buf.append(f"  |  [FAIL]  RAGAS evaluation FAILED for [{qid}]")
    buf.append(f"  +{'-' * (_W - 2)}")

    # ── STEP 4: DeepEval Metrics (reference-based, no LLM) ───────────────────
    buf.append(f"  +- STEP 4 . DeepEval Metrics  [reference-based, 0 LLM calls] {'-' * 17}")
    buf.append(f"  |  Computing DeepEval metrics vs golden dataset for [{qid}]...")

    de_row = {"question_id": qid, "question": question}
    try:
        buf += ["  |",
                f"  |  {'Metric':<38}  {'Score':>8}  Status  Computation",
                f"  |  {'-'*38}  {'-'*8}  {'-'*6}  {'-'*38}"]
        for col in DE_COLS:
            v      = m[col]
            st     = "PASS" if v >= 0.5 else "FAIL"
            reason = DE_REASONS[col]
            rsn    = reason[:55] + ("..." if len(reason) > 55 else "")
            buf.append(f"  |  {DE_DISPLAY[col]:<38}  {v:>8.4f}  [{st}]  {rsn}")
            de_row[col]             = v
            de_row[col + "_reason"] = reason

        buf += ["  |",
                f"  |  [OK]  DeepEval evaluation COMPLETED for query {global_idx}/{total} [{qid}]"]
    except Exception as exc:
        buf.append(f"  |  ERROR: {exc}")
        for col in DE_COLS:
            de_row[col]             = None
            de_row[col + "_reason"] = str(exc)
        buf.append(f"  |  [FAIL]  DeepEval evaluation FAILED for [{qid}]")

    buf += [
        f"  +{'-' * (_W - 2)}",
        "",
        f"  [END-TO-END COMPLETE]  Query {global_idx}/{total} [{qid}]"
        f"  |  key #{primary_idx + 1}  |  HyDE Doc -> Retrieval -> Answer -> RAGAS -> DeepEval",
        "",
    ]

    _flush(buf)
    return rag_r, ragas_row, de_row


# ══════════════════════════════════════════════════════════════════════════════
# Batch runner  (one thread per key group)
# ══════════════════════════════════════════════════════════════════════════════

def _run_batch(batch_rows: list, global_offsets: List[int], total: int,
               primary_idx: int) -> List[tuple]:
    results: List[tuple] = []
    n = len(batch_rows)
    for local_i, (row, g_idx) in enumerate(zip(batch_rows, global_offsets)):
        rag_r, ragas_r, de_r = _evaluate_one(row, g_idx, total, primary_idx)
        results.append((g_idx, rag_r, ragas_r, de_r))

        if local_i < n - 1:
            delay = random.uniform(5, 20)
            with _print_lock:
                print(f"  [sleep]  [key #{primary_idx + 1}] Sleeping {delay:.1f}s before next query...")
                sys.stdout.flush()
            time.sleep(delay)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Parallel evaluation coordinator
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(golden_df: pd.DataFrame, batch_size: int = 20,
                  sparse_n: int = None) -> tuple:
    """
    Split queries into key-aligned batches and run in parallel threads.
    Each query makes exactly 2 Groq calls (hypothetical doc + answer generation).
    Metrics are reference-based (zero LLM judge calls).
    """
    total_dataset = len(golden_df)
    eval_mode     = "reference-based (token-overlap + TF-IDF, no LLM judge)"
    rows_list     = list(golden_df.itertuples(index=False))

    batches: List[tuple] = []

    if sparse_n:
        for k in range(_PRIMARY_KEY_COUNT):
            start   = k * batch_size
            b_rows  = []
            b_off   = []
            for i in range(sparse_n):
                idx = start + i
                if idx < total_dataset:
                    b_rows.append(rows_list[idx])
                    b_off.append(idx + 1)
            if b_rows:
                batches.append((b_rows, b_off, k))
    else:
        for g_idx, row in enumerate(rows_list):
            p_idx = _primary_key_idx(g_idx, batch_size)
            while len(batches) <= p_idx:
                batches.append(([], [], len(batches)))
            batches[p_idx][0].append(row)
            batches[p_idx][1].append(g_idx + 1)

    n_selected = sum(len(b[0]) for b in batches)

    print(f"\n{'#' * _W}")
    print(f"#  END-TO-END HyDE RAG EVALUATION  --  {n_selected} queries selected"
          f"  (dataset size: {total_dataset})")
    print(f"#  LLM (2 calls per query)  : {GROQ_MODEL}")
    print(f"#    Step 1a : hypothetical doc generation  (temp={HYDE_TEMPERATURE}, max_tokens=256)")
    print(f"#    Step 2  : final answer generation       (temp={ANSWER_TEMPERATURE}, max_tokens=512)")
    print(f"#  Evaluation method        : {eval_mode}")
    if sparse_n:
        print(f"#  Mode                     : SPARSE  --  {sparse_n} queries per key"
              f"  (first {sparse_n} of each {batch_size}-query range)")
    else:
        print(f"#  Key layout               : 1-{batch_size} -> key#1 | "
              f"{batch_size+1}-{2*batch_size} -> key#2 | ...")
    print(f"#  Fallbacks                : keys #6-#{len(GROQ_API_KEYS)} auto-used when primary exhausted")
    print(f"#  Delay                    : random 5-20 s between consecutive queries in each batch")
    print(f"{'#' * _W}")

    print(f"\n  Batches: {len(batches)}  |  Parallel workers: {len(batches)}")
    for b_rows, b_off, p_idx in batches:
        positions = ", ".join(str(x) for x in b_off)
        print(f"    Groq key #{p_idx + 1}  ->  dataset positions [{positions}]  ({len(b_rows)} queries)")
    print()

    all_raw: List[tuple] = []
    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = {
            executor.submit(_run_batch, b_rows, b_off, total_dataset, p_idx): p_idx
            for (b_rows, b_off, p_idx) in batches
        }
        for fut in as_completed(futures):
            p_idx = futures[fut]
            try:
                batch_res = fut.result()
                all_raw.extend(batch_res)
                with _print_lock:
                    print(f"\n  [OK]  Batch key#{p_idx + 1} finished -- "
                          f"{len(batch_res)} queries done\n")
                    sys.stdout.flush()
            except Exception as exc:
                with _print_lock:
                    log.error("Batch key#%d raised an unhandled error: %s", p_idx + 1, exc)

    all_raw.sort(key=lambda x: x[0])
    rag_results   = [x[1] for x in all_raw]
    ragas_rows    = [x[2] for x in all_raw]
    deepeval_rows = [x[3] for x in all_raw]

    ragas_df    = pd.DataFrame(ragas_rows)
    deepeval_df = pd.DataFrame(deepeval_rows)

    print(f"\n{'#' * _W}")
    print(f"#  ALL {n_selected} QUERIES COMPLETE -- AGGREGATE SCORES")
    print(f"{'#' * _W}")

    sc = [c for c in RAGAS_COLS if c in ragas_df.columns]
    print(f"\n  RAGAS (mean across {n_selected} queries):")
    for c in sc:
        vals = pd.to_numeric(ragas_df[c], errors="coerce").dropna()
        print(f"    {RAGAS_DISPLAY[c]:<38}  {round(vals.mean(), 4) if len(vals) else 'N/A'}")

    dc = [c for c in DE_COLS if c in deepeval_df.columns]
    print(f"\n  DeepEval (mean across {n_selected} queries):")
    for c in dc:
        vals = pd.to_numeric(deepeval_df[c], errors="coerce").dropna()
        print(f"    {DE_DISPLAY[c]:<38}  {round(vals.mean(), 4) if len(vals) else 'N/A'}")
    print()

    return rag_results, ragas_df, deepeval_df, eval_mode


# ══════════════════════════════════════════════════════════════════════════════
# Excel export
# ══════════════════════════════════════════════════════════════════════════════

def export_excel(rag_results, ragas_df, deepeval_df, eval_mode, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ragas_metric_cols   = [c for c in ragas_df.columns   if c in RAGAS_COLS]
    deepeval_score_cols = [c for c in deepeval_df.columns if c in DE_COLS]

    # ── Sheet 1: Combined Results ─────────────────────────────────────────────
    combined_rows = []
    for r in rag_results:
        qid       = r["question_id"]
        ragas_row = ragas_df[ragas_df["question_id"] == qid]
        de_row    = deepeval_df[deepeval_df["question_id"] == qid]

        row = {
            "question_id":      qid,
            "difficulty":       r["difficulty"],
            "question_type":    r["question_type"],
            "best_kb_layer":    r["best_kb_layer"],
            "question":         r["question"],
            "expected_answer":  r["expected_answer"],
            "hypothetical_doc": r.get("hypothetical_doc", ""),
            "generated_answer": r["generated_answer"],
        }
        for col in ragas_metric_cols:
            display  = RAGAS_DISPLAY.get(col, f"RAGAS_{col}")
            row[display] = (round(float(ragas_row[col].values[0]), 4)
                            if len(ragas_row) and pd.notna(ragas_row[col].values[0]) else None)
        for col in deepeval_score_cols:
            display  = DE_DISPLAY.get(col, f"DE_{col}")
            row[display] = (round(float(de_row[col].values[0]), 4)
                            if len(de_row) and pd.notna(de_row[col].values[0]) else None)
        combined_rows.append(row)

    combined_df = pd.DataFrame(combined_rows)

    final = {
        "question_id": "FINAL (mean)", "difficulty": "", "question_type": "",
        "best_kb_layer": "", "question": "", "expected_answer": "",
        "hypothetical_doc": "", "generated_answer": "",
    }
    all_metric_display = (
        [RAGAS_DISPLAY[c] for c in ragas_metric_cols] +
        [DE_DISPLAY[c]    for c in deepeval_score_cols]
    )
    for col in all_metric_display:
        vals = pd.to_numeric(combined_df[col], errors="coerce").dropna()
        final[col] = round(vals.mean(), 4) if len(vals) else None
    combined_df = pd.concat([combined_df, pd.DataFrame([final])], ignore_index=True)

    # ── Sheet 5: Summary ─────────────────────────────────────────────────────
    summary_rows = [{
        "Framework": "META", "Metric": "Evaluation method",
        "Mean": eval_mode, "Min": "", "Max": "", "Std": "", "N": "",
    }, {
        "Framework": "META", "Metric": "RAG type",
        "Mean": "HyDE RAG (2 Groq calls per query: hypothetical doc + answer)",
        "Min": "", "Max": "", "Std": "", "N": "",
    }]
    for col in ragas_metric_cols:
        v = pd.to_numeric(ragas_df[col], errors="coerce").dropna()
        summary_rows.append({
            "Framework": "RAGAS",   "Metric": RAGAS_DISPLAY[col],
            "Mean": round(v.mean(), 4) if len(v) else None,
            "Min":  round(v.min(),  4) if len(v) else None,
            "Max":  round(v.max(),  4) if len(v) else None,
            "Std":  round(v.std(),  4) if len(v) else None,
            "N":    len(v),
        })
    for col in deepeval_score_cols:
        v = pd.to_numeric(deepeval_df[col], errors="coerce").dropna()
        summary_rows.append({
            "Framework": "DeepEval", "Metric": DE_DISPLAY[col],
            "Mean": round(v.mean(), 4) if len(v) else None,
            "Min":  round(v.min(),  4) if len(v) else None,
            "Max":  round(v.max(),  4) if len(v) else None,
            "Std":  round(v.std(),  4) if len(v) else None,
            "N":    len(v),
        })
    summary_df = pd.DataFrame(summary_rows)

    # ── Sheet 2: RAG Responses (includes hypothetical_doc) ───────────────────
    rag_rows = []
    for r in rag_results:
        ctx = r["retrieved_contexts"]
        rag_rows.append({
            "question_id":    r["question_id"],    "difficulty":    r["difficulty"],
            "question_type":  r["question_type"],  "best_kb_layer": r["best_kb_layer"],
            "question":       r["question"],        "expected_answer": r["expected_answer"],
            "hypothetical_doc": r.get("hypothetical_doc", ""),
            "generated_answer": r["generated_answer"],
            **{f"retrieved_context_{i+1}": ctx[i] if i < len(ctx) else "" for i in range(5)},
        })
    rag_sheet_df = pd.DataFrame(rag_rows)

    # ── Sheet 4: DeepEval ─────────────────────────────────────────────────────
    de_display_df = deepeval_df.copy()
    de_display_df = de_display_df.rename(
        columns={c: DE_DISPLAY.get(c, c) for c in deepeval_score_cols})

    # ── Write workbook ────────────────────────────────────────────────────────
    log.info("Writing Excel -> %s", output_path)
    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        combined_df.to_excel(   writer, sheet_name="Combined Results", index=False)
        rag_sheet_df.to_excel(  writer, sheet_name="RAG Responses",    index=False)
        ragas_df.to_excel(      writer, sheet_name="RAGAS Metrics",    index=False)
        de_display_df.to_excel( writer, sheet_name="DeepEval Metrics", index=False)
        summary_df.to_excel(    writer, sheet_name="Summary",          index=False)

    log.info("Saved -> %s", output_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",      type=int, default=None,
                        help="Max questions to evaluate (default: all 100)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Queries per Groq key group (default 20)")
    parser.add_argument("--sparse-n",   type=int, default=None,
                        help="Sparse mode: take first N queries from each key's range.")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output .xlsx path (auto-generated if omitted)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else _default_output()
    log.info("Output file: %s", output_path.name)

    golden_df = pd.read_csv(GOLDEN_PATH)
    if args.limit:
        golden_df = golden_df.head(args.limit)
    log.info("Dataset rows loaded: %d", len(golden_df))

    _ensure_index()
    rag_results, ragas_df, deepeval_df, eval_mode = evaluate_all(
        golden_df, batch_size=args.batch_size, sparse_n=args.sparse_n)
    export_excel(rag_results, ragas_df, deepeval_df, eval_mode, output_path)

    # ── Final console results table ───────────────────────────────────────────
    all_metric_cols = (
        [RAGAS_DISPLAY[c] for c in RAGAS_COLS    if c in ragas_df.columns] +
        [DE_DISPLAY[c]    for c in DE_COLS        if c in deepeval_df.columns]
    )
    combined_rows = []
    for r in rag_results:
        row = {"question_id": r["question_id"], "question": r["question"][:60]}
        rr  = ragas_df[ragas_df["question_id"] == r["question_id"]]
        dr  = deepeval_df[deepeval_df["question_id"] == r["question_id"]]
        for c in RAGAS_COLS:
            if c in ragas_df.columns:
                v = rr[c].values[0] if len(rr) else None
                row[RAGAS_DISPLAY[c]] = round(float(v), 3) if v is not None and pd.notna(v) else "-"
        for c in DE_COLS:
            if c in deepeval_df.columns:
                v = dr[c].values[0] if len(dr) else None
                row[DE_DISPLAY[c]] = round(float(v), 3) if v is not None and pd.notna(v) else "-"
        combined_rows.append(row)

    print_df = pd.DataFrame(combined_rows).set_index("question_id")
    means    = {}
    for col in all_metric_cols:
        if col in print_df.columns:
            vals = pd.to_numeric(print_df[col], errors="coerce").dropna()
            means[col] = round(vals.mean(), 3) if len(vals) else "-"
    means["question"] = "FINAL (mean)"
    print_df.loc["FINAL"] = means

    print("\n" + "=" * 100)
    print("EVALUATION RESULTS -- each row = one query, each column = one metric")
    print("=" * 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 55)
    print(print_df.to_string())
    print(f"\n  Eval method : {eval_mode}")
    print(f"  Results     : {output_path}\n")


if __name__ == "__main__":
    main()

"""
Unit tests for preprocessing/step5_build_golden_dataset.py (5-key rotation).

All tests that would call Gemini replace _call_gemini with a deterministic stub.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

import preprocessing.step5_build_golden_dataset as step5_mod
from preprocessing.step5_build_golden_dataset import (
    GOLDEN_COLUMNS,
    QUERIES_PER_KEY,
    NUM_KEYS,
    _DOC_TYPE_TO_LAYER,
    _group_and_sample,
    _build_job_list,
    _Job,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_kb_doc(doc_type: str, idx: int) -> dict:
    layer = _DOC_TYPE_TO_LAYER.get(doc_type, "unknown")
    return {
        "id":   f"{layer}_{idx}",
        "text": (
            f"Document Type: {doc_type}\n"
            f"Source ID: {layer}_{idx}\n"
            f"Total Orders: {100 + idx}\n"
            f"Average Review Score: 4.{idx % 10}\n"
            f"Late Delivery Rate: {idx % 20:.1f}%\n"
        ),
        "metadata": {"document_type": doc_type, "source_id": f"{layer}_{idx}"},
    }


def _make_all_layer_docs(n: int = 5) -> list:
    return [_make_kb_doc(dt, i) for dt in _DOC_TYPE_TO_LAYER for i in range(n)]


def _stub_gemini_ok(_client, _prompt: str, _key_idx: int):
    return {"question": "How many orders?", "expected_answer": "100"}


def _stub_gemini_empty(_client, _prompt: str, _key_idx: int):
    return None


# ── Constants ─────────────────────────────────────────────────────────────────

def test_golden_columns_complete():
    assert GOLDEN_COLUMNS == [
        "question_id", "question", "expected_answer",
        "expected_context", "expected_source_ids",
        "question_type", "difficulty", "best_kb_layer",
    ]


def test_queries_per_key_times_num_keys():
    assert QUERIES_PER_KEY * NUM_KEYS == 100


def test_layer_targets_sum_to_100():
    total = sum(
        v for targets in step5_mod._LAYER_TARGETS.values()
        for v in targets.values()
    )
    assert total == 100


# ── _read_api_keys ────────────────────────────────────────────────────────────

def test_read_api_keys_reads_env_vars(monkeypatch):
    for i in range(1, 6):
        monkeypatch.setenv(f"GOOGLE_API_KEY_{i}", f"fake-key-{i}")
    keys = step5_mod._read_api_keys()
    assert keys == [f"fake-key-{i}" for i in range(1, 6)]


def test_read_api_keys_empty_when_none_set(monkeypatch):
    for i in range(1, 6):
        monkeypatch.delenv(f"GOOGLE_API_KEY_{i}", raising=False)
    keys = step5_mod._read_api_keys()
    assert keys == []


# ── _group_and_sample ─────────────────────────────────────────────────────────

def test_group_and_sample_covers_all_layers():
    grouped = _group_and_sample(_make_all_layer_docs(5))
    assert set(grouped.keys()) == set(_DOC_TYPE_TO_LAYER.values())


def test_group_and_sample_respects_max_docs(monkeypatch):
    monkeypatch.setattr(step5_mod, "_LAYER_MAX_DOCS", {k: 2 for k in _DOC_TYPE_TO_LAYER.values()})
    grouped = _group_and_sample(_make_all_layer_docs(10))
    for layer, docs in grouped.items():
        assert len(docs) <= 2


# ── _build_job_list ───────────────────────────────────────────────────────────

def test_build_job_list_returns_100_jobs():
    docs = _group_and_sample(_make_all_layer_docs(15))
    jobs = _build_job_list(docs)
    assert len(jobs) == 100


def test_build_job_list_all_have_docs():
    docs = _group_and_sample(_make_all_layer_docs(15))
    for job in _build_job_list(docs):
        assert job.doc is not None


def test_build_job_list_cross_layer_has_two_docs():
    docs = _group_and_sample(_make_all_layer_docs(15))
    cross_jobs = [j for j in _build_job_list(docs) if j.layer == "cross_layer"]
    for job in cross_jobs:
        assert job.doc2 is not None
        assert job.layer1_name is not None
        assert job.layer2_name is not None


# ── _Job dataclass ────────────────────────────────────────────────────────────

def test_job_best_kb_layer_single():
    doc = _make_kb_doc("category_level", 0)
    job = _Job(layer="category", difficulty="easy", doc=doc)
    assert job.best_kb_layer == "category"


def test_job_best_kb_layer_cross():
    d1 = _make_kb_doc("category_level", 0)
    d2 = _make_kb_doc("customer_state_level", 0)
    job = _Job(layer="cross_layer", difficulty="hard", doc=d1, doc2=d2,
               layer1_name="category", layer2_name="state")
    assert job.best_kb_layer == "category+state"


def test_job_question_type_mapping():
    d = _make_kb_doc("order_level", 0)
    assert _Job(layer="order", difficulty="easy",   doc=d).question_type == "factual"
    assert _Job(layer="order", difficulty="medium", doc=d).question_type == "analytical"
    assert _Job(layer="order", difficulty="hard",   doc=d).question_type == "comparison"


# ── Full generation with stubbed Gemini ───────────────────────────────────────

@pytest.fixture
def tiny_run(monkeypatch, tmp_path):
    """Patch to use 2 keys × 2 queries each for fast testing."""
    monkeypatch.setattr(step5_mod, "QUERIES_PER_KEY", 2)
    monkeypatch.setattr(step5_mod, "NUM_KEYS",        2)
    monkeypatch.setattr(step5_mod, "_LAYER_TARGETS", {
        "order":    {"easy": 2, "medium": 0, "hard": 0},
        "category": {"easy": 2, "medium": 0, "hard": 0},
    })
    monkeypatch.setattr(step5_mod, "_DELAY_MIN_SEC", 0)
    monkeypatch.setattr(step5_mod, "_DELAY_MAX_SEC", 0)
    monkeypatch.setattr(step5_mod, "_CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(step5_mod, "_call_gemini", _stub_gemini_ok)
    monkeypatch.setattr(step5_mod.genai, "Client", lambda api_key: object())
    for i in range(1, 3):
        monkeypatch.setenv(f"GOOGLE_API_KEY_{i}", f"fake-{i}")


def test_generate_returns_dataframe(monkeypatch, tiny_run):
    result = step5_mod.generate_golden_dataset(pd.DataFrame(), _make_all_layer_docs(5))
    assert isinstance(result, pd.DataFrame)


def test_generate_has_correct_columns(monkeypatch, tiny_run):
    result = step5_mod.generate_golden_dataset(pd.DataFrame(), _make_all_layer_docs(5))
    assert list(result.columns) == GOLDEN_COLUMNS


def test_generate_question_ids_sequential(monkeypatch, tiny_run):
    result = step5_mod.generate_golden_dataset(pd.DataFrame(), _make_all_layer_docs(5))
    if len(result) > 0:
        assert result["question_id"].iloc[0] == "q001"
        assert result["question_id"].is_unique


def test_generate_expected_context_valid_json(monkeypatch, tiny_run):
    result = step5_mod.generate_golden_dataset(pd.DataFrame(), _make_all_layer_docs(5))
    for ctx in result["expected_context"]:
        parsed = json.loads(ctx)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1


def test_generate_difficulty_values_valid(monkeypatch, tiny_run):
    result = step5_mod.generate_golden_dataset(pd.DataFrame(), _make_all_layer_docs(5))
    assert set(result["difficulty"].unique()).issubset({"easy", "medium", "hard"})


def test_generate_no_api_keys_returns_empty(monkeypatch, tmp_path):
    for i in range(1, 6):
        monkeypatch.delenv(f"GOOGLE_API_KEY_{i}", raising=False)
    result = step5_mod.generate_golden_dataset(pd.DataFrame(), [])
    assert len(result) == 0
    assert list(result.columns) == GOLDEN_COLUMNS


def test_checkpoint_loaded_on_second_run(monkeypatch, tmp_path):
    """Second run should load checkpoints instead of calling Gemini."""
    monkeypatch.setattr(step5_mod, "QUERIES_PER_KEY", 2)
    monkeypatch.setattr(step5_mod, "NUM_KEYS",        1)
    monkeypatch.setattr(step5_mod, "_LAYER_TARGETS", {
        "order": {"easy": 2, "medium": 0, "hard": 0},
    })
    monkeypatch.setattr(step5_mod, "_DELAY_MIN_SEC", 0)
    monkeypatch.setattr(step5_mod, "_DELAY_MAX_SEC", 0)
    monkeypatch.setattr(step5_mod, "_CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(step5_mod.genai, "Client", lambda api_key: object())
    monkeypatch.setenv("GOOGLE_API_KEY_1", "fake")

    # Pre-write a checkpoint for key 1
    ckpt_rows = [
        {
            "question": "Pre-saved question?", "expected_answer": "Pre-saved answer",
            "expected_context": '["ctx"]', "expected_source_ids": '["id"]',
            "question_type": "factual", "difficulty": "easy", "best_kb_layer": "order",
        }
    ]
    (tmp_path / "golden_checkpoint_key1.json").write_text(
        json.dumps(ckpt_rows), encoding="utf-8"
    )

    # _call_gemini should never be called
    call_count = {"n": 0}
    def counting_stub(*a, **kw):
        call_count["n"] += 1
        return {"question": "q", "expected_answer": "a"}

    monkeypatch.setattr(step5_mod, "_call_gemini", counting_stub)
    step5_mod.generate_golden_dataset(pd.DataFrame(), _make_all_layer_docs(5))
    assert call_count["n"] == 0

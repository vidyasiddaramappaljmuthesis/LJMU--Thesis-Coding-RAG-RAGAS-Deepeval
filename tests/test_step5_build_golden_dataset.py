"""
Unit tests for preprocessing/step5_build_golden_dataset.py.

Real generation requires GOOGLE_API_KEY + network access, so all tests
that would call Gemini use monkeypatching to replace _call_gemini with a
deterministic stub.  The API-key-missing path is tested without any mocking.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

import preprocessing.step5_build_golden_dataset as step5_mod
from preprocessing.step5_build_golden_dataset import (
    generate_golden_dataset,
    GOLDEN_COLUMNS,
    _group_and_sample,
    _DOC_TYPE_TO_LAYER,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_kb_doc(doc_type: str, idx: int) -> dict:
    """Minimal KB document that matches the structure produced by step4."""
    layer = _DOC_TYPE_TO_LAYER.get(doc_type, "unknown")
    return {
        "id":   f"{layer}_{idx}",
        "text": (
            f"Document Type: {doc_type.replace('_', ' ').title()}\n"
            f"Source ID: {layer}_{idx}\n"
            f"Total Orders: 100\n"
            f"Average Review Score: 4.20\n"
            f"Late Delivery Rate: 8.50%\n"
        ),
        "metadata": {
            "document_type": doc_type,
            "source_id":     f"{layer}_{idx}",
        },
    }


def _make_all_layer_docs(n_per_layer: int = 3) -> list:
    """Return a small but complete set of KB docs covering all 6 layers."""
    doc_types = list(_DOC_TYPE_TO_LAYER.keys())
    return [_make_kb_doc(dt, i) for dt in doc_types for i in range(n_per_layer)]


def _stub_call_gemini(_client, _prompt: str):
    """Deterministic stub that always returns one valid Q&A pair."""
    return [{"question": "What is the total orders?", "expected_answer": "100"}]


# ── Schema / constant tests (no API) ─────────────────────────────────────────

def test_golden_columns_complete():
    expected = [
        "question_id", "question", "expected_answer",
        "expected_context", "expected_source_ids",
        "question_type", "difficulty", "best_kb_layer",
    ]
    assert GOLDEN_COLUMNS == expected


def test_doc_type_to_layer_has_six_entries():
    assert len(_DOC_TYPE_TO_LAYER) == 6


# ── No API key → empty DataFrame ─────────────────────────────────────────────

def test_returns_empty_when_no_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = generate_golden_dataset(pd.DataFrame(), [])
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == GOLDEN_COLUMNS
    assert len(result) == 0


# ── _group_and_sample ─────────────────────────────────────────────────────────

def test_group_and_sample_covers_all_layers():
    kb_docs = _make_all_layer_docs(n_per_layer=5)
    grouped = _group_and_sample(kb_docs)
    expected_layers = set(_DOC_TYPE_TO_LAYER.values())
    assert set(grouped.keys()) == expected_layers


def test_group_and_sample_respects_max_docs(monkeypatch):
    monkeypatch.setattr(step5_mod, "_LAYER_MAX_DOCS", {k: 2 for k in _DOC_TYPE_TO_LAYER.values()})
    kb_docs = _make_all_layer_docs(n_per_layer=10)
    grouped = _group_and_sample(kb_docs)
    for layer, docs in grouped.items():
        assert len(docs) <= 2, f"Layer '{layer}' has {len(docs)} docs, expected <= 2"


# ── Full generation with stubbed Gemini ───────────────────────────────────────

@pytest.fixture
def small_targets(monkeypatch):
    """Override targets to 1 question per difficulty to keep tests fast."""
    small = {
        "order":           {"easy": 1, "medium": 1, "hard": 0},
        "category":        {"easy": 1, "medium": 0, "hard": 0},
        "seller":          {"easy": 1, "medium": 0, "hard": 0},
        "state":           {"easy": 1, "medium": 0, "hard": 0},
        "month":           {"easy": 1, "medium": 0, "hard": 0},
        "delivery_status": {"easy": 1, "medium": 0, "hard": 0},
        "cross_layer":     {"easy": 0, "medium": 1, "hard": 0},
    }
    monkeypatch.setattr(step5_mod, "_LAYER_TARGETS", small)
    # Zero out random delay so tests complete instantly
    monkeypatch.setattr(step5_mod, "_DELAY_MIN_SEC", 0)
    monkeypatch.setattr(step5_mod, "_DELAY_MAX_SEC", 0)


def test_generate_returns_dataframe(monkeypatch, small_targets):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(step5_mod, "_call_gemini", _stub_call_gemini)
    # Stub genai.Client so no real HTTP call is made
    monkeypatch.setattr(step5_mod.genai, "Client", lambda **kw: object())

    kb_docs = _make_all_layer_docs(n_per_layer=5)
    result  = generate_golden_dataset(pd.DataFrame(), kb_docs)
    assert isinstance(result, pd.DataFrame)


def test_generate_has_correct_columns(monkeypatch, small_targets):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(step5_mod, "_call_gemini", _stub_call_gemini)
    monkeypatch.setattr(step5_mod.genai, "Client", lambda **kw: object())

    kb_docs = _make_all_layer_docs(n_per_layer=5)
    result  = generate_golden_dataset(pd.DataFrame(), kb_docs)
    assert list(result.columns) == GOLDEN_COLUMNS


def test_generate_question_ids_sequential(monkeypatch, small_targets):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(step5_mod, "_call_gemini", _stub_call_gemini)
    monkeypatch.setattr(step5_mod.genai, "Client", lambda **kw: object())

    kb_docs = _make_all_layer_docs(n_per_layer=5)
    result  = generate_golden_dataset(pd.DataFrame(), kb_docs)
    if len(result) > 0:
        assert result["question_id"].iloc[0] == "q001"
        assert result["question_id"].is_unique


def test_generate_expected_context_is_valid_json(monkeypatch, small_targets):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(step5_mod, "_call_gemini", _stub_call_gemini)
    monkeypatch.setattr(step5_mod.genai, "Client", lambda **kw: object())

    kb_docs = _make_all_layer_docs(n_per_layer=5)
    result  = generate_golden_dataset(pd.DataFrame(), kb_docs)
    for ctx in result["expected_context"]:
        parsed = json.loads(ctx)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1


def test_generate_difficulty_values_valid(monkeypatch, small_targets):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(step5_mod, "_call_gemini", _stub_call_gemini)
    monkeypatch.setattr(step5_mod.genai, "Client", lambda **kw: object())

    kb_docs = _make_all_layer_docs(n_per_layer=5)
    result  = generate_golden_dataset(pd.DataFrame(), kb_docs)
    assert set(result["difficulty"].unique()).issubset({"easy", "medium", "hard"})


def test_generate_question_type_values_valid(monkeypatch, small_targets):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(step5_mod, "_call_gemini", _stub_call_gemini)
    monkeypatch.setattr(step5_mod.genai, "Client", lambda **kw: object())

    kb_docs = _make_all_layer_docs(n_per_layer=5)
    result  = generate_golden_dataset(pd.DataFrame(), kb_docs)
    assert set(result["question_type"].unique()).issubset(
        {"factual", "analytical", "comparison"}
    )


def test_generate_no_empty_questions(monkeypatch, small_targets):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(step5_mod, "_call_gemini", _stub_call_gemini)
    monkeypatch.setattr(step5_mod.genai, "Client", lambda **kw: object())

    kb_docs = _make_all_layer_docs(n_per_layer=5)
    result  = generate_golden_dataset(pd.DataFrame(), kb_docs)
    assert result["question"].str.strip().ne("").all()
    assert result["expected_answer"].str.strip().ne("").all()

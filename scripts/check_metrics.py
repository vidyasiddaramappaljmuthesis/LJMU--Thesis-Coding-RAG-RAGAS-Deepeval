"""
Dev utility — checks which RAGAS non-LLM metrics are importable.

Run this once after installing requirements to verify the evaluation
dependencies.  Prints OK or a warning for each metric family.

Usage::

    python scripts/check_metrics.py
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Core non-LLM context metrics (required by all evaluation runners) ─────────
try:
    from ragas.metrics import NonLLMContextPrecision, NonLLMContextRecall
    log.info("NonLLMContextPrecision, NonLLMContextRecall: OK")
except ImportError as exc:
    log.error("NonLLM context metrics not available: %s", exc)

# ── Optional string-similarity metrics ───────────────────────────────────────
try:
    from ragas.metrics import BleuScore, RougeScore
    log.info("BleuScore, RougeScore: OK")
except ImportError as exc:
    log.warning("BleuScore/RougeScore not available: %s", exc)

try:
    from ragas.metrics import NonLLMStringSimilarity
    log.info("NonLLMStringSimilarity: OK")
except ImportError as exc:
    log.warning("NonLLMStringSimilarity not available: %s", exc)

# ── FactualCorrectness (used by all evaluation runners) ───────────────────────
try:
    from ragas.metrics import FactualCorrectness
    log.info("FactualCorrectness: OK")
except ImportError as exc:
    log.warning("FactualCorrectness not available: %s", exc)

# ── Version info ──────────────────────────────────────────────────────────────
import ragas
log.info("ragas version: %s", ragas.__version__)

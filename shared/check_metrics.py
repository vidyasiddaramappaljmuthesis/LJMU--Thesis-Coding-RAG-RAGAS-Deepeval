"""
Diagnostic script — checks which RAGAS non-LLM metrics are importable in this environment.

  python shared/check_metrics.py
"""
from ragas.metrics import NonLLMContextPrecision, NonLLMContextRecall
print("NonLLM context metrics: OK")
try:
    from ragas.metrics import BleuScore, RougeScore
    print("BleuScore, RougeScore: OK")
except ImportError as e:
    print("BleuScore/RougeScore not available:", e)
try:
    from ragas.metrics import NonLLMStringSimilarity
    print("NonLLMStringSimilarity: OK")
except ImportError as e:
    print("NonLLMStringSimilarity:", e)
try:
    from ragas.metrics import FactualCorrectness
    print("FactualCorrectness: OK")
except ImportError as e:
    print("FactualCorrectness:", e)
import ragas; print("ragas version:", ragas.__version__)

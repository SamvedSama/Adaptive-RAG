"""
router.py — Budget-Aware ML Micro-Router
Owner: Samved Jain

Classifies an incoming query based on its semantic embedding AND a numerical
system budget, returning a strict routing decision:

    "Multi_Hop_FAISS"  — optimal path  (budget ~1.0)  dense retrieval + reranking
    "Single_Hop_BM25"  — stressed path (budget ~0.5)  fast sparse retrieval
    "Direct_LLM"       — failsafe path (budget ~0.1)  bypass retrieval entirely

Artifact compatibility:
    Handles BOTH artifact shapes produced by train_router.py:
      • New (dict)  → {"model": clf, "encoder_name": ..., "feature_dim": ..., ...}
      • Legacy (clf) → raw RandomForestClassifier (old train_router.py output)

Public API (used by adaptive_pipeline.py):
    router = QueryRouter()
    label  = router.route(query, budget)          # → str

Extended API (used by ablation_runner.py):
    result = router.route_full(query, budget)     # → RoutingResult
    probs  = router.predict_proba(query, budget)  # → dict[str, float]

Module singleton (used by Streamlit app.py):
    router = get_router()                         # cached; no re-loading on reruns
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

# ── Custom Wrapper ─────────────────────────────────────────────────────────────
class StringLabelXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.clf = XGBClassifier(**kwargs)
        self.le_ = LabelEncoder()
        
    def fit(self, X, y, **kwargs):
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        # Forward kwargs (e.g. sample_weight) to the underlying XGBClassifier
        self.clf.fit(X, y_enc, **kwargs)
        return self
        
    def predict(self, X):
        return self.le_.inverse_transform(self.clf.predict(X))
        
    def predict_proba(self, X):
        return self.clf.predict_proba(X)

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

VALID_LABELS      = frozenset({"Multi_Hop_FAISS", "Single_Hop_BM25", "Direct_LLM"})
BUDGET_MIN        = 0.0
BUDGET_MAX        = 1.0
_DEFAULT_MODEL    = Path("data/router_model.pkl")
_DEFAULT_ENCODER  = "sentence-transformers/all-MiniLM-L6-v2"

# When the model is unavailable, degrade to the safest (fastest) path
_FALLBACK_LABEL   = "Direct_LLM"


# ── Result container ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoutingResult:
    label:         str               # chosen route
    confidence:    float             # probability of the chosen label
    probabilities: dict[str, float]  # full distribution over all labels
    latency_ms:    float             # wall-clock time for this call (ms)
    method:        str               # "ml_router" | "fallback"
    fallback:      bool              # True if model was unavailable


# ── QueryRouter ────────────────────────────────────────────────────────────────

class QueryRouter:
    """
    Stateful router — instantiate once, call many times.

    SentenceTransformer.encode() and sklearn predict() are both safe for
    concurrent inference, so one instance can be shared across Streamlit reruns.
    """

    def __init__(
        self,
        model_path: Path | str = _DEFAULT_MODEL,
    ) -> None:
        self._model_path  = Path(model_path)
        self._classifier  = None
        self._encoder: SentenceTransformer | None = None
        self._feature_dim: int | None = None
        self._labels: list[str] = sorted(VALID_LABELS)
        self._loaded      = False

        self._load_artifact()

    # ── Public API ─────────────────────────────────────────────────────────────

    def route(self, query: str, budget: float) -> str:
        """
        Simplest call: return the routing label for (query, budget).
        Falls back to Direct_LLM if the model is unavailable.
        """
        return self._route_internal(query, budget).label

    def route_full(self, query: str, budget: float) -> RoutingResult:
        """Return the full RoutingResult including probabilities and latency."""
        return self._route_internal(query, budget)

    def predict_proba(self, query: str, budget: float) -> dict[str, float]:
        """
        Return a probability dict for every routing label.
        Used by ablation_runner.py for soft decision analysis.
        """
        return self._route_internal(query, budget).probabilities

    # Back-compat alias for any code still calling .classify()
    def classify(self, query: str, budget: float) -> tuple[str, float, str]:
        """
        Legacy interface — returns (label, confidence, method).
        Prefer route() or route_full() for new code.
        """
        r = self._route_internal(query, budget)
        return r.label, r.confidence, r.method

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_path(self) -> Path:
        return self._model_path

    # ── Artifact Loading ───────────────────────────────────────────────────────

    def _load_artifact(self) -> None:
        """
        Load the joblib artifact from train_router.py.

        Handles two shapes:
          • New dict   → {"model": clf, "encoder_name": str, "feature_dim": int, ...}
          • Legacy clf → raw RandomForestClassifier

        Never raises — logs errors and leaves self._loaded = False so every
        call site falls back gracefully instead of crashing at startup.
        """
        if not self._model_path.exists():
            log.error(
                "Router model not found at '%s'. Run train_router.py first.",
                self._model_path,
            )
            return

        try:
            raw = joblib.load(self._model_path)
        except Exception as exc:
            log.error("Failed to deserialise artifact '%s': %s", self._model_path, exc)
            return

        # ── Compatibility branch ───────────────────────────────────────────────
        if isinstance(raw, dict) and "model" in raw:
            self._classifier  = raw["model"]
            encoder_name      = raw.get("encoder_name", _DEFAULT_ENCODER)
            self._feature_dim = raw.get("feature_dim")
            self._labels      = raw.get("labels", sorted(VALID_LABELS))
            log.info(
                "Loaded dict artifact | encoder='%s' | feature_dim=%s | labels=%s",
                encoder_name, self._feature_dim, self._labels,
            )
        else:
            # Legacy: artifact IS the raw classifier
            self._classifier = raw
            encoder_name     = _DEFAULT_ENCODER
            log.warning(
                "Loaded legacy (raw-classifier) artifact from '%s'. "
                "Re-run train_router.py to upgrade to the provenance-tracked format.",
                self._model_path,
            )

        # ── Classifier interface check ─────────────────────────────────────────
        for attr in ("predict", "predict_proba", "classes_"):
            if not hasattr(self._classifier, attr):
                log.error(
                    "Artifact is missing '%s' — likely corrupt. Re-run train_router.py.",
                    attr,
                )
                return

        # ── Load encoder ───────────────────────────────────────────────────────
        try:
            log.info("Loading sentence encoder '%s' ...", encoder_name)
            self._encoder = SentenceTransformer(encoder_name)
        except Exception as exc:
            log.error("Failed to load encoder '%s': %s", encoder_name, exc)
            return

        self._loaded = True
        log.info(
            "QueryRouter ready | model='%s' | classes=%s",
            self._model_path.name,
            list(self._classifier.classes_),
        )

    # ── Internal Routing ───────────────────────────────────────────────────────

    def _route_internal(self, query: str, budget: float) -> RoutingResult:
        query, budget = self._validate_inputs(query, budget)
        t0 = time.perf_counter()

        if not self._loaded:
            log.warning("Router not loaded — returning fallback label '%s'.", _FALLBACK_LABEL)
            return self._make_fallback(t0)

        try:
            vec       = self._build_feature(query, budget)
            label     = str(self._classifier.predict(vec)[0])
            proba_arr = self._classifier.predict_proba(vec)[0]
            proba_map = {
                str(cls): float(p)
                for cls, p in zip(self._classifier.classes_, proba_arr)
            }
            confidence = float(proba_map.get(label, 0.0))

            if label not in VALID_LABELS:
                log.error("Model returned unexpected label '%s' — falling back.", label)
                return self._make_fallback(t0)

            latency_ms = (time.perf_counter() - t0) * 1000
            log.debug(
                "route(budget=%.2f) → %-18s [%.1f ms]  probs=%s",
                budget, label, latency_ms,
                {k: f"{v:.3f}" for k, v in proba_map.items()},
            )
            return RoutingResult(
                label=label,
                confidence=confidence,
                probabilities=proba_map,
                latency_ms=latency_ms,
                method="ml_router",
                fallback=False,
            )

        except Exception as exc:  # noqa: BLE001 — never let inference crash the app
            log.error("Inference error: %s — falling back.", exc, exc_info=True)
            return self._make_fallback(t0)

    # ── Feature Engineering ────────────────────────────────────────────────────

    def _build_feature(self, query: str, budget: float) -> np.ndarray:
        """
        Encode query → L2-normalised embedding, append scalar budget.
        Returns shape (1, embedding_dim + 1) for sklearn.
        """
        embedding = self._encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        budget_col  = np.array([[budget]], dtype=np.float32)
        feature_vec = np.hstack((embedding, budget_col))

        # Dimension guard — catches encoder/model mismatch after a retrain
        if self._feature_dim is not None and feature_vec.shape[1] != self._feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: model expects {self._feature_dim} "
                f"but encoder produced {feature_vec.shape[1]}. "
                "Re-run train_router.py with the current encoder."
            )

        return feature_vec

    # ── Input Validation ───────────────────────────────────────────────────────

    @staticmethod
    def _validate_inputs(query: str, budget: float) -> tuple[str, float]:
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}.")
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty.")
        if not isinstance(budget, (int, float)):
            raise TypeError(f"budget must be numeric, got {type(budget).__name__}.")
        if not (BUDGET_MIN <= budget <= BUDGET_MAX):
            clamped = float(np.clip(budget, BUDGET_MIN, BUDGET_MAX))
            log.warning(
                "budget=%.4f out of [%.1f, %.1f] — clamped to %.4f.",
                budget, BUDGET_MIN, BUDGET_MAX, clamped,
            )
            budget = clamped
        return query, float(budget)

    # ── Fallback ───────────────────────────────────────────────────────────────

    def _make_fallback(self, t0: float) -> RoutingResult:
        proba = {lbl: (1.0 if lbl == _FALLBACK_LABEL else 0.0) for lbl in VALID_LABELS}
        return RoutingResult(
            label=_FALLBACK_LABEL,
            confidence=1.0,
            probabilities=proba,
            latency_ms=(time.perf_counter() - t0) * 1000,
            method="fallback",
            fallback=True,
        )


# ── Module-level singleton (for Streamlit / adaptive_pipeline.py) ──────────────

_singleton: QueryRouter | None = None


def get_router(model_path: Path | str = _DEFAULT_MODEL) -> QueryRouter:
    """
    Return (and lazily create) a module-level singleton QueryRouter.

    Streamlit re-executes the script on every interaction — this prevents
    the encoder and classifier from reloading on every user action.
    """
    global _singleton
    if _singleton is None or str(_singleton.model_path) != str(model_path):
        _singleton = QueryRouter(model_path)
    return _singleton


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    router = QueryRouter()
    if not router.is_loaded:
        print("\nRouter failed to load. Run train_router.py first.")
        sys.exit(1)

    test_cases = [
        ("What dataset is this paper evaluated on?",             1.0),
        ("How does the attention mechanism improve parallelism?", 0.5),
        ("Compare sparse retrieval with dense representations.", 0.1),
    ]

    print(f"\n{'─'*65}")
    print(f"{'Query':<45} {'Budget':>6}  {'Route':<18}  {'Conf':>5}  {'ms':>6}")
    print(f"{'─'*65}")
    for query, budget in test_cases:
        result = router.route_full(query, budget)
        short_q = query[:43] + ".." if len(query) > 43 else query
        print(
            f"{short_q:<45} {budget:>6.1f}  {result.label:<18}  "
            f"{result.confidence:>5.3f}  {result.latency_ms:>6.1f}"
        )
    print(f"{'─'*65}")
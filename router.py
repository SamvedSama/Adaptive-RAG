"""
router.py — Query Router Agent
Owner: Samved Jain

Classifies an incoming query as one of:
    factual   → BM25 sparse retrieval
    conceptual → FAISS dense retrieval
    complex    → Hybrid (BM25 + FAISS) retrieval

Strategy (two-stage, fast-first):
    Stage 1 — Rule-based heuristics (zero latency, no model inference)
        - Keyword pattern matching
        - Query length signal
    Stage 2 — Embedding similarity to prototype sentences (fast, ~2ms)
        - Compute cosine similarity to per-class prototype embeddings
        - Used when rule-based stage returns low-confidence result

    Fallback — LLM zero-shot classification via Ollama
        - Only triggered when both above stages fail to reach threshold
        - Avoids paying LLM latency on every query
"""

import re
import json
import numpy as np
from typing import Literal, Tuple, List
from sentence_transformers import SentenceTransformer
import ollama


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

QueryType = Literal["factual", "conceptual", "complex"]


# ---------------------------------------------------------------------------
# Constants — Heuristic rules
# ---------------------------------------------------------------------------

# Keywords strongly indicative of each query type
FACTUAL_KEYWORDS: List[str] = [
    "what is", "what was", "who is", "who was", "when did", "when was",
    "where is", "where was", "how many", "how much", "which", "name the",
    "list the", "define", "what year", "what date",
]

CONCEPTUAL_KEYWORDS: List[str] = [
    "why", "how does", "how do", "explain", "describe", "what is the difference",
    "what are the advantages", "what are the disadvantages", "how is",
    "what causes", "what leads to", "elaborate",
]

COMPLEX_KEYWORDS: List[str] = [
    "compare", "contrast", "analyze", "analyse", "evaluate", "discuss",
    "what are the implications", "what would happen if", "both", "multiple",
    "relationship between", "impact of", "trade-off", "trade off",
    "pros and cons", "advantages and disadvantages",
]

# Query length thresholds (in words)
# Short queries tend to be factual; long queries tend to be complex
SHORT_QUERY_MAX_WORDS = 8
LONG_QUERY_MIN_WORDS = 18

# Confidence threshold — below this, escalate to embedding stage
HEURISTIC_CONFIDENCE_THRESHOLD = 0.75

# Embedding similarity threshold — below this, escalate to LLM fallback
EMBEDDING_CONFIDENCE_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Prototype sentences for embedding similarity (one per class)
# ---------------------------------------------------------------------------

PROTOTYPES: dict[QueryType, List[str]] = {
    "factual": [
        "What is the name of the dataset used in this paper?",
        "Who proposed the transformer architecture?",
        "How many layers does BERT have?",
        "When was GPT-3 released?",
    ],
    "conceptual": [
        "Why does attention outperform recurrent networks?",
        "How does the BERT pre-training objective work?",
        "Explain the role of positional encoding in transformers.",
        "What is the difference between encoder-only and decoder-only models?",
    ],
    "complex": [
        "Compare the performance of BM25 and dense retrieval across multiple datasets.",
        "Analyze the trade-offs between model size and inference latency.",
        "What are the implications of scaling laws for future language model development?",
        "Discuss the relationship between pre-training data quality and downstream task performance.",
    ],
}


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Two-stage query classifier with LLM fallback.

    Stage 1: Rule-based (keyword + length heuristics) — ~0ms
    Stage 2: Embedding similarity to prototype sentences — ~2ms
    Fallback: Ollama LLM zero-shot classification — ~200–500ms

    The fast stages handle the vast majority of queries.  The LLM fallback
    exists for genuinely ambiguous queries to avoid misclassification.

    Why not just use the LLM for everything?
    ─────────────────────────────────────────
    Per your spec, Naive RAG has no router overhead.  In Adaptive RAG the
    router must be fast enough that it pays off in retrieval quality gains.
    Calling the LLM for every query would add ~300ms of routing latency
    before any retrieval happens — defeating the purpose.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "llama3.2:3b",
    ):
        """
        Initialise the router.

        Args:
            model_name: Encoder for embedding similarity stage.
            llm_model:  Ollama model name for the fallback stage.
        """
        print(f"[QueryRouter] Loading encoder: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.llm_model = llm_model

        # Pre-compute and cache mean prototype embeddings for each class
        print("[QueryRouter] Pre-computing prototype embeddings …")
        self._prototype_embeddings: dict[QueryType, np.ndarray] = {}
        for qtype, sentences in PROTOTYPES.items():
            vecs = self.encoder.encode(sentences, normalize_embeddings=True)
            # Use mean pooling over prototype sentences → single representative vec
            self._prototype_embeddings[qtype] = vecs.mean(axis=0)

        print("[QueryRouter] Router ready.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(self, query: str) -> Tuple[QueryType, float, str]:
        """
        Classify a query into factual / conceptual / complex.

        Args:
            query: Raw query string from the user.

        Returns:
            Tuple of (query_type, confidence, method_used)
            - query_type:   "factual" | "conceptual" | "complex"
            - confidence:   float in [0, 1]
            - method_used:  "heuristic" | "embedding" | "llm_fallback"
        """
        query_lower = query.lower().strip()

        # ── Stage 1: Rule-based heuristics ────────────────────────────
        qtype, confidence = self._heuristic_classify(query_lower)
        if confidence >= HEURISTIC_CONFIDENCE_THRESHOLD:
            return qtype, confidence, "heuristic"

        # ── Stage 2: Embedding similarity ─────────────────────────────
        qtype, confidence = self._embedding_classify(query)
        if confidence >= EMBEDDING_CONFIDENCE_THRESHOLD:
            return qtype, confidence, "embedding"

        # ── Fallback: LLM zero-shot ────────────────────────────────────
        qtype = self._llm_classify(query)
        return qtype, 1.0, "llm_fallback"   # LLM gives a hard label, confidence=1.0

    # ------------------------------------------------------------------
    # Stage 1: Rule-based heuristics
    # ------------------------------------------------------------------

    def _heuristic_classify(self, query_lower: str) -> Tuple[QueryType, float]:
        """
        Keyword matching + query length heuristics.

        Returns (query_type, confidence).
        Confidence is a rough measure: 1.0 = strong keyword match,
        0.6 = length signal only, 0.0 = no signal.
        """
        # Score each type by counting keyword matches
        scores: dict[QueryType, int] = {"factual": 0, "conceptual": 0, "complex": 0}

        for kw in FACTUAL_KEYWORDS:
            if kw in query_lower:
                scores["factual"] += 1

        for kw in CONCEPTUAL_KEYWORDS:
            if kw in query_lower:
                scores["conceptual"] += 1

        for kw in COMPLEX_KEYWORDS:
            if kw in query_lower:
                scores["complex"] += 1

        best_type = max(scores, key=lambda t: scores[t])
        best_score = scores[best_type]

        if best_score >= 2:
            return best_type, 1.0    # strong match
        if best_score == 1:
            return best_type, 0.80   # moderate match

        # No keyword match — use query length as a weak signal
        word_count = len(query_lower.split())
        if word_count <= SHORT_QUERY_MAX_WORDS:
            return "factual", 0.60
        if word_count >= LONG_QUERY_MIN_WORDS:
            return "complex", 0.60

        # No signal at all
        return "conceptual", 0.0

    # ------------------------------------------------------------------
    # Stage 2: Embedding similarity
    # ------------------------------------------------------------------

    def _embedding_classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Encode the query and compute cosine similarity to each class prototype.

        Returns (query_type, confidence) where confidence is the similarity
        to the winning class.
        """
        query_vec = self.encoder.encode([query], normalize_embeddings=True)[0]

        similarities: dict[QueryType, float] = {}
        for qtype, proto_vec in self._prototype_embeddings.items():
            # Both vectors are L2-normalised, so dot product == cosine similarity
            sim = float(np.dot(query_vec, proto_vec))
            similarities[qtype] = sim

        best_type = max(similarities, key=lambda t: similarities[t])
        best_sim = similarities[best_type]

        return best_type, best_sim

    # ------------------------------------------------------------------
    # Fallback: LLM zero-shot classification
    # ------------------------------------------------------------------

    def _llm_classify(self, query: str) -> QueryType:
        """
        Use the local Ollama LLM to classify the query via zero-shot prompting.

        Only called when both rule-based and embedding stages are uncertain.
        Returns a hard label — no confidence score from the LLM.
        """
        prompt = f"""Classify the following question into exactly one of these categories:
- factual: short, specific answer lookup (who, when, what exactly, how many)
- conceptual: requires explanation or reasoning (how does X work, why, explain)
- complex: requires connecting multiple pieces of information or comparing entities

Question: "{query}"

Respond with ONLY one word: factual, conceptual, or complex. No explanation."""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            label = response["message"]["content"].strip().lower()

            # Sanitise — LLM might add punctuation or extra text
            for valid in ("factual", "conceptual", "complex"):
                if valid in label:
                    return valid  # type: ignore[return-value]

            # If LLM gives garbage, default to conceptual (safest middle ground)
            print(f"[QueryRouter] LLM returned unexpected label: '{label}'. Defaulting to conceptual.")
            return "conceptual"

        except Exception as e:
            print(f"[QueryRouter] LLM fallback failed: {e}. Defaulting to conceptual.")
            return "conceptual"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Tests the router without needing Ollama running (LLM fallback won't fire
    # for these clear-cut examples)
    router = QueryRouter()

    test_queries = [
        "What is the F1 score of the model on SQuAD?",          # factual
        "How does attention mechanism work in transformers?",     # conceptual
        "Why does BERT use masked language modelling?",          # conceptual
        "Compare BM25 and dense retrieval on multiple datasets.", # complex
        "What are the implications of scaling laws?",             # complex
        "Who proposed the transformer architecture?",             # factual
    ]

    print(f"\n{'Query':<60} {'Type':<12} {'Confidence':<12} {'Method'}")
    print("-" * 100)
    for q in test_queries:
        qtype, conf, method = router.classify(q)
        print(f"{q:<60} {qtype:<12} {conf:<12.3f} {method}")
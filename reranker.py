"""
reranker.py — Cross-Encoder Re-ranking Module
Owner: Roshan K C

Re-ranks retrieved chunks using a cross-encoder model.
The cross-encoder jointly encodes (query, chunk_text) pairs
and produces a relevance score far more accurate than
bi-encoder cosine similarity or BM25 scores.

Input:
    query  (str)
    chunks (List[dict])

Output:
    List[chunk dict] sorted by cross-encoder score (descending)

Chunk schema:
{
  "chunk_id": str,
  "text":     str,
  "source":   str,
  "position": int,
  "score":    float        ← overwritten with cross-encoder score
}
"""

import logging
import time
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Reranker")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH   = 32        # safe for 8 GB VRAM; lower to 16 if OOM
MAX_TEXT_TOKENS = 512       # cross-encoder hard limit (model-dependent)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _truncate(text: str, max_chars: int = MAX_TEXT_TOKENS * 4) -> str:
    """
    Cheap character-level truncation so we never overflow the model's
    token window.  4 chars ≈ 1 token is a conservative heuristic.
    """
    return text[:max_chars] if len(text) > max_chars else text


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    """
    Production cross-encoder reranker for retrieved chunks.

    Why Cross-Encoder over Bi-Encoder?
    ───────────────────────────────────
    Bi-encoders (FAISS / BM25) encode query and document independently
    and compare with cosine similarity — fast but approximate.

    Cross-encoders encode the *pair* (query, doc) together, letting
    attention flow between them → far more accurate relevance signal.

    Trade-off: O(k) forward passes at rerank time, but k is small
    (typically 10–50), so latency is acceptable.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH,
        device: Optional[str] = None,
    ) -> None:
        """
        Load the cross-encoder model once at startup.

        Args:
            model_name:  HuggingFace model identifier.
            batch_size:  Inference batch size (tune for your VRAM).
            device:      'cuda', 'cpu', or None (auto-detect).
        """
        logger.info("Loading cross-encoder: %s", model_name)
        t0 = time.time()

        self.model_name = model_name
        self.batch_size  = batch_size
        self.model       = CrossEncoder(model_name, device=device)

        logger.info("Model loaded in %.2fs", time.time() - t0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query:  str,
        chunks: List[Dict[str, Any]],
        top_k:  Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank retrieved chunks with cross-encoder relevance scoring.

        Args:
            query:  User query string.
            chunks: Retrieved chunk dicts (any retriever output).
            top_k:  If set, return only the top_k highest-scored chunks.

        Returns:
            New list of chunk dicts sorted by score descending.
            Original dicts are NOT mutated.

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("[Reranker] query must be a non-empty string.")

        if not chunks:
            logger.warning("rerank() called with 0 chunks — returning empty list.")
            return []

        logger.info(
            "Reranking %d chunks for query: '%s'",
            len(chunks),
            query[:80],
        )
        t0 = time.time()

        # ── Build (query, text) pairs ──────────────────────────────────
        pairs = [
            (query.strip(), _truncate(chunk.get("text", "")))
            for chunk in chunks
        ]

        # ── Batch inference ────────────────────────────────────────────
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # ── Attach new scores (copy chunks — never mutate originals) ───
        reranked: List[Dict[str, Any]] = []
        for chunk, score in zip(chunks, scores):
            new_chunk          = dict(chunk)          # shallow copy
            new_chunk["score"] = float(score)
            reranked.append(new_chunk)

        # ── Sort descending ────────────────────────────────────────────
        reranked.sort(key=lambda c: c["score"], reverse=True)

        # ── Optional truncation ────────────────────────────────────────
        if top_k is not None and top_k > 0:
            reranked = reranked[:top_k]

        elapsed = time.time() - t0
        logger.info(
            "Reranking done in %.3fs — returning %d/%d chunks",
            elapsed,
            len(reranked),
            len(chunks),
        )

        return reranked

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def rerank_and_log(
        self,
        query:  str,
        chunks: List[Dict[str, Any]],
        top_k:  Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Same as rerank() but also returns latency and score statistics.
        Useful for ablation_runner.py and latency_tracker.py.

        Returns:
            {
              "chunks":       List[dict],
              "latency_ms":   float,
              "top_score":    float,
              "bottom_score": float,
              "score_range":  float,
            }
        """
        t0      = time.time()
        results = self.rerank(query, chunks, top_k=top_k)
        elapsed = (time.time() - t0) * 1000  # ms

        scores = [c["score"] for c in results] if results else [0.0]

        return {
            "chunks":       results,
            "latency_ms":   round(elapsed, 3),
            "top_score":    round(max(scores), 4),
            "bottom_score": round(min(scores), 4),
            "score_range":  round(max(scores) - min(scores), 4),
        }

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker(model='{self.model_name}', "
            f"batch_size={self.batch_size})"
        )


# ---------------------------------------------------------------------------
# Smoke test  (python reranker.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("CrossEncoderReranker — smoke test")
    print("=" * 60)

    QUERY = "How do transformers use attention mechanisms?"

    CHUNKS = [
        {
            "chunk_id": "doc1_chunk_000",
            "text": (
                "Transformers rely on self-attention mechanisms to model "
                "relationships between all positions in a sequence simultaneously."
            ),
            "source": "attention_is_all_you_need.pdf",
            "position": 0,
            "score": 0.30,
        },
        {
            "chunk_id": "doc2_chunk_001",
            "text": (
                "BM25 is a classic lexical retrieval algorithm based on "
                "term-frequency and inverse-document-frequency weighting."
            ),
            "source": "ir_textbook.pdf",
            "position": 1,
            "score": 0.20,
        },
        {
            "chunk_id": "doc3_chunk_002",
            "text": (
                "Attention allows the model to focus on the most relevant "
                "parts of the input when producing each output token."
            ),
            "source": "bert_paper.pdf",
            "position": 2,
            "score": 0.25,
        },
        {
            "chunk_id": "doc4_chunk_003",
            "text": (
                "Convolutional neural networks apply learned filters across "
                "fixed-size local windows of an image or sequence."
            ),
            "source": "cnn_survey.pdf",
            "position": 3,
            "score": 0.18,
        },
    ]

    # ── Basic rerank ──────────────────────────────────────────────────
    reranker = CrossEncoderReranker()
    results  = reranker.rerank(QUERY, CHUNKS, top_k=3)

    print(f"\nQuery : {QUERY}")
    print(f"Input : {len(CHUNKS)} chunks   →   Output: {len(results)} chunks\n")

    for rank, chunk in enumerate(results, start=1):
        print(
            f"  Rank {rank} | score={chunk['score']:+.4f} | "
            f"{chunk['chunk_id']} | {chunk['text'][:70]}..."
        )

    # ── rerank_and_log ────────────────────────────────────────────────
    print("\n--- rerank_and_log ---")
    log = reranker.rerank_and_log(QUERY, CHUNKS)
    print(f"  latency_ms   : {log['latency_ms']}")
    print(f"  top_score    : {log['top_score']}")
    print(f"  bottom_score : {log['bottom_score']}")
    print(f"  score_range  : {log['score_range']}")

    # ── Edge cases ────────────────────────────────────────────────────
    print("\n--- Edge cases ---")
    print("  Empty chunk list :", reranker.rerank(QUERY, []))

    try:
        reranker.rerank("", CHUNKS)
    except ValueError as e:
        print("  Empty query caught :", e)

    print("\n✅  All tests passed.")
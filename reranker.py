"""
reranker.py — Cross-Encoder Re-ranking Module
Owner: Roshan K C

Re-ranks retrieved chunks using a cross-encoder model.
The cross-encoder jointly encodes (query, chunk_text) pairs
and produces a relevance score far more accurate than
bi-encoder cosine similarity or BM25 scores.

Design:
- Accepts list[RetrievedChunk] (from faiss_retriever / bm25 / hybrid)
- Scores all pairs in a single batched tensor forward pass — no Python loop
- Returns list[RetrievedChunk] with score overwritten to cross-encoder logit
- Never mutates input chunks (frozen dataclass + fresh construction)

Public API:
    reranker = get_reranker()                            # module singleton
    chunks   = reranker.rerank(query, chunks, top_k=5)  # → list[RetrievedChunk]
    result   = reranker.rerank_full(query, chunks)       # → RerankResult
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional

from sentence_transformers import CrossEncoder

from faiss_retriever import RetrievedChunk

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_BATCH_SIZE = 32          # safe for 8 GB VRAM; lower to 16 if OOM
_MAX_CHARS          = 2048        # ~512 tokens at 4 chars/token heuristic


# ── Result container ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RerankResult:
    """
    Full output of rerank_full() — used by ablation_runner.py and
    latency_tracker.py for per-call profiling.
    """
    chunks:       list[RetrievedChunk]   # top-k chunks, sorted by CE score
    latency_ms:   float                  # wall-clock time for this call
    input_count:  int                    # number of chunks passed in
    output_count: int                    # number returned (≤ top_k)
    top_score:    float                  # highest cross-encoder logit
    bottom_score: float                  # lowest score among returned chunks


# ── CrossEncoderReranker ───────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Production cross-encoder reranker.

    Why Cross-Encoder over Bi-Encoder?
    ───────────────────────────────────
    Bi-encoders (FAISS / BM25) encode query and document independently.
    Cross-encoders encode the *pair* (query, doc) jointly, letting attention
    flow between them for a far more accurate relevance signal.

    Batching strategy:
    ──────────────────
    All (query, chunk) pairs are stacked into a single list and passed to
    CrossEncoder.predict() in one call. sentence-transformers handles the
    internal tensor batching and GPU dispatch — no Python-level loop over
    individual pairs. This is the maximum throughput approach for k ≤ ~200.
    """

    def __init__(
        self,
        model_name: str           = _DEFAULT_MODEL,
        batch_size: int           = _DEFAULT_BATCH_SIZE,
        device:     Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: HuggingFace cross-encoder model identifier.
            batch_size: Pairs per GPU batch (tune for your VRAM budget).
            device:     'cuda', 'cpu', or None (auto-detect via PyTorch).
        """
        if batch_size < 1:
            raise ValueError("batch_size must be ≥ 1.")

        self._model_name = model_name
        self._batch_size = batch_size
        self._model: CrossEncoder | None = None

        self._load_model(device)

    # ── Public API ─────────────────────────────────────────────────────────────

    def rerank(
        self,
        query:  str,
        chunks: list[RetrievedChunk],
        top_k:  Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Re-rank chunks by cross-encoder relevance. Simple call for pipeline use.

        Args:
            query:  User query string.
            chunks: Retriever output — list[RetrievedChunk].
            top_k:  Return only the top_k highest-scored chunks.
                    None returns all chunks, sorted.

        Returns:
            New list[RetrievedChunk] sorted by score descending.
            Input chunks are never mutated (frozen dataclass).

        Raises:
            RuntimeError: If the model failed to load.
            ValueError:   If query is empty.
        """
        return self._score_and_rank(query, chunks, top_k).chunks

    def rerank_full(
        self,
        query:  str,
        chunks: list[RetrievedChunk],
        top_k:  Optional[int] = None,
    ) -> RerankResult:
        """
        Same as rerank() but returns a RerankResult with latency and
        score statistics for ablation_runner.py and latency_tracker.py.
        """
        return self._score_and_rank(query, chunks, top_k)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker(model='{self._model_name}', "
            f"batch_size={self._batch_size}, loaded={self.is_loaded})"
        )

    # ── Internal scoring ───────────────────────────────────────────────────────

    def _score_and_rank(
        self,
        query:  str,
        chunks: list[RetrievedChunk],
        top_k:  Optional[int],
    ) -> RerankResult:
        """
        Core implementation shared by rerank() and rerank_full().
        Single code path — no duplication.
        """
        self._validate(query)

        if not self.is_loaded:
            raise RuntimeError(
                "CrossEncoderReranker model not loaded. "
                "Check logs for the load-time error."
            )

        if not chunks:
            log.warning("rerank() called with 0 chunks — returning empty result.")
            return RerankResult(
                chunks=[], latency_ms=0.0,
                input_count=0, output_count=0,
                top_score=0.0, bottom_score=0.0,
            )

        t0 = time.perf_counter()
        q  = query.strip()

        log.info(
            "Reranking %d chunks | model=%s | batch_size=%d",
            len(chunks), self._model_name, self._batch_size,
        )

        # ── Build all pairs in one list — single batched forward pass ──────────
        # _truncate() prevents overflow of the model's token window.
        # sentence-transformers CrossEncoder.predict() batches internally.
        pairs = [(q, _truncate(c.text)) for c in chunks]

        raw_scores: list[float] = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        ).tolist()

        # ── Construct new RetrievedChunks with overwritten score ───────────────
        # frozen=True means we cannot mutate originals — construct fresh objects.
        scored: list[RetrievedChunk] = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                position=chunk.position,
                score=float(score),
            )
            for chunk, score in zip(chunks, raw_scores)
        ]

        scored.sort(key=lambda c: c.score, reverse=True)

        if top_k is not None and top_k > 0:
            scored = scored[:top_k]

        latency_ms = (time.perf_counter() - t0) * 1000
        all_scores = [c.score for c in scored] if scored else [0.0]

        log.info(
            "Reranking complete | returned=%d/%d | top=%.4f | bottom=%.4f | %.1f ms",
            len(scored), len(chunks),
            max(all_scores), min(all_scores), latency_ms,
        )

        return RerankResult(
            chunks=scored,
            latency_ms=round(latency_ms, 3),
            input_count=len(chunks),
            output_count=len(scored),
            top_score=round(max(all_scores), 6),
            bottom_score=round(min(all_scores), 6),
        )

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self, device: Optional[str]) -> None:
        """
        Load the CrossEncoder model. Sets self._model = None on failure
        so callers fall back gracefully instead of crashing at import time.
        """
        log.info("Loading cross-encoder '%s' ...", self._model_name)
        t0 = time.perf_counter()
        try:
            self._model = CrossEncoder(self._model_name, device=device)
            log.info(
                "Cross-encoder ready | model='%s' | %.2fs",
                self._model_name, time.perf_counter() - t0,
            )
        except Exception as exc:
            log.error(
                "Failed to load cross-encoder '%s': %s. "
                "rerank() calls will raise RuntimeError until the model is available.",
                self._model_name, exc,
            )
            self._model = None

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(query: str) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")


# ── Truncation helper ──────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int = _MAX_CHARS) -> str:
    """
    Character-level truncation to stay within the cross-encoder token window.
    4 chars ≈ 1 token is a conservative heuristic for English text.
    Applied before building pairs — costs nothing compared to model inference.
    """
    return text[:max_chars] if len(text) > max_chars else text


# ── Module-level singleton ─────────────────────────────────────────────────────

_singleton: CrossEncoderReranker | None = None


def get_reranker(
    model_name: str           = _DEFAULT_MODEL,
    batch_size: int           = _DEFAULT_BATCH_SIZE,
    device:     Optional[str] = None,
) -> CrossEncoderReranker:
    """
    Return a module-level singleton CrossEncoderReranker.

    The cross-encoder weights (~90 MB) are loaded once per process.
    Streamlit reruns and repeated pipeline calls reuse the cached instance.
    """
    global _singleton
    if _singleton is None:
        log.info("Initialising CrossEncoderReranker singleton ...")
        _singleton = CrossEncoderReranker(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )
    return _singleton


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    QUERY = "How do transformers use attention mechanisms?"

    CHUNKS = [
        RetrievedChunk("doc1_chunk_000",
            "Transformers rely on self-attention to model relationships "
            "between all positions in a sequence simultaneously.",
            "attention_is_all_you_need.pdf", 0, 0.30),
        RetrievedChunk("doc2_chunk_001",
            "BM25 is a classic lexical retrieval algorithm based on "
            "term-frequency and inverse-document-frequency weighting.",
            "ir_textbook.pdf", 1, 0.20),
        RetrievedChunk("doc3_chunk_002",
            "Attention allows the model to focus on the most relevant "
            "parts of the input when producing each output token.",
            "bert_paper.pdf", 2, 0.25),
        RetrievedChunk("doc4_chunk_003",
            "Convolutional neural networks apply learned filters across "
            "fixed-size local windows of an image or sequence.",
            "cnn_survey.pdf", 3, 0.18),
    ]

    reranker = CrossEncoderReranker()

    # ── Basic rerank ───────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"Query: {QUERY}")
    results = reranker.rerank(QUERY, CHUNKS, top_k=3)
    print(f"\nTop-3 after reranking ({len(CHUNKS)} → {len(results)}):")
    for rank, chunk in enumerate(results, 1):
        print(f"  [{rank}] score={chunk.score:+.4f}  {chunk.chunk_id}  {chunk.text[:60]}...")

    # ── rerank_full ────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    full = reranker.rerank_full(QUERY, CHUNKS)
    print(f"RerankResult:")
    print(f"  latency_ms   : {full.latency_ms:.1f}")
    print(f"  input_count  : {full.input_count}")
    print(f"  output_count : {full.output_count}")
    print(f"  top_score    : {full.top_score:+.6f}")
    print(f"  bottom_score : {full.bottom_score:+.6f}")

    # ── Edge cases ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    empty = reranker.rerank(QUERY, [])
    print(f"Empty chunk list → {empty}")

    try:
        reranker.rerank("", CHUNKS)
    except ValueError as exc:
        print(f"Empty query correctly raised ValueError: {exc}")

    print(f"\n{'─'*65}")
    print("Smoke-test complete.")
    sys.exit(0)
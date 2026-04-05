"""
hybrid_retriever.py — Hybrid Retrieval Module (Reciprocal Rank Fusion)
Owner: Samved Jain

Combines BM25 (sparse) and FAISS (dense) retrieval results using
Reciprocal Rank Fusion (RRF), returning a unified ranked list of RetrievedChunk.

Used for the "Multi_Hop_FAISS" routing path — best accuracy, highest compute.

Why RRF instead of score averaging?
    BM25 scores (large positive floats) and cosine similarities ([0, 1]) live
    on incompatible scales. RRF sidesteps this by working purely on rank
    positions, making it scale-invariant by design.

    Formula: RRF(d) = Σ 1 / (k + rank_i(d))   [Cormack et al., 2009; k=60]

Public API:
    retriever = get_hybrid_retriever()              # module singleton
    chunks    = retriever.retrieve(query, k)        # → list[RetrievedChunk]
    breakdown = retriever.retrieve_with_breakdown(query, k)  # → FusionResult
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from faiss_retriever import (
    BaseRetriever,
    RetrievedChunk,
    FAISSRetriever,
    get_faiss_retriever,
)
from bm25_retriever import BM25Retriever, get_bm25_retriever

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_TOP_K   = 5
_DEFAULT_FETCH_K = 2   # multiplier: fetch fetch_k * top_k from each retriever
_RRF_K           = 60  # smoothing constant from Cormack et al. 2009


# ── Result container ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FusionResult:
    """
    Full breakdown returned by retrieve_with_breakdown().
    Carries the fused list alongside per-retriever sublists
    for ablation logging and debugging.
    """
    fused:      list[RetrievedChunk]   # final RRF-ranked results
    bm25:       list[RetrievedChunk]   # raw BM25 candidates
    faiss:      list[RetrievedChunk]   # raw FAISS candidates
    latency_ms: float                  # total wall-clock time (ms)


# ── HybridRetriever ────────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    RRF-based hybrid retriever over BM25 and FAISS.

    Accepts any BaseRetriever implementation for each slot, so the pipeline
    can inject mocks for testing without touching production paths.
    """

    def __init__(
        self,
        faiss_retriever: BaseRetriever,
        bm25_retriever:  BaseRetriever,
        rrf_k:           int = _RRF_K,
        fetch_multiplier: int = _DEFAULT_FETCH_K,
    ) -> None:
        """
        Args:
            faiss_retriever:  A ready-to-query dense retriever (BaseRetriever).
            bm25_retriever:   A ready-to-query sparse retriever (BaseRetriever).
            rrf_k:            RRF smoothing constant (default 60).
            fetch_multiplier: Each sub-retriever fetches top_k * this many
                              candidates before fusion (default 2).
        """
        if rrf_k < 1:
            raise ValueError("rrf_k must be ≥ 1.")
        if fetch_multiplier < 1:
            raise ValueError("fetch_multiplier must be ≥ 1.")

        self._faiss           = faiss_retriever
        self._bm25            = bm25_retriever
        self._rrf_k           = rrf_k
        self._fetch_multiplier = fetch_multiplier

    # ── BaseRetriever interface ────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._faiss.is_ready() and self._bm25.is_ready()

    def retrieve(self, query: str, top_k: int = _DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """
        Return top-k chunks fused from BM25 + FAISS via RRF.

        Args:
            query:  Natural language query string.
            top_k:  Final number of results to return.

        Returns:
            List of RetrievedChunk sorted by RRF score descending.
            Score field contains the RRF value (not a raw similarity).

        Raises:
            RuntimeError: If either sub-retriever is not ready.
            ValueError:   If query is empty.
        """
        return self._fuse(query, top_k).fused

    # ── Extended API ───────────────────────────────────────────────────────────

    def retrieve_with_breakdown(
        self, query: str, top_k: int = _DEFAULT_TOP_K
    ) -> FusionResult:
        """
        Same as retrieve() but also returns per-retriever sublists and latency.
        Used by ablation_runner.py and latency_tracker.py.
        """
        return self._fuse(query, top_k)

    # ── Internal fusion ────────────────────────────────────────────────────────

    def _fuse(self, query: str, top_k: int) -> FusionResult:
        """
        Core RRF fusion logic. Shared by retrieve() and retrieve_with_breakdown()
        so there's exactly one implementation path — no duplication.
        """
        self._validate_query(query)
        if not self.is_ready():
            raise RuntimeError(
                "HybridRetriever not ready — one or both sub-retrievers failed to load."
            )

        top_k   = max(1, top_k)
        fetch_k = top_k * self._fetch_multiplier
        t0      = time.perf_counter()

        # Step 1: Fetch candidates from each retriever independently
        bm25_chunks  = self._safe_retrieve(self._bm25,  query, fetch_k, "BM25")
        faiss_chunks = self._safe_retrieve(self._faiss, query, fetch_k, "FAISS")

        # Step 2: RRF fusion
        fused = self._rrf_merge(bm25_chunks, faiss_chunks, top_k)

        latency_ms = (time.perf_counter() - t0) * 1000
        log.debug(
            "Hybrid retrieve | top_k=%d | bm25=%d | faiss=%d | fused=%d | %.1f ms",
            top_k, len(bm25_chunks), len(faiss_chunks), len(fused), latency_ms,
        )
        return FusionResult(
            fused=fused,
            bm25=bm25_chunks,
            faiss=faiss_chunks,
            latency_ms=latency_ms,
        )

    def _rrf_merge(
        self,
        bm25_chunks:  list[RetrievedChunk],
        faiss_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Apply RRF across two ranked lists and return top_k fused chunks.

        Each list contributes 1 / (rrf_k + rank) per document.
        A document appearing in both lists accumulates contributions from both.
        Documents appearing in only one list still get their single contribution.
        """
        rrf_scores: dict[str, float] = {}
        # Store the chunk itself (prefer FAISS copy since it carries a meaningful
        # cosine score, but we overwrite score with the RRF value below)
        chunk_store: dict[str, RetrievedChunk] = {}

        for ranked_list in (bm25_chunks, faiss_chunks):
            for rank, chunk in enumerate(ranked_list, start=1):
                cid = chunk.chunk_id
                rrf_scores[cid] = (
                    rrf_scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank)
                )
                chunk_store.setdefault(cid, chunk)

        # Sort by descending RRF score and materialise top_k RetrievedChunks
        top_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        return [
            RetrievedChunk(
                chunk_id=chunk_store[cid].chunk_id,
                text=chunk_store[cid].text,
                source=chunk_store[cid].source,
                position=chunk_store[cid].position,
                score=round(rrf_scores[cid], 6),   # RRF score replaces raw similarity
            )
            for cid in top_ids
        ]

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_retrieve(
        retriever: BaseRetriever,
        query: str,
        top_k: int,
        label: str,
    ) -> list[RetrievedChunk]:
        """
        Call retriever.retrieve() and return an empty list on failure instead
        of propagating the exception. The fusion can still proceed with one
        retriever's results if the other is momentarily unavailable.
        """
        try:
            return retriever.retrieve(query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "%s retriever failed during hybrid fusion: %s — proceeding with empty list.",
                label, exc,
            )
            return []

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")


# ── Module-level singleton ─────────────────────────────────────────────────────

_singleton: HybridRetriever | None = None


def get_hybrid_retriever(
    rrf_k:            int = _RRF_K,
    fetch_multiplier: int = _DEFAULT_FETCH_K,
) -> HybridRetriever:
    """
    Return a module-level singleton HybridRetriever backed by the
    FAISS and BM25 singletons.

    All three singletons chain together:
        get_hybrid_retriever()
          └─ get_faiss_retriever()   (loads index once)
          └─ get_bm25_retriever()    (loads/builds index once)

    Safe to call repeatedly from Streamlit reruns — heavy I/O happens only once.
    """
    global _singleton
    if _singleton is None:
        log.info("Initialising HybridRetriever singleton ...")
        _singleton = HybridRetriever(
            faiss_retriever=get_faiss_retriever(),
            bm25_retriever=get_bm25_retriever(),
            rrf_k=rrf_k,
            fetch_multiplier=fetch_multiplier,
        )
        log.info(
            "HybridRetriever singleton ready | rrf_k=%d | fetch_multiplier=%d",
            rrf_k, fetch_multiplier,
        )
    return _singleton


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ── Mock sub-retrievers so the file is runnable without a built index ──────
    class _MockRetriever(BaseRetriever):
        def __init__(self, chunks: list[RetrievedChunk]) -> None:
            self._chunks = chunks
        def is_ready(self) -> bool:
            return True
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
            return self._chunks[:top_k]

    bm25_mock = _MockRetriever([
        RetrievedChunk("doc1_chunk_000", "BERT uses masked language modelling.", "p1.pdf", 0, 12.4),
        RetrievedChunk("doc1_chunk_002", "Attention is all you need.",            "p1.pdf", 2,  8.1),
        RetrievedChunk("doc1_chunk_003", "GPT uses autoregressive generation.",   "p1.pdf", 3,  5.3),
    ])
    faiss_mock = _MockRetriever([
        RetrievedChunk("doc1_chunk_001", "Transformers replaced RNNs in NLP.",    "p1.pdf", 1, 0.91),
        RetrievedChunk("doc1_chunk_000", "BERT uses masked language modelling.",  "p1.pdf", 0, 0.87),
        RetrievedChunk("doc1_chunk_004", "Self-attention computes query-key dot.", "p1.pdf", 4, 0.75),
    ])

    hybrid = HybridRetriever(faiss_mock, bm25_mock)
    breakdown = hybrid.retrieve_with_breakdown("how does BERT work?", top_k=3)

    print(f"\n{'─'*65}")
    print("FUSED results (RRF):")
    for chunk in breakdown.fused:
        print(f"  {chunk.score:.6f}  {chunk.chunk_id:<28}  {chunk.text[:50]}...")

    print("\nBM25 candidates:")
    for chunk in breakdown.bm25:
        print(f"  {chunk.score:>8.3f}  {chunk.chunk_id}")

    print("\nFAISS candidates:")
    for chunk in breakdown.faiss:
        print(f"  {chunk.score:>8.4f}  {chunk.chunk_id}")

    print(f"\nFusion latency: {breakdown.latency_ms:.1f} ms")
    print(f"{'─'*65}")
    print("Hybrid smoke-test complete.")
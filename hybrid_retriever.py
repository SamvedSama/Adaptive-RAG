"""
hybrid_retriever.py — Hybrid Retrieval Module (Score Fusion)
Owner: Samved Jain

Combines BM25 (sparse) and FAISS (dense) retrieval results using
Reciprocal Rank Fusion (RRF), returning a unified ranked chunk list.

Used for COMPLEX query types where both keyword precision (BM25) and
semantic understanding (FAISS) are needed together.

Chunk schema (output):
{
    "chunk_id": str,
    "text":     str,
    "source":   str,
    "position": int,
    "score":    float   # fused RRF score
}
"""

import json
from typing import List, Dict, Any, Optional

from faiss_retriever import FAISSRetriever
from bm25_retriever import BM25Retriever   # Nivi's module


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5

# RRF constant — controls how much early ranks are rewarded vs later ones.
# k=60 is the standard value from the original RRF paper (Cormack et al. 2009).
RRF_K = 60


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid retriever that fuses BM25 and FAISS results via Reciprocal Rank Fusion.

    Why RRF instead of simple score averaging?
    ─────────────────────────────────────────
    BM25 scores and cosine similarity scores live on completely different scales.
    BM25 scores can be large positive floats; cosine similarities are in [0, 1].
    Normalising and then averaging introduces arbitrary scale assumptions.

    RRF sidesteps this entirely by working only on *rank positions*, not raw
    scores.  A chunk ranked 1st by BM25 AND 1st by FAISS will get the highest
    fused score regardless of what its raw scores were.

    Formula:
        RRF(d) = Σ  1 / (k + rank_i(d))
    where rank_i(d) is the 1-based rank of document d in retriever i's list,
    and k=60 is a smoothing constant.

    This is the same fusion strategy used in many production hybrid search
    systems (e.g., Elastic, Weaviate, Cohere Rerank).
    """

    def __init__(
        self,
        faiss_retriever: FAISSRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = RRF_K,
    ):
        """
        Initialise the hybrid retriever with pre-built sparse and dense retrievers.

        Args:
            faiss_retriever: A ready-to-query FAISSRetriever instance.
            bm25_retriever:  A ready-to-query BM25Retriever instance (Nivi's).
            rrf_k:           RRF smoothing constant (default 60).
        """
        self.faiss = faiss_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks by fusing BM25 and FAISS ranked lists via RRF.

        Steps:
            1. Run BM25 retrieval (fetch 2*top_k for broader candidate pool).
            2. Run FAISS retrieval (fetch 2*top_k).
            3. Compute RRF score for every chunk that appears in either list.
            4. Sort by fused score descending.
            5. Return top-k results with "score" set to the RRF score.

        Args:
            query:  Natural language query string.
            top_k:  Number of results to return.

        Returns:
            List of chunk dicts sorted by fused RRF score descending.
        """
        fetch_k = top_k * 2   # fetch more from each retriever for better fusion

        # Step 1 & 2: get ranked lists from both retrievers
        bm25_results: List[Dict[str, Any]] = self.bm25.retrieve(query, top_k=fetch_k)
        faiss_results: List[Dict[str, Any]] = self.faiss.retrieve(query, top_k=fetch_k)

        # Step 3: compute RRF scores
        # Use chunk_id as the key to merge results from both retrievers
        rrf_scores: Dict[str, float] = {}
        chunk_store: Dict[str, Dict[str, Any]] = {}   # chunk_id → chunk dict

        def _apply_rrf(ranked_list: List[Dict[str, Any]]) -> None:
            """Add RRF contribution from one ranked list into rrf_scores."""
            for rank, chunk in enumerate(ranked_list, start=1):
                cid = chunk["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
                # Store the chunk data (prefer FAISS copy since it has cosine score,
                # but either is fine — we overwrite score later)
                if cid not in chunk_store:
                    chunk_store[cid] = chunk

        _apply_rrf(bm25_results)
        _apply_rrf(faiss_results)

        # Step 4 & 5: sort by fused score and slice top-k
        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        results: List[Dict[str, Any]] = []
        for cid in ranked_ids[:top_k]:
            chunk = dict(chunk_store[cid])
            chunk["score"] = round(rrf_scores[cid], 6)
            results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # Introspection helper (useful for debugging / ablation logging)
    # ------------------------------------------------------------------

    def retrieve_with_breakdown(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> Dict[str, Any]:
        """
        Same as retrieve() but also returns the individual BM25 and FAISS
        ranked lists for debugging or ablation analysis.

        Returns:
            {
                "fused":  List[chunk],   # final RRF-ranked results
                "bm25":   List[chunk],   # raw BM25 results
                "faiss":  List[chunk],   # raw FAISS results
            }
        """
        fetch_k = top_k * 2

        bm25_results = self.bm25.retrieve(query, top_k=fetch_k)
        faiss_results = self.faiss.retrieve(query, top_k=fetch_k)

        rrf_scores: Dict[str, float] = {}
        chunk_store: Dict[str, Dict[str, Any]] = {}

        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            chunk_store.setdefault(cid, chunk)

        for rank, chunk in enumerate(faiss_results, start=1):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            chunk_store.setdefault(cid, chunk)

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        fused = []
        for cid in ranked_ids[:top_k]:
            chunk = dict(chunk_store[cid])
            chunk["score"] = round(rrf_scores[cid], 6)
            fused.append(chunk)

        return {"fused": fused, "bm25": bm25_results, "faiss": faiss_results}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # This test uses mock retriever stubs so the file is runnable standalone.
    # In the real pipeline, pass actual FAISSRetriever + BM25Retriever instances.

    class _MockRetriever:
        """Minimal stub that returns a fixed ranked list."""

        def __init__(self, results):
            self._results = results

        def retrieve(self, query: str, top_k: int = 5):
            return self._results[:top_k]

    bm25_mock = _MockRetriever([
        {"chunk_id": "doc1_chunk_000", "text": "BERT uses masked language modelling.", "source": "p1.pdf", "position": 0, "score": 12.4},
        {"chunk_id": "doc1_chunk_002", "text": "Attention is all you need.", "source": "p1.pdf", "position": 2, "score": 8.1},
    ])

    faiss_mock = _MockRetriever([
        {"chunk_id": "doc1_chunk_001", "text": "Transformers replaced RNNs in NLP.", "source": "p1.pdf", "position": 1, "score": 0.91},
        {"chunk_id": "doc1_chunk_000", "text": "BERT uses masked language modelling.", "source": "p1.pdf", "position": 0, "score": 0.87},
    ])

    hybrid = HybridRetriever(faiss_mock, bm25_mock)   # type: ignore
    results = hybrid.retrieve("how does BERT work?", top_k=3)

    print("\n--- Hybrid Retrieval Results ---")
    for r in results:
        print(json.dumps(r, indent=2))
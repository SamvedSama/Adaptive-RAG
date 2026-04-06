"""
faiss_retriever.py — Dense Retrieval Module

Builds and queries a FAISS IndexFlatIP index using sentence-transformers.
Implements BaseRetriever so adaptive_pipeline.py can swap retrievers
without changing call sites.

Used for the "Multi_Hop_FAISS" routing path — semantically-rich dense
retrieval best suited for conceptual and complex multi-hop queries.

Public API:
    retriever = get_faiss_retriever()          # module singleton (load existing index)
    chunks    = retriever.retrieve(query, k)   # → list[RetrievedChunk]

CLI:
    python faiss_retriever.py --build-index            # encode chunks + save index
    python faiss_retriever.py --build-index --rebuild  # force re-encode even if valid
    python faiss_retriever.py                          # smoke-test (index must exist)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Shared types — imported from base_retriever, NOT redefined here ───────────
# This eliminates the circular import that previously existed when
# bm25_retriever imported BaseRetriever and RetrievedChunk from this file.
from base_retriever import BaseRetriever, RetrievedChunk, validate_chunk_schema

log = logging.getLogger(__name__)

__all__ = ["FAISSRetriever", "get_faiss_retriever"]

# ── Constants ──────────────────────────────────────────────────────────────────
_DEFAULT_ENCODER    = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_INDEX_DIR  = Path("data/faiss_index")
_DEFAULT_INDEX_PATH = _DEFAULT_INDEX_DIR / "index.faiss"   # ← file, not directory
_DEFAULT_META_PATH  = _DEFAULT_INDEX_DIR / "metadata.pkl"  # ← file, not directory
_DEFAULT_TOP_K      = 5
_ENCODE_BATCH_SIZE  = 64


# ── FAISSRetriever ─────────────────────────────────────────────────────────────

class FAISSRetriever(BaseRetriever):
    """
    Dense retriever backed by a FAISS IndexFlatIP (exact cosine similarity).

    Design decisions:
    - IndexFlatIP on L2-normalised vectors gives exact cosine similarity
      without the overhead of IndexFlatL2.
    - All embeddings are float32 — half the memory of float64 with no
      meaningful accuracy loss for MiniLM.
    - Encoder is loaded once in __init__; the FAISS index is loaded/built
      separately via load() or build_index() to keep startup fast when
      the index already exists.
    - No dependency on GPU at import time — sentence-transformers will use
      CUDA if available, CPU otherwise, staying within the ~8 GB VRAM budget.
    """

    def __init__(
        self,
        encoder_name: str       = _DEFAULT_ENCODER,
        index_path:   Path | str = _DEFAULT_INDEX_PATH,
        meta_path:    Path | str = _DEFAULT_META_PATH,
    ) -> None:
        self._index_path = Path(index_path)
        self._meta_path  = Path(meta_path)
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[dict[str, Any]]    = []
        self._dim: int = 0

        log.info("Loading sentence encoder '%s' ...", encoder_name)
        try:
            self._encoder = SentenceTransformer(encoder_name)
            self._dim = self._encoder.get_sentence_embedding_dimension()
            log.info("Encoder ready | dim=%d", self._dim)
        except Exception as exc:
            log.error("Failed to load encoder '%s': %s", encoder_name, exc)
            raise

    # ── BaseRetriever interface ───────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Return True if the FAISS index is loaded and contains at least one vector."""
        return self._index is not None and self._index.ntotal > 0

    def retrieve(self, query: str, top_k: int = _DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """
        Return the top-k most semantically similar chunks for query.

        Args:
            query:  Natural language query string.
            top_k:  Maximum number of results to return.

        Returns:
            List of RetrievedChunk sorted by cosine similarity descending.

        Raises:
            RuntimeError: If the index is not loaded.
            ValueError:   If query is not a non-empty string.
        """
        if not self.is_ready():
            raise RuntimeError(
                "FAISS index not ready. "
                "Run: python faiss_retriever.py --build-index"
            )
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        # Clamp top_k to actual index size — prevents FAISS assertion errors
        effective_k = max(1, min(top_k, self._index.ntotal))

        t0 = time.perf_counter()

        query_vec: np.ndarray = self._encoder.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores_arr, indices_arr = self._index.search(query_vec, effective_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores_arr[0], indices_arr[0]):
            if idx < 0:
                # FAISS returns -1 when the index has fewer vectors than top_k
                continue
            results.append(
                RetrievedChunk.from_dict(self._chunks[idx], score=float(score))
            )

        # IndexFlatIP already returns descending order — sort explicitly for safety
        results.sort(key=lambda c: c.score, reverse=True)

        log.debug(
            "FAISS retrieve | query='%s' | top_k=%d | returned=%d | %.1f ms",
            query[:60],
            top_k,
            len(results),
            (time.perf_counter() - t0) * 1000,
        )
        return results

    # ── Index construction ────────────────────────────────────────────────────

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Encode all chunks and build the FAISS IndexFlatIP in memory.

        Call save() after this to persist to disk.

        Args:
            chunks: List of chunk dicts from ingestion.py / chunks.json.

        Raises:
            ValueError: If chunks is empty or schema is invalid.
        """
        if not chunks:
            raise ValueError("Cannot build FAISS index from an empty chunk list.")

        validate_chunk_schema(chunks[:5])  # public, from base_retriever

        log.info(
            "Encoding %d chunks | encoder='%s' | batch_size=%d ...",
            len(chunks),
            self._encoder.__class__.__name__,
            _ENCODE_BATCH_SIZE,
        )
        t0 = time.perf_counter()

        texts: list[str] = [c["text"] for c in chunks]
        embeddings: np.ndarray = self._encoder.encode(
            texts,
            batch_size=_ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)
        self._chunks = list(chunks)  # defensive copy

        log.info(
            "Index built | vectors=%d | dim=%d | elapsed=%.1fs",
            self._index.ntotal,
            self._dim,
            time.perf_counter() - t0,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        index_path: Path | str | None = None,
        meta_path:  Path | str | None = None,
    ) -> None:
        """
        Atomically persist the FAISS index and chunk metadata to disk.

        Writes .tmp files first then renames — safe against mid-write crashes.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if not self.is_ready():
            raise RuntimeError("Index not built. Call build_index() before save().")

        idx_p  = Path(index_path) if index_path else self._index_path
        meta_p = Path(meta_path)  if meta_path  else self._meta_path

        idx_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        # Atomically write FAISS index
        tmp_idx = idx_p.with_suffix(".tmp.faiss")
        try:
            faiss.write_index(self._index, str(tmp_idx))
            tmp_idx.replace(idx_p)
        except Exception:
            tmp_idx.unlink(missing_ok=True)
            raise

        # Atomically write metadata
        tmp_meta = meta_p.with_suffix(".tmp.pkl")
        try:
            with open(tmp_meta, "wb") as fh:
                pickle.dump(self._chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_meta.replace(meta_p)
        except Exception:
            tmp_meta.unlink(missing_ok=True)
            raise

        log.info(
            "Saved FAISS index → %s (%.1f MB) | metadata → %s (%.1f KB)",
            idx_p,  idx_p.stat().st_size  / (1024 ** 2),
            meta_p, meta_p.stat().st_size / 1024,
        )

    def load(
        self,
        index_path: Path | str | None = None,
        meta_path:  Path | str | None = None,
    ) -> None:
        """
        Load a previously saved FAISS index and chunk metadata from disk.

        Raises:
            FileNotFoundError: If either file is missing.
            RuntimeError:      If encoder/index dimension mismatch or
                               vector count ≠ metadata chunk count.
        """
        idx_p  = Path(index_path) if index_path else self._index_path
        meta_p = Path(meta_path)  if meta_path  else self._meta_path

        if not idx_p.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{idx_p}'. "
                "Run: python faiss_retriever.py --build-index"
            )
        if not meta_p.exists():
            raise FileNotFoundError(
                f"Metadata not found at '{meta_p}'. "
                "Run: python faiss_retriever.py --build-index"
            )

        log.info("Loading FAISS index from '%s' ...", idx_p)
        t0 = time.perf_counter()

        self._index = faiss.read_index(str(idx_p))

        # Encoder / index dimension sanity check
        if self._index.d != self._dim:
            raise RuntimeError(
                f"Encoder dim ({self._dim}) ≠ index dim ({self._index.d}). "
                "The index was built with a different encoder. "
                "Re-run: python faiss_retriever.py --build-index --rebuild"
            )

        with open(meta_p, "rb") as fh:
            self._chunks = pickle.load(fh)

        # Vector / metadata count consistency check
        if len(self._chunks) != self._index.ntotal:
            raise RuntimeError(
                f"Metadata ({len(self._chunks)} chunks) and FAISS index "
                f"({self._index.ntotal} vectors) are out of sync. "
                "Re-run: python faiss_retriever.py --build-index --rebuild"
            )

        log.info(
            "FAISS index loaded | vectors=%d | dim=%d | elapsed=%.2fs",
            self._index.ntotal,
            self._index.d,
            time.perf_counter() - t0,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vector_count(self) -> int:
        """Number of vectors currently in the FAISS index."""
        return self._index.ntotal if self._index is not None else 0

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension of the loaded encoder."""
        return self._dim


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: FAISSRetriever | None = None


def get_faiss_retriever(
    index_path:   Path | str = _DEFAULT_INDEX_PATH,
    meta_path:    Path | str = _DEFAULT_META_PATH,
    encoder_name: str        = _DEFAULT_ENCODER,
) -> FAISSRetriever:
    """
    Return the module-level singleton FAISSRetriever, loading the index on first call.

    Streamlit re-executes the script on every UI interaction. This singleton
    prevents re-loading the FAISS index (potentially hundreds of MB) on every rerun.

    Args:
        index_path:   Path to the saved .faiss file.
        meta_path:    Path to the saved metadata .pkl file.
        encoder_name: Sentence-transformers model name.

    Returns:
        Initialised FAISSRetriever with index already loaded.

    Raises:
        FileNotFoundError: propagated from load() if the index hasn't been built yet.
    """
    global _singleton
    if _singleton is None:
        log.info("Initialising FAISSRetriever singleton ...")
        instance = FAISSRetriever(
            encoder_name=encoder_name,
            index_path=index_path,
            meta_path=meta_path,
        )
        instance.load()
        _singleton = instance
        log.info(
            "FAISSRetriever singleton ready | %d vectors.", _singleton.vector_count
        )
    return _singleton


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAISS dense retriever — build index or run smoke-test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python faiss_retriever.py --build-index             # encode chunks + cache index
  python faiss_retriever.py --build-index --rebuild   # force re-encode all chunks
  python faiss_retriever.py                           # smoke-test (index must exist)
        """,
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Encode chunks.json, build FAISS index, and save to disk.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force full rebuild even if a valid index already exists.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/chunks.json"),  # canonical ingestion.py output path
        help="Path to chunks.json (default: data/processed/chunks.json)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=_DEFAULT_ENCODER,
        help=f"Sentence-transformers encoder to use (default: {_DEFAULT_ENCODER})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help=f"Number of results for smoke-test queries (default: {_DEFAULT_TOP_K})",
    )
    return parser.parse_args()


def _run_smoke_test(retriever: FAISSRetriever, top_k: int) -> None:
    """Run representative queries and print formatted results."""
    test_cases = [
        ("FACTUAL",    "What datasets were used for evaluation?"),
        ("CONCEPTUAL", "How does the attention mechanism work in transformers?"),
        ("COMPLEX",    "What accuracy is reported in the main experiments?"),
    ]

    print(f"\n{'─' * 70}")
    print(f"{'Type':<12} {'Score':>7}  {'chunk_id':<30}  Preview")
    print(f"{'─' * 70}")

    total = 0
    for qtype, query in test_cases:
        print(f"\n[{qtype}] '{query}'")
        results = retriever.retrieve(query, top_k=top_k)
        if not results:
            print("  (no results)")
            continue
        for chunk in results:
            preview = chunk.text[:80].replace("\n", " ")
            print(f"  {chunk.score:>7.4f}  {chunk.chunk_id:<30}  {preview}...")
            total += 1

    print(f"\n{'─' * 70}")
    print(f"Smoke-test complete. {total} total results across {len(test_cases)} queries.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    args = _parse_args()

    # ── --build-index mode ────────────────────────────────────────────────────
    if args.build_index:
        log.info("=== FAISS Index Build ===")

        if not args.chunks.exists():
            log.error(
                "Chunks file not found at '%s'. Run: python ingestion.py", args.chunks
            )
            sys.exit(1)

        with open(args.chunks, encoding="utf-8") as fh:
            chunks = json.load(fh)

        if not isinstance(chunks, list) or not chunks:
            log.error("Chunks file is empty or malformed.")
            sys.exit(1)

        # Skip rebuild if a valid index already exists (unless --rebuild forced)
        if not args.rebuild and _DEFAULT_INDEX_PATH.exists() and _DEFAULT_META_PATH.exists():
            log.info(
                "Existing FAISS index found at '%s'. Use --rebuild to force re-encoding.",
                _DEFAULT_INDEX_PATH,
            )
            sys.exit(0)

        try:
            retriever = FAISSRetriever(encoder_name=args.encoder)
            retriever.build_index(chunks)
            retriever.save()
        except Exception as exc:
            log.exception("Index build failed: %s", exc)
            sys.exit(1)

        if not retriever.is_ready():
            log.error("Build completed but retriever reports not-ready.")
            sys.exit(1)

        log.info(
            "Index build successful. %d vectors indexed. "
            "Ready for: python train_router.py",
            retriever.vector_count,
        )
        sys.exit(0)

    # ── Smoke-test mode ───────────────────────────────────────────────────────
    log.info("=== FAISS Smoke-Test ===")
    try:
        retriever = FAISSRetriever(encoder_name=args.encoder)
        retriever.load()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Failed to load FAISS index: %s", exc)
        sys.exit(1)

    if not retriever.is_ready():
        log.error("Retriever failed to initialise — run with --build-index first.")
        sys.exit(1)

    _run_smoke_test(retriever, top_k=args.top_k)
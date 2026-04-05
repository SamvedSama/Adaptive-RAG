"""
bm25_retriever.py — Sparse Keyword Retrieval Module
Owner: Samved Jain

Builds and queries a BM25Okapi index over the chunk corpus.
Implements BaseRetriever so adaptive_pipeline.py can treat it
interchangeably with FAISSRetriever.

Used for the "Single_Hop_BM25" routing path — fast, zero-GPU,
keyword-frequency matching best suited for factual queries.

Public API:
    retriever = get_bm25_retriever()           # module singleton
    chunks    = retriever.retrieve(query, k)   # → list[RetrievedChunk]
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from faiss_retriever import BaseRetriever, RetrievedChunk, _validate_chunk_schema

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

_DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks.json")
_DEFAULT_INDEX_PATH  = Path("data/bm25_index.pkl")
_DEFAULT_META_PATH   = Path("data/bm25_index_meta.json")
_DEFAULT_TOP_K       = 5

# Compiled once at module load — used by every tokenize() call
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ── Tokenizer ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Lowercase alphanumeric tokenizer.
    Strips punctuation and special chars — more robust than naive .split()
    which leaves tokens like "model," and "results." in the index.
    """
    return _TOKEN_RE.findall(text.lower())


# ── BM25Retriever ──────────────────────────────────────────────────────────────

class BM25Retriever(BaseRetriever):
    """
    Sparse BM25Okapi retriever.

    Loads chunks from disk on init, then either loads a valid cached index
    or rebuilds it. Index validity is checked against a stored chunk-count
    fingerprint — prevents the classic IndexError when chunks.json grows
    but the stale pickle is still on disk.

    The heavy BM25 build happens once; subsequent instantiations via the
    module singleton skip straight to the cached index load.
    """

    def __init__(
        self,
        chunks_path: Path | str = _DEFAULT_CHUNKS_PATH,
        index_path:  Path | str = _DEFAULT_INDEX_PATH,
        meta_path:   Path | str = _DEFAULT_META_PATH,
        rebuild:     bool       = False,
    ) -> None:
        self._chunks_path = Path(chunks_path)
        self._index_path  = Path(index_path)
        self._meta_path   = Path(meta_path)

        self._bm25:   BM25Okapi | None     = None
        self._chunks: list[dict[str, Any]] = []

        self._load_chunks()
        self._ensure_index(force_rebuild=rebuild)

    # ── BaseRetriever interface ────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._bm25 is not None and len(self._chunks) > 0

    def retrieve(self, query: str, top_k: int = _DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """
        Return top-k chunks ranked by BM25 relevance.

        Args:
            query:  Natural language query string.
            top_k:  Number of results to return.

        Returns:
            List of RetrievedChunk sorted by BM25 score descending.
            Zero-score chunks are excluded.

        Raises:
            RuntimeError: If the index is not ready.
            ValueError:   If query is empty.
        """
        if not self.is_ready():
            raise RuntimeError("BM25 index not ready. Initialisation must have failed.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        top_k = max(1, top_k)
        t0 = time.perf_counter()

        tokens = tokenize(query)
        if not tokens:
            log.warning("Query '%s' produced no tokens after cleaning — returning empty.", query)
            return []

        scores: list[float] = self._bm25.get_scores(tokens).tolist()

        # Integrity guard — should never fail if _ensure_index() passed
        if len(scores) != len(self._chunks):
            raise RuntimeError(
                f"Score/chunk mismatch: {len(scores)} scores vs {len(self._chunks)} chunks. "
                "Delete the BM25 index files and re-run to rebuild."
            )

        # Pair scores with chunks, filter zeros, sort, take top_k
        scored = sorted(
            (
                (score, idx)
                for idx, score in enumerate(scores)
                if score > 0.0
            ),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        results = [
            RetrievedChunk.from_dict(self._chunks[idx], score=round(score, 6))
            for score, idx in scored
        ]

        log.debug(
            "BM25 retrieve | top_k=%d | returned=%d | %.1f ms",
            top_k, len(results), (time.perf_counter() - t0) * 1000,
        )
        return results

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_chunks(self) -> None:
        """Load and validate chunks from disk."""
        if not self._chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found at '{self._chunks_path}'. Run ingestion.py first."
            )
        with open(self._chunks_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list) or not data:
            raise ValueError(f"Chunks file '{self._chunks_path}' is empty or malformed.")
        _validate_chunk_schema(data[:5])
        self._chunks = data
        log.info("Loaded %d chunks from '%s'.", len(self._chunks), self._chunks_path)

    def _ensure_index(self, force_rebuild: bool = False) -> None:
        """
        Load the cached BM25 index if it's valid, otherwise rebuild and save it.
        Validity is determined by comparing the stored chunk count to the current one.
        """
        if not force_rebuild and self._index_is_valid():
            self._load_index()
        else:
            self._build_and_save()

    def _index_is_valid(self) -> bool:
        """
        Return True if both index files exist and the stored chunk count
        matches the currently loaded chunk list.
        """
        if not self._index_path.exists() or not self._meta_path.exists():
            log.info("No cached BM25 index found — will build fresh.")
            return False
        try:
            with open(self._meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            saved_count = int(meta["chunk_count"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("Corrupt BM25 metadata (%s) — rebuilding.", exc)
            return False

        if saved_count != len(self._chunks):
            log.warning(
                "Stale BM25 index: built on %d chunks, current corpus has %d. Rebuilding.",
                saved_count, len(self._chunks),
            )
            return False

        log.info("Cached BM25 index is valid (%d chunks).", saved_count)
        return True

    def _build_and_save(self) -> None:
        """Tokenize corpus, build BM25Okapi, and atomically persist to disk."""
        log.info("Building BM25 index over %d chunks ...", len(self._chunks))
        t0 = time.perf_counter()

        tokenized_corpus = [tokenize(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

        log.info("BM25 index built in %.2fs.", time.perf_counter() - t0)
        self._save_index()

    def _save_index(self) -> None:
        """Atomically write the BM25 pickle and metadata JSON."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_idx  = self._index_path.with_suffix(".tmp.pkl")
        tmp_meta = self._meta_path.with_suffix(".tmp.json")

        try:
            with open(tmp_idx, "wb") as fh:
                pickle.dump(self._bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_idx.replace(self._index_path)
        except Exception:
            tmp_idx.unlink(missing_ok=True)
            raise

        try:
            with open(tmp_meta, "w", encoding="utf-8") as fh:
                json.dump({"chunk_count": len(self._chunks)}, fh)
            tmp_meta.replace(self._meta_path)
        except Exception:
            tmp_meta.unlink(missing_ok=True)
            raise

        log.info(
            "BM25 index saved → %s (%.1f KB) | metadata → %s",
            self._index_path,
            self._index_path.stat().st_size / 1024,
            self._meta_path,
        )

    def _load_index(self) -> None:
        """Deserialise the BM25 pickle from disk."""
        log.info("Loading cached BM25 index from '%s' ...", self._index_path)
        t0 = time.perf_counter()
        try:
            with open(self._index_path, "rb") as fh:
                self._bm25 = pickle.load(fh)
        except (pickle.UnpicklingError, EOFError, Exception) as exc:
            log.error("Failed to deserialise BM25 index: %s — rebuilding.", exc)
            self._build_and_save()
            return
        log.info("BM25 index loaded in %.2fs.", time.perf_counter() - t0)


# ── Module-level singleton ─────────────────────────────────────────────────────

_singleton: BM25Retriever | None = None


def get_bm25_retriever(
    chunks_path: Path | str = _DEFAULT_CHUNKS_PATH,
    index_path:  Path | str = _DEFAULT_INDEX_PATH,
    meta_path:   Path | str = _DEFAULT_META_PATH,
    rebuild:     bool       = False,
) -> BM25Retriever:
    """
    Return a module-level singleton BM25Retriever.

    Streamlit re-executes scripts on every interaction — this ensures the
    BM25 index (which can take several seconds to build) is loaded only once
    per process lifetime, regardless of how many times the UI rerenders.
    """
    global _singleton
    if _singleton is None:
        log.info("Initialising BM25Retriever singleton ...")
        _singleton = BM25Retriever(
            chunks_path=chunks_path,
            index_path=index_path,
            meta_path=meta_path,
            rebuild=rebuild,
        )
        log.info("BM25Retriever singleton ready | %d chunks indexed.", _singleton.chunk_count)
    return _singleton


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    retriever = BM25Retriever()
    if not retriever.is_ready():
        log.error("Retriever failed to initialise.")
        sys.exit(1)

    test_cases = [
        ("FACTUAL",    "What datasets were used for evaluation?"),
        ("CONCEPTUAL", "How does the attention mechanism work in transformers?"),
        ("COMPLEX",    "What accuracy is reported in the main experiments?"),
    ]

    print(f"\n{'─'*65}")
    print(f"{'Type':<12} {'Score':>6}  {'chunk_id':<30}  Preview")
    print(f"{'─'*65}")

    for qtype, query in test_cases:
        print(f"\n[{qtype}] '{query}'")
        results = retriever.retrieve(query, top_k=3)
        if not results:
            print("  (no results)")
            continue
        for chunk in results:
            preview = chunk.text[:80].replace("\n", " ")
            print(f"  {chunk.score:>6.4f}  {chunk.chunk_id:<30}  {preview}...")

    print(f"\n{'─'*65}")
    print("BM25 smoke-test complete.")
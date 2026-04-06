"""
bm25_retriever.py — Sparse Keyword Retrieval Module

Builds and queries a BM25Okapi index over the chunk corpus.
Implements BaseRetriever so adaptive_pipeline.py can treat it
interchangeably with FAISSRetriever.

Used for the "Single_Hop_BM25" routing path — fast, zero-GPU,
keyword-frequency matching best suited for factual queries.

Public API:
    retriever = get_bm25_retriever()           # module singleton
    chunks    = retriever.retrieve(query, k)   # → list[RetrievedChunk]

CLI:
    python bm25_retriever.py --build-index            # build and cache index
    python bm25_retriever.py --build-index --rebuild  # force-rebuild even if cached
    python bm25_retriever.py                          # smoke-test only
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import time
import concurrent.futures
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

# ── Shared types live in base_retriever, NOT in faiss_retriever ───────────────
# This keeps bm25_retriever fully independent of FAISS / GPU dependencies.
# faiss_retriever.py should also import from base_retriever.
from base_retriever import BaseRetriever, RetrievedChunk, validate_chunk_schema

log = logging.getLogger(__name__)

__all__ = ["BM25Retriever", "get_bm25_retriever", "tokenize"]

# ── Paths ─────────────────────────────────────────────────────────────────────
# Canonical path matches ingestion.py output: data/chunks/chunks.json
_DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks.json")
_DEFAULT_INDEX_PATH  = Path("data/bm25_index.pkl")
_DEFAULT_META_PATH   = Path("data/bm25_index_meta.json")
_DEFAULT_TOP_K       = 5

# Compiled once at module load — used by every tokenize() call
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Lowercase alphanumeric tokenizer.

    Strips punctuation and special chars — more robust than naive .split()
    which leaves tokens like "model," and "results." in the index.

    Args:
        text: Raw string to tokenize.

    Returns:
        List of lowercase alphanumeric tokens.
    """
    return _TOKEN_RE.findall(text.lower())


# ── BM25Retriever ─────────────────────────────────────────────────────────────

class BM25Retriever(BaseRetriever):
    """
    Sparse BM25Okapi retriever.

    Loads chunks from disk on init, then either loads a valid cached index
    or rebuilds it. Index validity is checked against a stored chunk-count
    fingerprint — prevents the classic IndexError when chunks.json grows
    but the stale pickle is still on disk.

    The heavy BM25 build happens once; subsequent instantiations via the
    module singleton skip straight to the cached index load.

    Intentionally has NO dependency on faiss_retriever, GPU, or VRAM.
    This preserves the "Single_Hop_BM25 = zero GPU" contract.
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

    # ── BaseRetriever interface ───────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Return True if the index and chunks are loaded and consistent."""
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
            Returns an empty list (not an error) if the query tokenizes to nothing.

        Raises:
            RuntimeError: If the index is not initialised.
            ValueError:   If query is not a string.
        """
        if not self.is_ready():
            raise RuntimeError(
                "BM25 index not ready. Run: python bm25_retriever.py --build-index"
            )
        if not isinstance(query, str):
            raise ValueError(f"query must be a str, got {type(query).__name__}.")

        top_k = max(1, top_k)
        t0    = time.perf_counter()

        tokens = tokenize(query.strip())
        if not tokens:
            log.warning(
                "Query '%s' produced no tokens after cleaning — returning empty list.", query
            )
            return []

        scores: list[float] = self._bm25.get_scores(tokens).tolist()

        # Integrity guard — should never fire if _ensure_index() passed correctly
        if len(scores) != len(self._chunks):
            raise RuntimeError(
                f"Score/chunk count mismatch: {len(scores)} scores vs "
                f"{len(self._chunks)} chunks. "
                "Delete index files and re-run: python bm25_retriever.py --build-index"
            )

        # Pair scores with chunk indices, discard zero-score results, take top_k
        scored = sorted(
            ((score, idx) for idx, score in enumerate(scores) if score > 0.0),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        results = [
            RetrievedChunk.from_dict(self._chunks[idx], score=round(score, 6))
            for score, idx in scored
        ]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.debug(
            "BM25 retrieve | query='%s' | top_k=%d | returned=%d | %.1f ms",
            query[:60],
            top_k,
            len(results),
            elapsed_ms,
        )
        return results

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently loaded in this retriever."""
        return len(self._chunks)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_chunks(self) -> None:
        """Load and validate chunks from disk."""
        if not self._chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found at '{self._chunks_path}'. "
                "Run: python ingestion.py"
            )
        with open(self._chunks_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list) or not data:
            raise ValueError(
                f"Chunks file '{self._chunks_path}' is empty or not a JSON array."
            )
        validate_chunk_schema(data[:5])  # public symbol from base_retriever
        self._chunks = data
        log.info("Loaded %d chunks from '%s'.", len(self._chunks), self._chunks_path)

    def _ensure_index(self, force_rebuild: bool = False) -> None:
        """
        Load the cached BM25 index if it is valid, otherwise rebuild and save it.
        Validity check compares the stored chunk count to the current corpus size.
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
            log.info("No cached BM25 index found — will build from scratch.")
            return False
        try:
            with open(self._meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            saved_count = int(meta["chunk_count"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            log.warning("Corrupt or unreadable BM25 metadata (%s) — rebuilding.", exc)
            return False

        if saved_count != len(self._chunks):
            log.warning(
                "Stale BM25 index: built on %d chunks, current corpus has %d. Rebuilding.",
                saved_count,
                len(self._chunks),
            )
            return False

        log.info("Cached BM25 index is valid (%d chunks).", saved_count)
        return True

    def _build_and_save(self) -> None:
        """Tokenize corpus, build BM25Okapi index, and persist atomically to disk."""
        log.info("Building BM25 index over %d chunks ...", len(self._chunks))
        t0 = time.perf_counter()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            tokenized_corpus = list(executor.map(tokenize, [c["text"] for c in self._chunks], chunksize=100))
            
        self._bm25 = BM25Okapi(tokenized_corpus)

        elapsed = time.perf_counter() - t0
        log.info("BM25 index built in %.2fs.", elapsed)
        self._save_index()

    def _save_index(self) -> None:
        """Atomically write the BM25 pickle and metadata JSON to disk."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_idx  = self._index_path.with_suffix(".tmp.pkl")
        tmp_meta = self._meta_path.with_suffix(".tmp.json")

        # Write pickle atomically
        try:
            with open(tmp_idx, "wb") as fh:
                pickle.dump(self._bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_idx.replace(self._index_path)
        except Exception:
            tmp_idx.unlink(missing_ok=True)
            raise

        # Write metadata atomically
        try:
            with open(tmp_meta, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "chunk_count": len(self._chunks),
                        "chunks_source": str(self._chunks_path),
                    },
                    fh,
                    indent=2,
                )
            tmp_meta.replace(self._meta_path)
        except Exception:
            tmp_meta.unlink(missing_ok=True)
            raise

        size_kb = self._index_path.stat().st_size / 1024
        log.info(
            "BM25 index saved → %s (%.1f KB) | metadata → %s",
            self._index_path,
            size_kb,
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
            log.error(
                "Failed to deserialise BM25 index (%s) — falling back to rebuild.", exc
            )
            self._build_and_save()
            return
        log.info("BM25 index loaded in %.2fs.", time.perf_counter() - t0)


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: BM25Retriever | None = None


def get_bm25_retriever(
    chunks_path: Path | str = _DEFAULT_CHUNKS_PATH,
    index_path:  Path | str = _DEFAULT_INDEX_PATH,
    meta_path:   Path | str = _DEFAULT_META_PATH,
    rebuild:     bool       = False,
) -> BM25Retriever:
    """
    Return the module-level singleton BM25Retriever.

    Streamlit re-executes the script on every UI interaction. This singleton
    ensures the BM25 index is loaded only once per process lifetime, regardless
    of how many times the UI re-renders.

    Args:
        chunks_path: Path to chunks.json (override for testing).
        index_path:  Path to BM25 pickle file.
        meta_path:   Path to index metadata JSON.
        rebuild:     If True, forces a fresh rebuild even if a cached index exists.
                     NOTE: rebuild=True replaces an existing singleton.

    Returns:
        Initialised BM25Retriever ready for querying.
    """
    global _singleton
    # Allow rebuild=True to replace a stale singleton (e.g. after ingestion re-run)
    if _singleton is None or rebuild:
        log.info("Initialising BM25Retriever singleton (rebuild=%s) ...", rebuild)
        _singleton = BM25Retriever(
            chunks_path=chunks_path,
            index_path=index_path,
            meta_path=meta_path,
            rebuild=rebuild,
        )
        log.info(
            "BM25Retriever singleton ready | %d chunks indexed.", _singleton.chunk_count
        )
    return _singleton


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BM25 sparse retriever — build index or run smoke-test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bm25_retriever.py --build-index            # build & cache index
  python bm25_retriever.py --build-index --rebuild  # force full rebuild
  python bm25_retriever.py                          # smoke-test (index must exist)
        """,
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build (or verify) the BM25 index and exit.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if a valid cached index already exists.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=_DEFAULT_CHUNKS_PATH,
        help=f"Path to chunks.json (default: {_DEFAULT_CHUNKS_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help=f"Number of results for smoke-test queries (default: {_DEFAULT_TOP_K})",
    )
    return parser.parse_args()


def _run_smoke_test(retriever: BM25Retriever, top_k: int) -> None:
    """Run a set of representative queries and print formatted results."""
    test_cases = [
        ("FACTUAL",    "What datasets were used for evaluation?"),
        ("CONCEPTUAL", "How does the attention mechanism work in transformers?"),
        ("COMPLEX",    "What accuracy is reported in the main experiments?"),
    ]

    print(f"\n{'─' * 70}")
    print(f"{'Type':<12} {'Score':>7}  {'chunk_id':<30}  Preview")
    print(f"{'─' * 70}")

    total_results = 0
    for qtype, query in test_cases:
        print(f"\n[{qtype}] '{query}'")
        results = retriever.retrieve(query, top_k=top_k)
        if not results:
            print("  (no results — query may not match corpus vocabulary)")
            continue
        for chunk in results:
            preview = chunk.text[:80].replace("\n", " ")
            print(f"  {chunk.score:>7.4f}  {chunk.chunk_id:<30}  {preview}...")
            total_results += 1

    print(f"\n{'─' * 70}")
    print(f"Smoke-test complete. Returned {total_results} total results across {len(test_cases)} queries.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    args = _parse_args()

    # ── --build-index mode: build, validate, exit ─────────────────────────────
    if args.build_index:
        log.info("=== BM25 Index Build ===")
        try:
            retriever = BM25Retriever(
                chunks_path=args.chunks,
                rebuild=args.rebuild,
            )
        except FileNotFoundError as exc:
            log.error("%s", exc)
            sys.exit(1)
        except Exception as exc:
            log.exception("Unexpected error during index build: %s", exc)
            sys.exit(1)

        if not retriever.is_ready():
            log.error("Index build completed but retriever reports not-ready.")
            sys.exit(1)

        log.info(
            "Index build successful. %d chunks indexed. "
            "Ready for: python faiss_retriever.py --build-index",
            retriever.chunk_count,
        )
        sys.exit(0)

    # ── Smoke-test mode: load existing index and run test queries ─────────────
    log.info("=== BM25 Smoke-Test ===")
    try:
        retriever = BM25Retriever(chunks_path=args.chunks)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error loading retriever: %s", exc)
        sys.exit(1)

    if not retriever.is_ready():
        log.error("Retriever failed to initialise — run with --build-index first.")
        sys.exit(1)

    _run_smoke_test(retriever, top_k=args.top_k)
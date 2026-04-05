"""
faiss_retriever.py — Dense Retrieval Module
Owner: Samved Jain

Builds and queries a FAISS flat inner-product index using sentence-transformers.
Implements the BaseRetriever ABC so adaptive_pipeline.py can swap retrievers
without changing call sites.

Chunk schema (input & output) — do NOT alter:
{
    "chunk_id": str,
    "text":     str,
    "source":   str,
    "position": int,
    "score":    float   # cosine similarity, populated by retrieve()
}

Public API:
    retriever = get_faiss_retriever()          # module singleton
    chunks    = retriever.retrieve(query, k)   # → list[RetrievedChunk]

Build-time API (called by build_index.py):
    retriever = FAISSRetriever()
    retriever.build_index(chunks)
    retriever.save()
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_ENCODER   = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_INDEX_DIR = Path("data/faiss_index")
_DEFAULT_TOP_K     = 5
_ENCODE_BATCH_SIZE = 64


# ── Shared chunk schema ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievedChunk:
    """
    Typed, immutable representation of one retrieved chunk.
    Replaces the raw dict[str, Any] that was passed around before.
    Downstream code (reranker, pipeline) imports this class directly.
    """
    chunk_id: str
    text:     str
    source:   str
    position: int
    score:    float

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to the agreed JSON schema for logging / evaluation."""
        return {
            "chunk_id": self.chunk_id,
            "text":     self.text,
            "source":   self.source,
            "position": self.position,
            "score":    self.score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], score: float | None = None) -> "RetrievedChunk":
        return cls(
            chunk_id=d["chunk_id"],
            text=d["text"],
            source=d["source"],
            position=int(d["position"]),
            score=float(score if score is not None else d.get("score", 0.0)),
        )


# ── Abstract base retriever ────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """
    Common interface for all retriever engines.
    adaptive_pipeline.py types its retriever slots as BaseRetriever,
    so FAISSRetriever and BM25Retriever are interchangeable.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = _DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """Return top_k chunks ranked by relevance to query."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the retriever is loaded and ready to serve queries."""


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FAISSConfig:
    index_dir:    Path = _DEFAULT_ENCODER and _DEFAULT_INDEX_DIR  # evaluated at runtime
    encoder_name: str  = _DEFAULT_ENCODER
    top_k:        int  = _DEFAULT_TOP_K
    batch_size:   int  = _ENCODE_BATCH_SIZE

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be ≥ 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be ≥ 1.")

    @property
    def index_path(self) -> Path:
        return Path("data/faiss_index") if self.index_dir is _DEFAULT_INDEX_DIR \
               else self.index_dir / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        return Path("data/faiss_index") if self.index_dir is _DEFAULT_INDEX_DIR \
               else self.index_dir / "metadata.pkl"


# Simpler fixed paths used by default (avoids the dataclass property quirk above)
_DEFAULT_INDEX_PATH = _DEFAULT_INDEX_DIR / "index.faiss"
_DEFAULT_META_PATH  = _DEFAULT_INDEX_DIR / "metadata.pkl"


# ── FAISSRetriever ─────────────────────────────────────────────────────────────

class FAISSRetriever(BaseRetriever):
    """
    Dense retriever backed by a FAISS IndexFlatIP (exact cosine similarity).

    Design decisions:
    - IndexFlatIP on L2-normalised vectors gives exact cosine similarity
      without the overhead of IndexFlatL2.
    - All encoding is float32 — half the memory of float64 with no accuracy loss
      for MiniLM embeddings.
    - The encoder is loaded once in __init__; the FAISS index is loaded lazily
      via load() or built via build_index().
    """

    def __init__(
        self,
        encoder_name: str = _DEFAULT_ENCODER,
        index_path:   Path | str = _DEFAULT_INDEX_PATH,
        meta_path:    Path | str = _DEFAULT_META_PATH,
    ) -> None:
        self._index_path = Path(index_path)
        self._meta_path  = Path(meta_path)
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[dict[str, Any]] = []
        self._dim: int = 0

        log.info("Loading sentence encoder '%s' ...", encoder_name)
        try:
            self._encoder = SentenceTransformer(encoder_name)
            self._dim = self._encoder.get_sentence_embedding_dimension()
            log.info("Encoder ready | dim=%d", self._dim)
        except Exception as exc:
            log.error("Failed to load encoder '%s': %s", encoder_name, exc)
            raise

    # ── BaseRetriever interface ────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def retrieve(self, query: str, top_k: int = _DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """
        Return the top-k most semantically similar chunks for query.

        Args:
            query:  Natural language query string.
            top_k:  Number of results (default 5).

        Returns:
            List of RetrievedChunk sorted by cosine similarity descending.

        Raises:
            RuntimeError: If the index is not ready.
            ValueError:   If query is empty.
        """
        if not self.is_ready():
            raise RuntimeError(
                "FAISS index not ready. Call build_index() or load() first."
            )
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        top_k = max(1, min(top_k, self._index.ntotal))  # clamp to index size

        t0 = time.perf_counter()

        query_vec = self._encoder.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores, indices = self._index.search(query_vec, top_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 when index has < top_k vectors
                continue
            results.append(
                RetrievedChunk.from_dict(self._chunks[idx], score=float(score))
            )

        # FAISS already returns descending order for IP; be explicit
        results.sort(key=lambda c: c.score, reverse=True)

        log.debug(
            "FAISS retrieve | top_k=%d | returned=%d | %.1f ms",
            top_k, len(results), (time.perf_counter() - t0) * 1000,
        )
        return results

    # ── Index construction ─────────────────────────────────────────────────────

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Encode all chunks and populate the FAISS index.

        Args:
            chunks: List of chunk dicts from ingestion.py.

        Raises:
            ValueError: If chunks is empty or missing required keys.
        """
        if not chunks:
            raise ValueError("Cannot build FAISS index from an empty chunk list.")

        _validate_chunk_schema(chunks[:5])  # spot-check first 5

        log.info("Encoding %d chunks (batch_size=%d) ...", len(chunks), _ENCODE_BATCH_SIZE)
        t0 = time.perf_counter()

        texts = [c["text"] for c in chunks]
        embeddings: np.ndarray = self._encoder.encode(
            texts,
            batch_size=_ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)
        self._chunks = list(chunks)  # store a copy

        log.info(
            "Index built | vectors=%d | dim=%d | %.1fs",
            self._index.ntotal, self._dim, time.perf_counter() - t0,
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(
        self,
        index_path: Path | str | None = None,
        meta_path:  Path | str | None = None,
    ) -> None:
        """
        Atomically persist the FAISS index and chunk metadata.
        Writes to .tmp files first, then renames — safe against mid-write crashes.
        """
        if not self.is_ready():
            raise RuntimeError("Index not built. Call build_index() first.")

        idx_path  = Path(index_path)  if index_path else self._index_path
        meta_path = Path(meta_path)   if meta_path  else self._meta_path

        idx_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Write FAISS index
        tmp_idx = idx_path.with_suffix(".tmp.faiss")
        try:
            faiss.write_index(self._index, str(tmp_idx))
            tmp_idx.replace(idx_path)
        except Exception:
            tmp_idx.unlink(missing_ok=True)
            raise

        # Write metadata
        tmp_meta = meta_path.with_suffix(".tmp.pkl")
        try:
            with open(tmp_meta, "wb") as fh:
                pickle.dump(self._chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_meta.replace(meta_path)
        except Exception:
            tmp_meta.unlink(missing_ok=True)
            raise

        log.info(
            "Saved FAISS index → %s (%.1f MB) | metadata → %s (%.1f KB)",
            idx_path,  idx_path.stat().st_size  / (1024 ** 2),
            meta_path, meta_path.stat().st_size / 1024,
        )

    def load(
        self,
        index_path: Path | str | None = None,
        meta_path:  Path | str | None = None,
    ) -> None:
        """
        Load a previously saved FAISS index and chunk metadata.

        Raises:
            FileNotFoundError: If either file is missing.
            RuntimeError:      If the loaded index dimension mismatches the encoder.
        """
        idx_path  = Path(index_path)  if index_path else self._index_path
        meta_path = Path(meta_path)   if meta_path  else self._meta_path

        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index not found at '{idx_path}'.")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at '{meta_path}'.")

        log.info("Loading FAISS index from '%s' ...", idx_path)
        t0 = time.perf_counter()

        self._index = faiss.read_index(str(idx_path))

        # Dimension sanity check — catches encoder/index mismatch
        if self._index.d != self._dim:
            raise RuntimeError(
                f"Encoder dim ({self._dim}) ≠ index dim ({self._index.d}). "
                "Re-run build_index.py with the current encoder."
            )

        with open(meta_path, "rb") as fh:
            self._chunks = pickle.load(fh)

        if len(self._chunks) != self._index.ntotal:
            raise RuntimeError(
                f"Metadata ({len(self._chunks)} chunks) and index "
                f"({self._index.ntotal} vectors) are out of sync. "
                "Re-run build_index.py."
            )

        log.info(
            "FAISS index loaded | vectors=%d | dim=%d | %.2fs",
            self._index.ntotal, self._index.d, time.perf_counter() - t0,
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def vector_count(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def embedding_dim(self) -> int:
        return self._dim


# ── Schema validator (shared with bm25_retriever) ─────────────────────────────

_REQUIRED_CHUNK_KEYS = frozenset({"chunk_id", "text", "source", "position"})


def _validate_chunk_schema(chunks: list[dict[str, Any]]) -> None:
    for i, chunk in enumerate(chunks):
        missing = _REQUIRED_CHUNK_KEYS - chunk.keys()
        if missing:
            raise ValueError(f"Chunk at index {i} missing keys: {missing}")
        if not str(chunk.get("text", "")).strip():
            raise ValueError(f"Chunk '{chunk.get('chunk_id', i)}' has empty text.")


# ── Module-level singleton ─────────────────────────────────────────────────────

_singleton: FAISSRetriever | None = None


def get_faiss_retriever(
    index_path: Path | str = _DEFAULT_INDEX_PATH,
    meta_path:  Path | str = _DEFAULT_META_PATH,
    encoder_name: str      = _DEFAULT_ENCODER,
) -> FAISSRetriever:
    """
    Return a module-level singleton FAISSRetriever, loading the index on first call.

    Streamlit re-executes scripts on every interaction — this prevents
    re-loading the FAISS index (potentially hundreds of MB) on every rerun.

    Raises:
        FileNotFoundError: propagated from FAISSRetriever.load() if index missing.
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
        log.info("FAISSRetriever singleton ready | %d vectors.", _singleton.vector_count)
    return _singleton


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    chunk_path = Path("data/chunks/chunks.json")
    if not chunk_path.exists():
        log.error("Chunks not found at '%s'. Run ingestion.py first.", chunk_path)
        sys.exit(1)

    with open(chunk_path, encoding="utf-8") as fh:
        chunks = json.load(fh)

    log.info("Loaded %d chunks.", len(chunks))

    retriever = FAISSRetriever()
    retriever.build_index(chunks)
    retriever.save()

    query = "How do transformer attention mechanisms work?"
    log.info("\n── Smoke-test query: '%s' ──", query)
    results = retriever.retrieve(query, top_k=3)

    for i, chunk in enumerate(results, 1):
        print(f"\n[{i}] {chunk.chunk_id}  score={chunk.score:.4f}  source={chunk.source}")
        print(f"    {chunk.text[:120]}...")
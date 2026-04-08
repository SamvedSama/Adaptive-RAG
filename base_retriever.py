"""
base_retriever.py — Shared Retriever Contracts

Defines the abstract base class and shared data types used by ALL retrievers
in Eco-RAG. Both bm25_retriever.py and faiss_retriever.py import from here.

This module has ZERO heavy dependencies (no FAISS, no torch, no sentence-transformers)
so it can be safely imported anywhere — including on CPU-only or low-VRAM paths.

Public API:
    BaseRetriever        — ABC that all retrievers implement
    RetrievedChunk       — Dataclass returned by every retrieve() call
    validate_chunk_schema — Validates raw chunk dicts from chunks.json
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["BaseRetriever", "RetrievedChunk", "validate_chunk_schema"]

# Required keys in every chunk dict loaded from chunks.json
_REQUIRED_CHUNK_KEYS = {"chunk_id", "text", "source"}


# ── RetrievedChunk ────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    Uniform result type returned by all retrievers.

    Using a dataclass rather than a plain dict gives:
      - Attribute access (chunk.text vs chunk["text"])
      - Easy repr/equality for debugging and tests
      - A stable contract that adaptive_pipeline.py can depend on

    Fields:
        chunk_id:  Unique identifier matching the source chunks.json entry.
        text:      Raw chunk text to be passed to the LLM as context.
        source:    Document / PDF source identifier.
        score:     Retrieval score (BM25 okapi score, cosine sim, etc.).
                   Higher = more relevant. Scale is retriever-dependent.
        metadata:  Any extra fields from the original chunk dict (page, section, etc.)
    """
    chunk_id: str
    text:     str
    source:   str
    score:    float             = 0.0
    metadata: dict[str, Any]   = field(default_factory=dict)

    @classmethod
    def from_dict(cls, chunk: dict[str, Any], score: float = 0.0) -> "RetrievedChunk":
        """
        Construct a RetrievedChunk from a raw chunk dict (as loaded from chunks.json).

        Known keys (chunk_id, text, source) are promoted to dataclass fields.
        All other keys are collected into `metadata` for downstream use.

        Args:
            chunk: A single chunk dict from chunks.json.
            score: Retrieval score assigned by the calling retriever.

        Returns:
            RetrievedChunk instance.
        """
        known = {"chunk_id", "text", "source"}
        return cls(
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            source=chunk["source"],
            score=score,
            metadata={k: v for k, v in chunk.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to a plain dict (useful for logging / RAGAS eval)."""
        return {
            "chunk_id": self.chunk_id,
            "text":     self.text,
            "source":   self.source,
            "score":    self.score,
            **self.metadata,
        }

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"RetrievedChunk(chunk_id={self.chunk_id!r}, score={self.score:.4f}, "
            f"text={preview!r}...)"
        )


# ── Schema validation ─────────────────────────────────────────────────────────

def validate_chunk_schema(chunks: list[dict[str, Any]]) -> None:
    """
    Validate that a sample of chunks contains all required keys.

    Raises ValueError with a descriptive message if validation fails.
    Only inspects the provided sample — call with [:5] for a fast check.

    Args:
        chunks: A list (or sample) of raw chunk dicts.

    Raises:
        ValueError: If any chunk is missing a required key.
    """
    for i, chunk in enumerate(chunks):
        missing = _REQUIRED_CHUNK_KEYS - chunk.keys()
        if missing:
            raise ValueError(
                f"Chunk at index {i} is missing required keys: {sorted(missing)}. "
                f"Present keys: {sorted(chunk.keys())}. "
                "Re-run ingestion.py to regenerate chunks.json."
            )
    log.debug("Chunk schema validation passed for %d sample chunks.", len(chunks))


# ── BaseRetriever ─────────────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """
    Abstract base class for all Eco-RAG retrievers.

    Concrete implementations:
        BM25Retriever  — sparse keyword search (bm25_retriever.py)
        FAISSRetriever — dense embedding search (faiss_retriever.py)

    adaptive_pipeline.py accepts any BaseRetriever — this is the seam that
    allows the router to swap retrieval strategies without knowing their internals.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievedChunk, sorted by descending relevance score.
            May return fewer than top_k results if the corpus is small or
            no chunks exceed the relevance threshold.

        Raises:
            RuntimeError: If the retriever is not ready (index not built).
            ValueError:   If query is invalid.
        """

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Return True if the retriever is fully initialised and ready to serve queries.

        adaptive_pipeline.py checks this before routing to any retriever path.
        """
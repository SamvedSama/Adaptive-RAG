"""
faiss_retriever.py — Dense Retrieval Module
Owner: Samved Jain

Builds and queries a FAISS index using sentence-transformers (all-MiniLM-L6-v2).
Accepts chunks produced by ingestion.py and returns ranked chunk lists
conforming to the agreed chunk schema.

Chunk schema (input & output):
{
    "chunk_id": str,
    "text":     str,
    "source":   str,
    "position": int,
    "score":    float   # cosine similarity score, populated here
}
"""

import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
INDEX_PATH = "data/faiss_index/index.faiss"
METADATA_PATH = "data/faiss_index/metadata.pkl"


# ---------------------------------------------------------------------------
# FAISSRetriever
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """
    Dense retriever backed by a FAISS flat inner-product index.

    Why FAISS?
    - Exact nearest-neighbour search with cosine similarity at low latency.
    - Runs fully on CPU (faiss-cpu), respecting the 8GB VRAM constraint.
    - IndexFlatIP operates on L2-normalised vectors, giving cosine similarity.

    Why all-MiniLM-L6-v2?
    - 22M parameters — tiny, fits easily within memory.
    - 384-dimensional embeddings — fast to encode and store.
    - Strong semantic retrieval quality for short passages.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialise the retriever and load the embedding model.

        Args:
            model_name: HuggingFace model identifier for the encoder.
        """
        print(f"[FAISSRetriever] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()

        # FAISS index — built later via build_index()
        self.index: Optional[faiss.IndexFlatIP] = None

        # Parallel list of chunk dicts, index i ↔ FAISS row i
        self.chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Encode all chunks and build the FAISS index.

        Args:
            chunks: List of chunk dicts from ingestion.py.
                    Each must have at least "chunk_id", "text", "source",
                    "position" keys. "score" is ignored here.

        Design note:
            IndexFlatIP stores raw inner products.  We L2-normalise every
            vector before insertion so inner product == cosine similarity.
            This avoids the overhead of IndexFlatL2 + manual normalisation
            at query time.
        """
        if not chunks:
            raise ValueError("Cannot build FAISS index from an empty chunk list.")

        print(f"[FAISSRetriever] Encoding {len(chunks)} chunks …")
        texts = [c["text"] for c in chunks]

        # Encode in batches; show a progress bar for large corpora
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → cosine sim via IP
        ).astype(np.float32)

        # Build flat index (exact search — no approximation)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)  # type: ignore[arg-type]
        self.chunks = chunks

        print(f"[FAISSRetriever] Index built: {self.index.ntotal} vectors, "
              f"dim={self.embedding_dim}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str = INDEX_PATH, meta_path: str = METADATA_PATH) -> None:
        """
        Persist the FAISS index and chunk metadata to disk.

        Args:
            index_path: Path for the .faiss binary file.
            meta_path:  Path for the pickled chunk list.
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"[FAISSRetriever] Saved index → {index_path}")
        print(f"[FAISSRetriever] Saved metadata → {meta_path}")

    def load(self, index_path: str = INDEX_PATH, meta_path: str = METADATA_PATH) -> None:
        """
        Load a previously saved FAISS index and chunk metadata from disk.

        Args:
            index_path: Path to the .faiss binary file.
            meta_path:  Path to the pickled chunk list.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at: {meta_path}")

        self.index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"[FAISSRetriever] Loaded index with {self.index.ntotal} vectors.")

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most semantically similar chunks for a query.

        Args:
            query:  Natural language query string.
            top_k:  Number of results to return (default 5, per spec).

        Returns:
            List of chunk dicts sorted by score descending.
            The "score" field is set to the cosine similarity (0–1).

        Raises:
            RuntimeError: If the index has not been built or loaded.
        """
        if self.index is None:
            raise RuntimeError("Index not ready. Call build_index() or load() first.")

        # Encode and normalise the query vector
        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # FAISS search returns (scores, row_indices) arrays of shape (1, top_k)
        scores, indices = self.index.search(query_vec, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 when the index has fewer than top_k vectors
                continue
            chunk = dict(self.chunks[idx])   # shallow copy — don't mutate stored data
            chunk["score"] = float(round(score, 6))
            results.append(chunk)

        # Already sorted descending by FAISS, but be explicit
        results.sort(key=lambda c: c["score"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_faiss_retriever(chunks: List[Dict[str, Any]]) -> FAISSRetriever:
    """
    One-shot helper: build and return a FAISSRetriever from a chunk list.

    Args:
        chunks: Output of ingestion.py — list of chunk dicts.

    Returns:
        A ready-to-query FAISSRetriever instance.
    """
    retriever = FAISSRetriever()
    retriever.build_index(chunks)
    return retriever


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Build FAISS index from real chunks
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    CHUNK_PATH = "data/chunks/chunks.json"

    if not os.path.exists(CHUNK_PATH):
        raise FileNotFoundError(
            f"Chunk file not found at {CHUNK_PATH}. Run ingestion.py first."
        )

    print(f"[FAISSRetriever] Loading chunks from {CHUNK_PATH}")

    with open(CHUNK_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[FAISSRetriever] Loaded {len(chunks)} chunks")

    # Build index
    retriever = FAISSRetriever()
    retriever.build_index(chunks)

    # Save index
    retriever.save()

    # Test retrieval
    print("\n--- Test Query ---")
    query = "How do transformer attention mechanisms work?"

    results = retriever.retrieve(query, top_k=5)

    for r in results:
        print(json.dumps(r, indent=2))
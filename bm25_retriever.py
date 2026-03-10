"""
bm25_retriever.py
-----------------
What this does:
- Loads chunks from data/chunks/chunks.json
- Validates saved index matches current chunk count (prevents IndexError)
- Builds a BM25 keyword search index over all chunk texts
- Given a query string, returns the top-k most relevant chunks
- Used by the router for FACTUAL queries ("what", "who", "when", "how many")
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi  # pip install rank-bm25

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
CHUNKS_PATH     = Path("data/chunks/chunks.json")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
BM25_META_PATH  = Path("data/bm25_index_meta.json")  # stores chunk count for validation


# ── Tokenizer ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Lowercase + whitespace tokenizer for BM25."""
    return text.lower().split()


# ── Index Management ───────────────────────────────────────────────────────────

def build_bm25_index(chunks: List[Dict]) -> BM25Okapi:
    """Build BM25 index from chunk list."""
    logger.info(f"Building BM25 index over {len(chunks)} chunks...")
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25 index built successfully.")
    return bm25


def save_bm25_index(bm25: BM25Okapi, chunk_count: int) -> None:
    """
    Save BM25 index + metadata.
    Metadata records chunk count so we can detect stale indexes later.
    """
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_META_PATH, "w") as f:
        json.dump({"chunk_count": chunk_count}, f)
    logger.info(f"BM25 index saved ({chunk_count} chunks)")


def load_bm25_index() -> BM25Okapi:
    """Load saved BM25 index from disk."""
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    logger.info(f"BM25 index loaded from {BM25_INDEX_PATH}")
    return bm25


def is_index_valid(current_chunk_count: int) -> bool:
    """
    Check if saved index matches current chunk count.

    This prevents the IndexError:
      'index 687 is out of bounds for axis 0 with size 687'
    which happens when index was built on 687 chunks but
    chunks.json now has 1806.

    Returns True if index can be reused, False if rebuild needed.
    """
    if not BM25_INDEX_PATH.exists() or not BM25_META_PATH.exists():
        logger.info("No saved index found — will build fresh.")
        return False

    with open(BM25_META_PATH, "r") as f:
        meta = json.load(f)

    saved_count = meta.get("chunk_count", 0)

    if saved_count != current_chunk_count:
        logger.warning(
            f"Stale index: built on {saved_count} chunks, "
            f"current chunks.json has {current_chunk_count}. Rebuilding..."
        )
        return False

    logger.info(f"Index is valid — {saved_count} chunks match.")
    return True


# ── BM25 Retriever Class ───────────────────────────────────────────────────────

class BM25Retriever:
    """
    BM25 sparse retriever.

    Automatically validates the saved index against current chunk count.
    Rebuilds index if stale — prevents IndexError on chunk count mismatch.

    Usage:
        retriever = BM25Retriever()
        results = retriever.retrieve("What datasets were used?", top_k=5)
    """

    def __init__(self, chunks_path: Path = CHUNKS_PATH, rebuild: bool = False):
        """
        Initialize retriever. Auto-detects and fixes stale indexes.

        Args:
            chunks_path: Path to chunks.json
            rebuild:     Force rebuild even if index looks valid
        """
        logger.info(f"Loading chunks from {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info(f"Loaded {len(self.chunks)} chunks")

        # Auto-validate: rebuild if stale or forced
        if rebuild or not is_index_valid(len(self.chunks)):
            self.bm25 = build_bm25_index(self.chunks)
            save_bm25_index(self.bm25, len(self.chunks))
        else:
            self.bm25 = load_bm25_index()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query:  Search query string.
            top_k:  Number of chunks to return (default 5).

        Returns:
            List of top-k chunk dicts sorted by score descending.
            Score field is filled with BM25 relevance score.
        """
        if not query.strip():
            logger.warning("Empty query — returning empty results.")
            return []

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Safety check — should never fail now but kept as guard
        if len(scores) != len(self.chunks):
            raise RuntimeError(
                f"Score/chunk mismatch: {len(scores)} scores vs {len(self.chunks)} chunks. "
                "Delete data/bm25_index.pkl and data/bm25_index_meta.json, then re-run."
            )

        # Attach scores to chunks and sort
        scored_chunks = []
        for idx, chunk in enumerate(self.chunks):
            chunk_copy = dict(chunk)
            chunk_copy["score"] = round(float(scores[idx]), 4)
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]


# ── Main: Test ────────────────────────────────────────────────────────────────

def main():
    """Test BM25 retriever with sample queries. Run: python bm25_retriever.py"""
    print("\n" + "=" * 60)
    print("BM25 Retriever — Quick Test")
    print("=" * 60)

    retriever = BM25Retriever()
    print(f"\nTotal chunks indexed: {len(retriever.chunks)}")

    test_queries = [
        ("FACTUAL",     "What datasets were used for evaluation?"),
        ("CONCEPTUAL",  "How does attention mechanism work in transformers?"),
        ("COMPLEX",     "What is the accuracy reported in the experiments?"),
    ]

    for query_type, query in test_queries:
        print(f"\n{'='*60}")
        print(f"Type  : {query_type}")
        print(f"Query : '{query}'")
        print("-" * 60)

        results = retriever.retrieve(query, top_k=5)

        for i, chunk in enumerate(results, 1):
            print(f"\n  Rank {i}:")
            print(f"    chunk_id : {chunk['chunk_id']}")
            print(f"    source   : {chunk['source']}")
            print(f"    score    : {chunk['score']}")
            print(f"    text     : {chunk['text'][:150]}...")

    print("\n" + "=" * 60)
    print("BM25 test complete.")


if __name__ == "__main__":
    main()
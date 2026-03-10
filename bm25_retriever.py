"""
bm25_retriever.py
-----------------
What this does:
- Loads chunks from data/chunks/chunks.json
- Builds a BM25 keyword search index over all chunk texts
- Given a query string, returns the top-k most relevant chunks
- Used by the router for FACTUAL queries ("what", "who", "when", "how many")

Why BM25?
- BM25 (Best Match 25) is a classic keyword ranking algorithm
- It scores chunks based on term frequency + document length normalization
- No GPU needed — runs entirely on CPU
- Very fast (~1-3ms per query)
- Best for exact keyword/factual lookups

Input:  query string (e.g. "What datasets were used for evaluation?")
Output: List of top-k chunk dicts sorted by score descending, score field filled in

Agreed chunk schema (DO NOT change):
{
    "chunk_id": str,
    "text":     str,
    "source":   str,
    "position": int,
    "score":    float   ← BM25 relevance score filled in here
}
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi  # pip install rank-bm25

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
CHUNKS_PATH = Path("data/chunks/chunks.json")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")  # saved index for fast reload


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """
    Simple whitespace + lowercase tokenizer for BM25.

    BM25 works on lists of tokens. We lowercase everything so that
    "BERT" and "bert" are treated as the same token.

    Args:
        text: Raw text string.

    Returns:
        List of lowercase word tokens.
    """
    return text.lower().split()


# ── BM25 Index Builder ────────────────────────────────────────────────────────

def build_bm25_index(chunks: List[Dict]) -> BM25Okapi:
    """
    Build a BM25 index from a list of chunk dicts.

    Steps:
    1. Extract the text from each chunk
    2. Tokenize each text into a list of words
    3. Pass all tokenized documents to BM25Okapi

    Args:
        chunks: List of chunk dicts loaded from chunks.json

    Returns:
        A fitted BM25Okapi index object.
    """
    logger.info(f"Building BM25 index over {len(chunks)} chunks...")

    # Tokenize all chunk texts
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    logger.info("BM25 index built successfully.")
    return bm25


def save_bm25_index(bm25: BM25Okapi, path: Path = BM25_INDEX_PATH) -> None:
    """
    Save the BM25 index to disk using pickle so we don't rebuild every run.

    Args:
        bm25: Fitted BM25Okapi object.
        path: File path to save to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"BM25 index saved to {path}")


def load_bm25_index(path: Path = BM25_INDEX_PATH) -> BM25Okapi:
    """
    Load a previously saved BM25 index from disk.

    Args:
        path: File path to load from.

    Returns:
        Loaded BM25Okapi object.
    """
    with open(path, "rb") as f:
        bm25 = pickle.load(f)
    logger.info(f"BM25 index loaded from {path}")
    return bm25


# ── BM25 Retriever Class ──────────────────────────────────────────────────────

class BM25Retriever:
    """
    BM25 sparse retriever.

    Loads chunks and builds (or reloads) a BM25 index.
    Provides a retrieve() method that returns top-k chunks for a query.

    Usage:
        retriever = BM25Retriever()
        results = retriever.retrieve("What datasets were used?", top_k=5)
    """

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        index_path: Path = BM25_INDEX_PATH,
        rebuild: bool = False
    ):
        """
        Initialize the BM25 retriever.

        Args:
            chunks_path: Path to chunks.json
            index_path:  Path to saved BM25 index (pickle)
            rebuild:     If True, always rebuild index even if saved one exists
        """
        # Load chunks
        logger.info(f"Loading chunks from {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info(f"Loaded {len(self.chunks)} chunks")

        # Build or load BM25 index
        if index_path.exists() and not rebuild:
            self.bm25 = load_bm25_index(index_path)
        else:
            self.bm25 = build_bm25_index(self.chunks)
            save_bm25_index(self.bm25, index_path)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the top-k most relevant chunks for a query using BM25.

        Steps:
        1. Tokenize the query
        2. Score all chunks using BM25
        3. Sort by score descending
        4. Return top-k chunks with score field filled in

        Args:
            query:  The search query string.
            top_k:  Number of chunks to return (default 5).

        Returns:
            List of top-k chunk dicts, sorted by score descending.
            Each chunk has its "score" field set to the BM25 relevance score.
        """
        # Tokenize the query the same way we tokenized the corpus
        tokenized_query = tokenize(query)

        # Get BM25 scores for all chunks
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each chunk with its score and sort descending
        scored_chunks = []
        for idx, chunk in enumerate(self.chunks):
            # Make a copy so we don't modify the original chunk objects
            chunk_copy = dict(chunk)
            chunk_copy["score"] = round(float(scores[idx]), 4)
            scored_chunks.append(chunk_copy)

        # Sort by score descending and return top-k
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]


# ── Main: Test the retriever ──────────────────────────────────────────────────

def main():
    """
    Quick test to verify BM25 retriever works correctly.
    Run this directly: python bm25_retriever.py
    """
    print("\n" + "=" * 55)
    print("BM25 Retriever — Quick Test")
    print("=" * 55)

    # Build retriever (will build and save index on first run)
    retriever = BM25Retriever()

    # Test queries — one factual, one conceptual, one complex
    test_queries = [
        "What datasets were used for evaluation?",
        "How does attention mechanism work in transformers?",
        "What is the accuracy reported in the experiments?"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 55)

        results = retriever.retrieve(query, top_k=5)

        for i, chunk in enumerate(results, 1):
            print(f"\n  Rank {i}:")
            print(f"    chunk_id : {chunk['chunk_id']}")
            print(f"    source   : {chunk['source']}")
            print(f"    position : {chunk['position']}")
            print(f"    score    : {chunk['score']}")
            print(f"    text     : {chunk['text'][:120]}...")

    print("\n" + "=" * 55)
    print("BM25 test complete.")


if __name__ == "__main__":
    main()
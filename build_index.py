"""
build_index.py — One-Time FAISS Index Builder
Owner: Samved Jain

Run this ONCE after ingestion.py has produced data/chunks/chunks.json.
Builds and saves the FAISS dense vector index to data/faiss_index/.

Usage:
    python build_index.py

Output:
    data/faiss_index/index.faiss     ← FAISS binary index
    data/faiss_index/metadata.pkl    ← chunk metadata (parallel to index rows)
"""

from ingestion import load_chunks
from faiss_retriever import FAISSRetriever


def main():
    print("=" * 50)
    print("FAISS Index Builder")
    print("=" * 50)

    # Step 1: Load chunks produced by Nivi's ingestion.py
    print("\n[1/3] Loading chunks from data/chunks/chunks.json ...")
    chunks = load_chunks()
    print(f"      Loaded {len(chunks)} chunks.")

    # Step 2: Build the FAISS index (encodes all chunks — takes ~1–2 min)
    print("\n[2/3] Building FAISS index ...")
    retriever = FAISSRetriever()
    retriever.build_index(chunks)

    # Step 3: Save to disk so it can be loaded by pipelines without rebuilding
    print("\n[3/3] Saving index to data/faiss_index/ ...")
    retriever.save()

    print("\n" + "=" * 50)
    print(f"Done. {len(chunks)} vectors indexed and saved.")
    print("Roshan can now load the index with: faiss.load()")
    print("=" * 50)


if __name__ == "__main__":
    main()
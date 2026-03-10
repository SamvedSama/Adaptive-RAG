"""
run_qa_generator.py
Runs QA generation against real chunks from data/chunks/chunks.json.
Produces 30 QA pairs (10 factual, 10 conceptual, 10 complex).

Usage:
    python run_qa_generator.py
"""

from ingestion import load_chunks
from qa_generator import QAGenerator

def main():
    print("=" * 50)
    print("QA Pair Generation — Real Pipeline")
    print("=" * 50)

    # Load real chunks from Nivi's ingestion output
    print("\nLoading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    # Generate 30 QA pairs (10 per type)
    gen = QAGenerator()
    gen.generate_from_chunks(
        chunks,
        target_counts={"factual": 10, "conceptual": 10, "complex": 10},
    )

    # Save to data/qa_pairs.json
    gen.save()
    gen.print_stats()

    print("\nDone. Share data/qa_pairs.json with Nivi and Roshan.")

if __name__ == "__main__":
    main()
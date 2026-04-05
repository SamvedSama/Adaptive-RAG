"""
build_index.py — One-Time FAISS Index Builder
Owner: Samved Jain

Run this ONCE after ingestion.py has produced data/chunks/chunks.json.
Builds and saves the FAISS dense vector index to data/faiss_index/.

Usage:
    python build_index.py [--chunks-path data/chunks/chunks.json]
                          [--index-dir data/faiss_index]
                          [--force]

Output:
    data/faiss_index/index.faiss     ← FAISS binary index
    data/faiss_index/metadata.pkl    ← chunk metadata (parallel to index rows)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from ingestion import load_chunks
from faiss_retriever import FAISSRetriever

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("build_index.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

_DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks.json")
_DEFAULT_INDEX_DIR   = Path("data/faiss_index")
_INDEX_FILE          = "index.faiss"
_METADATA_FILE       = "metadata.pkl"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _index_exists(index_dir: Path) -> bool:
    return (
        (index_dir / _INDEX_FILE).exists()
        and (index_dir / _METADATA_FILE).exists()
    )


def _validate_chunks(chunks: list[dict]) -> None:
    """Spot-check that chunks conform to the agreed schema before indexing."""
    required = {"chunk_id", "text", "source", "position"}
    for i, chunk in enumerate(chunks[:10]):  # sample first 10
        missing = required - chunk.keys()
        if missing:
            raise ValueError(f"Chunk at index {i} is missing keys: {missing}")
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError(f"Chunk '{chunk.get('chunk_id', i)}' has empty text.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build and persist a FAISS dense vector index from chunk corpus."
    )
    p.add_argument(
        "--chunks-path", type=Path, default=_DEFAULT_CHUNKS_PATH,
        help=f"Path to chunks.json (default: {_DEFAULT_CHUNKS_PATH})",
    )
    p.add_argument(
        "--index-dir", type=Path, default=_DEFAULT_INDEX_DIR,
        help=f"Directory to write index artefacts (default: {_DEFAULT_INDEX_DIR})",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Rebuild even if a valid index already exists.",
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    chunks_path: Path = args.chunks_path
    index_dir: Path   = args.index_dir
    force: bool       = args.force

    log.info("=" * 60)
    log.info("FAISS Index Builder")
    log.info("  Chunks path : %s", chunks_path)
    log.info("  Index dir   : %s", index_dir)
    log.info("  Force rebuild: %s", force)
    log.info("=" * 60)

    # ── Guard: skip if already built (unless --force) ──────────────────────────
    if _index_exists(index_dir) and not force:
        log.info(
            "Index already exists at '%s'. "
            "Use --force to rebuild.",
            index_dir,
        )
        sys.exit(0)

    # ── Step 1: Validate inputs ────────────────────────────────────────────────
    if not chunks_path.exists():
        log.error(
            "Chunks file not found at '%s'. Run ingestion.py first.", chunks_path
        )
        sys.exit(1)

    # ── Step 2: Load chunks ────────────────────────────────────────────────────
    log.info("[1/3] Loading chunks from %s ...", chunks_path)
    t0 = time.perf_counter()
    try:
        chunks = load_chunks(chunks_path)
    except Exception as exc:
        log.error("Failed to load chunks: %s", exc)
        sys.exit(1)

    if not chunks:
        log.error("Chunk file is empty. Re-run ingestion.py.")
        sys.exit(1)

    try:
        _validate_chunks(chunks)
    except ValueError as exc:
        log.error("Chunk schema validation failed: %s", exc)
        sys.exit(1)

    log.info("  Loaded and validated %d chunks (%.2fs).", len(chunks), time.perf_counter() - t0)

    # ── Step 3: Build FAISS index ──────────────────────────────────────────────
    log.info("[2/3] Building FAISS index (encoding %d chunks — this may take a few minutes) ...", len(chunks))
    t1 = time.perf_counter()
    try:
        retriever = FAISSRetriever()
        retriever.build_index(chunks)
    except Exception as exc:
        log.error("Index build failed: %s", exc, exc_info=True)
        sys.exit(1)

    build_time = time.perf_counter() - t1
    log.info("  Index built in %.1fs (%.0f chunks/sec).", build_time, len(chunks) / build_time)

    # ── Step 4: Persist to disk ────────────────────────────────────────────────
    log.info("[3/3] Saving index to %s ...", index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    t2 = time.perf_counter()
    try:
        retriever.save(
            index_path=index_dir / _INDEX_FILE,
            meta_path=index_dir / _METADATA_FILE
        )
    except Exception as exc:
        log.error("Failed to save index: %s", exc, exc_info=True)
        sys.exit(1)

    log.info("  Saved in %.2fs.", time.perf_counter() - t2)

    # ── Verify artefacts were written ─────────────────────────────────────────
    if not _index_exists(index_dir):
        log.error(
            "Save appeared to succeed but artefacts are missing in '%s'. "
            "Check FAISSRetriever.save() output paths.",
            index_dir,
        )
        sys.exit(1)

    index_size_mb = (index_dir / _INDEX_FILE).stat().st_size / (1024 ** 2)
    meta_size_kb  = (index_dir / _METADATA_FILE).stat().st_size / 1024

    log.info("=" * 60)
    log.info("Index build complete.")
    log.info("  Vectors indexed : %d", len(chunks))
    log.info("  Total wall time : %.1fs", time.perf_counter() - t0)
    log.info("  index.faiss     : %.1f MB", index_size_mb)
    log.info("  metadata.pkl    : %.1f KB", meta_size_kb)
    log.info("=" * 60)
    log.info("Next step: python qa_generator.py")


if __name__ == "__main__":
    main()
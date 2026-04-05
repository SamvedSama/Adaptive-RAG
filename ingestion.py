"""
ingestion.py
------------
Responsibilities:
- Load PDF files from data/raw_pdfs/
- Clean and normalize extracted text
- Split text into overlapping chunks
- Attach metadata: chunk_id, source, position
- Save all chunks to data/chunks/chunks.json

Output schema (agreed interface — do NOT change):
{
    "chunk_id":  str,   # e.g. "paper1_chunk_042"
    "text":      str,   # actual chunk content
    "source":    str,   # filename e.g. "paper1.pdf"
    "position":  int,   # chunk index within document
    "score":     float  # always 0.0 at ingestion time
}

Usage:
    python ingestion.py [--pdf-dir data/raw_pdfs] [--output data/chunks/chunks.json]
                        [--chunk-size 400] [--overlap 80] [--workers 4] [--min-words 50]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingestion.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

_DEFAULT_PDF_DIR = Path("data/raw_pdfs")
_DEFAULT_OUTPUT = Path("data/chunks/chunks.json")


@dataclass(frozen=True)
class IngestionConfig:
    pdf_dir: Path = _DEFAULT_PDF_DIR
    output_path: Path = _DEFAULT_OUTPUT
    chunk_size: int = 400       # words per chunk
    chunk_overlap: int = 80     # words shared between consecutive chunks
    min_chunk_words: int = 50   # discard chunks shorter than this
    workers: int = 4            # parallel PDF-processing workers

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        if self.min_chunk_words > self.chunk_size:
            raise ValueError("min_chunk_words must be ≤ chunk_size.")
        if self.workers < 1:
            raise ValueError("workers must be ≥ 1.")


# ── Chunk Schema ───────────────────────────────────────────────────────────────

ChunkDict = dict  # typed alias for readability

CHUNK_SCHEMA_KEYS = frozenset({"chunk_id", "text", "source", "position", "score"})


def _make_chunk(stem: str, source: str, idx: int, text: str) -> ChunkDict:
    return {
        "chunk_id": f"{stem}_chunk_{idx:03d}",
        "text": text,
        "source": source,
        "position": idx,
        "score": 0.0,
    }


# ── PDF Text Extraction ────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract full text from a PDF using PyMuPDF.
    Raises RuntimeError on corrupt / unreadable files.
    """
    pages: list[str] = []
    try:
        with fitz.open(str(pdf_path)) as doc:
            if doc.is_encrypted:
                raise RuntimeError(f"{pdf_path.name} is encrypted.")
            for page in doc:
                page_text = page.get_text("text")
                if page_text.strip():
                    pages.append(page_text)
    except fitz.FileDataError as exc:
        raise RuntimeError(f"PyMuPDF cannot open {pdf_path.name}: {exc}") from exc

    if not pages:
        raise RuntimeError(f"{pdf_path.name} yielded no extractable text.")

    return "\n".join(pages)


# ── Text Cleaning ──────────────────────────────────────────────────────────────

# Compile once at module level for performance
_RE_HYPHEN_BREAK = re.compile(r"-\n")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_NON_ASCII = re.compile(r"[^\x00-\x7F]+")
_RE_PAGE_ARTIFACTS = re.compile(
    r"(?i)(arxiv:\S+|doi:\S+|\bpage\s+\d+\s+of\s+\d+\b)"
)


def clean_text(text: str) -> str:
    """
    Normalize raw PDF text:
    1. Repair hyphenated line breaks  ("meth-\\nod" → "method")
    2. Remove common PDF artifacts   (arXiv IDs, "Page N of M")
    3. Collapse whitespace           (tabs, multiple spaces/newlines → single space)
    4. Strip non-ASCII noise
    """
    text = _RE_HYPHEN_BREAK.sub("", text)
    text = _RE_PAGE_ARTIFACTS.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text)
    text = _RE_NON_ASCII.sub(" ", text)
    return text.strip()


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    min_words: int,
) -> Iterator[str]:
    """
    Yield overlapping word-window chunks from *text*.

    Overlap preserves context at chunk boundaries — any fact spanning two
    windows appears fully in at least one of them.
    """
    words = text.split()
    total = len(words)
    if total == 0:
        return

    step = chunk_size - overlap
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        window = words[start:end]
        if len(window) >= min_words:
            yield " ".join(window)
        if end == total:
            break
        start += step


# ── Per-Document Processing (runs in worker process) ──────────────────────────

def _process_pdf(args: tuple[Path, IngestionConfig]) -> tuple[str, list[ChunkDict], str | None]:
    """
    Top-level function executed in each worker process.
    Returns (pdf_name, chunks, error_message_or_None).
    Must be module-level for pickling (ProcessPoolExecutor requirement).
    """
    pdf_path, cfg = args
    try:
        raw = extract_text_from_pdf(pdf_path)
        clean = clean_text(raw)
        stem = pdf_path.stem
        source = pdf_path.name
        chunks = [
            _make_chunk(stem, source, idx, text)
            for idx, text in enumerate(
                chunk_text(clean, cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_words)
            )
        ]
        return pdf_path.name, chunks, None
    except Exception as exc:  # noqa: BLE001 — worker must not crash the pool
        return pdf_path.name, [], str(exc)


# ── Parallel PDF Orchestrator ──────────────────────────────────────────────────

def process_all_pdfs(cfg: IngestionConfig) -> list[ChunkDict]:
    """
    Process every PDF in cfg.pdf_dir in parallel.
    Returns a flat, deterministically ordered list of chunk dicts.
    Logs and skips failed files; does not abort the pipeline.
    """
    pdf_files = sorted(cfg.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDF files found in %s.", cfg.pdf_dir)
        return []

    log.info(
        "Processing %d PDFs | chunk_size=%d | overlap=%d | workers=%d",
        len(pdf_files),
        cfg.chunk_size,
        cfg.chunk_overlap,
        cfg.workers,
    )

    # Preserve source-document ordering for reproducibility
    results_by_name: dict[str, list[ChunkDict]] = {}
    failed: list[str] = []

    work_items = [(p, cfg) for p in pdf_files]

    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(_process_pdf, item): item[0] for item in work_items}
        for future in as_completed(futures):
            pdf_name, chunks, error = future.result()
            if error:
                log.error("  ✗ %s — %s", pdf_name, error)
                failed.append(pdf_name)
            else:
                log.info("  ✓ %s → %d chunks", pdf_name, len(chunks))
                results_by_name[pdf_name] = chunks

    # Re-sort by original file order for reproducibility
    all_chunks: list[ChunkDict] = []
    for pdf_path in pdf_files:
        if pdf_path.name in results_by_name:
            all_chunks.extend(results_by_name[pdf_path.name])

    log.info(
        "Ingestion complete. Total chunks: %d | Failed PDFs: %d",
        len(all_chunks),
        len(failed),
    )
    if failed:
        log.warning("Failed files: %s", ", ".join(failed))

    return all_chunks


# ── Persistence ────────────────────────────────────────────────────────────────

def save_chunks(chunks: list[ChunkDict], output_path: Path) -> None:
    """
    Atomically write chunks to JSON (write to temp → rename).
    Prevents partial/corrupt output files on crash mid-write.
    """
    if not chunks:
        raise ValueError("Refusing to write an empty chunks file.")

    # Validate schema before writing
    for i, chunk in enumerate(chunks[:5]):  # spot-check first 5
        missing = CHUNK_SCHEMA_KEYS - chunk.keys()
        if missing:
            raise ValueError(f"Chunk {i} missing keys: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.json")

    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(chunks, fh, indent=2, ensure_ascii=False)
        tmp_path.replace(output_path)  # atomic on POSIX; near-atomic on Windows
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    log.info("Saved %d chunks → %s", len(chunks), output_path)


def load_chunks(chunks_path: Path = _DEFAULT_OUTPUT) -> list[ChunkDict]:
    """
    Load and validate chunks from disk.
    Called by retrievers, pipelines, and evaluation scripts.

    Raises:
        FileNotFoundError: if the JSON file does not exist.
        ValueError: if the JSON structure is not a list.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at '{chunks_path}'. "
            "Run ingestion.py first."
        )

    with open(chunks_path, encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {chunks_path}, got {type(data)}.")

    log.info("Loaded %d chunks from %s", len(data), chunks_path)
    return data


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> IngestionConfig:
    p = argparse.ArgumentParser(description="Ingest QASPER PDFs into a chunk corpus.")
    p.add_argument("--pdf-dir", type=Path, default=_DEFAULT_PDF_DIR)
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--overlap", type=int, default=80)
    p.add_argument("--min-words", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    return IngestionConfig(
        pdf_dir=args.pdf_dir,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        min_chunk_words=args.min_words,
        workers=args.workers,
    )


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _parse_args()

    log.info("=" * 60)
    log.info("Ingestion Pipeline")
    log.info("  PDF dir    : %s", cfg.pdf_dir)
    log.info("  Output     : %s", cfg.output_path)
    log.info("  Chunk size : %d words | Overlap: %d words", cfg.chunk_size, cfg.chunk_overlap)
    log.info("  Min words  : %d | Workers: %d", cfg.min_chunk_words, cfg.workers)
    log.info("=" * 60)

    if not cfg.pdf_dir.exists():
        log.error("PDF directory does not exist: %s", cfg.pdf_dir)
        sys.exit(1)

    all_chunks = process_all_pdfs(cfg)

    if not all_chunks:
        log.error("No chunks produced. Check %s for valid PDFs.", cfg.pdf_dir)
        sys.exit(1)

    save_chunks(all_chunks, cfg.output_path)

    # Print one sample chunk for quick sanity check
    log.info("\n── Sample chunk (index 0) ──")
    print(json.dumps(all_chunks[0], indent=2))

    log.info("\nNext step: python build_index.py")


if __name__ == "__main__":
    main()
"""
ingestion.py
------------
Responsibilities:
- Load PDF files from data/raw_pdfs/
- Clean and normalize extracted text (preserving math/Greek symbols)
- Split text into overlapping chunks
- Deduplicate chunks via SHA-256 content hashing
- Attach metadata: chunk_id, source, position
- Save all chunks to data/chunks/chunks.json

Output schema (agreed interface -- do NOT change):
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
                        [--no-dedup]
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import logging
import multiprocessing
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

# ── Logging ────────────────────────────────────────────────────────────────────

def _build_stream_handler() -> logging.StreamHandler:
    """Return a StreamHandler that always writes UTF-8, even on Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        return logging.StreamHandler(sys.stdout)
    utf8_stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    return logging.StreamHandler(utf8_stdout)  # type: ignore[arg-type]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        _build_stream_handler(),
        logging.FileHandler("ingestion.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

_DEFAULT_PDF_DIR = Path("data/raw_pdfs")
_DEFAULT_OUTPUT  = Path("data/chunks/chunks.json")


@dataclass(frozen=True)
class IngestionConfig:
    pdf_dir:         Path  = _DEFAULT_PDF_DIR
    output_path:     Path  = _DEFAULT_OUTPUT
    chunk_size:      int   = 400    # words per chunk
    chunk_overlap:   int   = 80     # words shared between consecutive chunks
    min_chunk_words: int   = 50     # discard chunks shorter than this
    workers:         int   = 4      # parallel PDF-processing workers
    dedup:           bool  = True   # deduplicate identical chunk content

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        if self.min_chunk_words > self.chunk_size:
            raise ValueError("min_chunk_words must be <= chunk_size.")
        if self.workers < 1:
            raise ValueError("workers must be >= 1.")


# ── Chunk Schema ───────────────────────────────────────────────────────────────

ChunkDict = dict  # typed alias for readability

CHUNK_SCHEMA_KEYS = frozenset({"chunk_id", "text", "source", "position", "score"})


def _make_chunk(stem: str, source: str, idx: int, text: str) -> ChunkDict:
    return {
        "chunk_id": f"{stem}_chunk_{idx:03d}",
        "text":     text,
        "source":   source,
        "position": idx,
        "score":    0.0,
    }


# ── PDF Text Extraction ────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract full text from a PDF using PyMuPDF.
    Uses 'blocks' mode to preserve paragraph structure better than 'text'.
    Raises RuntimeError on corrupt / unreadable files.
    """
    pages: list[str] = []
    try:
        with fitz.open(str(pdf_path)) as doc:
            if doc.is_encrypted:
                raise RuntimeError(f"{pdf_path.name} is encrypted.")
            for page in doc:
                # 'blocks' gives better paragraph separation than flat 'text'
                blocks = page.get_text("blocks")
                page_text = "\n".join(
                    b[4].strip()
                    for b in blocks
                    if isinstance(b[4], str) and b[4].strip()
                )
                if page_text:
                    pages.append(page_text)
    except fitz.FileDataError as exc:
        raise RuntimeError(f"PyMuPDF cannot open {pdf_path.name}: {exc}") from exc

    if not pages:
        raise RuntimeError(f"{pdf_path.name} yielded no extractable text.")

    return "\n".join(pages)


# ── Text Cleaning ──────────────────────────────────────────────────────────────

# Compile all regexes once at module load for performance.
_RE_HYPHEN_BREAK   = re.compile(r"-\n\s*")
_RE_WHITESPACE     = re.compile(r"[ \t]+")          # collapse horizontal space only
_RE_MULTI_NEWLINE  = re.compile(r"\n{3,}")          # collapse 3+ newlines -> 2
_RE_PAGE_ARTIFACTS = re.compile(
    r"(?i)(arxiv:\S+|doi:\S+|\bpage\s+\d+\s+of\s+\d+\b|\bproceed(?:ings?)?\b.*?(?=\n|$))"
)
# Selective noise removal: control chars and surrogates only.
# Preserves Unicode letters (α β σ θ), math symbols (∑ ∈ ≤), and accented chars.
_RE_CONTROL_NOISE  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")
# Collapse runs of replacement characters that appear after binary decode failures.
_RE_REPLACEMENT    = re.compile(r"(\ufffd){2,}")


def clean_text(text: str) -> str:
    """
    Normalize raw PDF text while preserving scientific notation.

    Pipeline:
    1. Repair hyphenated line breaks  ("meth-\\nod" -> "method")
    2. Remove common PDF artifacts    (arXiv IDs, "Page N of M")
    3. Remove invisible control chars (preserves Greek/math Unicode)
    4. Collapse replacement char runs (artifact of bad PDF encodings)
    5. Collapse horizontal whitespace (not newlines -- preserve paragraphs)
    6. Collapse excessive blank lines
    """
    text = _RE_HYPHEN_BREAK.sub("", text)
    text = _RE_PAGE_ARTIFACTS.sub(" ", text)
    text = _RE_CONTROL_NOISE.sub("", text)
    text = _RE_REPLACEMENT.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(
    text:       str,
    chunk_size: int,
    overlap:    int,
    min_words:  int,
) -> Iterator[str]:
    """
    Yield overlapping word-window chunks from *text*.

    Overlap preserves context at chunk boundaries -- any fact spanning two
    windows appears fully in at least one of them.

    Words are split on whitespace (including newlines), so paragraph breaks
    inside a chunk are naturally preserved in the joined output.
    """
    words = text.split()
    total = len(words)
    if total == 0:
        return

    step  = chunk_size - overlap
    start = 0
    while start < total:
        end    = min(start + chunk_size, total)
        window = words[start:end]
        if len(window) >= min_words:
            yield " ".join(window)
        if end == total:
            break
        start += step


# ── Deduplication ──────────────────────────────────────────────────────────────

def _content_hash(text: str) -> str:
    """SHA-256 fingerprint of normalised chunk text."""
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


def deduplicate_chunks(chunks: list[ChunkDict]) -> tuple[list[ChunkDict], int]:
    """
    Remove chunks with identical content across the entire corpus.
    Keeps the first occurrence (deterministic: PDFs are sorted before processing).
    Returns (deduped_list, n_removed).
    """
    seen: set[str] = set()
    unique: list[ChunkDict] = []
    for chunk in chunks:
        h = _content_hash(chunk["text"])
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique, len(chunks) - len(unique)


# ── Per-Document Processing (runs in worker process) ──────────────────────────

def _process_pdf(
    args: tuple[Path, IngestionConfig],
) -> tuple[str, list[ChunkDict], str | None]:
    """
    Top-level function executed in each worker process.
    Returns (pdf_name, chunks, error_message_or_None).
    Must be module-level for pickling (ProcessPoolExecutor requirement).
    """
    pdf_path, cfg = args
    try:
        raw    = extract_text_from_pdf(pdf_path)
        clean  = clean_text(raw)
        stem   = pdf_path.stem
        source = pdf_path.name
        chunks = [
            _make_chunk(stem, source, idx, text)
            for idx, text in enumerate(
                chunk_text(clean, cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_words)
            )
        ]
        return pdf_path.name, chunks, None
    except Exception as exc:  # noqa: BLE001 -- worker must not crash the pool
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
        "Processing %d PDFs | chunk_size=%d | overlap=%d | workers=%d | dedup=%s",
        len(pdf_files),
        cfg.chunk_size,
        cfg.chunk_overlap,
        cfg.workers,
        cfg.dedup,
    )

    results_by_name: dict[str, list[ChunkDict]] = {}
    failed: list[str] = []
    completed = 0

    work_items = [(p, cfg) for p in pdf_files]

    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(_process_pdf, item): item[0] for item in work_items}
        for future in as_completed(futures):
            pdf_name, chunks, error = future.result()
            completed += 1
            pct = 100 * completed // len(pdf_files)
            if error:
                log.error("  [%3d%%] FAIL  %s -- %s", pct, pdf_name, error)
                failed.append(pdf_name)
            else:
                log.info("  [%3d%%] OK    %s -> %d chunks", pct, pdf_name, len(chunks))
                results_by_name[pdf_name] = chunks

    # Re-sort by original file order for reproducibility
    all_chunks: list[ChunkDict] = []
    for pdf_path in pdf_files:
        if pdf_path.name in results_by_name:
            all_chunks.extend(results_by_name[pdf_path.name])

    raw_count = len(all_chunks)

    if cfg.dedup:
        all_chunks, n_removed = deduplicate_chunks(all_chunks)
        if n_removed:
            log.info("Deduplication removed %d duplicate chunks.", n_removed)

    log.info(
        "Ingestion complete. Raw: %d | Final: %d | Failed PDFs: %d",
        raw_count,
        len(all_chunks),
        len(failed),
    )
    if failed:
        log.warning("Failed files: %s", ", ".join(failed))

    return all_chunks


# ── Persistence ────────────────────────────────────────────────────────────────

def save_chunks(chunks: list[ChunkDict], output_path: Path) -> None:
    """
    Atomically write chunks to JSON (write to .tmp -> rename).
    Prevents partial/corrupt output files on crash mid-write.
    """
    if not chunks:
        raise ValueError("Refusing to write an empty chunks file.")

    # Validate schema before writing -- spot-check first 5
    for i, chunk in enumerate(chunks[:5]):
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

    log.info("Saved %d chunks -> %s", len(chunks), output_path)


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
        raise ValueError(
            f"Expected a JSON list in {chunks_path}, got {type(data)}."
        )

    # DEBUG level -- this is called on every retriever init, not just once
    log.debug("Loaded %d chunks from %s", len(data), chunks_path)
    return data


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> IngestionConfig:
    p = argparse.ArgumentParser(
        description="Ingest QASPER PDFs into a chunk corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdf-dir",    type=Path, default=_DEFAULT_PDF_DIR)
    p.add_argument("--output",     type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--chunk-size", type=int,  default=400)
    p.add_argument("--overlap",    type=int,  default=80)
    p.add_argument("--min-words",  type=int,  default=50)
    p.add_argument("--workers",    type=int,  default=4)
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable cross-corpus chunk deduplication.",
    )
    args = p.parse_args()
    return IngestionConfig(
        pdf_dir         = args.pdf_dir,
        output_path     = args.output,
        chunk_size      = args.chunk_size,
        chunk_overlap   = args.overlap,
        min_chunk_words = args.min_words,
        workers         = args.workers,
        dedup           = not args.no_dedup,
    )


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Required on Windows for ProcessPoolExecutor to not re-spawn infinitely.
    multiprocessing.freeze_support()

    cfg = _parse_args()

    log.info("=" * 60)
    log.info("Eco-RAG Ingestion Pipeline")
    log.info("  PDF dir    : %s", cfg.pdf_dir)
    log.info("  Output     : %s", cfg.output_path)
    log.info(
        "  Chunk size : %d words | Overlap: %d words",
        cfg.chunk_size, cfg.chunk_overlap,
    )
    log.info(
        "  Min words  : %d | Workers: %d | Dedup: %s",
        cfg.min_chunk_words, cfg.workers, cfg.dedup,
    )
    log.info("=" * 60)

    if not cfg.pdf_dir.exists():
        log.error("PDF directory does not exist: %s", cfg.pdf_dir)
        sys.exit(1)

    all_chunks = process_all_pdfs(cfg)

    if not all_chunks:
        log.error("No chunks produced. Check %s for valid PDFs.", cfg.pdf_dir)
        sys.exit(1)

    save_chunks(all_chunks, cfg.output_path)

    # Sanity-check: print first chunk
    log.info("-- Sample chunk (index 0) --")
    print(json.dumps(all_chunks[0], indent=2, ensure_ascii=False))

    log.info("Next step: python faiss_retriever.py --build-index")


if __name__ == "__main__":
    main()
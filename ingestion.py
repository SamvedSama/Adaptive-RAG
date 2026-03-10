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
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict

import fitz 

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 400        # number of words per chunk
CHUNK_OVERLAP = 80      # number of words overlapping between consecutive chunks
RAW_PDF_DIR = Path("data/raw_pdfs")
CHUNKS_OUTPUT_DIR = Path("data/chunks")
CHUNKS_OUTPUT_FILE = CHUNKS_OUTPUT_DIR / "chunks.json"


# ── Step 1: PDF Text Extraction ────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract raw text from a single PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A single string containing all text from the PDF, page by page.
    """
    logger.info(f"Extracting text from: {pdf_path.name}")
    full_text = []

    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")  
            if page_text.strip():
                full_text.append(page_text)

    combined = "\n".join(full_text)
    logger.info(f"  → Extracted {len(combined)} characters from {len(full_text)} pages")
    return combined


# ── Step 2: Text Cleaning ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize and clean raw extracted PDF text.

    Cleaning steps:
    - Remove hyphenated line breaks (e.g. "meth-\nod" → "method")
    - Collapse multiple whitespace/newlines into a single space
    - Remove non-ASCII noise characters
    - Strip leading/trailing whitespace

    Args:
        text: Raw text extracted from PDF.

    Returns:
        Cleaned, normalized text string.
    """
    # Fix hyphenated line breaks common in academic PDFs
    text = re.sub(r'-\n', '', text)

    # Collapse newlines and extra spaces into a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove non-ASCII characters (noise from PDF encoding artifacts)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    return text.strip()


# ── Step 3: Chunking with Overlap ──────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Why overlap?
    Overlap ensures that sentences/facts near chunk boundaries are not
    lost. If a key fact spans two chunks, at least one chunk will fully
    contain it.

    Strategy: word-level sliding window.
    - Tokenize text into words
    - Slide a window of `chunk_size` words
    - Advance by (chunk_size - overlap) words each step

    Args:
        text:       Cleaned text to split.
        chunk_size: Number of words per chunk.
        overlap:    Number of words to repeat at chunk boundaries.

    Returns:
        List of text chunk strings.
    """
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return []

    chunks = []
    step = chunk_size - overlap  # how far we advance each iteration

    start = 0
    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        # Only keep chunks with enough content (filter very short tail chunks)
        if len(chunk_words) >= 50:
            chunks.append(chunk_text_str)

        if end == total_words:
            break  # reached end of document

        start += step

    return chunks


# ── Step 4: Build Chunk Objects ────────────────────────────────────────────────

def build_chunk_objects(chunks: List[str], source_filename: str) -> List[Dict]:
    """
    Wrap raw text chunks into the agreed chunk schema.

    Each chunk gets:
    - chunk_id:  "{stem}_chunk_{index:03d}"  e.g. "paper1_chunk_042"
    - text:      the actual chunk content
    - source:    original PDF filename
    - position:  index of this chunk within the document
    - score:     0.0 at ingestion (filled in by retrievers later)

    Args:
        chunks:           List of text strings from chunk_text().
        source_filename:  The PDF filename (e.g. "paper1.pdf").

    Returns:
        List of chunk dicts conforming to the agreed schema.
    """
    stem = Path(source_filename).stem  # "paper1.pdf" → "paper1"
    chunk_objects = []

    for idx, chunk in enumerate(chunks):
        chunk_objects.append({
            "chunk_id": f"{stem}_chunk_{idx:03d}",
            "text":     chunk,
            "source":   source_filename,
            "position": idx,
            "score":    0.0
        })

    return chunk_objects


# ── Step 5: Process All PDFs ───────────────────────────────────────────────────

def process_all_pdfs(pdf_dir: Path = RAW_PDF_DIR) -> List[Dict]:
    """
    Load, clean, chunk, and package all PDFs in the given directory.

    Args:
        pdf_dir: Directory containing raw PDF files.

    Returns:
        A flat list of all chunk dicts from all PDFs combined.
    """
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    all_chunks = []

    for pdf_path in pdf_files:
        try:
            # Extract → clean → chunk → package
            raw_text = extract_text_from_pdf(pdf_path)
            clean = clean_text(raw_text)
            chunks = chunk_text(clean)
            chunk_objects = build_chunk_objects(chunks, pdf_path.name)

            logger.info(f"  → {pdf_path.name}: {len(chunk_objects)} chunks created")
            all_chunks.extend(chunk_objects)

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            continue

    logger.info(f"\nTotal chunks across all documents: {len(all_chunks)}")
    return all_chunks


# ── Step 6: Save Chunks to Disk ────────────────────────────────────────────────

def save_chunks(chunks: List[Dict], output_path: Path = CHUNKS_OUTPUT_FILE) -> None:
    """
    Save all chunk objects to a JSON file.

    Args:
        chunks:      List of chunk dicts to save.
        output_path: File path to write JSON output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


# ── Step 7: Load Chunks from Disk (used by other modules) ─────────────────────

def load_chunks(chunks_path: Path = CHUNKS_OUTPUT_FILE) -> List[Dict]:
    """
    Load saved chunks from JSON file.

    This is the function other team members (Samved, Roshan) will call
    to load chunks into their retrievers and pipelines.

    Args:
        chunks_path: Path to the chunks JSON file.

    Returns:
        List of chunk dicts.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {chunks_path}. "
            "Run ingestion.py first to generate it."
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main():
    """
    Run the full ingestion pipeline:
    1. Load all PDFs from data/raw_pdfs/
    2. Clean and chunk each document
    3. Save all chunks to data/chunks/chunks.json
    """
    logger.info("=" * 50)
    logger.info("Starting ingestion pipeline")
    logger.info(f"Chunk size: {CHUNK_SIZE} words | Overlap: {CHUNK_OVERLAP} words")
    logger.info("=" * 50)

    all_chunks = process_all_pdfs(RAW_PDF_DIR)

    if not all_chunks:
        logger.error("No chunks produced. Check your data/raw_pdfs/ directory.")
        return

    save_chunks(all_chunks, CHUNKS_OUTPUT_FILE)

    # Print a sample chunk so you can visually verify output
    logger.info("\n── Sample chunk (first one) ──")
    sample = all_chunks[0]
    print(json.dumps(sample, indent=2))

    logger.info("\nIngestion complete.")


if __name__ == "__main__":
    main()
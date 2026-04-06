"""
1. load_qasper.py
--------------
Downloads official QASPER papers from arXiv.
All paper IDs come directly from the official QASPER dataset JSON.

Run this before ingestion.py.

Usage:
    python load_qasper.py [--num-papers 100] [--workers 4] [--output-dir data]
"""

from __future__ import annotations

import argparse
import codecs
import json
import logging
import os
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Logging Setup ──────────────────────────────────────────────────────────────
# Force UTF-8 on the stream handler so Windows cp1252 consoles never raise
# UnicodeEncodeError on non-ASCII characters in log messages.
# All log symbols use plain ASCII (OK / FAIL / SKIP) — safe on every platform.

def _build_stream_handler() -> logging.StreamHandler:
    """Return a StreamHandler that always writes UTF-8, even on Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        # Python 3.7+ TextIOWrapper — reconfigure in-place
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        return logging.StreamHandler(sys.stdout)
    # Fallback: wrap stdout in a UTF-8 writer
    utf8_stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    return logging.StreamHandler(utf8_stdout)  # type: ignore[arg-type]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        _build_stream_handler(),
        logging.FileHandler("load_qasper.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

QASPER_ARCHIVE_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
)
ARXIV_PDF_BASE = "https://arxiv.org/pdf/{paper_id}.pdf"
QASPER_JSON_NAME = "qasper-train-v0.3.json"
REQUEST_HEADERS = {"User-Agent": "QASPER-Downloader/1.0 (research; polite-bot)"}

# Retry config for transient network failures
HTTP_RETRIES = Retry(
    total=3,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)

# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class DownloadConfig:
    data_dir: Path = Path("data")
    num_papers: int = 100
    workers: int = 4           # parallel download threads
    request_timeout: int = 45  # seconds per PDF request
    inter_request_delay: float = 2.0  # seconds between downloads (per worker)
    min_pdf_size_bytes: int = 1024  # reject files smaller than 1 KB

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "raw_pdfs"

    @property
    def tgz_path(self) -> Path:
        return self.data_dir / "qasper-train-dev.tgz"


# ── Result Tracking ────────────────────────────────────────────────────────────


class DownloadResult(NamedTuple):
    paper_id: str
    success: bool
    size_kb: int = 0
    reason: str = ""


# ── HTTP Session Factory ───────────────────────────────────────────────────────


def _build_session() -> requests.Session:
    """Return a session with retry logic and polite headers."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=HTTP_RETRIES)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(REQUEST_HEADERS)
    return session


# ── Download Helpers ───────────────────────────────────────────────────────────


def _download_archive(cfg: DownloadConfig, session: requests.Session) -> None:
    """Download the QASPER .tgz archive if not already present."""
    if cfg.tgz_path.exists():
        log.info("Archive already present at %s -- skipping download.", cfg.tgz_path)
        return

    log.info("Downloading QASPER archive from AllenAI S3...")
    try:
        with session.get(QASPER_ARCHIVE_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(cfg.tgz_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        # ASCII-only progress line — safe on all consoles
                        print(
                            f"\r  Progress: {pct:5.1f}%  ({downloaded >> 20} MB)",
                            end="",
                            flush=True,
                        )
        print()
        log.info("Archive downloaded: %s", cfg.tgz_path)
    except requests.RequestException as exc:
        cfg.tgz_path.unlink(missing_ok=True)  # remove partial file
        raise RuntimeError(f"Failed to download QASPER archive: {exc}") from exc


def _extract_archive(cfg: DownloadConfig) -> None:
    """Extract the .tgz archive into data_dir."""
    log.info("Extracting archive %s ...", cfg.tgz_path)
    try:
        with tarfile.open(cfg.tgz_path, "r:gz") as tar:
            tar.extractall(path=cfg.data_dir)
        log.info("Extraction complete.")
    except tarfile.TarError as exc:
        raise RuntimeError(f"Archive extraction failed: {exc}") from exc


def _find_json(cfg: DownloadConfig) -> Path:
    """Walk data_dir to find the QASPER JSON file."""
    for root, _, files in os.walk(cfg.data_dir):
        if QASPER_JSON_NAME in files:
            found = Path(root) / QASPER_JSON_NAME
            log.info("Found QASPER JSON at: %s", found)
            return found
    raise FileNotFoundError(
        f"{QASPER_JSON_NAME} not found under {cfg.data_dir}. "
        "Re-extract the archive or check for corruption."
    )


def _load_paper_ids(json_path: Path) -> list[str]:
    """Load and return all paper IDs from the QASPER JSON."""
    log.info("Reading paper IDs from %s ...", json_path)
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    ids = list(data.keys())
    log.info("Total papers in QASPER: %d", len(ids))
    return ids


def _already_downloaded(cfg: DownloadConfig) -> set[str]:
    """Return the set of paper IDs whose PDFs are already on disk."""
    return {p.stem for p in cfg.pdf_dir.glob("*.pdf")}


def _is_valid_pdf(content: bytes) -> bool:
    """Basic magic-byte check — real PDFs start with %PDF."""
    return content[:4] == b"%PDF"


def _download_one(
    paper_id: str,
    cfg: DownloadConfig,
    session: requests.Session,
) -> DownloadResult:
    """
    Download a single PDF. Returns a DownloadResult regardless of outcome
    so the caller can aggregate statistics without try/except at the call site.
    """
    pdf_path = cfg.pdf_dir / f"{paper_id}.pdf"

    # Idempotency: skip if valid file already exists
    if pdf_path.exists() and pdf_path.stat().st_size >= cfg.min_pdf_size_bytes:
        return DownloadResult(paper_id, success=True, reason="already_exists")

    url = ARXIV_PDF_BASE.format(paper_id=paper_id)
    try:
        resp = session.get(url, timeout=cfg.request_timeout)

        if resp.status_code == 403:
            return DownloadResult(paper_id, success=False, reason="403_forbidden")
        if resp.status_code == 404:
            return DownloadResult(paper_id, success=False, reason="404_not_found")
        resp.raise_for_status()

        content = resp.content
        if not _is_valid_pdf(content):
            return DownloadResult(paper_id, success=False, reason="invalid_pdf_magic")
        if len(content) < cfg.min_pdf_size_bytes:
            return DownloadResult(paper_id, success=False, reason="file_too_small")

        pdf_path.write_bytes(content)
        size_kb = len(content) // 1024
        return DownloadResult(paper_id, success=True, size_kb=size_kb)

    except requests.Timeout:
        return DownloadResult(paper_id, success=False, reason="timeout")
    except requests.RequestException as exc:
        return DownloadResult(paper_id, success=False, reason=str(exc)[:120])
    finally:
        # Throttle each worker to be polite to arXiv
        time.sleep(cfg.inter_request_delay)


# ── Parallel Download Orchestrator ─────────────────────────────────────────────


def _download_pdfs(paper_ids: list[str], cfg: DownloadConfig) -> dict:
    """
    Download up to cfg.num_papers PDFs using a thread pool.
    Returns a summary dict with counts.
    """
    already = _already_downloaded(cfg)
    log.info("PDFs already on disk: %d", len(already))

    still_needed = cfg.num_papers - len(already)
    if still_needed <= 0:
        log.info("Target of %d PDFs already met. Nothing to download.", cfg.num_papers)
        return {"downloaded": 0, "skipped": len(already), "failed": 0}

    pending = [pid for pid in paper_ids if pid not in already][:still_needed]
    log.info(
        "Need %d more PDFs. Queuing %d candidates with %d workers.",
        still_needed,
        len(pending),
        cfg.workers,
    )

    stats: dict[str, list[str]] = {"downloaded": [], "skipped": [], "failed": []}
    total_done = len(already)

    # One session per thread is thread-safe; share here for simplicity
    session = _build_session()

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {
            pool.submit(_download_one, pid, cfg, session): pid for pid in pending
        }
        for future in as_completed(futures):
            result: DownloadResult = future.result()
            total_done += 1

            if result.success:
                if result.reason == "already_exists":
                    stats["skipped"].append(result.paper_id)
                    log.debug("  SKIP  %s (already_exists)", result.paper_id)
                else:
                    stats["downloaded"].append(result.paper_id)
                    log.info(
                        "  [%d/%d]  OK  %s  (%d KB)",   # ASCII: was ✓
                        total_done,
                        cfg.num_papers,
                        result.paper_id,
                        result.size_kb,
                    )
            else:
                stats["failed"].append(result.paper_id)
                log.warning(
                    "  [%d/%d]  FAIL  %s  -- %s",       # ASCII: was ✗
                    total_done,
                    cfg.num_papers,
                    result.paper_id,
                    result.reason,
                )

    return {k: len(v) for k, v in stats.items()}


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> DownloadConfig:
    parser = argparse.ArgumentParser(
        description="Download QASPER papers from arXiv."
    )
    parser.add_argument(
        "--num-papers", type=int, default=100,
        help="Target number of PDFs to download (default: 100)."
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download threads (default: 4). Keep <= 5 to be polite to arXiv."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"),
        help="Root data directory (default: ./data)."
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to wait between requests per worker (default: 2.0)."
    )
    args = parser.parse_args()
    return DownloadConfig(
        data_dir=args.output_dir,
        num_papers=args.num_papers,
        workers=min(args.workers, 5),  # hard cap — arXiv rate limits aggressively
        inter_request_delay=args.delay,
    )


# ── Entry Point ────────────────────────────────────────────────────────────────


def main() -> None:
    cfg = _parse_args()

    # Create directories
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.pdf_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("QASPER Downloader -- target: %d PDFs", cfg.num_papers)
    log.info("Output dir : %s", cfg.pdf_dir)
    log.info("Workers    : %d", cfg.workers)
    log.info("=" * 60)

    session = _build_session()

    # Phase 1: Ensure archive is present and extracted
    _download_archive(cfg, session)
    _extract_archive(cfg)

    # Phase 2: Locate the JSON and read paper IDs
    json_path = _find_json(cfg)
    paper_ids = _load_paper_ids(json_path)

    if not paper_ids:
        log.error("No paper IDs found in the QASPER JSON. Aborting.")
        sys.exit(1)

    # Phase 3: Download PDFs
    stats = _download_pdfs(paper_ids, cfg)

    # Final summary
    total_on_disk = len(list(cfg.pdf_dir.glob("*.pdf")))
    log.info("=" * 60)
    log.info("Download complete!")
    log.info("  New downloads : %d", stats["downloaded"])
    log.info("  Already had   : %d", stats["skipped"])
    log.info("  Failed        : %d", stats["failed"])
    log.info("  Total on disk : %d", total_on_disk)
    log.info("=" * 60)
    log.info("Next step: python ingestion.py")

    if stats["failed"] > 0:
        log.warning(
            "%d papers failed. Re-run to retry -- the downloader is fully resumable.",
            stats["failed"],
        )
        sys.exit(2)  # non-zero exit for CI pipelines to detect partial failure


if __name__ == "__main__":
    main()
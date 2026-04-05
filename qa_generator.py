"""
qa_generator.py — Synthetic QA Pair Generator + Router Training Data Exporter
Owner: Samved Jain

Reads chunks from data/chunks/chunks.json, prompts a local LLM (via Ollama)
to generate factual / conceptual / complex question-answer pairs, and exports:
  - data/qa_pairs.json              ← full QA records for evaluation
  - data/router_training_data.csv   ← trifold budget-labelled training set

Usage:
    python qa_generator.py [--chunks data/chunks/chunks.json]
                           [--qa-output data/qa_pairs.json]
                           [--csv-output data/router_training_data.csv]
                           [--model phi3:mini]
                           [--factual 10] [--conceptual 10] [--complex 10]
                           [--seed 42] [--max-retries 3] [--retry-delay 2.0]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ollama
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("qa_generator.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

QUERY_TYPES = ("factual", "conceptual", "complex")

# Budget tiers written to the training CSV — order matters for train_router.py
BUDGET_TIERS: list[tuple[float, str]] = [
    (0.9, "Multi_Hop_FAISS"),
    (0.5, "Single_Hop_BM25"),
    (0.1, "Direct_LLM"),
]

CSV_HEADER = ["Query_Text", "Budget_Value", "Target_Label"]

# Characters of chunk text sent to the LLM — longer = more context, more tokens
_PROMPT_CHUNK_CHARS = 800


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    chunks_path: Path = Path("data/chunks/chunks.json")
    qa_output_path: Path = Path("data/qa_pairs.json")
    csv_output_path: Path = Path("data/router_training_data.csv")
    llm_model: str = "phi3:mini"
    target_counts: dict[str, int] = field(
        default_factory=lambda: {"factual": 10, "conceptual": 10, "complex": 10}
    )
    seed: int = 42
    max_retries: int = 3
    retry_delay: float = 2.0       # seconds between LLM retry attempts
    unique_retries: int = 5        # attempts to get a non-duplicate question

    def __post_init__(self) -> None:
        unknown = set(self.target_counts) - set(QUERY_TYPES)
        if unknown:
            raise ValueError(f"Unknown query types: {unknown}. Allowed: {QUERY_TYPES}")
        if self.max_retries < 1:
            raise ValueError("max_retries must be ≥ 1.")


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def _build_prompt(chunk_text: str, query_type: str) -> str:
    """
    Build a focused, diversity-encouraging prompt for the given query type.
    Chunk text is truncated to avoid blowing up LLM context limits.
    """
    snippet = chunk_text[:_PROMPT_CHUNK_CHARS].replace('"""', "'''")
    return f"""\
You are generating HIGH-QUALITY evaluation questions for a Retrieval-Augmented Generation system.

Generate ONE UNIQUE {query_type.upper()} question and a concise answer grounded solely in the text below.

Rules:
- Be SPECIFIC to this chunk — do NOT ask generic questions like "What dataset was used?"
- The answer MUST be answerable from the text only
- Output ONLY valid JSON with exactly these keys: question, answer, query_type
- Do NOT include any explanation, markdown, or extra keys

Format:
{{
  "question": "<your question here>",
  "answer": "<your answer here>",
  "query_type": "{query_type}"
}}

TEXT:
\"\"\"
{snippet}
\"\"\"
"""


# ── LLM Response Parser ────────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?.*?```", re.DOTALL)
_REQUIRED_KEYS = frozenset({"question", "answer", "query_type"})


def _parse_response(raw: str, expected_type: str) -> dict[str, Any] | None:
    """
    Extract and validate a JSON object from the LLM's raw output.
    Returns None if the response is malformed, missing keys, or empty.
    """
    # Strip markdown code fences the model sometimes adds
    cleaned = _CODE_FENCE_RE.sub("", raw).strip()

    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    if not _REQUIRED_KEYS.issubset(data.keys()):
        return None

    if not str(data.get("question", "")).strip():
        return None
    if not str(data.get("answer", "")).strip():
        return None

    # Normalise query_type in case the model drifts from the expected value
    data["query_type"] = expected_type
    return data


# ── QA Generator ──────────────────────────────────────────────────────────────

class QAGenerator:
    """
    Drives per-type QA generation from a chunk corpus using a local Ollama LLM.
    Tracks seen questions to prevent duplicates across the entire session.
    """

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self._pairs: list[dict[str, Any]] = []
        self._seen: set[str] = set()  # normalised question fingerprints

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, chunks: list[dict]) -> list[dict[str, Any]]:
        """
        Generate QA pairs across all configured query types.
        Chunks are shuffled once, then partitioned by type to maximise diversity.
        """
        if not chunks:
            raise ValueError("Cannot generate QA pairs from an empty chunk list.")

        random.seed(self.cfg.seed)
        self._pairs = []
        self._seen = set()

        shuffled = chunks.copy()
        random.shuffle(shuffled)

        pools = self._partition_pools(shuffled)

        for qtype, count in self.cfg.target_counts.items():
            pool = pools[qtype]
            if not pool:
                log.warning("No chunks available for query type '%s'. Skipping.", qtype)
                continue

            selected = random.sample(pool, min(count, len(pool)))
            log.info("Generating %d '%s' questions from %d candidate chunks ...", count, qtype, len(pool))

            for chunk in tqdm(selected, desc=qtype, unit="chunk"):
                pair = self._generate_unique(chunk, qtype)
                if pair:
                    self._pairs.append(pair)
                else:
                    log.debug("No unique pair produced for chunk '%s'.", chunk.get("chunk_id"))

        log.info("Generation complete. Produced %d / %d requested pairs.",
                 len(self._pairs), sum(self.cfg.target_counts.values()))
        return self._pairs

    def save(self) -> None:
        """Persist QA pairs to JSON and router training data to CSV."""
        if not self._pairs:
            log.warning("No QA pairs to save.")
            return
        self._save_qa_json()
        self._save_training_csv()

    def print_stats(self) -> None:
        counts = Counter(p["query_type"] for p in self._pairs)
        log.info("QA pair breakdown: %s", dict(counts))

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _partition_pools(self, shuffled: list[dict]) -> dict[str, list[dict]]:
        """Split shuffled chunks into one non-overlapping pool per query type."""
        n = len(shuffled)
        third = max(1, n // len(QUERY_TYPES))
        pools: dict[str, list[dict]] = {}
        for i, qtype in enumerate(QUERY_TYPES):
            start = i * third
            end = (i + 1) * third if i < len(QUERY_TYPES) - 1 else n
            pools[qtype] = shuffled[start:end]
        return pools

    def _generate_unique(self, chunk: dict, query_type: str) -> dict[str, Any] | None:
        """
        Attempt up to cfg.unique_retries times to produce a non-duplicate pair
        for a single chunk. Returns None if all attempts fail or yield duplicates.
        """
        for _ in range(self.cfg.unique_retries):
            pair = self._call_llm(chunk, query_type)
            if pair is None:
                continue

            fingerprint = pair["question"].strip().lower()
            if fingerprint in self._seen:
                log.debug("Duplicate question discarded: %s", pair["question"][:60])
                continue

            self._seen.add(fingerprint)
            return self._enrich(pair, chunk)

        return None

    def _call_llm(self, chunk: dict, query_type: str) -> dict[str, Any] | None:
        """
        Call the Ollama LLM with retry/backoff.
        Returns a parsed dict or None on persistent failure.
        """
        prompt = _build_prompt(chunk["text"], query_type)

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                response = ollama.chat(
                    model=self.cfg.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.7, "num_predict": 300},
                )
                raw = response["message"]["content"].strip()
                parsed = _parse_response(raw, query_type)
                if parsed:
                    return parsed
                log.debug("Attempt %d: unparseable LLM response.", attempt)

            except ollama.ResponseError as exc:
                log.warning("Ollama ResponseError (attempt %d/%d): %s", attempt, self.cfg.max_retries, exc)
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM call failed (attempt %d/%d): %s", attempt, self.cfg.max_retries, exc)

            if attempt < self.cfg.max_retries:
                time.sleep(self.cfg.retry_delay)

        return None

    @staticmethod
    def _enrich(pair: dict, chunk: dict) -> dict[str, Any]:
        """Attach chunk provenance metadata and normalise the answer key."""
        pair["ground_truth_answer"] = pair.pop("answer")
        pair["source_document"] = chunk.get("source", "")
        pair["relevant_chunk_ids"] = [chunk.get("chunk_id", "")]
        return pair

    def _save_qa_json(self) -> None:
        self.cfg.qa_output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cfg.qa_output_path.with_suffix(".tmp.json")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._pairs, fh, indent=2, ensure_ascii=False)
            tmp.replace(self.cfg.qa_output_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        log.info("Saved %d QA pairs → %s", len(self._pairs), self.cfg.qa_output_path)

    def _save_training_csv(self) -> None:
        """
        Trifold expansion: each question is written once per budget tier,
        producing 3 × N rows covering the full routing decision space.
        """
        self.cfg.csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cfg.csv_output_path.with_suffix(".tmp.csv")
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(CSV_HEADER)
                for pair in self._pairs:
                    q = pair["question"]
                    for budget, label in BUDGET_TIERS:
                        writer.writerow([q, budget, label])
            tmp.replace(self.cfg.csv_output_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        rows_written = len(self._pairs) * len(BUDGET_TIERS)
        log.info(
            "Saved %d training rows (%d questions × %d tiers) → %s",
            rows_written, len(self._pairs), len(BUDGET_TIERS), self.cfg.csv_output_path,
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> GeneratorConfig:
    p = argparse.ArgumentParser(
        description="Generate synthetic QA pairs and router training data from chunk corpus."
    )
    p.add_argument("--chunks", type=Path, default=Path("data/chunks/chunks.json"))
    p.add_argument("--qa-output", type=Path, default=Path("data/qa_pairs.json"))
    p.add_argument("--csv-output", type=Path, default=Path("data/router_training_data.csv"))
    p.add_argument("--model", default="phi3:mini")
    p.add_argument("--factual", type=int, default=10)
    p.add_argument("--conceptual", type=int, default=10)
    p.add_argument("--complex", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=2.0)
    args = p.parse_args()
    return GeneratorConfig(
        chunks_path=args.chunks,
        qa_output_path=args.qa_output,
        csv_output_path=args.csv_output,
        llm_model=args.model,
        target_counts={
            "factual": args.factual,
            "conceptual": args.conceptual,
            "complex": args.complex,
        },
        seed=args.seed,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _parse_args()

    log.info("=" * 60)
    log.info("QA Generator")
    log.info("  Model       : %s", cfg.llm_model)
    log.info("  Targets     : %s", cfg.target_counts)
    log.info("  Chunks path : %s", cfg.chunks_path)
    log.info("=" * 60)

    if not cfg.chunks_path.exists():
        log.error("Chunks file not found at '%s'. Run ingestion.py first.", cfg.chunks_path)
        sys.exit(1)

    with open(cfg.chunks_path, encoding="utf-8") as fh:
        chunks = json.load(fh)

    if not isinstance(chunks, list) or not chunks:
        log.error("Chunks file is empty or malformed.")
        sys.exit(1)

    log.info("Loaded %d chunks.", len(chunks))

    generator = QAGenerator(cfg)
    generator.generate(chunks)
    generator.print_stats()
    generator.save()

    log.info("Next step: python train_router.py")


if __name__ == "__main__":
    main()
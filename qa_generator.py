"""
qa_generator.py — Synthetic QA Pair Generator + Router Training Data Exporter
Owner: Samved Jain

Reads chunks from data/chunks/chunks.json, prompts a local LLM (via Ollama)
to generate factual / conceptual / complex question-answer pairs, and exports:
  - data/qa_pairs.json              ← full QA records for RAGAS evaluation
  - data/router_training_data.csv   ← semantic-labelled training set for train_router.py

Key design decisions:
  - Routing labels are derived from QUESTION SEMANTICS (keyword heuristics),
    NOT from budget tiers. Budget remains a numeric feature but is NOT the label.
  - Per-type temperature: factual=0.3, conceptual=0.6, complex=0.85
  - Checkpoint/resume: pairs saved after every successful generation.
  - Near-duplicate detection via token-set Jaccard similarity.
  - Budget values are uniformly sampled from [0.0, 1.0] — the model
    learns the full budget space, not three fixed points.

Usage:
    python qa_generator.py [--chunks data/chunks/chunks.json]
                           [--qa-output data/qa_pairs.json]
                           [--csv-output data/router_training_data.csv]
                           [--model phi3:mini]
                           [--factual 50] [--conceptual 50] [--complex 50]
                           [--budget-samples 5]
                           [--seed 42] [--max-retries 3] [--retry-delay 2.0]
                           [--resume]
"""

from __future__ import annotations

import argparse
import codecs
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

def _build_stream_handler() -> logging.StreamHandler:
    """UTF-8 safe stream handler."""
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
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        _build_stream_handler(),
        logging.FileHandler("qa_generator.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

QUERY_TYPES = ("factual", "conceptual", "complex")

# Per-type LLM temperature
QUERY_TYPE_TEMPERATURE: dict[str, float] = {
    "factual":    0.3,
    "conceptual": 0.6,
    "complex":    0.85,
}

CSV_HEADER = ["Query_Text", "Budget_Value", "Target_Label"]

# Max chars of chunk text sent to LLM
_PROMPT_CHUNK_CHARS = 900

# Jaccard similarity threshold for near-duplicate detection
_DEDUP_JACCARD_THRESHOLD = 0.70

# Checkpoint filename suffix
_CHECKPOINT_SUFFIX = ".checkpoint.json"

# Budget samples per question (all drawn from [0.0, 1.0])
# Budget stays as a numeric FEATURE — not the label.
_BUDGET_SAMPLES_PER_QUESTION = 15   # more samples → better budget boundary learning

# ── Routing label heuristics ───────────────────────────────────────────────────
# Labels are derived from QUESTION SEMANTICS, not budget.

_MULTI_HOP_KEYWORDS: list[str] = [
    "compare", "contrast", "analyze", "analyse", "why", "how does",
    "how do", "impact", "difference", "difference between",
    "trade-off", "trade off", "relationship", "effect of",
    "effect on", "explain", "role of", "advantage", "disadvantage",
    "versus", " vs ", "evaluate", "critically",
]

_SINGLE_HOP_KEYWORDS: list[str] = [
    "what is", "what are", "what was", "what were",
    "who is", "who are", "who was", "who were",
    "when", "where", "which", "how many", "how much",
    "name the", "list the", "define", "identify",
]

_SINGLE_HOP_MAX_WORDS = 15  # factual lookup questions tend to be concise


def assign_routing_label(question: str) -> str:
    """
    Assign a routing label to *question* using deterministic keyword heuristics.

    Priority:
        1. Multi-hop reasoning keywords  → Multi_Hop_FAISS
        2. Short factual lookup keywords  → Single_Hop_BM25  (length < 15 words)
        3. Fallback                        → Direct_LLM
    """
    q = question.lower().strip()
    word_count = len(re.findall(r"\b\w+\b", q))

    if any(kw in q for kw in _MULTI_HOP_KEYWORDS):
        return "Multi_Hop_FAISS"

    # Priority 2: factual lookup (starts with or contains lookup keywords)
    if any(kw in q for kw in _SINGLE_HOP_KEYWORDS):
        return "Single_Hop_BM25"

    return "Direct_LLM"


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    chunks_path:    Path            = Path("data/chunks/chunks.json")
    qa_output_path: Path            = Path("data/qa_pairs.json")
    csv_output_path: Path           = Path("data/router_training_data.csv")
    llm_model:      str             = "phi3:mini"
    target_counts:  dict[str, int]  = field(
        default_factory=lambda: {"factual": 50, "conceptual": 50, "complex": 50}
    )
    budget_samples: int   = _BUDGET_SAMPLES_PER_QUESTION
    seed:           int   = 42
    max_retries:    int   = 3
    retry_delay:    float = 2.0
    unique_retries: int   = 5
    resume:         bool  = False

    def __post_init__(self) -> None:
        unknown = set(self.target_counts) - set(QUERY_TYPES)
        if unknown:
            raise ValueError(f"Unknown query types: {unknown}. Allowed: {QUERY_TYPES}")
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        if self.budget_samples < 1:
            raise ValueError("budget_samples must be >= 1.")

    @property
    def checkpoint_path(self) -> Path:
        return self.qa_output_path.with_suffix(_CHECKPOINT_SUFFIX)


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def _build_prompt(chunk_text: str, query_type: str) -> str:
    """
    Build a type-specific prompt.

    factual    — asks for a specific, directly stated fact
    conceptual — asks for explanation of a method/idea/term
    complex    — asks for multi-step reasoning or comparison
    """
    snippet = chunk_text[:_PROMPT_CHUNK_CHARS].replace('"""', "'''")

    type_guidance = {
        "factual": (
            "Ask about ONE specific, concrete fact directly stated in the text "
            "(e.g. a number, name, model, result, or dataset). "
            "The answer must be a single sentence or short phrase."
        ),
        "conceptual": (
            "Ask the reader to explain a METHOD, CONCEPT, or TERM introduced in the text. "
            "The answer should be 2-3 sentences explaining how or why it works."
        ),
        "complex": (
            "Ask a MULTI-STEP reasoning question that requires synthesising "
            "two or more ideas from the text (e.g. cause-effect, comparison, trade-off). "
            "The answer should be 3-4 sentences."
        ),
    }[query_type]

    return f"""\
You are generating HIGH-QUALITY evaluation questions for a Retrieval-Augmented Generation benchmark.

Task: Generate ONE UNIQUE {query_type.upper()} question and its answer, grounded solely in the text below.

Guidance for this question type:
{type_guidance}

Hard rules:
- Be SPECIFIC to THIS text — do NOT ask generic questions like "What dataset was used?" or "What is the main contribution?"
- The answer MUST be fully answerable from the provided text only — no outside knowledge
- Output ONLY a single valid JSON object with exactly these three keys: question, answer, query_type
- Do NOT include markdown fences, preamble, explanation, or any text outside the JSON object

Required output format (nothing else):
{{"question": "<your question>", "answer": "<your answer>", "query_type": "{query_type}"}}

TEXT:
\"\"\"
{snippet}
\"\"\"
"""


# ── LLM Response Parser ────────────────────────────────────────────────────────

_JSON_BLOCK_RE  = re.compile(r"\{.*?\}", re.DOTALL)
_CODE_FENCE_RE  = re.compile(r"```(?:json)?\s*|\s*```", re.DOTALL)
_REQUIRED_KEYS  = frozenset({"question", "answer", "query_type"})
_MIN_ANSWER_LEN = 10
_MIN_QUESTION_LEN = 15


def _parse_response(raw: str, expected_type: str) -> dict[str, Any] | None:
    cleaned = _CODE_FENCE_RE.sub("", raw).strip()
    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        log.debug("No JSON object found in LLM response.")
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        log.debug("JSON parse error: %s", exc)
        return None

    if not _REQUIRED_KEYS.issubset(data.keys()):
        log.debug("Missing keys in LLM response: %s", _REQUIRED_KEYS - data.keys())
        return None

    question = str(data.get("question", "")).strip()
    answer   = str(data.get("answer",   "")).strip()

    if len(question) < _MIN_QUESTION_LEN:
        log.debug("Question too short (%d chars).", len(question))
        return None
    if len(answer) < _MIN_ANSWER_LEN:
        log.debug("Answer too short (%d chars).", len(answer))
        return None

    data["question"]   = question
    data["answer"]     = answer
    data["query_type"] = expected_type
    return data


# ── Near-Duplicate Detection ───────────────────────────────────────────────────

def _tokenise(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\b\w+\b", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


# ── Checkpoint Helpers ────────────────────────────────────────────────────────

def _load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        log.info("Resumed %d pairs from checkpoint: %s", len(data), path)
        return data if isinstance(data, list) else []
    except Exception as exc:
        log.warning("Could not load checkpoint (%s). Starting fresh.", exc)
        return []


def _save_checkpoint(pairs: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp.json")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(pairs, fh, ensure_ascii=False)
        tmp.replace(path)
    except Exception as exc:
        log.warning("Checkpoint save failed: %s", exc)


# ── Ollama Connectivity Check ─────────────────────────────────────────────────

def _verify_ollama(model: str) -> None:
    try:
        models_response = ollama.list()
        available = [m.model for m in models_response.models]
        base = model.split(":")[0]
        if not any(base in name for name in available):
            raise RuntimeError(
                f"Model '{model}' not found in Ollama. "
                f"Available: {available}. Run: ollama pull {model}"
            )
        log.info("Ollama OK | model '%s' available.", model)
    except ollama.ResponseError as exc:
        raise RuntimeError(f"Ollama is not reachable: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama connectivity check failed: {exc}") from exc


# ── QA Generator ──────────────────────────────────────────────────────────────

class QAGenerator:
    """
    Drives per-type QA generation from a chunk corpus using a local Ollama LLM.

    Every generated pair receives a 'routing_label' derived from question
    semantics (keyword heuristics). The training CSV uses that label as the
    target — NOT budget tier — so the router learns semantic routing.
    """

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self._pairs: list[dict[str, Any]] = []
        self._seen_tokens: list[frozenset[str]] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, chunks: list[dict]) -> list[dict[str, Any]]:
        if not chunks:
            raise ValueError("Cannot generate QA pairs from an empty chunk list.")

        random.seed(self.cfg.seed)

        if self.cfg.resume:
            self._pairs = _load_checkpoint(self.cfg.checkpoint_path)
            for p in self._pairs:
                self._seen_tokens.append(_tokenise(p["question"]))
        else:
            self._pairs = []
            self._seen_tokens = []

        shuffled = chunks.copy()
        random.shuffle(shuffled)
        pools = self._partition_pools(shuffled)

        for qtype in QUERY_TYPES:
            count = self.cfg.target_counts.get(qtype, 0)
            if count == 0:
                continue

            already_done = sum(1 for p in self._pairs if p["query_type"] == qtype)
            remaining = count - already_done
            if remaining <= 0:
                log.info("'%s': already have %d/%d — skipping.", qtype, already_done, count)
                continue

            pool = pools[qtype]
            if not pool:
                log.warning("No chunks available for query type '%s'. Skipping.", qtype)
                continue

            if remaining <= len(pool):
                selected = random.sample(pool, remaining)
            else:
                selected = random.choices(pool, k=remaining)
                log.warning(
                    "'%s': pool (%d) smaller than target (%d). Sampling with replacement.",
                    qtype, len(pool), remaining,
                )

            log.info(
                "Generating %d '%s' questions (temp=%.2f) ...",
                remaining, qtype, QUERY_TYPE_TEMPERATURE[qtype],
            )

            for chunk in tqdm(selected, desc=qtype, unit="chunk"):
                pair = self._generate_unique(chunk, qtype)
                if pair:
                    self._pairs.append(pair)
                    self._seen_tokens.append(_tokenise(pair["question"]))
                    _save_checkpoint(self._pairs, self.cfg.checkpoint_path)
                else:
                    log.debug(
                        "No unique pair produced for chunk '%s'.",
                        chunk.get("chunk_id"),
                    )

        produced = len(self._pairs)
        requested = sum(self.cfg.target_counts.values())
        log.info("Generation complete. Produced %d / %d requested pairs.", produced, requested)
        return self._pairs

    def save(self) -> None:
        """Persist QA pairs to JSON and router training data to CSV."""
        if not self._pairs:
            log.warning("No QA pairs to save.")
            return
        self._save_qa_json()
        self._save_training_csv()
        if self.cfg.checkpoint_path.exists():
            self.cfg.checkpoint_path.unlink()
            log.info("Checkpoint file removed after successful save.")

    def print_stats(self) -> None:
        counts = Counter(p["query_type"] for p in self._pairs)
        label_dist = Counter(p.get("routing_label", "unset") for p in self._pairs)
        log.info("QA pair breakdown by type: %s", dict(counts))
        log.info("Routing label distribution: %s", dict(label_dist))
        log.info("Total unique questions: %d", len(self._pairs))

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _partition_pools(self, shuffled: list[dict]) -> dict[str, list[dict]]:
        n = len(shuffled)
        pools: dict[str, list[dict]] = {}
        size = -(-n // len(QUERY_TYPES))  # ceiling division
        for i, qtype in enumerate(QUERY_TYPES):
            pools[qtype] = shuffled[i * size: (i + 1) * size]
        return pools

    def _is_near_duplicate(self, question: str) -> bool:
        tokens = _tokenise(question)
        for seen in self._seen_tokens:
            if _jaccard(tokens, seen) >= _DEDUP_JACCARD_THRESHOLD:
                return True
        return False

    def _generate_unique(self, chunk: dict, query_type: str) -> dict[str, Any] | None:
        for attempt in range(self.cfg.unique_retries):
            pair = self._call_llm(chunk, query_type)
            if pair is None:
                continue
            if self._is_near_duplicate(pair["question"]):
                log.debug(
                    "Near-duplicate discarded (attempt %d): %s",
                    attempt + 1, pair["question"][:60],
                )
                continue
            return self._enrich(pair, chunk)
        return None

    def _call_llm(self, chunk: dict, query_type: str) -> dict[str, Any] | None:
        prompt      = _build_prompt(chunk["text"], query_type)
        temperature = QUERY_TYPE_TEMPERATURE[query_type]

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                response = ollama.chat(
                    model=self.cfg.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature, "num_predict": 350},
                )
                raw    = response["message"]["content"].strip()
                parsed = _parse_response(raw, query_type)
                if parsed:
                    return parsed
                log.debug("Attempt %d/%d: unparseable response.", attempt, self.cfg.max_retries)

            except ollama.ResponseError as exc:
                log.warning(
                    "Ollama ResponseError (attempt %d/%d): %s",
                    attempt, self.cfg.max_retries, exc,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt, self.cfg.max_retries, exc,
                )

            if attempt < self.cfg.max_retries:
                time.sleep(self.cfg.retry_delay)

        return None

    @staticmethod
    def _enrich(pair: dict, chunk: dict) -> dict[str, Any]:
        """
        Attach chunk provenance + routing label derived from question semantics.

        'routing_label' is set by keyword heuristics on the question text,
        NOT by budget — this is the key fix for semantic router training.
        """
        question = pair["question"]
        pair["ground_truth_answer"] = pair.pop("answer")
        pair["source_document"]     = chunk.get("source", "")
        pair["relevant_chunk_ids"]  = [chunk.get("chunk_id", "")]
        pair["routing_label"]       = assign_routing_label(question)
        return pair

    def _save_qa_json(self) -> None:
        """Atomically write QA pairs to JSON for RAGAS evaluation."""
        self.cfg.qa_output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cfg.qa_output_path.with_suffix(".tmp.json")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._pairs, fh, indent=2, ensure_ascii=False)
            tmp.replace(self.cfg.qa_output_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        log.info("Saved %d QA pairs -> %s", len(self._pairs), self.cfg.qa_output_path)

    def _save_training_csv(self) -> None:
        """
        Write the router training CSV.

        KEY CHANGE vs. old implementation:
          - Target label = pair["routing_label"]  (semantic, from question)
          - Budget       = uniform random [0.0, 1.0]  (numeric feature only)

        This teaches the router:
            factual questions    → Single_Hop_BM25   (regardless of budget)
            reasoning questions  → Multi_Hop_FAISS   (regardless of budget)
            vague questions      → Direct_LLM        (regardless of budget)

        Budget modulates confidence at inference time but is not the decision axis.
        """
        self.cfg.csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cfg.csv_output_path.with_suffix(".tmp.csv")

        total_rows = 0
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(CSV_HEADER)
                for pair in self._pairs:
                    q             = pair["question"]
                    routing_label = pair.get(
                        "routing_label", assign_routing_label(q)
                    )
                    # Sample budget uniformly — it stays as a feature, not a label
                    for _ in range(self.cfg.budget_samples):
                        budget = round(random.uniform(0.0, 1.0), 4)
                        writer.writerow([q, budget, routing_label])
                        total_rows += 1
            tmp.replace(self.cfg.csv_output_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        # Print routing label distribution in CSV
        label_counts: Counter = Counter(
            pair.get("routing_label", "unset") for pair in self._pairs
        )
        log.info(
            "Saved %d training rows (%d questions x %d budget samples) -> %s",
            total_rows, len(self._pairs), self.cfg.budget_samples,
            self.cfg.csv_output_path,
        )
        log.info(
            "Routing label distribution in CSV: %s",
            dict(label_counts),
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> GeneratorConfig:
    p = argparse.ArgumentParser(
        description="Generate synthetic QA pairs and router training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--chunks",         type=Path,  default=Path("data/chunks/chunks.json"))
    p.add_argument("--qa-output",      type=Path,  default=Path("data/qa_pairs.json"))
    p.add_argument("--csv-output",     type=Path,  default=Path("data/router_training_data.csv"))
    p.add_argument("--model",          default="phi3:mini")
    p.add_argument("--factual",        type=int,   default=50)
    p.add_argument("--conceptual",     type=int,   default=50)
    p.add_argument("--complex",        type=int,   default=50)
    p.add_argument("--budget-samples", type=int,   default=_BUDGET_SAMPLES_PER_QUESTION,
                   help="Budget value samples per question in training CSV.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--max-retries",    type=int,   default=3)
    p.add_argument("--retry-delay",    type=float, default=2.0)
    p.add_argument("--unique-retries", type=int,   default=5)
    p.add_argument("--resume",         action="store_true",
                   help="Resume from checkpoint instead of starting fresh.")
    args = p.parse_args()
    return GeneratorConfig(
        chunks_path     = args.chunks,
        qa_output_path  = args.qa_output,
        csv_output_path = args.csv_output,
        llm_model       = args.model,
        target_counts   = {
            "factual":    args.factual,
            "conceptual": args.conceptual,
            "complex":    args.complex,
        },
        budget_samples  = args.budget_samples,
        seed            = args.seed,
        max_retries     = args.max_retries,
        retry_delay     = args.retry_delay,
        unique_retries  = args.unique_retries,
        resume          = args.resume,
    )


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _parse_args()

    log.info("=" * 60)
    log.info("Eco-RAG QA Generator")
    log.info("  Model          : %s", cfg.llm_model)
    log.info("  Targets        : %s", cfg.target_counts)
    log.info("  Budget samples : %d per question (uniform [0,1])", cfg.budget_samples)
    log.info("  Routing labels : SEMANTIC (keyword heuristics, not budget)")
    log.info("  Resume         : %s", cfg.resume)
    log.info("  Chunks path    : %s", cfg.chunks_path)
    log.info("=" * 60)

    if not cfg.chunks_path.exists():
        log.error("Chunks file not found at '%s'. Run ingestion.py first.", cfg.chunks_path)
        sys.exit(1)

    try:
        _verify_ollama(cfg.llm_model)
    except RuntimeError as exc:
        log.error("%s", exc)
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

    log.info("Next step: python train_router.py --force")


if __name__ == "__main__":
    main()
"""
patch_qa_labels.py — One-off script to add 'routing_label' to existing qa_pairs.json

Loads data/qa_pairs.json, assigns a routing_label to each entry based on
keyword heuristics over the question text, and writes the result back in-place.

Logic:
  Multi_Hop_FAISS   — reasoning / comparative / causal keywords
  Single_Hop_BM25   — short factual lookup keywords (AND length < 15 words)
  Direct_LLM        — fallback for everything else

Usage:
    python patch_qa_labels.py [--input data/qa_pairs.json] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ── Heuristic keyword sets ──────────────────────────────────────────────────────

MULTI_HOP_KEYWORDS: list[str] = [
    "compare", "contrast", "analyze", "analyse", "why", "how does",
    "how do", "impact", "difference", "difference between",
    "trade-off", "trade off", "relationship", "effect of",
    "effect on", "explain", "role of", "advantage", "disadvantage",
    "versus", " vs ", "evaluate", "critically",
]

SINGLE_HOP_KEYWORDS: list[str] = [
    "what is", "what are", "what was", "what were",
    "who is", "who are", "who was", "who were",
    "when", "where", "which", "how many", "how much",
    "name the", "list the", "define", "identify",
]

# A single-hop label is only assigned when the question is ALSO short
_SINGLE_HOP_MAX_WORDS = 15


# ── Core labelling function ────────────────────────────────────────────────────

def assign_routing_label(question: str) -> str:
    """
    Return a routing label for *question* using deterministic heuristics.

    Priority order:
        1. Multi-hop / reasoning keywords  → Multi_Hop_FAISS
        2. Short factual / lookup keywords  → Single_Hop_BM25
        3. Fallback                         → Direct_LLM
    """
    q_lower = question.lower().strip()
    word_count = len(re.findall(r"\b\w+\b", q_lower))

    # Priority 1: reasoning / comparative / causal
    if any(kw in q_lower for kw in MULTI_HOP_KEYWORDS):
        return "Multi_Hop_FAISS"

    # Priority 2: factual / lookup keywords — no word count restriction
    if any(kw in q_lower for kw in SINGLE_HOP_KEYWORDS):
        return "Single_Hop_BM25"

    # Fallback
    return "Direct_LLM"


# ── File helpers ───────────────────────────────────────────────────────────────

def load_qa_pairs(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at '{path}', got {type(data).__name__}.")
    return data


def save_qa_pairs(pairs: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp.json")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(pairs, fh, indent=2, ensure_ascii=False)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Patch qa_pairs.json to add 'routing_label' to every entry."
    )
    p.add_argument(
        "--input", type=Path, default=Path("data/qa_pairs.json"),
        help="Path to the QA pairs JSON file.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print label distribution without writing to disk.",
    )
    args = p.parse_args()

    if not args.input.exists():
        print(f"[ERROR] File not found: '{args.input}'")
        sys.exit(1)

    pairs = load_qa_pairs(args.input)
    print(f"Loaded {len(pairs)} QA pairs from '{args.input}'.")

    # Patch
    already_labelled = sum(1 for p in pairs if "routing_label" in p)
    if already_labelled:
        print(f"  Note: {already_labelled} pairs already have 'routing_label' — overwriting.")

    for pair in pairs:
        question = pair.get("question", "")
        pair["routing_label"] = assign_routing_label(question)

    # Stats
    dist = Counter(p["routing_label"] for p in pairs)
    total = len(pairs)
    print("\nRouting label distribution:")
    print(f"  {'Label':<20} {'Count':>5}  {'%':>6}")
    print(f"  {'-'*35}")
    for label in ("Multi_Hop_FAISS", "Single_Hop_BM25", "Direct_LLM"):
        n = dist.get(label, 0)
        print(f"  {label:<20} {n:>5}  {n/total*100:>5.1f}%")

    if args.dry_run:
        print("\n[Dry run] — file NOT written.")
        return

    save_qa_pairs(pairs, args.input)
    print(f"\n[OK] '{args.input}' updated in-place with routing_label.")
    print("Next: regenerate router_training_data.csv, then run:")
    print("      python train_router.py --force")


if __name__ == "__main__":
    main()

"""
router_eval.py — Router Evaluation Benchmark
Owner: Nivi

Evaluates QueryRouter performance across all three budget tiers so the
degradation curve (expensive → cheap routing as budget drops) is fully
visible in the results.

Budget sweep:
    1.0  →  expects Multi_Hop_FAISS   (dense retrieval + reranking)
    0.5  →  expects Single_Hop_BM25   (sparse keyword retrieval)
    0.1  →  expects Direct_LLM        (bypass retrieval entirely)

Router API contract (router.py):
    router.classify(query: str, budget: float)
        → tuple[str, float, str]
           (label, confidence, method)
    where label ∈ {"Multi_Hop_FAISS", "Single_Hop_BM25", "Direct_LLM"}

Input:
    data/qa_pairs.json
        Required keys per item: "question", "query_type"
        "query_type" ∈ {"factual", "conceptual", "complex"}

Output:
    results/router_eval.json   ← per-budget confusion matrices + metrics

QA query_type → expected router label mapping:
    factual    → Multi_Hop_FAISS  (precise retrieval needed)
    conceptual → Single_Hop_BM25  (keyword match sufficient)
    complex    → Multi_Hop_FAISS  (multi-hop reasoning needed)

    This mapping is applied at eval time — the router itself is unaware
    of query_type labels and only sees (query, budget).
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from router import get_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RouterEval")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
QA_PATH     = "data/qa_pairs.json"
OUTPUT_PATH = "results/router_eval.json"

# Budget tiers — must match qa_generator.py / ablation_runner.py
BUDGET_TIERS: list[float] = [1.0, 0.5, 0.1]

# Router output labels (what router.classify() actually returns)
ROUTER_LABELS: list[str] = ["Multi_Hop_FAISS", "Single_Hop_BM25", "Direct_LLM"]

# QA dataset query_type values
QUERY_TYPES: list[str] = ["factual", "conceptual", "complex"]

# ---------------------------------------------------------------------------
# Ground-truth label: read routing_label from qa_pairs.json directly.
# The router should be evaluated against the SEMANTIC label assigned to each
# question, NOT a hardcoded budget→label mapping (that was the old bug).
# ---------------------------------------------------------------------------

# Fallback query_type → routing_label for pairs that lack routing_label
_QUERY_TYPE_FALLBACK: dict[str, str] = {
    "factual":    "Single_Hop_BM25",
    "conceptual": "Multi_Hop_FAISS",
    "complex":    "Multi_Hop_FAISS",
}

def _true_label(item: dict) -> str:
    """Return the ground-truth routing label for a QA pair."""
    # Prefer the explicit routing_label field set by qa_generator / patch script
    if "routing_label" in item and item["routing_label"] in ROUTER_LABELS:
        return item["routing_label"]
    # Fallback: derive from query_type
    return _QUERY_TYPE_FALLBACK.get(item.get("query_type", "factual"), "Direct_LLM")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atomic_json_write(path: str, data: Any) -> None:
    """Write JSON atomically via .tmp → rename (POSIX atomic)."""
    p   = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


def load_qa_pairs(path: str) -> list[dict[str, Any]]:
    """
    Load and validate QA pairs from disk.

    Raises:
        FileNotFoundError: QA file missing.
        ValueError:        Empty file or missing required keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"QA dataset not found at '{path}'. Run load_qasper.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"QA file '{path}' is empty or not a list.")

    required = {"question", "query_type"}
    missing  = required - set(data[0].keys())
    if missing:
        raise ValueError(
            f"QA items missing required keys: {missing}. "
            f"Check qa_generator.py output schema."
        )

    return data


def _empty_confusion() -> dict[str, dict[str, int]]:
    """Zero-initialised confusion matrix over ROUTER_LABELS."""
    return {true: {pred: 0 for pred in ROUTER_LABELS} for true in ROUTER_LABELS}


def _per_label_metrics(
    confusion: dict[str, dict[str, int]]
) -> dict[str, dict[str, float]]:
    """
    Derive precision, recall, and F1 per router label from a confusion matrix.

    Returns:
        { label: { "precision": float, "recall": float, "f1": float,
                   "support": int } }
    """
    metrics: dict[str, dict[str, float]] = {}

    for label in ROUTER_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in ROUTER_LABELS if other != label)
        fn = sum(confusion[label][other] for other in ROUTER_LABELS if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support   = tp + fn

        metrics[label] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   support,
        }

    return metrics


# ---------------------------------------------------------------------------
# Per-budget evaluation
# ---------------------------------------------------------------------------

def evaluate_at_budget(
    qa_pairs: list[dict[str, Any]],
    budget:   float,
) -> dict[str, Any]:
    """
    Run router evaluation for a single budget tier.

    For each QA item:
      1. Map item["query_type"] → expected router label via EXPECTED_LABEL[budget].
      2. Call router.classify(query, budget) → (pred_label, confidence, method).
      3. Accumulate confusion matrix over ROUTER_LABELS.

    Args:
        qa_pairs: Loaded QA pairs list.
        budget:   Budget scalar in [0.0, 1.0].

    Returns:
        Per-budget result dict with confusion matrix, accuracy, and
        per-label precision/recall/F1.
    """
    router = get_router()

    total   = len(qa_pairs)
    correct = 0

    confusion        = _empty_confusion()
    label_total:   dict[str, int] = defaultdict(int)
    label_correct: dict[str, int] = defaultdict(int)
    confidences:   list[float]    = []
    fallback_count: int           = 0

    for item in qa_pairs:
        query      = item["question"]

        # Ground-truth label from routing_label field (semantic), not budget
        true_label = _true_label(item)

        # router.classify(query, budget) → (label, confidence, method)
        pred_label, confidence, method = router.classify(query, budget)

        confidences.append(confidence)
        if method != "ml_router":
            fallback_count += 1

        label_total[true_label] += 1

        if pred_label == true_label:
            correct += 1
            label_correct[true_label] += 1

        # Guard: pred_label might be unexpected if router falls back
        if pred_label not in ROUTER_LABELS:
            logger.warning(
                "Unexpected pred_label '%s' at budget=%.1f — "
                "mapping to Direct_LLM fallback.",
                pred_label, budget,
            )
            pred_label = "Direct_LLM"

        confusion[true_label][pred_label] += 1

    accuracy    = correct / total if total > 0 else 0.0
    mean_conf   = sum(confidences) / len(confidences) if confidences else 0.0
    label_metrics = _per_label_metrics(confusion)

    per_label_accuracy: dict[str, float] = {
        lbl: round(label_correct[lbl] / label_total[lbl], 4)
        if label_total[lbl] > 0 else 0.0
        for lbl in ROUTER_LABELS
    }

    return {
        "budget":               budget,
        "total":                total,
        "correct":              correct,
        "overall_accuracy":     round(accuracy, 4),
        "mean_confidence":      round(mean_conf, 4),
        "fallback_count":       fallback_count,
        "per_label_accuracy":   per_label_accuracy,
        "per_label_metrics":    label_metrics,
        "confusion_matrix":     confusion,
    }


# ---------------------------------------------------------------------------
# Console printer
# ---------------------------------------------------------------------------

def _print_budget_result(result: dict[str, Any]) -> None:
    """Pretty-print one budget tier's evaluation result to stdout."""
    budget   = result["budget"]
    total    = result["total"]
    correct  = result["correct"]
    accuracy = result["overall_accuracy"]

    sep = "─" * 58
    print(f"\n{sep}")
    print(f"  Budget = {budget:.1f}")
    print(sep)
    print(f"  Queries   : {total}")
    print(f"  Correct   : {correct}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Mean conf : {result['mean_confidence']:.4f}")
    print(f"  Fallbacks : {result['fallback_count']}")

    print("\n  Per-label accuracy:")
    for lbl in ROUTER_LABELS:
        acc = result["per_label_accuracy"].get(lbl, 0.0)
        print(f"    {lbl:<20} {acc:.4f}")

    print("\n  Per-label metrics (P / R / F1):")
    for lbl in ROUTER_LABELS:
        m = result["per_label_metrics"].get(lbl, {})
        print(
            f"    {lbl:<20}  "
            f"P={m.get('precision',0):.4f}  "
            f"R={m.get('recall',0):.4f}  "
            f"F1={m.get('f1',0):.4f}  "
            f"(n={m.get('support',0)})"
        )

    # Confusion matrix
    cm    = result["confusion_matrix"]
    col_w = 20
    print("\n  Confusion Matrix  (rows=True, cols=Predicted):")
    header = f"  {'True \\ Pred':<20}" + "".join(f"{lbl:>{col_w}}" for lbl in ROUTER_LABELS)
    print(header)
    print("  " + "─" * (20 + col_w * len(ROUTER_LABELS)))
    for true_lbl in ROUTER_LABELS:
        row = f"  {true_lbl:<20}" + "".join(
            f"{cm[true_lbl].get(pred_lbl, 0):>{col_w}}" for pred_lbl in ROUTER_LABELS
        )
        print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_router(
    qa_path:    str = QA_PATH,
    output_path: str = OUTPUT_PATH,
) -> dict[str, Any]:
    """
    Evaluate the router across all BUDGET_TIERS and save results.

    Returns:
        Full output dict written to output_path.
    """
    print("=" * 60)
    print("  Router Evaluation — Budget-Tier Sweep")
    print("=" * 60)
    print(f"  QA path     : {qa_path}")
    print(f"  Output path : {output_path}")
    print(f"  Budgets     : {BUDGET_TIERS}")
    print()

    qa_pairs = load_qa_pairs(qa_path)
    logger.info("Loaded %d QA pairs.", len(qa_pairs))

    results_per_budget: dict[str, dict[str, Any]] = {}

    for budget in BUDGET_TIERS:
        logger.info("Evaluating at budget=%.1f …", budget)
        result = evaluate_at_budget(qa_pairs, budget)
        results_per_budget[str(budget)] = result
        _print_budget_result(result)

    # ── Aggregate across budgets ──────────────────────────────────────
    all_correct = sum(r["correct"] for r in results_per_budget.values())
    all_total   = sum(r["total"]   for r in results_per_budget.values())
    macro_acc   = all_correct / all_total if all_total > 0 else 0.0

    output: dict[str, Any] = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "qa_path":         qa_path,
        "budget_tiers":    BUDGET_TIERS,
        "router_labels":   ROUTER_LABELS,
        "macro_accuracy":  round(macro_acc, 4),
        "per_budget":      results_per_budget,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _atomic_json_write(output_path, output)

    print("─" * 60)
    print(f"  Macro accuracy (all budgets) : {macro_acc:.4f}")
    print(f"  Saved → {output_path}")
    print("─" * 60)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate QueryRouter across budget tiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qa-path", type=str, default=QA_PATH,
        help="Path to QA pairs JSON.",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_PATH,
        help="Path to write router_eval.json.",
    )
    args = parser.parse_args()

    evaluate_router(qa_path=args.qa_path, output_path=args.output)
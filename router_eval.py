"""
router_eval.py — Router Evaluation
Owner: Nivi

Evaluates QueryRouter performance using QA dataset.

Input:
    data/qa_pairs.json

Output:
    results/router_eval.json
"""

import json
import os
from collections import defaultdict

from router import QueryRouter

QA_PATH = "data/qa_pairs.json"
OUTPUT_PATH = "results/router_eval.json"


# -------------------------------------------------------
# Load QA pairs
# -------------------------------------------------------

def load_qa_pairs(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

def evaluate_router():
    print("=" * 60)
    print("Router Evaluation")
    print("=" * 60)

    qa_pairs = load_qa_pairs(QA_PATH)
    router = QueryRouter()

    total = len(qa_pairs)
    correct = 0

    labels = ["factual", "conceptual", "complex"]

    # confusion matrix
    confusion = {l: {ll: 0 for ll in labels} for l in labels}

    # per-class stats
    class_total = defaultdict(int)
    class_correct = defaultdict(int)

    for item in qa_pairs:
        query = item["question"]
        true_type = item["query_type"]

        pred_type, confidence, method = router.classify(query)

        class_total[true_type] += 1

        if pred_type == true_type:
            correct += 1
            class_correct[true_type] += 1

        confusion[true_type][pred_type] += 1

    # -------------------------------------------------------
    # Metrics
    # -------------------------------------------------------

    accuracy = correct / total if total > 0 else 0.0

    per_class_accuracy = {}
    for label in labels:
        if class_total[label] == 0:
            per_class_accuracy[label] = 0.0
        else:
            per_class_accuracy[label] = round(
                class_correct[label] / class_total[label], 4
            )

    # -------------------------------------------------------
    # Print nicely
    # -------------------------------------------------------

    print(f"\nTotal queries : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.4f}")

    print("\nPer-class accuracy:")
    for label in labels:
        print(f"  {label:<10}: {per_class_accuracy[label]:.4f}")

    print("\nConfusion Matrix:")
    print("True \\ Pred  | factual | conceptual | complex")
    print("-" * 50)

    for true_label in labels:
        row = confusion[true_label]
        print(f"{true_label:<12} | "
              f"{row['factual']:^7} | "
              f"{row['conceptual']:^10} | "
              f"{row['complex']:^7}")

    # -------------------------------------------------------
    # Save results
    # -------------------------------------------------------

    os.makedirs("results", exist_ok=True)

    output = {
        "overall_accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved → {OUTPUT_PATH}")


# -------------------------------------------------------
# Run
# -------------------------------------------------------

if __name__ == "__main__":
    evaluate_router()
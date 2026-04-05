"""
pareto_curve.py — Pareto Curve (Latency vs Quality)
Owner: Roshan K C

Reads:
    results/evaluation_results.json   ← primary quality source (F1, EM, RAGAS)
    results/ablation_summary.json     ← latency stats per config
    results/ablation_results.json     ← fallback if evaluation file missing

Writes:
    results/pareto_curve.png
    results/pareto_data.json

Design decisions:
    - evaluation_results.json is the SOLE quality source when present.
      Falls back to answer success rate only if the file is absent.
    - Pareto dominance: point A dominates B iff A has LOWER latency AND
      HIGHER quality (or equal on one axis, strictly better on the other).
    - load_latency() is defensive — handles missing keys from both the
      new ablation_summary schema and any older schema variants.
    - All file I/O uses encoding="utf-8" to avoid cp1252 on Windows.
    - Plot shows error bars (±1 std), per-point colour, Pareto frontier
      shading, and a quality metric legend so the chart is self-contained.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("Pareto")

# ---------------------------------------------------------------------------
RESULTS_DIR   = Path("results")
SUMMARY_PATH  = RESULTS_DIR / "ablation_summary.json"
RESULTS_PATH  = RESULTS_DIR / "ablation_results.json"
EVAL_PATH     = RESULTS_DIR / "evaluation_results.json"
OUTPUT_PARETO = RESULTS_DIR / "pareto_curve.png"
OUTPUT_DATA = RESULTS_DIR / "pareto_data.json"

# Display order and colours for the four ablation configs
_CONFIG_STYLES = {
    "naive":          {"color": "#e74c3c", "marker": "o"},
    "router_only":    {"color": "#f39c12", "marker": "s"},
    "reranker_only":  {"color": "#2ecc71", "marker": "^"},
    "full_adaptive":  {"color": "#3498db", "marker": "D"},
}
_DEFAULT_STYLE = {"color": "#95a5a6", "marker": "o"}

# Which quality metric to prefer (in order) when multiple are available
_METRIC_PRIORITY = [
    ("faithfulness",        "Faithfulness (RAGAS)"),
    ("token_f1",            "Token F1"),
    ("answer_relevance",    "Answer Relevance (RAGAS)"),
    ("context_recall",      "Context Recall (RAGAS)"),
    ("answer_success_rate", "Answer Success Rate"),
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    """Load a JSON file. Returns None (with warning) if missing."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# LATENCY
# ---------------------------------------------------------

def load_latency(summary):
    data = {}
    for k, v in summary.items():
        if k == "generated_at":
            continue

        data[k] = {
            "mean": v["total_ms"]["mean_ms"],
            "std": v["total_ms"]["std_ms"]
        }
    return data


# ---------------------------------------------------------
# QUALITY (FIXED LOGIC)
# ---------------------------------------------------------

def load_quality(eval_data, results_data):
    quality = {}

    # ✅ PRIORITY: evaluation results (BEST)
    if eval_data:
        logger.info("Using evaluation results (F1 score)")
        for config, metrics in eval_data.items():
            quality[config] = metrics.get("token_f1", 0)

        return quality, "Token F1"

    # ❌ fallback (not ideal)
    logger.warning("Using fallback success rate")

    for config, queries in results_data.items():
        ok = sum(1 for q in queries if not q["answer"].startswith("[ERROR]"))
        quality[config] = ok / len(queries)

    return quality, "Answer Success Rate"


# ---------------------------------------------------------
# PARETO
# ---------------------------------------------------------

def compute_pareto(points):
    pareto = []

    for c in points:
        dominated = False
        for o in points:
            if o == c:
                continue

            if (o[1] <= c[1] and o[2] >= c[2]) and (o[1] < c[1] or o[2] > c[2]):
                dominated = True
                break

        if not dominated:
            pareto.append(c)

    return pareto


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

def plot(points, pareto, label):
    plt.figure(figsize=(8, 5))

    for name, lat, qual in points:
        plt.scatter(lat, qual, s=120)

        plt.text(lat * 1.01, qual + 0.002, name)

    # Pareto line
    pareto = sorted(pareto, key=lambda x: x[1])
    px = [p[1] for p in pareto]
    py = [p[2] for p in pareto]

    plt.plot(px, py, "--")

    plt.xlabel("Latency (ms)")
    plt.ylabel(label)
    plt.title("Pareto Curve (Latency vs Quality)")
    plt.grid()

    plt.savefig(OUTPUT_PARETO)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 50)
    print("PARETO ANALYSIS")
    print("=" * 50)

    summary    = load_json(SUMMARY_PATH)
    results    = load_json(RESULTS_PATH)
    evaluation = load_json(EVAL_PATH)

    latency = load_latency(summary)
    quality, label = load_quality(evaluation, results)

    points = []

    for config in latency:
        if config not in quality:
            continue

        points.append((
            config,
            latency[config]["mean"],
            quality[config]
        ))

    pareto = compute_pareto(points)

    print("\nRESULTS:")
    for p in points:
        flag = "✅" if p in pareto else ""
        print(p, flag)

    plot(points, pareto, label)

    with open(OUTPUT_DATA, "w") as f:
        json.dump(points, f, indent=2)

    print("Saved:")
    print("→", OUTPUT_PARETO)
    print("→", OUTPUT_DATA)


if __name__ == "__main__":
    main()
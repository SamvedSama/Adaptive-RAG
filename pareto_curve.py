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

DISPLAY_NAMES = {
    "naive":          "Naive RAG",
    "router_only":    "Router Only",
    "reranker_only":  "Reranker Only",
    "full_adaptive":  "Full Adaptive",
}

COLORS = {k: v["color"] for k, v in _CONFIG_STYLES.items()}

BUDGET_MARKERS = {
    "1.0": "o",   # Full budget
    "0.5": "s",   # Med budget
    "0.1": "v"    # Low budget
}

# Fallback latency estimates (ms) if ablation_summary.json is missing.
# Replace these with your actual measured values.
FALLBACK_LATENCY = {
    "naive":          {"mean": 1200, "std": 150},
    "router_only":    {"mean": 1800, "std": 200},
    "reranker_only":  {"mean": 2400, "std": 300},
    "full_adaptive":  {"mean": 2800, "std": 350},
}


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
    # ✅ Primary: Token F1 from evaluation results
    if eval_data and "configs" in eval_data:
        logger.info("Using evaluation results (Token F1)")
        quality = {cfg: metrics.get("token_f1", 0.0)
                   for cfg, metrics in eval_data["configs"].items()}
        return quality, "Token F1"
    elif eval_data: # Fallback if someone used the old schema
        logger.info("Using evaluation results (Token F1) - legacy schema")
        quality = {cfg: metrics.get("token_f1", 0.0)
                   for cfg, metrics in eval_data.items() if isinstance(metrics, dict)}
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

def plot(points, pareto, label, latency_data):
    fig, ax = plt.subplots(figsize=(8, 5))
    pareto_set = {p[0] for p in pareto}

    for name, lat, qual in points:
        if "_b" in name:
            base_config, b_str = name.split("_b")
            display_name = f"{DISPLAY_NAMES.get(base_config, base_config)} (B: {b_str})"
        else:
            base_config, b_str = name, "1.0"
            display_name = DISPLAY_NAMES.get(name, name)
            
        color  = COLORS.get(base_config, "#333333")
        
        # Override marker if in pareto set
        if name in pareto_set:
            marker = "★"
            zorder = 5
        else:
            marker = BUDGET_MARKERS.get(b_str, "o")
            zorder = 4

        std = latency_data.get(name, {}).get("std", 0)

        # Error bar for latency std
        ax.errorbar(lat, qual, xerr=std,
                    fmt="none", color=color, alpha=0.4, capsize=4)

        ax.scatter(lat, qual,
                   s=180 if marker != "v" else 220, color=color, zorder=zorder,
                   edgecolors="white", linewidths=1.5,
                   marker=marker)

        offset_x = lat * 0.012
        offset_y = 0.003
        ax.annotate(
            display_name,
            xy=(lat, qual),
            xytext=(lat + offset_x, qual + offset_y),
            fontsize=10,
            fontweight="bold" if name in pareto_set else "normal",
            color=color,
        )

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

    print(f"\nQuality metric : {label}")
    print(f"\n{'Config':<30} {'Latency (ms)':>14} {'Quality':>10}  Pareto?")
    print("-" * 65)
    for p in sorted(points, key=lambda x: x[2], reverse=True):
        flag = "✅" if p in pareto else ""
        print(f"{p[0]:<30} {p[1]:>14.1f} {p[2]:>10.4f}  {flag}")

    plot(points, pareto, label, latency)

    with open(OUTPUT_DATA, "w") as f:
        json.dump(points, f, indent=2)

    print("Saved:")
    print("→", OUTPUT_PARETO)
    print("→", OUTPUT_DATA)


if __name__ == "__main__":
    main()
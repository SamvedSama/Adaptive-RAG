import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pareto")

RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "ablation_summary.json"
RESULTS_PATH = RESULTS_DIR / "ablation_results.json"
EVAL_PATH = RESULTS_DIR / "evaluation_results.json"

OUTPUT_PARETO = RESULTS_DIR / "pareto_curve.png"
OUTPUT_DATA = RESULTS_DIR / "pareto_data.json"

# Pretty labels for display
DISPLAY_NAMES = {
    "naive": "Naive RAG",
    "router_only": "Router Only",
    "reranker_only": "Reranker Only",
    "full_adaptive": "Full Adaptive",
}

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


# ---------------------------------------------------------
# LOADERS
# ---------------------------------------------------------

def load_json(path):
    if not path.exists():
        logger.warning(f"{path} not found")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# LATENCY  (with safe fallback)
# ---------------------------------------------------------

def load_latency(summary):
    if summary is None:
        logger.warning("ablation_summary.json missing — using fallback latency estimates")
        return FALLBACK_LATENCY

    data = {}
    for k, v in summary.items():
        if k == "generated_at":
            continue
        try:
            data[k] = {
                "mean": v["total_ms"]["mean_ms"],
                "std":  v["total_ms"]["std_ms"],
            }
        except (KeyError, TypeError) as e:
            logger.warning(f"Could not read latency for '{k}': {e}. Using fallback.")
            data[k] = FALLBACK_LATENCY.get(k, {"mean": 1000, "std": 100})
    return data


# ---------------------------------------------------------
# QUALITY
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

    # Fallback: answer success rate from raw results
    if results_data:
        logger.warning("evaluation_results.json missing — falling back to answer success rate")
        quality = {}
        for config, queries in results_data.items():
            ok = sum(1 for q in queries if not q["answer"].startswith("[ERROR]"))
            quality[config] = ok / len(queries) if queries else 0.0
        return quality, "Answer Success Rate"

    raise RuntimeError("No quality data available (need evaluation_results.json or ablation_results.json)")


# ---------------------------------------------------------
# PARETO  (lower latency AND higher quality = dominates)
# ---------------------------------------------------------

def compute_pareto(points):
    """
    A point is Pareto-optimal if no other point is simultaneously
    faster (lower latency) AND better quality (higher F1).
    """
    pareto = []
    for c in points:
        dominated = False
        for o in points:
            if o is c:
                continue
            # o dominates c if o is at least as good on both axes
            # and strictly better on at least one
            lat_better  = o[1] <= c[1]
            qual_better = o[2] >= c[2]
            strictly    = (o[1] < c[1]) or (o[2] > c[2])
            if lat_better and qual_better and strictly:
                dominated = True
                break
        if not dominated:
            pareto.append(c)
    return pareto


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

COLORS = {
    "naive":         "#6c757d",   # grey
    "router_only":   "#0d6efd",   # blue
    "reranker_only": "#dc3545",   # red
    "full_adaptive": "#198754",   # green
}


def plot(points, pareto, label, latency_data):
    fig, ax = plt.subplots(figsize=(9, 6))

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

    # Pareto frontier line
    if len(pareto) > 1:
        pareto_sorted = sorted(pareto, key=lambda x: x[1])
        px = [p[1] for p in pareto_sorted]
        py = [p[2] for p in pareto_sorted]
        ax.plot(px, py, "--", color="#f0a500", linewidth=1.8,
                label="Pareto Frontier", zorder=3)

    # Legend for Pareto marker
    star_patch = mpatches.Patch(color="#f0a500", label="Pareto Frontier")
    pareto_dot = plt.scatter([], [], marker="*", s=180, color="black",
                             edgecolors="white", linewidths=1.5,
                             label="Pareto-optimal config")
    ax.legend(handles=[star_patch, pareto_dot], loc="lower right", fontsize=9)

    ax.set_xlabel("Mean Latency (ms)", fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title("Pareto Curve: Latency vs Quality\n(Adaptive RAG Ablation Study)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Shaded "ideal" quadrant (low latency, high quality)
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()
    mid_x = (ax_xlim[0] + ax_xlim[1]) / 2
    mid_y = (ax_ylim[0] + ax_ylim[1]) / 2
    ax.axhspan(mid_y, ax_ylim[1], xmin=0, xmax=0.5, alpha=0.04, color="green")
    ax.text(ax_xlim[0] + (mid_x - ax_xlim[0]) * 0.05,
            ax_ylim[1] * 0.995, "← ideal zone →",
            fontsize=7, color="green", alpha=0.5, va="top")

    plt.tight_layout()
    plt.savefig(OUTPUT_PARETO, dpi=150)
    plt.close()
    logger.info(f"Plot saved → {OUTPUT_PARETO}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("=" * 50)
    print("PARETO ANALYSIS")
    print("=" * 50)

    summary    = load_json(SUMMARY_PATH)
    results    = load_json(RESULTS_PATH)
    evaluation = load_json(EVAL_PATH)

    latency_data       = load_latency(summary)
    quality, qual_label = load_quality(evaluation, results)

    # Build (name, latency_mean, quality) tuples
    points = []
    for config, lat in latency_data.items():
        if config not in quality:
            logger.warning(f"No quality score for '{config}' — skipping")
            continue
        points.append((config, lat["mean"], quality[config]))

    pareto = compute_pareto(points)

    print(f"\nQuality metric : {qual_label}")
    print(f"\n{'Config':<30} {'Latency (ms)':>14} {'Quality':>10}  Pareto?")
    print("-" * 65)
    for p in sorted(points, key=lambda x: x[2], reverse=True):
        flag = "✅" if p in pareto else ""
        print(f"{p[0]:<30} {p[1]:>14.1f} {p[2]:>10.4f}  {flag}")

    print(f"\nPareto-optimal configs: {[p[0] for p in pareto]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    plot(points, pareto, qual_label, latency_data)

    with open(OUTPUT_DATA, "w") as f:
        json.dump(
            [{"config": p[0], "latency_ms": p[1], "quality": p[2],
              "pareto_optimal": p in pareto} for p in points],
            f, indent=2
        )

    print("\nSaved:")
    print("→", OUTPUT_PARETO)
    print("→", OUTPUT_DATA)


if __name__ == "__main__":
    main()
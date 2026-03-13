"""
pareto_curve.py — Latency vs Quality Pareto Curve
Owner: Roshan K C

Reads ablation_summary.json (latency) and ablation_results.json (quality)
and produces a publication-ready Pareto curve comparing all four
ablation configurations.

Inputs:
    results/ablation_summary.json   ← latency stats from ablation_runner.py
    results/ablation_results.json   ← per-query answers from ablation_runner.py
    results/evaluation_results.json ← RAGAS scores from evaluation.py (optional)

Outputs:
    results/pareto_curve.png        ← main Pareto scatter plot
    results/pareto_curve_stages.png ← stacked bar of per-stage latency
    results/pareto_data.json        ← extracted plot points (for auditing)

Quality metric priority (uses whichever is available):
    1. RAGAS answer_relevance     (from evaluation_results.json)
    2. RAGAS faithfulness         (from evaluation_results.json)
    3. answer_success_rate        (fraction of non-error answers in results)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ParetoCurve")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR      = Path("results")
SUMMARY_PATH     = RESULTS_DIR / "ablation_summary.json"
RESULTS_PATH     = RESULTS_DIR / "ablation_results.json"
EVAL_PATH        = RESULTS_DIR / "evaluation_results.json"
OUTPUT_PARETO    = RESULTS_DIR / "pareto_curve.png"
OUTPUT_STAGES    = RESULTS_DIR / "pareto_curve_stages.png"
OUTPUT_DATA      = RESULTS_DIR / "pareto_data.json"

# ---------------------------------------------------------------------------
# Visual config — consistent across both plots
# ---------------------------------------------------------------------------
CONFIG_STYLES: Dict[str, Dict[str, Any]] = {
    "naive":          {"color": "#e74c3c", "marker": "o", "label": "Naive RAG"},
    "router_only":    {"color": "#f39c12", "marker": "s", "label": "Router Only"},
    "reranker_only":  {"color": "#3498db", "marker": "^", "label": "Reranker Only"},
    "full_adaptive":  {"color": "#2ecc71", "marker": "D", "label": "Full Adaptive"},
}

KNOWN_STAGES = ["routing", "retrieval", "reranking", "generation"]

STAGE_COLORS: Dict[str, str] = {
    "routing":    "#9b59b6",
    "retrieval":  "#3498db",
    "reranking":  "#f39c12",
    "generation": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(path: Path, label: str) -> Optional[Dict]:
    if not path.exists():
        logger.warning("%s not found at '%s' — skipping.", label, path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latency(summary: Dict) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-config latency stats from ablation_summary.json.

    Returns:
        {config_name: {"total_mean": float, "total_std": float,
                       "stages": {stage: mean_ms}}}
    """
    out: Dict[str, Dict[str, Any]] = {}
    for key, data in summary.items():
        if key == "generated_at":
            continue
        if not isinstance(data, dict):
            continue
        total  = data.get("total_ms", {})
        stages = data.get("stages", {})
        out[key] = {
            "total_mean": total.get("mean_ms", 0.0),
            "total_std":  total.get("std_ms",  0.0),
            "stages": {
                s: stages.get(s, {}).get("mean_ms", 0.0)
                for s in KNOWN_STAGES
            },
        }
    return out


def load_quality_from_eval(eval_data: Dict) -> Dict[str, float]:
    """
    Extract quality scores from evaluation_results.json (RAGAS output).
    Tries answer_relevance first, then faithfulness, then context_recall.

    Returns:
        {config_name: score_0_to_1}
    """
    quality: Dict[str, float] = {}
    for config, metrics in eval_data.items():
        if isinstance(metrics, dict):
            score = (
                metrics.get("answer_relevance")
                or metrics.get("faithfulness")
                or metrics.get("context_recall")
                or 0.0
            )
            quality[config] = round(float(score), 4)
    return quality


def load_quality_from_results(results_data: Dict) -> Dict[str, float]:
    """
    Compute answer_success_rate from ablation_results.json as a fallback
    when RAGAS scores are not yet available.

    success = answer does not start with "[ERROR]"

    Returns:
        {config_name: fraction_0_to_1}
    """
    quality: Dict[str, float] = {}
    for config, queries in results_data.items():
        if not isinstance(queries, list) or len(queries) == 0:
            quality[config] = 0.0
            continue
        n_ok = sum(
            1 for q in queries
            if isinstance(q.get("answer"), str)
            and not q["answer"].startswith("[ERROR]")
        )
        quality[config] = round(n_ok / len(queries), 4)
    return quality


# ---------------------------------------------------------------------------
# Pareto-front computation
# ---------------------------------------------------------------------------

def compute_pareto_front(
    points: List[Tuple[str, float, float]]
) -> List[Tuple[str, float, float]]:
    """
    Identify Pareto-optimal configurations (lower latency AND higher quality,
    neither dimension dominated by another point).

    Args:
        points: List of (config_name, latency_ms, quality_score)

    Returns:
        Pareto-optimal subset, sorted by latency ascending.
    """
    pareto: List[Tuple[str, float, float]] = []
    for candidate in points:
        c_name, c_lat, c_qual = candidate
        dominated = any(
            (
                o_lat <= c_lat and o_qual >= c_qual
                and (o_lat < c_lat or o_qual > c_qual)
            )
            for o_name, o_lat, o_qual in points
            if o_name != c_name
        )
        if not dominated:
            pareto.append(candidate)
    return sorted(pareto, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Plot 1 — Pareto scatter
# ---------------------------------------------------------------------------

def plot_pareto_scatter(
    points:        List[Tuple[str, float, float]],
    latency_data:  Dict[str, Dict[str, Any]],
    quality_label: str,
    output_path:   Path,
) -> None:
    """
    Publication-ready Pareto scatter with:
        - ±1 std error bars on latency axis
        - Pareto-front dashed line
        - Per-config styled markers (larger + black edge if on front)
        - Naive baseline reference lines
        - "← faster / better ↑" orientation hint
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    pareto_front  = compute_pareto_front(points)
    pareto_names  = {p[0] for p in pareto_front}

    # ── Error bars + scatter ──────────────────────────────────────────
    for config, latency, quality in points:
        style    = CONFIG_STYLES.get(config, {"color": "#7f8c8d", "marker": "o", "label": config})
        std      = latency_data.get(config, {}).get("total_std", 0.0)
        on_front = config in pareto_names

        ax.errorbar(
            latency, quality,
            xerr=std,
            fmt=style["marker"],
            color=style["color"],
            markersize=14 if on_front else 10,
            linewidth=1.5,
            capsize=4,
            zorder=3,
            markeredgecolor="black" if on_front else style["color"],
            markeredgewidth=1.5   if on_front else 0,
        )

        # Annotation — small nudge so labels don't overlap the marker
        x_off = latency * 0.012
        y_off = 0.008
        ax.annotate(
            style["label"],
            xy=(latency, quality),
            xytext=(latency + x_off, quality + y_off),
            fontsize=10,
            fontweight="bold" if on_front else "normal",
            color=style["color"],
        )

    # ── Pareto-front dashed line ───────────────────────────────────────
    if len(pareto_front) > 1:
        px = [p[1] for p in pareto_front]
        py = [p[2] for p in pareto_front]
        ax.plot(
            px, py,
            linestyle="--", color="#555555",
            linewidth=1.2, alpha=0.6, zorder=2,
            label="Pareto front",
        )

    # ── Naive baseline reference lines ────────────────────────────────
    naive = next((p for p in points if p[0] == "naive"), None)
    if naive:
        ax.axvline(naive[1], color="#e74c3c", linestyle=":", alpha=0.3, linewidth=1)
        ax.axhline(naive[2], color="#e74c3c", linestyle=":", alpha=0.3, linewidth=1)

    # ── Orientation hint ──────────────────────────────────────────────
    ax.annotate(
        "← faster\nbetter ↑",
        xy=(0.02, 0.97), xycoords="axes fraction",
        fontsize=8, color="#999999",
        verticalalignment="top",
    )

    # ── Legend ────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=v["color"], label=v["label"])
        for v in CONFIG_STYLES.values()
    ]
    if len(pareto_front) > 1:
        handles.append(
            plt.Line2D([0], [0], linestyle="--", color="#555555", label="Pareto front")
        )
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    ax.set_xlabel("Mean Total Latency (ms)", fontsize=12)
    ax.set_ylabel(quality_label, fontsize=12)
    ax.set_title(
        "Adaptive RAG — Latency vs Quality Pareto Curve",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0.0, min(p[2] for p in points) - 0.05))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pareto scatter saved → %s", output_path)


# ---------------------------------------------------------------------------
# Plot 2 — Per-stage latency stacked bar
# ---------------------------------------------------------------------------

def plot_stage_breakdown(
    latency_data: Dict[str, Dict[str, Any]],
    output_path:  Path,
) -> None:
    """
    Stacked horizontal bar chart showing per-stage latency breakdown.
    Helps the paper explain exactly where latency comes from in each config.
    """
    configs = [c for c in CONFIG_STYLES if c in latency_data]
    if not configs:
        logger.warning("No matching configs in latency_data — skipping stage breakdown.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))

    y_pos   = np.arange(len(configs))
    bottoms = np.zeros(len(configs))

    for stage in KNOWN_STAGES:
        values = np.array([
            latency_data[c]["stages"].get(stage, 0.0)
            for c in configs
        ])
        bars = ax.barh(
            y_pos, values, left=bottoms,
            color=STAGE_COLORS[stage],
            label=stage.capitalize(),
            height=0.5,
        )
        # In-bar value labels — only when bar is wide enough to be readable
        for i, (val, bot) in enumerate(zip(values, bottoms)):
            if val > 8:
                ax.text(
                    bot + val / 2, i,
                    f"{val:.0f}",
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )
        bottoms += values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [CONFIG_STYLES.get(c, {}).get("label", c) for c in configs],
        fontsize=10,
    )
    ax.set_xlabel("Mean Latency (ms)", fontsize=11)
    ax.set_title(
        "Per-Stage Latency Breakdown by Configuration",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(
        loc="lower right", fontsize=9,
        title="Pipeline stage", title_fontsize=9,
    )
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Stage breakdown saved → %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Pareto Curve Generator")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────────
    summary_raw = _load_json(SUMMARY_PATH, "ablation_summary.json")
    results_raw = _load_json(RESULTS_PATH, "ablation_results.json")
    eval_raw    = _load_json(EVAL_PATH,    "evaluation_results.json")

    if summary_raw is None:
        raise FileNotFoundError(
            f"'{SUMMARY_PATH}' not found. Run ablation_runner.py first."
        )
    if results_raw is None:
        raise FileNotFoundError(
            f"'{RESULTS_PATH}' not found. Run ablation_runner.py first."
        )

    # ── Extract latency ───────────────────────────────────────────────
    latency_data = load_latency(summary_raw)
    logger.info("Latency loaded for: %s", list(latency_data.keys()))

    # ── Extract quality — RAGAS preferred, success_rate fallback ──────
    if eval_raw:
        quality       = load_quality_from_eval(eval_raw)
        quality_label = "Answer Relevance (RAGAS)"
        logger.info("Using RAGAS quality scores.")
    else:
        quality       = load_quality_from_results(results_raw)
        quality_label = "Answer Success Rate"
        logger.warning(
            "evaluation_results.json not found — using answer_success_rate "
            "as quality proxy. Run evaluation.py for RAGAS metrics."
        )

    # ── Build plot points ─────────────────────────────────────────────
    points: List[Tuple[str, float, float]] = []
    for config, lat in latency_data.items():
        q = quality.get(config)
        if q is None:
            logger.warning("No quality score for '%s' — skipping.", config)
            continue
        points.append((config, lat["total_mean"], q))

    if not points:
        logger.error("No valid plot points found. Cannot generate curve.")
        return

    # ── Console summary table ─────────────────────────────────────────
    pareto = compute_pareto_front(points)
    pareto_names = {p[0] for p in pareto}

    print(f"\n{'Config':<20} {'Latency (ms)':>14} {'±Std':>8} {'Quality':>10}  {'Pareto':>7}")
    print("─" * 64)
    for config, latency, q_score in sorted(points, key=lambda x: x[1]):
        std  = latency_data[config]["total_std"]
        flag = "  ✅" if config in pareto_names else ""
        print(f"  {config:<18} {latency:>12.1f} {std:>8.1f} {q_score:>10.4f}{flag}")
    print()
    print(f"Quality metric : {quality_label}")
    print(f"Pareto-optimal : {[p[0] for p in pareto]}")
    print()

    # ── Persist extracted data for auditing / CI ──────────────────────
    pareto_data = {
        "quality_metric": quality_label,
        "configs": [
            {
                "name":            name,
                "latency_mean_ms": round(lat, 2),
                "latency_std_ms":  round(latency_data[name]["total_std"], 2),
                "quality":         round(q, 4),
                "pareto_optimal":  name in pareto_names,
                "stages_ms":       latency_data[name]["stages"],
            }
            for name, lat, q in sorted(points, key=lambda x: x[1])
        ],
    }
    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(pareto_data, f, indent=2)
    logger.info("Plot data saved → %s", OUTPUT_DATA)

    # ── Generate plots ────────────────────────────────────────────────
    plot_pareto_scatter(points, latency_data, quality_label, OUTPUT_PARETO)
    plot_stage_breakdown(latency_data, OUTPUT_STAGES)

    print(f"  Pareto curve    → {OUTPUT_PARETO}")
    print(f"  Stage breakdown → {OUTPUT_STAGES}")
    print(f"  Plot data       → {OUTPUT_DATA}")
    print("\n✅  Done.")


if __name__ == "__main__":
    main()
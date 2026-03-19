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
OUTPUT_DATA   = RESULTS_DIR / "pareto_data.json"

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


# ---------------------------------------------------------------------------
# Latency loader — defensive against schema variants
# ---------------------------------------------------------------------------

def load_latency(summary: dict) -> dict[str, dict]:
    """
    Extract mean ± std total latency per config from ablation_summary.json.

    Handles both:
        summary[config]["total_ms"]["mean_ms"]   ← new schema (aggregate_logs)
        summary[config]["mean_ms"]               ← flat fallback schema

    Returns:
        { config_name: {"mean": float, "std": float} }
    """
    data: dict[str, dict] = {}

    for key, val in summary.items():
        if key in ("generated_at", "device"):
            continue
        if not isinstance(val, dict):
            continue

        # New schema: total_ms sub-dict
        if "total_ms" in val and isinstance(val["total_ms"], dict):
            mean = val["total_ms"].get("mean_ms", 0.0)
            std  = val["total_ms"].get("std_ms",  0.0)

        # Flat fallback schema
        elif "mean_ms" in val:
            mean = val.get("mean_ms", 0.0)
            std  = val.get("std_ms",  0.0)

        else:
            logger.warning(
                "Config '%s' has no recognisable latency schema — skipping.", key
            )
            continue

        data[key] = {"mean": float(mean), "std": float(std)}
        logger.info("Latency  %-18s  mean=%.1f ms  std=%.1f ms", key, mean, std)

    return data


# ---------------------------------------------------------------------------
# Quality loader — evaluation_results.json is the primary source
# ---------------------------------------------------------------------------

def load_quality(
    eval_data:    dict | None,
    results_data: dict | None,
) -> tuple[dict[str, float], str]:
    """
    Build a quality score per config.

    Priority:
        1. evaluation_results.json  → best available metric from _METRIC_PRIORITY
        2. ablation_results.json    → answer success rate (fallback only)

    Returns:
        (quality_dict, metric_label)
    """
    # ── Primary: evaluation results ──────────────────────────────────
    if eval_data:
        # Find the highest-priority metric present in ALL configs
        configs = list(eval_data.keys())
        for metric_key, metric_label in _METRIC_PRIORITY:
            if all(metric_key in eval_data.get(c, {}) for c in configs):
                quality = {
                    c: float(eval_data[c][metric_key])
                    for c in configs
                }
                logger.info(
                    "Quality source: evaluation_results.json  metric='%s'",
                    metric_key,
                )
                for c, v in quality.items():
                    logger.info("Quality  %-18s  %s=%.4f", c, metric_key, v)
                return quality, metric_label

        # No metric present across ALL configs — take whatever is available
        # per config using priority order
        logger.warning(
            "No single metric present in all configs; using per-config best."
        )
        quality = {}
        for c, metrics in eval_data.items():
            for metric_key, metric_label in _METRIC_PRIORITY:
                if metric_key in metrics:
                    quality[c] = float(metrics[metric_key])
                    break
            else:
                quality[c] = 0.0
        # label = last matched metric_label (imperfect but workable)
        return quality, metric_label  # type: ignore[possibly-undefined]

    # ── Fallback: raw ablation results (success rate) ─────────────────
    if results_data:
        logger.warning(
            "evaluation_results.json missing — falling back to "
            "answer success rate from ablation_results.json"
        )
        quality = {}
        for config, queries in results_data.items():
            if not queries:
                quality[config] = 0.0
                continue
            ok = sum(
                1 for q in queries
                if not str(q.get("answer", "")).startswith("[ERROR]")
                and str(q.get("answer", "")) not in ("[NO ANSWER]", "")
            )
            quality[config] = ok / len(queries)
            logger.info("Quality  %-18s  success_rate=%.4f", config, quality[config])
        return quality, "Answer Success Rate"

    raise RuntimeError(
        "No quality data available. "
        "Run ablation_runner.py then evaluation.py first."
    )


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------

def compute_pareto(
    points: list[tuple[str, float, float]]
) -> list[tuple[str, float, float]]:
    """
    Identify Pareto-optimal points.

    A point (name, latency, quality) is Pareto-optimal if no other point
    has BOTH lower-or-equal latency AND higher-or-equal quality, with at
    least one strict inequality.

    Args:
        points: List of (config_name, mean_latency_ms, quality_score)

    Returns:
        Subset of points that lie on the Pareto frontier.
    """
    pareto = []
    for candidate in points:
        _, lat_c, qual_c = candidate
        dominated = False
        for other in points:
            if other is candidate:
                continue
            _, lat_o, qual_o = other
            # other dominates candidate if it's at least as good on both
            # axes and strictly better on at least one
            if lat_o <= lat_c and qual_o >= qual_c:
                if lat_o < lat_c or qual_o > qual_c:
                    dominated = True
                    break
        if not dominated:
            pareto.append(candidate)
    return pareto


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(
    points: list[tuple[str, float, float]],
    pareto: list[tuple[str, float, float]],
    latency_std: dict[str, float],
    quality_label: str,
) -> None:
    """
    Render the Pareto curve with error bars and save to OUTPUT_PARETO.

    Args:
        points:        All (name, lat_mean, quality) triples.
        pareto:        Pareto-optimal subset.
        latency_std:   { config_name: latency_std_ms }
        quality_label: Y-axis label string.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    pareto_names = {p[0] for p in pareto}

    for name, lat, qual in points:
        style  = _CONFIG_STYLES.get(name, _DEFAULT_STYLE)
        color  = style["color"]
        marker = style["marker"]
        std    = latency_std.get(name, 0.0)
        zorder = 4 if name in pareto_names else 3

        ax.errorbar(
            lat, qual,
            xerr=std,
            fmt=marker,
            color=color,
            markersize=11,
            capsize=5,
            linewidth=1.5,
            zorder=zorder,
            label=name,
        )

        # Config label — offset slightly to avoid overlap
        ax.annotate(
            name,
            xy=(lat, qual),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )

    # ── Pareto frontier line ──────────────────────────────────────────
    if len(pareto) >= 2:
        pareto_sorted = sorted(pareto, key=lambda x: x[1])
        px = [p[1] for p in pareto_sorted]
        py = [p[2] for p in pareto_sorted]

        ax.plot(
            px, py,
            linestyle="--",
            color="#2c3e50",
            linewidth=1.5,
            alpha=0.7,
            zorder=2,
            label="Pareto frontier",
        )

        # Shade under the frontier to emphasise the optimal region
        ax.fill_betweenx(
            py, px,
            ax.get_xlim()[0] if ax.get_xlim()[0] < min(px) else min(px) - 50,
            alpha=0.06,
            color="#2c3e50",
        )

    # ── Pareto-optimal badges ─────────────────────────────────────────
    for name, lat, qual in pareto:
        ax.annotate(
            "★",
            xy=(lat, qual),
            xytext=(-14, 4),
            textcoords="offset points",
            fontsize=11,
            color="#f39c12",
        )

    ax.set_xlabel("Mean Total Latency (ms)", fontsize=11)
    ax.set_ylabel(quality_label, fontsize=11)
    ax.set_title("Pareto Curve — Latency vs Quality\n(★ = Pareto-optimal)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    fig.savefig(OUTPUT_PARETO, dpi=150)
    plt.close(fig)
    logger.info("Pareto plot saved → %s", OUTPUT_PARETO)


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

    if summary is None:
        raise RuntimeError(
            f"ablation_summary.json not found at {SUMMARY_PATH}. "
            "Run ablation_runner.py first."
        )

    latency = load_latency(summary)
    quality, quality_label = load_quality(evaluation, results)

    # Build combined points — only configs present in BOTH dicts
    points: list[tuple[str, float, float]] = []
    for config in latency:
        if config not in quality:
            logger.warning("Config '%s' in latency but not in quality — skipping.", config)
            continue
        points.append((config, latency[config]["mean"], quality[config]))

    if not points:
        raise RuntimeError("No configs with both latency and quality data.")

    pareto = compute_pareto(points)

    # ── Console summary ───────────────────────────────────────────────
    print(f"\n{'Config':<20}  {'Latency (ms)':>14}  {'±Std':>8}  {quality_label:>22}  Pareto")
    print("─" * 80)
    for name, lat, qual in sorted(points, key=lambda x: x[1]):
        std  = latency.get(name, {}).get("std", 0.0)
        flag = "  ★" if (name, lat, qual) in pareto else ""
        print(f"{name:<20}  {lat:>14.1f}  {std:>8.1f}  {qual:>22.4f}{flag}")
    print()

    # ── Save artefacts ────────────────────────────────────────────────
    latency_std = {k: v["std"] for k, v in latency.items()}
    plot(points, pareto, latency_std, quality_label)

    pareto_data = {
        "quality_metric": quality_label,
        "points": [
            {
                "config":        name,
                "latency_mean":  lat,
                "latency_std":   latency.get(name, {}).get("std", 0.0),
                "quality":       qual,
                "pareto_optimal": (name, lat, qual) in pareto,
            }
            for name, lat, qual in sorted(points, key=lambda x: x[1])
        ],
    }

    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(pareto_data, f, indent=2)
    logger.info("Pareto data saved  → %s", OUTPUT_DATA)

    print("Saved:")
    print("→", OUTPUT_PARETO)
    print("→", OUTPUT_DATA)


if __name__ == "__main__":
    main()
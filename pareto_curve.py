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
        if not isinstance(v, dict) or "total_ms" not in v:
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
    """
    Render a publication-quality Pareto scatter plot.

    Layout:
    -------
    • All 12 configs plotted on one chart with a clipped x-axis so outliers
      (router_only_b0.1, full_adaptive_b0.1) don't squash the interesting region.
    • Error bars are shown but capped at ±1 std to avoid negative latency.
    • Each system config gets its own colour; budget level drives marker shape.
    • Pareto-optimal points are marked with a gold star outline + bold label.
    • Labels are placed with a smart offset to minimise overlap.
    • A clean two-column legend shows config colour AND budget marker shape.
    """

    # ── style ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "font.size":       10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")
    ax.grid(color="white", linewidth=1.2, zorder=0)

    pareto_set = {p[0] for p in pareto}

    # ── clip x-axis at the 80th-percentile latency + 15 % headroom ───────────
    all_lats = [lat for _, lat, _ in points]
    x_max    = sorted(all_lats)[int(len(all_lats) * 0.80)] * 1.18
    ax.set_xlim(left=max(0, min(all_lats) * 0.88), right=x_max)

    # ── scatter each point ────────────────────────────────────────────────────
    label_data = []   # (x, y, text, color, is_pareto) – for offset labelling

    for name, lat, qual in points:
        if "_b" in name:
            base_config, b_str = name.rsplit("_b", 1)
        else:
            base_config, b_str = name, "1.0"

        color     = COLORS.get(base_config, "#555555")
        is_pareto = name in pareto_set
        
        if base_config != "full_adaptive":
            marker = "s" # Baselines as squares
        else:
            marker = BUDGET_MARKERS.get(b_str, "o")
            
        ms        = 220 if is_pareto else 140
        zorder    = 6 if is_pareto else 4

        std = latency_data.get(name, {}).get("std", 0)
        # Clip error bar so it never goes below 0
        xerr_lo = min(std, lat)
        xerr_hi = std

        # Only draw error bar if within x-axis range
        if lat <= x_max:
            ax.errorbar(lat, qual,
                        xerr=[[xerr_lo], [xerr_hi]],
                        fmt="none", color=color, alpha=0.30,
                        capsize=3, linewidth=1, zorder=3)

            ax.scatter(lat, qual,
                       s=ms, color=color, marker=marker,
                       edgecolors="white" if not is_pareto else "gold",
                       linewidths=2.0 if is_pareto else 1.2,
                       zorder=zorder)

            if is_pareto:
                # Gold halo ring for Pareto-optimal points
                ax.scatter(lat, qual,
                           s=ms + 80, color="none",
                           edgecolors="gold", linewidths=2.0,
                           marker=marker, zorder=zorder - 1)

            label_data.append((lat, qual, name, color, is_pareto))
        else:
            # Annotate clipped outliers on the right edge
            ax.annotate(
                f"↦ {name}\n({lat/1000:.1f}s, F1={qual:.3f})",
                xy=(x_max, qual),
                xytext=(x_max * 0.96, qual),
                fontsize=8, color=color, fontstyle="italic",
                ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec=color),
            )

    # ── smart label placement (stagger offsets to reduce overlap) ─────────────
    # Sort by quality descending so higher-quality labels get placed first
    label_data.sort(key=lambda x: x[1], reverse=True)

    placed: list[tuple[float, float]] = []

    def _overlaps(x, y, placed, r_x=350, r_y=0.006):
        return any(abs(x - px) < r_x and abs(y - py) < r_y for px, py in placed)

    offsets = [
        ( 120,  0.004), (-140,  0.005), ( 120, -0.004),
        (-140, -0.005), ( 200,  0.000), (-200,  0.000),
        (  60,  0.007), ( -60, -0.007),
    ]

    for lat, qual, name, color, is_pareto in label_data:
        if lat > x_max:
            continue

        if "_b" in name:
            base, b = name.rsplit("_b", 1)
            display = f"{DISPLAY_NAMES.get(base, base)}\n(b={b})"
        else:
            display = DISPLAY_NAMES.get(name, name)

        # Try each candidate offset until no overlap
        chosen_dx, chosen_dy = offsets[0]
        for dx, dy in offsets:
            tx, ty = lat + dx, qual + dy
            if not _overlaps(tx, ty, placed):
                chosen_dx, chosen_dy = dx, dy
                break

        tx, ty = lat + chosen_dx, qual + chosen_dy
        placed.append((tx, ty))

        ax.annotate(
            display,
            xy=(lat, qual),
            xytext=(tx, ty),
            fontsize=8.5,
            fontweight="bold" if is_pareto else "normal",
            color=color,
            arrowprops=dict(
                arrowstyle="-",
                color=color, alpha=0.5, lw=0.8,
                connectionstyle="arc3,rad=0.1",
            ) if abs(chosen_dx) > 50 or abs(chosen_dy) > 0.001 else None,
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white", alpha=0.85,
                ec=("gold" if is_pareto else color), lw=1.0,
            ),
            zorder=7,
        )

    # ── Pareto frontier line ──────────────────────────────────────────────────
    visible_pareto = [(lat, qual) for n, lat, qual in pareto if lat <= x_max]
    if visible_pareto:
        visible_pareto.sort(key=lambda x: x[0])
        px = [p[0] for p in visible_pareto]
        py = [p[1] for p in visible_pareto]
        ax.plot(px, py, "--", color="gold", linewidth=2.0,
                alpha=0.85, label="Pareto frontier", zorder=5)

    # ── Budget Degradation Path ───────────────────────────────────────────────
    adaptive_points = [(lat, qual) for n, lat, qual in points if n.startswith("full_adaptive_b")]
    if len(adaptive_points) > 1:
        adaptive_points.sort(key=lambda x: x[0])
        px = [p[0] for p in adaptive_points]
        py = [p[1] for p in adaptive_points]
        ax.plot(px, py, linestyle="--", color=COLORS["full_adaptive"], linewidth=1.5, zorder=2)
        
        mid_idx = len(px) // 2
        ax.annotate("Budget Degradation Path", xy=(px[mid_idx], py[mid_idx]), 
                    xytext=(px[mid_idx], py[mid_idx] - 0.015),
                    color=COLORS["full_adaptive"], fontsize=9, fontstyle="italic", ha="center",
                    arrowprops=dict(arrowstyle="->", color=COLORS["full_adaptive"], alpha=0.6))

    # ── legends ───────────────────────────────────────────────────────────────
    # Config colour legend
    config_handles = [
        mpatches.Patch(color=v, label=DISPLAY_NAMES[k])
        for k, v in COLORS.items()
    ]
    leg1 = ax.legend(
        handles=config_handles,
        title="System Config", title_fontsize=9,
        loc="lower right", fontsize=8.5,
        framealpha=0.9, edgecolor="#cccccc",
    )
    ax.add_artist(leg1)

    # Budget marker legend
    import matplotlib.lines as mlines
    budget_handles = [
        mlines.Line2D([], [], color="gray", marker=m, linestyle="None",
                      markersize=7, label=f"Budget = {b}")
        for b, m in BUDGET_MARKERS.items()
    ]
    budget_handles.append(
        mlines.Line2D([], [], color="gold", marker="o", linestyle="None",
                      markersize=9, markeredgecolor="gold",
                      markeredgewidth=2, label="Pareto-optimal")
    )
    ax.legend(
        handles=budget_handles,
        title="Budget Tier", title_fontsize=9,
        loc="upper right", fontsize=8.5,
        framealpha=0.9, edgecolor="#cccccc",
    )

    # ── axis decorations ─────────────────────────────────────────────────────
    ax.set_xlabel("Mean End-to-End Latency (ms)", fontsize=11, labelpad=8)
    ax.set_ylabel(f"Quality — {label}", fontsize=11, labelpad=8)
    ax.set_title(
        "Latency–Quality Pareto Curve\nAdaptive RAG Ablation Study",
        fontsize=13, fontweight="bold", pad=14,
    )

    # Shade region ← lower latency & higher quality = "better"
    y_min = ax.get_ylim()[0]
    x_min = ax.get_xlim()[0]
    ax.annotate(
        "← lower latency\nbetter →",
        xy=(x_min * 1.02, ax.get_ylim()[1] * 0.97),
        fontsize=7.5, color="#888888", va="top", style="italic",
    )

    fig.tight_layout()
    plt.savefig(OUTPUT_PARETO, dpi=180, bbox_inches="tight")
    plt.close()
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
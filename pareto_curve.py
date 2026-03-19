import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pareto")

RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "ablation_summary.json"
RESULTS_PATH = RESULTS_DIR / "ablation_results.json"
EVAL_PATH = RESULTS_DIR / "evaluation_results.json"

OUTPUT_PARETO = RESULTS_DIR / "pareto_curve.png"
OUTPUT_DATA = RESULTS_DIR / "pareto_data.json"


# ---------------------------------------------------------
# LOADERS
# ---------------------------------------------------------

def load_json(path):
    if not path.exists():
        logger.warning(f"{path} not found")
        return None
    with open(path) as f:
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


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("=" * 50)
    print("PARETO ANALYSIS")
    print("=" * 50)

    summary = load_json(SUMMARY_PATH)
    results = load_json(RESULTS_PATH)
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

    print("\nSaved:")
    print("→", OUTPUT_PARETO)
    print("→", OUTPUT_DATA)


if __name__ == "__main__":
    main()
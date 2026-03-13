"""
ablation_runner.py — Ablation Study Experiment Runner
Owner: Roshan K C

Runs all four ablation configurations on the QA dataset and produces
a single structured results file consumed by evaluation.py and
pareto_curve.py.

Configurations:
    1. naive            Router=❌  Reranker=❌
    2. router_only      Router=✅  Reranker=❌
    3. reranker_only    Router=❌  Reranker=✅
    4. full_adaptive    Router=✅  Reranker=✅

Design decisions:
    - Uses AdaptiveRAGPipeline(use_router, use_reranker) — one class,
      four configs — no duplicated retrieval logic here.
    - Generates a real LLM answer for every query so evaluation.py can
      run RAGAS faithfulness / relevance metrics on real answers.
    - Per-query errors are caught and logged — one failure never aborts
      a 150-query run.
    - Progress bar via tqdm so long runs are observable.
    - Atomic JSON writes so results are never partially written.
    - Latency aggregation (mean ± std per stage) appended to the
      results file so pareto_curve.py has everything it needs.

Outputs:
    results/ablation_results.json      ← per-query results, all configs
    results/ablation_summary.json      ← latency aggregation per config
    latency_logs/                      ← per-query latency logs
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from adaptive_pipeline import AdaptiveRAGPipeline, _atomic_json_write
from latency_tracker import aggregate_logs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AblationRunner")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
QA_PATH          = "data/qa_pairs.json"
RESULTS_DIR      = "results"
RESULT_PATH      = os.path.join(RESULTS_DIR, "ablation_results.json")
SUMMARY_PATH     = os.path.join(RESULTS_DIR, "ablation_summary.json")
LATENCY_LOG_DIR  = "latency_logs"

# Four ablation configurations: (use_router, use_reranker, label)
ABLATION_CONFIGS: List[tuple] = [
    (False, False, "naive"),
    (True,  False, "router_only"),
    (False, True,  "reranker_only"),
    (True,  True,  "full_adaptive"),
]


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------
class AblationRunner:
    """
    Orchestrates all four ablation runs over the QA dataset.

    Components are NOT loaded here — each AdaptiveRAGPipeline instance
    manages its own components.  This keeps memory predictable on 8 GB
    VRAM: we load one pipeline config at a time, run it fully, then
    replace it with the next config.
    """

    def __init__(
        self,
        qa_path:     str = QA_PATH,
        results_dir: str = RESULTS_DIR,
        max_queries: Optional[int] = None,
    ) -> None:
        """
        Args:
            qa_path:     Path to QA pairs JSON.
            results_dir: Directory for result files.
            max_queries: Cap number of queries (useful for quick smoke tests).
                         None = use all QA pairs.
        """
        self.qa_path     = qa_path
        self.results_dir = results_dir
        self.max_queries = max_queries

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(LATENCY_LOG_DIR, exist_ok=True)

        logger.info(
            "AblationRunner ready | qa_path='%s' max_queries=%s",
            qa_path, max_queries,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_qa_pairs(self) -> List[Dict[str, Any]]:
        """
        Load QA pairs from disk.

        Expected schema per item:
            {
              "question":             str,
              "ground_truth_answer":  str,
              "query_type":           str   (optional — used for router eval)
            }

        Returns:
            List of QA dicts, capped at max_queries if set.

        Raises:
            FileNotFoundError: If QA file does not exist.
            ValueError:        If file is empty or malformed.
        """
        if not os.path.exists(self.qa_path):
            raise FileNotFoundError(
                f"QA dataset not found at '{self.qa_path}'. "
                f"Run load_qasper.py first."
            )

        with open(self.qa_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"QA file '{self.qa_path}' is empty or not a list.")

        # Validate required keys on first item
        required = {"question", "ground_truth_answer"}
        missing  = required - set(data[0].keys())
        if missing:
            raise ValueError(
                f"QA items missing required keys: {missing}. "
                f"Check load_qasper.py output schema."
            )

        if self.max_queries:
            data = data[: self.max_queries]

        logger.info("Loaded %d QA pairs from '%s'.", len(data), self.qa_path)
        return data

    # ------------------------------------------------------------------
    # Single-config runner
    # ------------------------------------------------------------------

    def run_config(
        self,
        qa_pairs:    List[Dict[str, Any]],
        use_router:  bool,
        use_reranker: bool,
        config_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Run one ablation configuration over the full QA set.

        Loads AdaptiveRAGPipeline fresh for this config so GPU memory
        from the previous config can be freed before loading the next.

        Args:
            qa_pairs:     List of QA dicts.
            use_router:   Enable router for this config.
            use_reranker: Enable reranker for this config.
            config_name:  Label for logging and output files.

        Returns:
            List of per-query result dicts.
        """
        logger.info(
            "━━━  Config: %-16s  Router=%-5s  Reranker=%s  ━━━",
            config_name, use_router, use_reranker,
        )

        pipeline = AdaptiveRAGPipeline(
            use_router   = use_router,
            use_reranker = use_reranker,
        )

        results: List[Dict[str, Any]] = []
        n = len(qa_pairs)

        for idx, qa in enumerate(
            tqdm(qa_pairs, desc=f"[{config_name}]", unit="q"), start=1
        ):
            query        = qa["question"]
            ground_truth = qa.get("ground_truth_answer", "")
            gold_type    = qa.get("query_type", "unknown")   # for router eval

            logger.debug("  [%d/%d] %s", idx, n, query[:70])

            try:
                out = pipeline.run(
                    query,
                    save_latency=True,
                    save_result=False,
                )

                results.append({
                    # ── identifiers ───────────────────────────────────
                    "query":               query,
                    "config":              config_name,
                    # ── router analysis fields (Samved) ───────────────
                    "gold_query_type":     gold_type,
                    "predicted_query_type": out["query_type"],
                    "router_confidence":   out["confidence"],
                    "retriever_used":      out["retriever"],
                    # ── evaluation fields (Nivi / RAGAS) ─────────────
                    "ground_truth":        ground_truth,
                    "answer":              out["answer"],
                    "retrieved_chunk_ids": [c["chunk_id"] for c in out["chunks"]],
                    "retrieved_texts":     [c["text"]     for c in out["chunks"]],
                    # ── latency (pareto_curve.py) ─────────────────────
                    "latency_ms":          out["latency_log"].get("timings_ms", {}),
                    "total_latency_ms":    out["latency_log"].get("total_ms", 0.0),
                })

            except Exception as exc:
                logger.error("  Query %d/%d failed (%s): %s", idx, n, config_name, exc)
                results.append({
                    "query":               query,
                    "config":              config_name,
                    "gold_query_type":     gold_type,
                    "predicted_query_type": "error",
                    "router_confidence":   0.0,
                    "retriever_used":      "error",
                    "ground_truth":        ground_truth,
                    "answer":              f"[ERROR] {exc}",
                    "retrieved_chunk_ids": [],
                    "retrieved_texts":     [],
                    "latency_ms":          {},
                    "total_latency_ms":    0.0,
                })

        n_ok = sum(1 for r in results if not r["answer"].startswith("[ERROR]"))
        logger.info(
            "Config '%s' done — %d/%d succeeded.", config_name, n_ok, n
        )
        return results

    # ------------------------------------------------------------------
    # Latency summary builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        all_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Build per-config latency aggregation (mean ± std ± min/max).
        Used by pareto_curve.py.
        """
        summary: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        for config_name, results in all_results.items():
            # Convert per-query result dicts into LatencyTracker.to_log() schema
            pseudo_logs = [
                {
                    "timings_ms": r["latency_ms"],
                    "total_ms":   r["total_latency_ms"],
                }
                for r in results
                if r["latency_ms"]   # skip errored queries
            ]
            summary[config_name] = aggregate_logs(pseudo_logs)

        return summary

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Execute all four ablation configurations sequentially and save
        results to disk.

        Returns:
            {
              "ablation_results": Dict[config, List[result]],
              "summary":          Dict[config, latency_stats],
            }
        """
        t_start = time.time()
        logger.info("╔══  Ablation study started  ══╗")

        qa_pairs    = self.load_qa_pairs()
        all_results: Dict[str, List[Dict[str, Any]]] = {}

        for use_router, use_reranker, config_name in ABLATION_CONFIGS:
            all_results[config_name] = self.run_config(
                qa_pairs, use_router, use_reranker, config_name
            )

        # ── Persist per-query results ─────────────────────────────────
        _atomic_json_write(RESULT_PATH, all_results)
        logger.info("Ablation results saved → %s", RESULT_PATH)

        # ── Persist latency summary ───────────────────────────────────
        summary = self._build_summary(all_results)
        _atomic_json_write(SUMMARY_PATH, summary)
        logger.info("Latency summary saved  → %s", SUMMARY_PATH)

        elapsed = round(time.time() - t_start, 1)
        logger.info("╚══  Ablation study complete in %.1fs  ══╝", elapsed)

        self._print_summary_table(summary)

        return {"ablation_results": all_results, "summary": summary}

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary_table(summary: Dict[str, Any]) -> None:
        """Print a readable per-config latency table after all runs."""
        configs = [c for c in summary if c != "generated_at"]
        if not configs:
            return

        stages = list(next(
            summary[c]["stages"]
            for c in configs
            if "stages" in summary.get(c, {})
        ))

        col_w = 16
        header = f"{'Config':<20}" + "".join(f"{'  '+s+' (ms)':>{col_w}}" for s in stages) + f"{'  Total (ms)':>{col_w}}"
        print("\n" + "─" * len(header))
        print(header)
        print("─" * len(header))

        for cfg in configs:
            row = f"{cfg:<20}"
            for s in stages:
                mean = summary[cfg].get("stages", {}).get(s, {}).get("mean_ms", 0)
                row += f"{mean:>{col_w}.1f}"
            total_mean = summary[cfg].get("total_ms", {}).get("mean_ms", 0)
            row += f"{total_mean:>{col_w}.1f}"
            print(row)

        print("─" * len(header))
        print()


# ---------------------------------------------------------------------------
# Smoke test  (python ablation_runner.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="Limit number of QA pairs (e.g. --max-queries 10 for a smoke test)",
    )
    parser.add_argument(
        "--qa-path", type=str, default=QA_PATH,
        help="Path to QA pairs JSON file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Ablation Runner — Starting Experiment")
    print("=" * 60)
    print(f"  QA path     : {args.qa_path}")
    print(f"  Max queries : {args.max_queries or 'all'}")
    print(f"  Configs     : {[c[2] for c in ABLATION_CONFIGS]}")
    print()

    runner = AblationRunner(
        qa_path     = args.qa_path,
        max_queries = args.max_queries,
    )

    output = runner.run()

    # Quick sanity print
    for cfg_name, results in output["ablation_results"].items():
        n_ok = sum(1 for r in results if not r["answer"].startswith("[ERROR]"))
        print(f"  {cfg_name:<20} {n_ok}/{len(results)} queries answered")

    print(f"\nResults  → {RESULT_PATH}")
    print(f"Summary  → {SUMMARY_PATH}")
    print("\n✅  Experiment complete.")
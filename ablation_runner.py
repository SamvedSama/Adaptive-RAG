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

API Contract (as of adaptive_pipeline.py refactor):
    pipeline.run() returns PipelineResult (frozen dataclass), NOT a dict.
    All field access uses attribute syntax: out.answer, out.route_label, etc.
    Latency is read from out.latency (dict[str,float]) and
    out.total_latency_ms (@property).
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from adaptive_pipeline import AdaptiveRAGPipeline, PipelineResult, PipelineConfig, _atomic_json_write
from latency_tracker import aggregate_logs

_NO_ANSWER_SENTINEL = "[ERROR] Empty answer from LLM"

def _safe_answer(ans: Optional[str]) -> str:
    if not ans or not ans.strip():
        return _NO_ANSWER_SENTINEL
    return ans

# ---------------------------------------------------------------------------
# Force UTF-8 for stdout/stderr on Windows (fixes cp1252 UnicodeDecodeError)
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# Also propagate to any subprocesses spawned by this process
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ---------------------------------------------------------------------------
# GPU / CPU device detection  (graceful fallback)
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore

    if torch.cuda.is_available():
        DEVICE = "cuda"
        _gpu_name = torch.cuda.get_device_name(0)
    else:
        DEVICE = "cpu"
        _gpu_name = "N/A"
except ImportError:
    DEVICE = "cpu"
    _gpu_name = "N/A (torch not installed)"

# ---------------------------------------------------------------------------
# Logging — file handler uses UTF-8 explicitly to avoid cp1252 on Windows
# ---------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

_log_file_handler = logging.FileHandler(
    "results/ablation_run.log", encoding="utf-8"
)
_log_file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stderr),
        _log_file_handler,
    ],
)
logger = logging.getLogger("AblationRunner")
logger.info("Device selected: %s  (GPU: %s)", DEVICE, _gpu_name)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
QA_PATH         = "data/qa_pairs.json"
RESULTS_DIR     = "results"
RESULT_PATH     = os.path.join(RESULTS_DIR, "ablation_results.json")
SUMMARY_PATH    = os.path.join(RESULTS_DIR, "ablation_summary.json")
ERROR_LOG_PATH  = os.path.join(RESULTS_DIR, "ablation_errors.log")
LATENCY_LOG_DIR = "latency_logs"

# Four ablation configurations: (use_router, use_reranker, label)
ABLATION_CONFIGS: List[tuple] = [
    (False, False, "naive"),
    (True,  False, "router_only"),
    (False, True,  "reranker_only"),
    (True,  True,  "full_adaptive"),
]

# Budget tiers to sweep per config (matches qa_generator.py trifold scheme)
BUDGET_TIERS: List[float] = [1.0, 0.5, 0.1]


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------

class AblationRunner:
    """
    Orchestrates all four ablation runs over the QA dataset.

    Components are NOT loaded here — each AdaptiveRAGPipeline instance
    manages its own components via the singleton chain.  This keeps
    memory predictable on 8 GB VRAM: we load one pipeline config at a
    time, run it fully, then replace it with the next config.

    API Contract
    ────────────
    pipeline.run() → PipelineResult (frozen dataclass)
      ├── out.answer              str
      ├── out.route_label         str   (e.g. "Multi_Hop_FAISS")
      ├── out.route_confidence    float
      ├── out.route_fallback      bool
      ├── out.retriever_used      str
      ├── out.retrieved_chunks    list[RetrievedChunk]
      ├── out.latency             dict[str, float]  (per-stage ms)
      ├── out.total_latency_ms    float (@property — sum of latency dict)
      └── out.error               Optional[str]
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
            max_queries: Cap number of queries (None = use all).
        """
        self.qa_path     = qa_path
        self.results_dir = results_dir
        self.max_queries = max_queries

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(LATENCY_LOG_DIR, exist_ok=True)

        logger.info(
            "AblationRunner ready | qa_path='%s' max_queries=%s device=%s",
            qa_path, max_queries, DEVICE,
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
              "query_type":           str   (optional)
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

        # Open explicitly as UTF-8 to avoid cp1252 on Windows
        with open(self.qa_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"QA file '{self.qa_path}' is empty or not a list.")

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
        qa_pairs:     List[Dict[str, Any]],
        use_router:   bool,
        use_reranker: bool,
        config_name:  str,
        budget:       float,
    ) -> List[Dict[str, Any]]:
        """
        Run one ablation configuration over the full QA set.

        Args:
            qa_pairs:     List of QA dicts.
            use_router:   Enable router for this config.
            use_reranker: Enable reranker for this config.
            config_name:  Label for logging and output files.
            budget:       Budget scalar passed to pipeline.run().

        Returns:
            List of per-query result dicts.

        Note on API contract:
            pipeline.run() returns PipelineResult (frozen dataclass).
            All access is via attributes (out.answer, out.route_label, …),
            never dict keys.  out.latency is a dict[str, float] of per-stage
            wall-clock milliseconds; out.total_latency_ms is the @property sum.
        """
        logger.info(
            "━━━  Config: %-20s  Router=%-5s  Reranker=%-5s  Budget=%.1f  ━━━",
            config_name, use_router, use_reranker, budget,
        )

        pipeline = AdaptiveRAGPipeline(
            PipelineConfig(
                use_router   = use_router,
                use_reranker = use_reranker,
            )
        )

        results: List[Dict[str, Any]] = []
        n = len(qa_pairs)

        for idx, qa in enumerate(
            tqdm(qa_pairs, desc=f"[{config_name}]", unit="q"), start=1
        ):
            query        = qa["question"]
            ground_truth = qa.get("ground_truth_answer", "")
            gold_type    = qa.get("query_type", "unknown")

            logger.debug("  [%d/%d] %s", idx, n, query[:70])

            try:
                out: PipelineResult = pipeline.run(
                    query,
                    budget      = budget,
                    save_result  = False,
                )

                # ── Guard against None / empty answer from LLM ────────
                answer = _safe_answer(out.answer)

                results.append({
                    # ── identifiers ───────────────────────────────────
                    "query":    query,
                    "config":   config_name,
                    "budget":   budget,

                    # ── router analysis fields ─────────────────────────
                    # out.route_label   → routed path name (e.g. "Multi_Hop_FAISS")
                    # out.route_confidence → ML classifier confidence [0,1]
                    # out.route_fallback   → True when router was unavailable
                    "gold_query_type":      gold_type,
                    "predicted_query_type": out.route_label,
                    "router_confidence":    out.route_confidence,
                    "router_fallback":      out.route_fallback,
                    "retriever_used":       out.retriever_used,

                    # ── evaluation fields (RAGAS) ─────────────────────
                    "ground_truth":         ground_truth,
                    "answer":               out.answer,
                    "retrieved_chunk_ids":  [c.chunk_id for c in out.retrieved_chunks],
                    "retrieved_texts":      [c.text     for c in out.retrieved_chunks],

                    # ── latency (pareto_curve.py) ─────────────────────
                    # out.latency          → dict[str, float] of per-stage ms
                    # out.total_latency_ms → @property sum across all stages
                    "latency_ms":       out.latency,
                    "total_latency_ms": out.total_latency_ms,
                })

            except Exception as exc:
                logger.error(
                    "  Query %d/%d failed (%s): %s", idx, n, config_name, exc,
                    exc_info=True,
                )
                results.append({
                    "query":                query,
                    "config":               config_name,
                    "budget":               budget,
                    "gold_query_type":      gold_type,
                    "predicted_query_type": "error",
                    "router_confidence":    0.0,
                    "router_fallback":      True,
                    "retriever_used":       "error",
                    "ground_truth":         ground_truth,
                    "answer":               f"[ERROR] {exc}",
                    "retrieved_chunk_ids":  [],
                    "retrieved_texts":      [],
                    "latency_ms":           {},
                    "total_latency_ms":     0.0,
                })

        n_ok  = sum(1 for r in results if not r["answer"].startswith("[ERROR]"))
        n_no  = sum(1 for r in results if r["answer"] == _NO_ANSWER_SENTINEL)
        logger.info(
            "Config '%s' done — %d/%d succeeded.", config_name, n_ok, n,
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

        Converts each query's out.latency dict and out.total_latency_ms
        into the pseudo-log schema that aggregate_logs() expects:
            { "timings_ms": dict[str,float], "total_ms": float }

        Errored queries (latency_ms == {}) are skipped so they don't
        drag down mean calculations.

        Used by pareto_curve.py.
        """
        summary: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device":       DEVICE,
        }

        for config_name, results in all_results.items():
            pseudo_logs = [
                {
                    "timings_ms": r["latency_ms"],
                    "total_ms":   r["total_latency_ms"],
                }
                for r in results
                if r["latency_ms"]          # skip errored / empty queries
            ]
            summary[config_name] = aggregate_logs(pseudo_logs)

        return summary

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Execute all four ablation configurations × 3 budget tiers
        sequentially and save results to disk.

        Returns:
            {
              "ablation_results": Dict[config_name, List[result_dict]],
              "summary":          Dict[config_name, latency_stats],
            }
        """
        t_start = time.time()
        logger.info("╔══  Ablation study started  ══╗  device=%s", DEVICE)

        # Clear / initialise error log for this run
        with open(ERROR_LOG_PATH, "w", encoding="utf-8") as fh:
            fh.write(
                f"Ablation error log — {datetime.now(timezone.utc).isoformat()}\n"
                + "=" * 72 + "\n"
            )

        qa_pairs    = self.load_qa_pairs()
        all_results: Dict[str, List[Dict[str, Any]]] = {}

        for use_router, use_reranker, config_base in ABLATION_CONFIGS:
            if config_base == "full_adaptive":
                for budget in BUDGET_TIERS:
                    config_name = f"{config_base}_b{budget}"
                    all_results[config_name] = self.run_config(
                        qa_pairs, use_router, use_reranker, config_name, budget,
                    )
            else:
                config_name = config_base
                all_results[config_name] = self.run_config(
                    qa_pairs, use_router, use_reranker, config_name, 1.0,
                )

        # Persist per-query results
        _atomic_json_write(RESULT_PATH, all_results)
        logger.info("Ablation results saved → %s", RESULT_PATH)

        # Persist latency summary
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
        configs = [c for c in summary if c not in ("generated_at", "device")]
        if not configs:
            return

        # Pull stage names from the first config that has them
        stages: List[str] = []
        for c in configs:
            if "stages" in summary.get(c, {}):
                stages = list(summary[c]["stages"].keys())
                break

        if not stages:
            logger.warning("No stage breakdown found in summary — skipping table.")
            return

        col_w  = 16
        header = (
            f"{'Config':<24}"
            + "".join(f"{'  ' + s + ' (ms)':>{col_w}}" for s in stages)
            + f"{'  Total (ms)':>{col_w}}"
        )
        sep = "─" * len(header)

        print(f"\n{sep}")
        print(header)
        print(sep)

        for cfg in configs:
            row = f"{cfg:<24}"
            for s in stages:
                mean = summary[cfg].get("stages", {}).get(s, {}).get("mean_ms", 0.0)
                row += f"{mean:>{col_w}.1f}"
            total_mean = summary[cfg].get("total_ms", {}).get("mean_ms", 0.0)
            row += f"{total_mean:>{col_w}.1f}"
            print(row)

        print(sep)
        print()


# ---------------------------------------------------------------------------
# CLI  (python ablation_runner.py [--max-queries N] [--qa-path PATH])
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the 4-config × 3-budget ablation study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="Limit QA pairs processed (e.g. --max-queries 2 for a smoke test).",
    )
    parser.add_argument(
        "--qa-path", type=str, default=QA_PATH,
        help="Path to QA pairs JSON file.",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  Ablation Runner — Starting Experiment")
    print("=" * 62)
    print(f"  QA path     : {args.qa_path}")
    print(f"  Max queries : {args.max_queries or 'all'}")
    print(f"  Configs     : {[c[2] for c in ABLATION_CONFIGS]}")
    print(f"  Budgets     : {BUDGET_TIERS}")
    print()

    runner = AblationRunner(
        qa_path     = args.qa_path,
        max_queries = args.max_queries,
    )

    output = runner.run()

    # ── Quick sanity summary ──────────────────────────────────────────
    print()
    for cfg_name, results in output["ablation_results"].items():
        n_ok = sum(1 for r in results if not r["answer"].startswith("[ERROR]"))
        print(f"  {cfg_name:<28}  {n_ok}/{len(results)} queries answered")

    print(f"\n  Results  → {RESULT_PATH}")
    print(f"  Summary  → {SUMMARY_PATH}")
    print("\n✅  Experiment complete.")
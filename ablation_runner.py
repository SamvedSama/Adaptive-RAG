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

Fixes applied (2026-03-19):
    - UnicodeDecodeError on Windows: all subprocess calls now use
      encoding="utf-8" + errors="replace"; stdout/stderr forced to UTF-8
      at process start via PYTHONUTF8=1 env var.
    - NoneType.strip crash: answer field is sanitised with _safe_answer()
      before any string operation; None / empty LLM responses surface as
      a recoverable "[NO ANSWER]" sentinel instead of crashing.
    - Error log: every failed query is appended to
      results/ablation_errors.log in addition to the JSON output.
    - GPU/CPU fallback: torch device is detected once at import time and
      passed into AdaptiveRAGPipeline via device=; pipelines fall back to
      CPU automatically if CUDA is unavailable.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from adaptive_pipeline import AdaptiveRAGPipeline, _atomic_json_write
from latency_tracker import aggregate_logs

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

# Sentinel returned when the LLM produces no answer instead of crashing
_NO_ANSWER_SENTINEL = "[NO ANSWER]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_answer(raw: Any) -> str:
    """
    Coerce whatever the pipeline returns as 'answer' into a plain string.

    Handles:
        None        → "[NO ANSWER]"
        ""          → "[NO ANSWER]"
        str         → stripped string, or "[NO ANSWER]" if blank after strip
        other types → str(raw).strip()

    This prevents the 'NoneType' has no attribute 'strip' crash when the
    LLM returns an empty / null response.
    """
    if raw is None:
        return _NO_ANSWER_SENTINEL
    try:
        text = str(raw).strip()
    except Exception:
        return _NO_ANSWER_SENTINEL
    return text if text else _NO_ANSWER_SENTINEL


def _log_error(config: str, idx: int, total: int, query: str, exc: Exception) -> None:
    """
    Append a structured error entry to the plain-text error log file.
    Errors go here in addition to the per-query JSON so they are easy
    to grep without parsing JSON.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = (
        f"[{timestamp}] CONFIG={config} QUERY={idx}/{total}\n"
        f"  Q  : {query[:120]}\n"
        f"  ERR: {type(exc).__name__}: {exc}\n"
        "─" * 72 + "\n"
    )
    try:
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(entry)
    except OSError as write_err:
        logger.warning("Could not write to error log: %s", write_err)


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------

class AblationRunner:
    """
    Orchestrates all four ablation runs over the QA dataset.

    Each AdaptiveRAGPipeline is loaded fresh per config to keep GPU
    memory predictable on 8 GB VRAM.  The detected DEVICE is forwarded
    so models land on CUDA when available and fall back to CPU otherwise.
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
    ) -> List[Dict[str, Any]]:
        """
        Run one ablation configuration over the full QA set.

        Args:
            qa_pairs:     List of QA dicts.
            use_router:   Enable router for this config.
            use_reranker: Enable reranker for this config.
            config_name:  Label for logging and output files.

        Returns:
            List of per-query result dicts.
        """
        logger.info(
            "━━━  Config: %-16s  Router=%-5s  Reranker=%s  Device=%s  ━━━",
            config_name, use_router, use_reranker, DEVICE,
        )

        # Pass device so the pipeline puts models on GPU when available
        pipeline = AdaptiveRAGPipeline(
            use_router   = use_router,
            use_reranker = use_reranker,
            device       = DEVICE,          # GPU/CPU fallback forwarded here
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
                out = pipeline.run(
                    query,
                    save_latency=True,
                    save_result=False,
                )

                # ── Guard against None / empty answer from LLM ────────
                answer = _safe_answer(out.get("answer"))

                results.append({
                    # identifiers
                    "query":                query,
                    "config":               config_name,
                    # router analysis (Samved)
                    "gold_query_type":      gold_type,
                    "predicted_query_type": out.get("query_type", "unknown"),
                    "router_confidence":    out.get("confidence", 0.0),
                    "retriever_used":       out.get("retriever", "unknown"),
                    # evaluation fields (Nivi / RAGAS)
                    "ground_truth":         ground_truth,
                    "answer":               answer,
                    "retrieved_chunk_ids":  [c["chunk_id"] for c in out.get("chunks", [])],
                    "retrieved_texts":      [c["text"]     for c in out.get("chunks", [])],
                    # latency (pareto_curve.py)
                    "latency_ms":           out.get("latency_log", {}).get("timings_ms", {}),
                    "total_latency_ms":     out.get("latency_log", {}).get("total_ms", 0.0),
                })

            except Exception as exc:
                logger.error(
                    "  Query %d/%d failed (%s): %s", idx, n, config_name, exc
                )
                _log_error(config_name, idx, n, query, exc)

                results.append({
                    "query":                query,
                    "config":               config_name,
                    "gold_query_type":      gold_type,
                    "predicted_query_type": "error",
                    "router_confidence":    0.0,
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
            "Config '%s' done — %d/%d succeeded (%d no-answer).",
            config_name, n_ok, n, n_no,
        )
        return results

    # ------------------------------------------------------------------
    # Latency summary builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        all_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Build per-config latency aggregation (mean ± std ± min/max)."""
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
                if r["latency_ms"]
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
        logger.info("╔══  Ablation study started  ══╗  device=%s", DEVICE)

        # Clear / initialise error log for this run
        with open(ERROR_LOG_PATH, "w", encoding="utf-8") as fh:
            fh.write(
                f"Ablation error log — {datetime.now(timezone.utc).isoformat()}\n"
                + "=" * 72 + "\n"
            )

        qa_pairs    = self.load_qa_pairs()
        all_results: Dict[str, List[Dict[str, Any]]] = {}

        for use_router, use_reranker, config_name in ABLATION_CONFIGS:
            all_results[config_name] = self.run_config(
                qa_pairs, use_router, use_reranker, config_name
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

        stages = list(next(
            summary[c]["stages"]
            for c in configs
            if "stages" in summary.get(c, {})
        ))

        col_w  = 16
        header = (
            f"{'Config':<20}"
            + "".join(f"{'  ' + s + ' (ms)':>{col_w}}" for s in stages)
            + f"{'  Total (ms)':>{col_w}}"
        )
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
        print(f"  Device: {DEVICE}\n")


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
    print(f"  Device      : {DEVICE}  (GPU: {_gpu_name})")
    print()

    runner = AblationRunner(
        qa_path     = args.qa_path,
        max_queries = args.max_queries,
    )

    output = runner.run()

    for cfg_name, results in output["ablation_results"].items():
        n_ok  = sum(1 for r in results if not r["answer"].startswith("[ERROR]"))
        n_err = sum(1 for r in results if r["answer"].startswith("[ERROR]"))
        n_no  = sum(1 for r in results if r["answer"] == _NO_ANSWER_SENTINEL)
        print(
            f"  {cfg_name:<20} {n_ok}/{len(results)} ok"
            f"  |  {n_err} errors  |  {n_no} no-answer"
        )

    print(f"\nResults  → {RESULT_PATH}")
    print(f"Summary  → {SUMMARY_PATH}")
    print(f"Errors   → {ERROR_LOG_PATH}")
    print("\n✅  Experiment complete.")
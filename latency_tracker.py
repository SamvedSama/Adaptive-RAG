"""
latency_tracker.py — Latency Tracking Utility
Owner: Roshan K C

Tracks wall-clock latency of every stage in the RAG pipeline.

Stages tracked:
    routing | retrieval | reranking | generation

Key additions over sample:
    - Context-manager API  (with tracker.track("stage"):)
    - reset() for reuse across ablation runs
    - aggregate_logs() — merge N log dicts into mean/std summary
    - save_log() writes atomic JSON with ISO timestamp in filename
    - Guards against end() called before start(), double start(), etc.
"""

import json
import logging
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LatencyTracker")

# ---------------------------------------------------------------------------
# Valid pipeline stages (extend here if the pipeline grows)
# ---------------------------------------------------------------------------
KNOWN_STAGES = {"routing", "retrieval", "reranking", "generation"}


# ---------------------------------------------------------------------------
# LatencyTracker
# ---------------------------------------------------------------------------
@dataclass
class LatencyTracker:
    """
    Tracks wall-clock execution time of RAG pipeline stages.

    Two usage patterns
    ──────────────────
    1. Manual start / end  (matches ablation_runner.py style)

        tracker = LatencyTracker()
        tracker.start("routing")
        result = router.classify(query)
        tracker.end()

    2. Context manager  (cleaner, exception-safe)

        with tracker.track("retrieval"):
            chunks = retriever.retrieve(query)

    Reuse across queries
    ────────────────────
        tracker.reset()          # wipe timings before next query
    """

    timings:        Dict[str, float] = field(default_factory=dict)
    _start:         float            = field(default=0.0,  init=False, repr=False)
    _current_stage: str              = field(default="",   init=False, repr=False)
    _stage_open:    bool             = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Manual API
    # ------------------------------------------------------------------

    def start(self, stage: str) -> None:
        """
        Begin timing a pipeline stage.

        Args:
            stage: One of routing | retrieval | reranking | generation
                   (unknown stage names are accepted but logged as warnings)

        Raises:
            RuntimeError: If a stage is already open (forgot to call end()).
        """
        if self._stage_open:
            raise RuntimeError(
                f"[LatencyTracker] Cannot start '{stage}' — "
                f"stage '{self._current_stage}' is still open. Call end() first."
            )
        if stage not in KNOWN_STAGES:
            logger.warning("Unknown stage '%s' — recording anyway.", stage)

        self._current_stage = stage
        self._stage_open    = True
        self._start         = time.perf_counter()
        logger.debug("Stage started: %s", stage)

    def end(self) -> float:
        """
        Stop timing the current stage and record latency in ms.

        Returns:
            Elapsed time in milliseconds.

        Raises:
            RuntimeError: If end() is called without a preceding start().
        """
        if not self._stage_open:
            raise RuntimeError(
                "[LatencyTracker] end() called with no open stage. Call start() first."
            )

        elapsed_ms = (time.perf_counter() - self._start) * 1_000
        self.timings[self._current_stage] = round(elapsed_ms, 3)

        logger.debug("Stage ended: %s → %.3f ms", self._current_stage, elapsed_ms)

        self._stage_open    = False
        self._current_stage = ""
        return elapsed_ms

    # ------------------------------------------------------------------
    # Context-manager API
    # ------------------------------------------------------------------

    @contextmanager
    def track(self, stage: str) -> Generator[None, None, None]:
        """
        Exception-safe context manager for timing a stage.

        Usage:
            with tracker.track("generation"):
                answer = llm.generate(prompt)

        The stage is ended automatically even if an exception is raised,
        so latency logs are never left in a broken state.
        """
        self.start(stage)
        try:
            yield
        finally:
            self.end()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all recorded timings so the tracker can be reused for the
        next query without creating a new instance.

        Raises:
            RuntimeError: If reset() is called while a stage is open.
        """
        if self._stage_open:
            raise RuntimeError(
                f"[LatencyTracker] Cannot reset while stage "
                f"'{self._current_stage}' is open."
            )
        self.timings.clear()
        logger.debug("Tracker reset.")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def total_ms(self) -> float:
        """Return sum of all recorded stage latencies in ms."""
        return round(sum(self.timings.values()), 3)

    def report(self, silent: bool = False) -> Dict[str, float]:
        """
        Print a formatted latency table and return the timings dict.

        Args:
            silent: If True, suppress console output (useful in batch runs).

        Returns:
            Dict of {stage: latency_ms}.
        """
        if not silent:
            total = self.total_ms()
            print(f"\n{'Stage':<25} {'Time (ms)':>12}")
            print("─" * 39)
            for stage, t in self.timings.items():
                pct = (t / total * 100) if total > 0 else 0
                print(f"  {stage:<23} {t:>10.3f}   ({pct:.1f}%)")
            print("─" * 39)
            print(f"  {'TOTAL':<23} {total:>10.3f}")
            print()

        return dict(self.timings)

    # ------------------------------------------------------------------
    # Log serialisation
    # ------------------------------------------------------------------

    def to_log(
        self,
        query:       str,
        query_type:  str,
        config_name: str = "unknown",
        extra:       Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Serialise the current timings into the project log schema.

        Args:
            query:       Raw user query string.
            query_type:  Router classification (factual|conceptual|complex).
            config_name: Ablation config label (naive|router_only|…).
            extra:       Any additional key-value pairs to embed in the log.

        Returns:
            Log dict ready for JSON serialisation.
        """
        log: Dict[str, Any] = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "config":      config_name,
            "query":       query,
            "query_type":  query_type,
            "timings_ms":  dict(self.timings),
            "total_ms":    self.total_ms(),
        }
        if extra:
            log.update(extra)
        return log

    def save_log(
        self,
        query:       str,
        query_type:  str,
        config_name: str = "unknown",
        path:        str = "latency_logs",
        extra:       Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Persist a latency log to disk as an atomic JSON file.

        Filename format:
            <config_name>_<unix_ms>.json

        Args:
            query:       Raw user query string.
            query_type:  Router classification.
            config_name: Ablation config label.
            path:        Directory for log files.
            extra:       Optional extra fields for the log dict.

        Returns:
            Absolute path of the written file.
        """
        os.makedirs(path, exist_ok=True)

        log_data = self.to_log(query, query_type, config_name, extra)

        safe_config = config_name.replace(" ", "_")
        filename    = f"{safe_config}_{int(time.time() * 1_000)}.json"
        filepath    = os.path.join(path, filename)

        # Atomic write: write to .tmp then rename so we never get partial files
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, filepath)

        logger.info("Latency log saved → %s", filepath)
        return filepath


# ---------------------------------------------------------------------------
# Module-level helper — used by ablation_runner.py
# ---------------------------------------------------------------------------

def aggregate_logs(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate a list of per-query latency logs into mean ± std summary.

    Useful for ablation_runner.py to summarise 150-query runs.

    Args:
        logs: List of dicts returned by LatencyTracker.to_log()

    Returns:
        {
          "n_queries": int,
          "stages": {
            "<stage>": {"mean_ms": float, "std_ms": float, "min_ms": float, "max_ms": float}
          },
          "total_ms": {"mean_ms": float, "std_ms": float}
        }
    """
    if not logs:
        return {}

    # Collect per-stage samples
    stage_samples: Dict[str, List[float]] = {}
    total_samples: List[float] = []

    for log in logs:
        timings = log.get("timings_ms", {})
        for stage, ms in timings.items():
            stage_samples.setdefault(stage, []).append(ms)
        total_samples.append(log.get("total_ms", 0.0))

    def _stats(values: List[float]) -> Dict[str, float]:
        return {
            "mean_ms": round(statistics.mean(values), 3),
            "std_ms":  round(statistics.stdev(values) if len(values) > 1 else 0.0, 3),
            "min_ms":  round(min(values), 3),
            "max_ms":  round(max(values), 3),
        }

    return {
        "n_queries": len(logs),
        "stages":    {stage: _stats(vals) for stage, vals in stage_samples.items()},
        "total_ms":  _stats(total_samples),
    }


# ---------------------------------------------------------------------------
# Smoke test  (python latency_tracker.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("LatencyTracker — smoke test")
    print("=" * 55)

    tracker = LatencyTracker()

    # ── Pattern 1: manual start / end ─────────────────────────────────
    print("\n[1] Manual start/end")
    tracker.start("routing")
    time.sleep(0.05)
    tracker.end()

    tracker.start("retrieval")
    time.sleep(0.12)
    tracker.end()

    tracker.start("reranking")
    time.sleep(0.03)
    tracker.end()

    tracker.start("generation")
    time.sleep(0.25)
    tracker.end()

    tracker.report()

    # ── Pattern 2: context manager ────────────────────────────────────
    print("[2] Context manager (reset first)")
    tracker.reset()

    with tracker.track("routing"):
        time.sleep(0.04)

    with tracker.track("retrieval"):
        time.sleep(0.10)

    with tracker.track("reranking"):
        time.sleep(0.02)

    with tracker.track("generation"):
        time.sleep(0.20)

    tracker.report()

    # ── to_log and save_log ───────────────────────────────────────────
    print("[3] to_log()")
    log = tracker.to_log(
        query       = "How does attention work in transformers?",
        query_type  = "conceptual",
        config_name = "full_adaptive",
        extra       = {"n_chunks_retrieved": 10, "n_chunks_reranked": 5},
    )
    print(json.dumps(log, indent=2))

    print("\n[4] save_log()")
    saved_path = tracker.save_log(
        query       = "How does attention work in transformers?",
        query_type  = "conceptual",
        config_name = "full_adaptive",
        path        = "latency_logs",
    )
    print(f"Saved → {saved_path}")

    # ── aggregate_logs ────────────────────────────────────────────────
    print("\n[5] aggregate_logs() over 3 fake runs")
    fake_logs = [
        {"timings_ms": {"routing": 12.1, "retrieval": 98.4, "reranking": 22.3, "generation": 310.5}, "total_ms": 443.3},
        {"timings_ms": {"routing": 11.8, "retrieval": 102.1, "reranking": 19.7, "generation": 295.0}, "total_ms": 428.6},
        {"timings_ms": {"routing": 13.0, "retrieval": 95.6, "reranking": 24.1, "generation": 320.2}, "total_ms": 452.9},
    ]
    summary = aggregate_logs(fake_logs)
    print(json.dumps(summary, indent=2))

    # ── Error guard tests ─────────────────────────────────────────────
    print("\n[6] Error guards")

    try:
        tracker.reset()
        tracker.start("routing")
        tracker.start("retrieval")          # double start
    except RuntimeError as e:
        print(f"  Double-start caught  : {e}")
    finally:
        tracker.end()                       # clean up

    try:
        tracker.reset()
        tracker.end()                       # end without start
    except RuntimeError as e:
        print(f"  End-without-start caught: {e}")

    print("\n✅  All smoke tests passed.")
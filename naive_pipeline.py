"""
naive_pipeline.py — Baseline RAG Pipeline
Owner: Roshan K C

Naive RAG system used as the baseline in the ablation study.

Pipeline:
    query
      ↓
    FAISS retrieval          (dense only — no router)
      ↓
    LLM generation (Ollama)  (no reranker)

Outputs per query:
    - answer         (str)
    - chunks         (List[dict])
    - latency_log    (dict)  — compatible with aggregate_logs()

Ablation config label: "naive"
"""

import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from faiss_retriever import FAISSRetriever
from latency_tracker import LatencyTracker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NaivePipeline")

# ---------------------------------------------------------------------------
# Constants  (single place to tune for the whole ablation study)
# ---------------------------------------------------------------------------
OLLAMA_MODEL    = "phi3:mini"       # swap to llama3.2 etc. without touching logic
TOP_K           = 5                  # chunks fed to the LLM
OLLAMA_TIMEOUT  = 120                # seconds before subprocess is killed
CONFIG_NAME     = "naive"            # used in latency log filenames + ablation CSV
LATENCY_LOG_DIR = "latency_logs"
RESULTS_DIR     = "results"


# ---------------------------------------------------------------------------
# NaiveRAGPipeline
# ---------------------------------------------------------------------------
class NaiveRAGPipeline:
    """
    Baseline RAG pipeline — FAISS retrieval + LLM generation.

    No router.  No reranker.
    This is Configuration 1 of the ablation study:

        Config      Router  Reranker
        ──────────  ──────  ────────
        naive        ❌      ❌
    """

    def __init__(
        self,
        top_k:        int = TOP_K,
        ollama_model: str = OLLAMA_MODEL,
        ollama_timeout: int = OLLAMA_TIMEOUT,
    ) -> None:
        """
        Load FAISS index once at startup.  Fail loudly if the index
        does not exist so the user knows to run build_index.py first.

        Args:
            top_k:           Number of chunks to retrieve.
            ollama_model:    Ollama model tag.
            ollama_timeout:  Seconds before the Ollama subprocess is killed.
        """
        self.top_k          = top_k
        self.ollama_model   = ollama_model
        self.ollama_timeout = ollama_timeout

        logger.info("Loading FAISS index…")
        self.retriever = FAISSRetriever()
        self.retriever.load()
        logger.info("FAISS index loaded — pipeline ready.")

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Construct the LLM prompt from the query and retrieved chunks.

        Context blocks are separated by a rule so the model can clearly
        distinguish one passage from another.

        Args:
            query:  User query string.
            chunks: Retrieved chunk dicts (must contain "text" key).

        Returns:
            Formatted prompt string.
        """
        if not chunks:
            logger.warning("build_prompt() called with 0 chunks.")

        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            text   = chunk.get("text", "").strip()
            context_blocks.append(f"[{i}] (source: {source})\n{text}")

        context = "\n\n---\n\n".join(context_blocks)

        prompt = (
            "You are a precise research assistant.\n"
            "Answer the question using ONLY the context passages provided below.\n"
            "If the context does not contain enough information, say so clearly.\n"
            "Do NOT add information not present in the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )
        return prompt

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    def generate_answer(self, prompt: str) -> str:
        """
        Call the local Ollama model via subprocess and return its output.

        Uses a timeout so a hanging model never blocks the ablation runner
        indefinitely.

        Args:
            prompt: Full prompt string.

        Returns:
            Model output string, or an error sentinel on failure.
        """
        logger.debug("Sending prompt to Ollama (%s)…", self.ollama_model)
        try:
            result = subprocess.run(
                ["ollama", "run", self.ollama_model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.ollama_timeout,
            )

            if result.returncode != 0:
                err = result.stderr.strip()
                logger.error("Ollama returned non-zero exit code. stderr: %s", err)
                return f"[ERROR] Ollama failed: {err}"

            answer = result.stdout.strip()
            if not answer:
                logger.warning("Ollama returned an empty response.")
                return "[ERROR] Empty response from model."

            return answer

        except subprocess.TimeoutExpired:
            logger.error("Ollama timed out after %ds.", self.ollama_timeout)
            return f"[ERROR] Model timed out after {self.ollama_timeout}s."

        except FileNotFoundError:
            logger.error("'ollama' binary not found. Is Ollama installed and on PATH?")
            return "[ERROR] Ollama not found on PATH."

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        query:          str,
        save_latency:   bool = True,
        save_result:    bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full naive RAG pipeline for a single query.

        Args:
            query:        User query string.
            save_latency: Persist latency log to LATENCY_LOG_DIR.
            save_result:  Persist full result dict to RESULTS_DIR.

        Returns:
            {
              "query":       str,
              "answer":      str,
              "chunks":      List[dict],
              "latency_log": dict,       ← compatible with aggregate_logs()
              "config":      "naive",
            }
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        logger.info("Running naive pipeline | query: '%s'", query[:80])
        tracker = LatencyTracker()

        # ── Retrieval ─────────────────────────────────────────────────
        with tracker.track("retrieval"):
            chunks = self.retriever.retrieve(query, top_k=self.top_k)

        logger.info("Retrieved %d chunks.", len(chunks))

        # ── Generation ────────────────────────────────────────────────
        with tracker.track("generation"):
            prompt = self.build_prompt(query, chunks)
            answer = self.generate_answer(prompt)

        logger.info("Answer generated (%d chars).", len(answer))

        # ── Latency report ────────────────────────────────────────────
        tracker.report(silent=False)

        latency_log = tracker.to_log(
            query       = query,
            query_type  = "naive",          # no router → always "naive"
            config_name = CONFIG_NAME,
            extra       = {
                "n_chunks_retrieved": len(chunks),
                "model":              self.ollama_model,
            },
        )

        if save_latency:
            tracker.save_log(
                query       = query,
                query_type  = "naive",
                config_name = CONFIG_NAME,
                path        = LATENCY_LOG_DIR,
            )

        # ── Result dict ───────────────────────────────────────────────
        result: Dict[str, Any] = {
            "query":       query,
            "answer":      answer,
            "chunks":      chunks,
            "latency_log": latency_log,
            "config":      CONFIG_NAME,
        }

        if save_result:
            self._save_result(result)

        return result

    # ------------------------------------------------------------------
    # Batch helper — used directly by ablation_runner.py
    # ------------------------------------------------------------------

    def run_batch(
        self,
        queries:      List[str],
        save_latency: bool = True,
        save_result:  bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run the pipeline over a list of queries and return all results.

        Args:
            queries:      List of query strings.
            save_latency: Persist per-query latency logs.
            save_result:  Persist aggregated results file.

        Returns:
            List of result dicts (same schema as run()).
        """
        logger.info("Starting batch run: %d queries, config='%s'", len(queries), CONFIG_NAME)
        results = []

        for idx, query in enumerate(queries, start=1):
            logger.info("Query %d/%d", idx, len(queries))
            try:
                result = self.run(query, save_latency=save_latency, save_result=False)
            except Exception as e:
                logger.error("Query %d failed: %s", idx, e)
                result = {
                    "query":       query,
                    "answer":      f"[ERROR] {e}",
                    "chunks":      [],
                    "latency_log": {},
                    "config":      CONFIG_NAME,
                }
            results.append(result)

        if save_result:
            self._save_batch_results(results)

        logger.info("Batch run complete. %d/%d succeeded.", 
                    sum(1 for r in results if not r["answer"].startswith("[ERROR]")),
                    len(results))
        return results

    # ------------------------------------------------------------------
    # Internal persistence helpers
    # ------------------------------------------------------------------

    def _save_result(self, result: Dict[str, Any]) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filename = f"{CONFIG_NAME}_{int(time.time() * 1000)}.json"
        filepath = os.path.join(RESULTS_DIR, filename)
        _atomic_json_write(filepath, result)
        logger.info("Result saved → %s", filepath)

    def _save_batch_results(self, results: List[Dict[str, Any]]) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(RESULTS_DIR, f"{CONFIG_NAME}_results.json")
        _atomic_json_write(filepath, results)
        logger.info("Batch results saved → %s", filepath)

    def __repr__(self) -> str:
        return (
            f"NaiveRAGPipeline(model='{self.ollama_model}', "
            f"top_k={self.top_k})"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _atomic_json_write(filepath: str, data: Any) -> None:
    """Write JSON atomically (tmp + rename) to avoid partial files on crash."""
    tmp = filepath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, filepath)


# ---------------------------------------------------------------------------
# Smoke test  (python naive_pipeline.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Naive RAG Pipeline — smoke test")
    print("=" * 60)

    pipeline = NaiveRAGPipeline()
    print(f"\nPipeline: {pipeline}\n")

    # ── Single query ──────────────────────────────────────────────────
    QUERY = "How does the transformer attention mechanism work?"

    result = pipeline.run(QUERY, save_latency=True, save_result=True)

    print("\n--- Answer ---")
    print(result["answer"])

    print("\n--- Retrieved Chunks ---")
    for i, chunk in enumerate(result["chunks"], start=1):
        print(
            f"  {i}. {chunk['chunk_id']:<35} "
            f"score={chunk['score']:.4f}  "
            f"{chunk['text'][:60]}…"
        )

    print("\n--- Latency Log ---")
    print(json.dumps(result["latency_log"], indent=2))

    # ── Edge case: empty query ────────────────────────────────────────
    print("\n--- Edge case: empty query ---")
    try:
        pipeline.run("")
    except ValueError as e:
        print(f"  Caught: {e}")

    print("\n✅  Smoke test complete.")
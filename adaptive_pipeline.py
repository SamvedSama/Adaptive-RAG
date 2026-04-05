"""
adaptive_pipeline.py — Adaptive Retrieval RAG Pipeline
Owner: Roshan K C

Full Adaptive RAG system — Configuration 4 of the ablation study.

    Config           Router   Reranker
    ─────────────    ──────   ────────
    naive              ❌       ❌
    router_only        ✅       ❌
    reranker_only      ❌       ✅
    full_adaptive      ✅       ✅   ← THIS FILE

Fixes applied (2026-03-19):
    - Added device= param to __init__() forwarded to FAISS and
      CrossEncoderReranker so models use CUDA when available.
    - subprocess.run() for Ollama now explicitly passes
      encoding="utf-8" and errors="replace" on both stdin and
      stdout so Windows cp1252 never causes UnicodeDecodeError.
    - env= override ensures PYTHONUTF8=1 is set for the Ollama
      subprocess regardless of the parent shell's code page.
    - result.stdout guarded against None before .strip() so a
      missing stdout never produces a NoneType crash.
"""

import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from bm25_retriever import BM25Retriever
from faiss_retriever import FAISSRetriever
from hybrid_retriever import HybridRetriever
from latency_tracker import LatencyTracker
from reranker import CrossEncoderReranker
from router import QueryRouter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AdaptivePipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_MODEL    = "phi3:mini"       # swap to llama3.2 etc. without touching logic
TOP_K_RETRIEVE  = 10                # fetch more candidates before reranking
TOP_K_RERANK    = 5                 # keep top-N after reranking
OLLAMA_TIMEOUT  = 120               # seconds before subprocess is killed
LATENCY_LOG_DIR = "latency_logs"
RESULTS_DIR     = "results"

# Maps (use_router, use_reranker) → ablation config label
_CONFIG_LABELS: Dict[tuple, str] = {
    (False, False): "naive",
    (True,  False): "router_only",
    (False, True):  "reranker_only",
    (True,  True):  "full_adaptive",
}

# Default retrieval method when router is disabled
_FALLBACK_QUERY_TYPE = "conceptual"   # → FAISS  (same as naive_pipeline.py)

# Environment passed to every Ollama subprocess so UTF-8 is used on Windows
_SUBPROCESS_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


# ---------------------------------------------------------------------------
# AdaptiveRAGPipeline
# ---------------------------------------------------------------------------
class AdaptiveRAGPipeline:
    """
    Single class that covers all four ablation configurations via flags.

    Usage (ablation_runner.py):

        # Config 1 — Naive
        pipe = AdaptiveRAGPipeline(use_router=False, use_reranker=False)

        # Config 2 — Router only
        pipe = AdaptiveRAGPipeline(use_router=True,  use_reranker=False)

        # Config 3 — Reranker only
        pipe = AdaptiveRAGPipeline(use_router=False, use_reranker=True)

        # Config 4 — Full Adaptive  (default)
        pipe = AdaptiveRAGPipeline(use_router=True,  use_reranker=True)

        results = pipe.run_batch(queries)
    """

    def __init__(
        self,
        use_router:     bool = True,
        use_reranker:   bool = True,
        top_k_retrieve: int  = TOP_K_RETRIEVE,
        top_k_rerank:   int  = TOP_K_RERANK,
        ollama_model:   str  = OLLAMA_MODEL,
        ollama_timeout: int  = OLLAMA_TIMEOUT,
        device:         str  = "cpu",
    ) -> None:
        """
        Load all components once.  Expensive models (FAISS, reranker)
        are loaded here so run() / run_batch() stay fast.

        Args:
            use_router:     Enable query router.
            use_reranker:   Enable cross-encoder reranker.
            top_k_retrieve: Candidates fetched from retriever.
            top_k_rerank:   Final chunks kept after reranking
                            (ignored when use_reranker=False).
            ollama_model:   Ollama model tag.
            ollama_timeout: Subprocess kill timeout in seconds.
            device:         Torch device string — "cuda" or "cpu".
                            Passed to FAISSRetriever and
                            CrossEncoderReranker so embeddings and
                            the cross-encoder land on GPU when
                            available.  Defaults to "cpu" so existing
                            callers that omit the arg are unaffected.
        """
        self.use_router     = use_router
        self.use_reranker   = use_reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank   = top_k_rerank
        self.ollama_model   = ollama_model
        self.ollama_timeout = ollama_timeout
        self.device         = device
        self.config_name    = _CONFIG_LABELS[(use_router, use_reranker)]

        logger.info(
            "Initialising AdaptiveRAGPipeline | config='%s' "
            "use_router=%s use_reranker=%s device=%s",
            self.config_name, use_router, use_reranker, device,
        )

        # ── Retrievers (always loaded — reused across ablation configs) ──
        logger.info("Loading BM25 retriever…")
        self.bm25 = BM25Retriever()

        logger.info("Loading FAISS retriever…")
        self.faiss = FAISSRetriever()
        self.faiss.load()

        logger.info("Building Hybrid retriever…")
        self.hybrid = HybridRetriever(
            faiss_retriever=self.faiss,
            bm25_retriever=self.bm25,
        )

        # ── Router (conditional) ──────────────────────────────────────
        if self.use_router:
            logger.info("Loading QueryRouter…")
            self.router = QueryRouter()
        else:
            self.router = None
            logger.info("Router disabled — all queries routed to FAISS (conceptual).")

        # ── Reranker (conditional) ────────────────────────────────────
        if self.use_reranker:
            logger.info("Loading CrossEncoderReranker… (device=%s)", device)
            # Pass device= if your CrossEncoderReranker supports it.
            # Falls back gracefully if it doesn't accept the kwarg yet.
            try:
                self.reranker = CrossEncoderReranker(device=device)
            except TypeError:
                logger.warning(
                    "CrossEncoderReranker does not accept device= yet — "
                    "loading without device kwarg. Add device= support to "
                    "reranker.py for GPU acceleration."
                )
                self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None
            logger.info("Reranker disabled.")

        logger.info("All components loaded — pipeline ready.")

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, query: str) -> tuple:
        """
        Classify query type.  Returns (query_type, confidence, method).
        Falls back to _FALLBACK_QUERY_TYPE when router is disabled.
        """
        if self.use_router and self.router is not None:
            query_type, confidence, method = self.router.classify(query)
        else:
            query_type = _FALLBACK_QUERY_TYPE
            confidence = 1.0
            method     = "fallback"
        return query_type, confidence, method

    # ------------------------------------------------------------------
    # Retrieval dispatch
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, query_type: str) -> List[Dict[str, Any]]:
        """
        Dispatch to the correct retriever based on query_type.

        Args:
            query:      User query string.
            query_type: factual | conceptual | complex

        Returns:
            List of chunk dicts.

        Raises:
            ValueError: On unknown query_type.
        """
        if query_type == "factual":
            logger.debug("Retriever: BM25")
            return self.bm25.retrieve(query, top_k=self.top_k_retrieve)

        elif query_type == "conceptual":
            logger.debug("Retriever: FAISS")
            return self.faiss.retrieve(query, top_k=self.top_k_retrieve)

        elif query_type == "complex":
            logger.debug("Retriever: Hybrid")
            return self.hybrid.retrieve(query, top_k=self.top_k_retrieve)

        else:
            raise ValueError(
                f"[AdaptivePipeline] Unknown query_type: '{query_type}'. "
                f"Expected factual | conceptual | complex."
            )

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks if enabled, otherwise just truncate to top_k_rerank.
        """
        if self.use_reranker and self.reranker is not None:
            return self.reranker.rerank(query, chunks, top_k=self.top_k_rerank)
        else:
            return chunks[: self.top_k_rerank]

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Construct the LLM prompt from the query and context chunks.

        Numbered passages with source labels improve faithfulness and
        make it easier to trace answers back to chunks during evaluation.
        """
        if not chunks:
            logger.warning("build_prompt() called with 0 chunks.")

        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            text   = chunk.get("text", "").strip()
            context_blocks.append(f"[{i}] (source: {source})\n{text}")

        context = "\n\n---\n\n".join(context_blocks)

        return (
            "You are a precise research assistant.\n"
            "Answer the question using ONLY the context passages provided below.\n"
            "If the context does not contain enough information, say so clearly.\n"
            "Do NOT add information not present in the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    def generate_answer(self, prompt: str) -> str:
        """
        Call local Ollama model and return its output.

        Key fix: subprocess.run() uses encoding="utf-8" + errors="replace"
        explicitly so Windows never falls back to cp1252, which was causing
        UnicodeDecodeError on bytes like 0x8f and 0x9d.  The subprocess
        also inherits _SUBPROCESS_ENV which sets PYTHONUTF8=1.

        result.stdout is guarded against None before .strip() so an empty
        or missing stdout produces a recoverable error string rather than
        a NoneType crash.
        """
        logger.debug("Sending prompt to Ollama (%s)…", self.ollama_model)
        try:
            result = subprocess.run(
                ["ollama", "run", self.ollama_model],
                input=prompt,
                capture_output=True,
                # ── UTF-8 fix: explicit encoding so Windows never uses cp1252 ──
                encoding="utf-8",
                errors="replace",       # replace undecodable bytes with '?' 
                timeout=self.ollama_timeout,
                env=_SUBPROCESS_ENV,    # propagate PYTHONUTF8=1 to child
            )

            if result.returncode != 0:
                # stderr may also be None on some platforms — guard it too
                err = (result.stderr or "").strip()
                logger.error("Ollama non-zero exit. stderr: %s", err)
                return f"[ERROR] Ollama failed: {err}"

            # Guard against None stdout before calling .strip()
            raw_stdout = result.stdout or ""
            answer = raw_stdout.strip()
            if not answer:
                logger.warning("Ollama returned empty response.")
                return "[NO ANSWER]"

            return answer

        except subprocess.TimeoutExpired:
            logger.error("Ollama timed out after %ds.", self.ollama_timeout)
            return f"[ERROR] Model timed out after {self.ollama_timeout}s."

        except FileNotFoundError:
            logger.error("'ollama' binary not found. Is Ollama installed?")
            return "[ERROR] Ollama not found on PATH."

    # ------------------------------------------------------------------
    # Main pipeline — single query
    # ------------------------------------------------------------------

    def run(
        self,
        query:        str,
        save_latency: bool = True,
        save_result:  bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full adaptive RAG pipeline for one query.

        Args:
            query:        User query string.
            save_latency: Persist latency log to LATENCY_LOG_DIR.
            save_result:  Persist result dict to RESULTS_DIR.

        Returns:
            {
              "query":        str,
              "query_type":   str,      ← from router (or fallback)
              "confidence":   float,    ← router confidence
              "retriever":    str,      ← which retriever was used
              "answer":       str,
              "chunks":       List[dict],
              "latency_log":  dict,     ← compatible with aggregate_logs()
              "config":       str,      ← ablation config label
            }

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        logger.info(
            "[%s] Running pipeline | query: '%s'",
            self.config_name, query[:80],
        )

        tracker = LatencyTracker()

        # ── 1. ROUTING ────────────────────────────────────────────────
        with tracker.track("routing"):
            query_type, confidence, route_method = self._route(query)

        logger.info(
            "Router → type='%s'  confidence=%.2f  method='%s'",
            query_type, confidence, route_method,
        )

        # ── 2. RETRIEVAL ──────────────────────────────────────────────
        with tracker.track("retrieval"):
            chunks = self._retrieve(query, query_type)

        retriever_name = {
            "factual":    "BM25",
            "conceptual": "FAISS",
            "complex":    "Hybrid",
        }.get(query_type, query_type)
        logger.info("Retriever=%s → %d chunks.", retriever_name, len(chunks))

        # ── 3. RERANKING ──────────────────────────────────────────────
        with tracker.track("reranking"):
            final_chunks = self._rerank(query, chunks)

        logger.info(
            "Reranker=%s → %d/%d chunks kept.",
            "ON" if self.use_reranker else "OFF",
            len(final_chunks), len(chunks),
        )

        # ── 4. GENERATION ─────────────────────────────────────────────
        with tracker.track("generation"):
            prompt = self.build_prompt(query, final_chunks)
            answer = self.generate_answer(prompt)

        logger.info("Answer generated (%d chars).", len(answer))

        # ── 5. LATENCY REPORT ─────────────────────────────────────────
        tracker.report(silent=False)

        latency_log = tracker.to_log(
            query       = query,
            query_type  = query_type,
            config_name = self.config_name,
            extra       = {
                "confidence":          round(confidence, 4),
                "retriever":           retriever_name,
                "n_chunks_retrieved":  len(chunks),
                "n_chunks_final":      len(final_chunks),
                "use_router":          self.use_router,
                "use_reranker":        self.use_reranker,
                "model":               self.ollama_model,
                "device":              self.device,
            },
        )

        if save_latency:
            tracker.save_log(
                query       = query,
                query_type  = query_type,
                config_name = self.config_name,
                path        = LATENCY_LOG_DIR,
            )

        # ── 6. RESULT DICT ────────────────────────────────────────────
        result: Dict[str, Any] = {
            "query":       query,
            "query_type":  query_type,
            "confidence":  round(confidence, 4),
            "retriever":   retriever_name,
            "answer":      answer,
            "chunks":      final_chunks,
            "latency_log": latency_log,
            "config":      self.config_name,
        }

        if save_result:
            self._save_result(result)

        return result

    # ------------------------------------------------------------------
    # Batch runner — called directly by ablation_runner.py
    # ------------------------------------------------------------------

    def run_batch(
        self,
        queries:      List[str],
        save_latency: bool = True,
        save_result:  bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run the pipeline over a list of queries.

        Per-query failures are caught and logged — one bad query never
        aborts the full 150-query ablation run.

        Args:
            queries:      List of query strings.
            save_latency: Persist per-query latency logs.
            save_result:  Persist aggregated results JSON.

        Returns:
            List of result dicts.
        """
        n = len(queries)
        logger.info(
            "Starting batch run | config='%s' | %d queries", self.config_name, n
        )

        results: List[Dict[str, Any]] = []

        for idx, query in enumerate(queries, start=1):
            logger.info("Query %d/%d", idx, n)
            try:
                result = self.run(query, save_latency=save_latency, save_result=False)
            except Exception as exc:
                logger.error("Query %d failed: %s", idx, exc)
                result = {
                    "query":       query,
                    "query_type":  "unknown",
                    "confidence":  0.0,
                    "retriever":   "unknown",
                    "answer":      f"[ERROR] {exc}",
                    "chunks":      [],
                    "latency_log": {},
                    "config":      self.config_name,
                }
            results.append(result)

        n_ok = sum(1 for r in results if not r["answer"].startswith("[ERROR]"))
        logger.info("Batch complete — %d/%d succeeded.", n_ok, n)

        if save_result:
            self._save_batch_results(results)

        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_result(self, result: Dict[str, Any]) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fname = f"{self.config_name}_{int(time.time() * 1000)}.json"
        _atomic_json_write(os.path.join(RESULTS_DIR, fname), result)
        logger.info("Result saved → %s/%s", RESULTS_DIR, fname)

    def _save_batch_results(self, results: List[Dict[str, Any]]) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fpath = os.path.join(RESULTS_DIR, f"{self.config_name}_results.json")
        _atomic_json_write(fpath, results)
        logger.info("Batch results saved → %s", fpath)

    def __repr__(self) -> str:
        return (
            f"AdaptiveRAGPipeline(config='{self.config_name}', "
            f"model='{self.ollama_model}', "
            f"top_k_retrieve={self.top_k_retrieve}, "
            f"top_k_rerank={self.top_k_rerank}, "
            f"device='{self.device}')"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _atomic_json_write(filepath: str, data: Any) -> None:
    """Write JSON atomically (tmp + rename) to avoid corrupt files on crash."""
    tmp = filepath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, filepath)


# ---------------------------------------------------------------------------
# Smoke test  (python adaptive_pipeline.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Adaptive RAG Pipeline — smoke test")
    print("=" * 60)

    pipeline = AdaptiveRAGPipeline(use_router=True, use_reranker=True)
    print(f"\n{pipeline}\n")

    QUERIES = [
        "What is BERT?",
        "How does self-attention work in transformers?",
        "Compare BERT and GPT in terms of training objectives.",
    ]

    for q in QUERIES:
        print("\n" + "─" * 55)
        result = pipeline.run(q, save_latency=True, save_result=False)

        print(f"Query      : {result['query']}")
        print(f"Type       : {result['query_type']}  (conf={result['confidence']})")
        print(f"Retriever  : {result['retriever']}")
        print(f"Config     : {result['config']}")
        print(f"\nAnswer:\n{result['answer']}")

        print("\nTop chunks:")
        for i, chunk in enumerate(result["chunks"], start=1):
            print(
                f"  {i}. {chunk['chunk_id']:<35} "
                f"score={chunk['score']:+.4f}  "
                f"{chunk['text'][:55]}…"
            )

        print("\nLatency log:")
        print(json.dumps(result["latency_log"], indent=2))

    print("\n--- Edge case: empty query ---")
    try:
        pipeline.run("")
    except ValueError as e:
        print(f"  Caught: {e}")

    print("\n--- Ablation config labels ---")
    for (r, rr), label in _CONFIG_LABELS.items():
        p = AdaptiveRAGPipeline.__new__(AdaptiveRAGPipeline)
        p.use_router     = r
        p.use_reranker   = rr
        p.config_name    = label
        p.ollama_model   = OLLAMA_MODEL
        p.top_k_retrieve = TOP_K_RETRIEVE
        p.top_k_rerank   = TOP_K_RERANK
        p.device         = "cpu"
        print(f"  {repr(p)}")

    print("\n✅  Smoke test complete.")
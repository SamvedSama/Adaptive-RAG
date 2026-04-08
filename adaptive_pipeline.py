"""
adaptive_pipeline.py — Adaptive Retrieval RAG Pipeline
Owner: Roshan K C

Full Adaptive RAG system — covers all 4 ablation configurations via flags.

    Config           Router   Reranker
    ─────────────    ──────   ────────
    naive              ❌       ❌
    router_only        ✅       ❌
    reranker_only      ❌       ✅
    full_adaptive      ✅       ✅   ← default

Pipeline (per query):
    query + budget
      ↓
    QueryRouter.route_full()          → RoutingResult
      ↓
    Direct_LLM  → skip retrieval
    Single_Hop_BM25  → BM25Retriever
    Multi_Hop_FAISS  → HybridRetriever   (dense + sparse RRF)
      ↓
    CrossEncoderReranker              (bypassed for Direct_LLM)
      ↓
    Ollama LLM  (subprocess, timeout-guarded)
      ↓
    PipelineResult

Public API:
    pipeline = get_pipeline()                 # module singleton
    result   = pipeline.run(query, budget)    # → PipelineResult
    results  = pipeline.run_batch(queries)    # → list[PipelineResult]
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from typing import Optional, TypedDict, Required

from langgraph.graph import StateGraph, START, END

class PipelineGraphState(TypedDict, total=False):
    query: Required[str]
    budget: Required[float]
    route_label: str
    route_confidence: float
    route_fallback: bool
    retriever_used: str
    raw_chunks: list
    final_chunks: list
    answer: str
    error: str
    latency: dict

from bm25_retriever import BM25Retriever, get_bm25_retriever
from faiss_retriever import FAISSRetriever, RetrievedChunk, get_faiss_retriever
from hybrid_retriever import HybridRetriever, get_hybrid_retriever
from reranker import CrossEncoderReranker, RerankResult, get_reranker
from router import QueryRouter, RoutingResult, get_router

log = logging.getLogger(__name__)

# ── Config maps ────────────────────────────────────────────────────────────────

_CONFIG_LABELS: dict[tuple[bool, bool], str] = {
    (False, False): "naive",
    (True,  False): "router_only",
    (False, True):  "reranker_only",
    (True,  True):  "full_adaptive",
}

_FALLBACK_ROUTE = "Multi_Hop_FAISS"   # used when router is disabled


# ── PipelineConfig ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    """
    All tunable parameters for AdaptiveRAGPipeline in one place.
    Pass a single config object instead of loose kwargs.
    """
    use_router:      bool  = True
    use_reranker:    bool  = True
    top_k_retrieve:  int   = 10      # candidates fetched from retriever
    top_k_rerank:    int   = 5       # chunks kept after reranking
    ollama_model:    str   = "phi3:mini"
    ollama_timeout:  int   = 120     # seconds before subprocess kill
    results_dir:     Path  = Path("results")
    latency_log_dir: Path  = Path("latency_logs")

    def __post_init__(self) -> None:
        if self.top_k_retrieve < 1:
            raise ValueError("top_k_retrieve must be ≥ 1.")
        if self.top_k_rerank < 1:
            raise ValueError("top_k_rerank must be ≥ 1.")
        if self.top_k_rerank > self.top_k_retrieve:
            raise ValueError("top_k_rerank must be ≤ top_k_retrieve.")
        if self.ollama_timeout < 5:
            raise ValueError("ollama_timeout must be ≥ 5 seconds.")

    @property
    def config_name(self) -> str:
        return _CONFIG_LABELS[(self.use_router, self.use_reranker)]


# ── PipelineResult ─────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Typed result container for a single pipeline run.
    Replaces the raw dict[str, Any] that was returned before.
    ablation_runner.py and app.py import and use this directly.
    """
    query:              str
    answer:             str
    config:             str
    route_label:        str                   # Multi_Hop_FAISS | Single_Hop_BM25 | Direct_LLM
    route_confidence:   float
    route_fallback:     bool
    retriever_used:     str
    chunks_retrieved:   int
    chunks_final:       int
    retrieved_chunks:   list[RetrievedChunk]
    latency: dict[str, float] = field(default_factory=dict)   # phase → ms
    error:   Optional[str]    = None                           # set on failure

    @property
    def total_latency_ms(self) -> float:
        return sum(self.latency.values())

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for disk persistence."""
        d = {
            "query":            self.query,
            "answer":           self.answer,
            "config":           self.config,
            "route_label":      self.route_label,
            "route_confidence": self.route_confidence,
            "route_fallback":   self.route_fallback,
            "retriever_used":   self.retriever_used,
            "chunks_retrieved": self.chunks_retrieved,
            "chunks_final":     self.chunks_final,
            "latency_ms":       self.latency,
            "total_latency_ms": self.total_latency_ms,
            "error":            self.error,
            "chunks": [c.to_dict() for c in self.retrieved_chunks],
        }
        return d


# ── AdaptiveRAGPipeline ────────────────────────────────────────────────────────

class AdaptiveRAGPipeline:
    """
    Single class covering all four ablation configurations via flags.

    Heavy components (FAISS index, BM25 index, cross-encoder weights,
    router model) are loaded once via module singletons — safe for Streamlit
    reruns which re-execute the script on every UI interaction.

    Usage (ablation_runner.py):
        pipe = AdaptiveRAGPipeline(PipelineConfig(use_router=True, use_reranker=True))
        results = pipe.run_batch(queries, budget=1.0)
    """

    def __init__(self, cfg: PipelineConfig = PipelineConfig()) -> None:
        self.cfg = cfg
        log.info(
            "Initialising AdaptiveRAGPipeline | config='%s' | "
            "router=%s | reranker=%s | model=%s",
            cfg.config_name, cfg.use_router, cfg.use_reranker, cfg.ollama_model,
        )

        # ── Lazy singleton loading — heavy I/O happens only once ──────────────
        # All three retriever singletons chain through get_hybrid_retriever()
        self._hybrid: HybridRetriever = get_hybrid_retriever()
        self._bm25:   BM25Retriever   = self._hybrid._bm25    # already loaded
        self._faiss:  FAISSRetriever  = self._hybrid._faiss   # already loaded

        self._router: Optional[QueryRouter]          = get_router()   if cfg.use_router   else None
        self._reranker: Optional[CrossEncoderReranker] = get_reranker() if cfg.use_reranker else None

        if cfg.use_router and not self._router.is_loaded:
            log.warning("Router artifact missing — will use fallback routing.")
        if cfg.use_reranker and not self._reranker.is_loaded:
            log.warning("Reranker model unavailable — will fall back to truncation.")

        self._graph = self._build_graph()

        log.info("AdaptiveRAGPipeline ready | config='%s'.", cfg.config_name)

    def _build_graph(self):
        def route_node(state: PipelineGraphState):
            t0 = time.perf_counter()
            routing = self._route(state["query"], state["budget"])
            lat = (time.perf_counter() - t0) * 1000
            
            l = state.get("latency", {}).copy()
            l["routing_ms"] = lat
            
            log.info("Route → label='%s' conf=%.3f fallback=%s", routing.label, routing.confidence, routing.fallback)
            
            return {
                "route_label": routing.label,
                "route_confidence": routing.confidence,
                "route_fallback": routing.fallback,
                "latency": l
            }

        def retrieve_bm25_node(state: PipelineGraphState):
            t0 = time.perf_counter()
            raw = self._bm25.retrieve(state["query"], top_k=self.cfg.top_k_retrieve)
            lat = (time.perf_counter() - t0) * 1000
            l = state.get("latency", {}).copy()
            l["retrieval_ms"] = lat
            log.info("Retriever='BM25' → %d chunks.", len(raw))
            return {"raw_chunks": raw, "retriever_used": "BM25", "latency": l}

        def retrieve_faiss_node(state: PipelineGraphState):
            t0 = time.perf_counter()
            raw = self._hybrid.retrieve(state["query"], top_k=self.cfg.top_k_retrieve)
            lat = (time.perf_counter() - t0) * 1000
            l = state.get("latency", {}).copy()
            l["retrieval_ms"] = lat
            log.info("Retriever='Hybrid(FAISS+BM25)' → %d chunks.", len(raw))
            return {"raw_chunks": raw, "retriever_used": "Hybrid(FAISS+BM25)", "latency": l}

        def direct_llm_node(state: PipelineGraphState):
            l = state.get("latency", {}).copy()
            l["retrieval_ms"] = 0.0
            log.info("Retriever='Direct_LLM' → 0 chunks.")
            return {"raw_chunks": [], "retriever_used": "Direct_LLM", "latency": l}

        def rerank_node(state: PipelineGraphState):
            t0 = time.perf_counter()
            raw = state.get("raw_chunks", [])
            label = state.get("route_label")
            budget = state.get("budget", 1.0)
            
            force_rerank = False
            if budget >= 0.8 and label == "Multi_Hop_FAISS":
                log.info("[ROUTE] High budget path: forcing reranker ON")
                force_rerank = True

            use_reranker = self.cfg.use_reranker or force_rerank

            if label == "Direct_LLM" or not raw:
                final = []
            elif use_reranker and self._reranker is not None and self._reranker.is_loaded:
                final = self._reranker.rerank(state["query"], raw, top_k=self.cfg.top_k_rerank)
            else:
                final = raw[: self.cfg.top_k_rerank]

            lat = (time.perf_counter() - t0) * 1000
            
            l = state.get("latency", {}).copy()
            l["reranking_ms"] = lat
            is_on = "ON" if use_reranker else "OFF"
            log.info("Reranker=%s → %d/%d chunks kept.", is_on, len(final), len(raw))
            
            return {"final_chunks": final, "latency": l}

        def generate_node(state: PipelineGraphState):
            t0 = time.perf_counter()
            prompt = self._build_prompt(state["query"], state.get("final_chunks", []))
            answer, err = self._generate(prompt)
            lat = (time.perf_counter() - t0) * 1000
            
            l = state.get("latency", {}).copy()
            l["generation_ms"] = lat
            
            log.info("Generation complete | %d chars | elapsed generation %.1f ms", len(answer if answer is not None else ""), lat)
            
            return {"answer": answer, "error": err, "latency": l}

        builder = StateGraph(PipelineGraphState)
        builder.add_node("route", route_node)
        builder.add_node("retrieve_bm25", retrieve_bm25_node)
        builder.add_node("retrieve_faiss", retrieve_faiss_node)
        builder.add_node("direct_llm", direct_llm_node)
        builder.add_node("rerank", rerank_node)
        builder.add_node("generate", generate_node)

        builder.add_edge(START, "route")

        def route_condition(state: PipelineGraphState) -> str:
            lbl = state["route_label"]
            if lbl == "Single_Hop_BM25": return "retrieve_bm25"
            if lbl == "Multi_Hop_FAISS": return "retrieve_faiss"
            return "direct_llm"

        builder.add_conditional_edges(
            "route",
            route_condition,
            {
                "retrieve_bm25": "retrieve_bm25",
                "retrieve_faiss": "retrieve_faiss",
                "direct_llm": "direct_llm"
            }
        )

        builder.add_edge("retrieve_bm25", "rerank")
        builder.add_edge("retrieve_faiss", "rerank")
        builder.add_edge("direct_llm", "rerank")
        builder.add_edge("rerank", "generate")
        builder.add_edge("generate", END)

        return builder.compile()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        query:  str,
        budget: float = 1.0,
        save_result: bool = False,
    ) -> PipelineResult:
        """
        Execute the full adaptive RAG pipeline for one query.

        Args:
            query:       Natural language question.
            budget:      System budget in [0.0, 1.0].
            save_result: Persist result JSON to cfg.results_dir.

        Returns:
            PipelineResult — typed, never raises (errors stored in .error field).

        Raises:
            ValueError: If query is empty.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        initial_state = {"query": query, "budget": budget, "latency": {}}
        final_state = self._graph.invoke(initial_state)

        answer = final_state.get("answer") or ""
        gen_error = final_state.get("error")
        latency = final_state.get("latency", {})
        raw_chunks = final_state.get("raw_chunks", [])
        final_chunks = final_state.get("final_chunks", [])

        result = PipelineResult(
            query=query,
            answer=answer,
            config=self.cfg.config_name,
            route_label=final_state.get("route_label", "unknown"),
            route_confidence=final_state.get("route_confidence", 0.0),
            route_fallback=final_state.get("route_fallback", True),
            retriever_used=final_state.get("retriever_used", "unknown"),
            chunks_retrieved=len(raw_chunks),
            chunks_final=len(final_chunks),
            retrieved_chunks=final_chunks,
            latency=latency,
            error=gen_error,
        )

        if save_result:
            self._save_result(result)

        return result

    def run_batch(
        self,
        queries:     list[str],
        budget:      float = 1.0,
        save_result: bool  = True,
    ) -> list[PipelineResult]:
        """
        Run the pipeline over multiple queries.
        Per-query failures are caught and stored in result.error —
        one bad query never aborts a 150-query ablation run.

        Args:
            queries:     List of query strings.
            budget:      System budget level for all queries.
            save_result: Persist aggregated results JSON on completion.

        Returns:
            list[PipelineResult] in the same order as input queries.
        """
        n = len(queries)
        log.info("Batch run | config='%s' | %d queries | budget=%.1f", self.cfg.config_name, n, budget)

        results: list[PipelineResult] = []
        for idx, query in enumerate(queries, 1):
            log.info("  [%d/%d] %s", idx, n, query[:60])
            try:
                result = self.run(query, budget=budget, save_result=False)
            except Exception as exc:  # noqa: BLE001
                log.error("  Query %d failed: %s", idx, exc, exc_info=True)
                result = PipelineResult(
                    query=query,
                    answer=f"[ERROR] {exc}",
                    config=self.cfg.config_name,
                    route_label="unknown",
                    route_confidence=0.0,
                    route_fallback=True,
                    retriever_used="unknown",
                    chunks_retrieved=0,
                    chunks_final=0,
                    retrieved_chunks=[],
                    latency={},
                    error=str(exc),
                )
            results.append(result)

        ok = sum(1 for r in results if r.error is None)
        log.info("Batch complete | %d/%d succeeded.", ok, n)

        if save_result:
            self._save_batch(results)

        return results

    # ── Pipeline phases ────────────────────────────────────────────────────────

    def _route(self, query: str, budget: float) -> RoutingResult:
        """
        Return a RoutingResult. Uses QueryRouter when enabled,
        otherwise returns a synthetic fallback RoutingResult.
        """
        if self.cfg.use_router and self._router is not None:
            return self._router.route_full(query, budget)

        # Router disabled — synthetic fallback result
        from router import VALID_LABELS, _FALLBACK_LABEL  # local import avoids circular
        proba = {lbl: (1.0 if lbl == _FALLBACK_ROUTE else 0.0) for lbl in VALID_LABELS}
        return RoutingResult(
            label=_FALLBACK_ROUTE,
            confidence=1.0,
            probabilities=proba,
            latency_ms=0.0,
            method="disabled",
            fallback=True,
        )

    def _retrieve(
        self, query: str, label: str
    ) -> tuple[list[RetrievedChunk], str]:
        """
        Dispatch to the correct retriever based on routing label.

        Returns:
            (chunks, retriever_name) — chunks is empty for Direct_LLM.
        """
        k = self.cfg.top_k_retrieve

        if label == "Direct_LLM":
            return [], "Direct_LLM"

        if label == "Single_Hop_BM25":
            return self._bm25.retrieve(query, top_k=k), "BM25"

        if label == "Multi_Hop_FAISS":
            # Use full Hybrid (BM25 + FAISS + RRF) for maximum recall
            return self._hybrid.retrieve(query, top_k=k), "Hybrid(FAISS+BM25)"

        # Unknown label — warn and fall back to Hybrid
        log.warning("Unknown routing label '%s' — falling back to Hybrid.", label)
        return self._hybrid.retrieve(query, top_k=k), f"Hybrid(fallback_from={label})"

    def _rerank(
        self,
        query:  str,
        chunks: list[RetrievedChunk],
        label:  str,
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks if enabled. Bypassed for Direct_LLM or empty chunk lists.
        Falls back to truncation if reranker model failed to load.
        """
        if label == "Direct_LLM" or not chunks:
            return []

        if self.cfg.use_reranker and self._reranker is not None and self._reranker.is_loaded:
            return self._reranker.rerank(query, chunks, top_k=self.cfg.top_k_rerank)

        # Reranker disabled or unavailable — simple truncation
        return chunks[: self.cfg.top_k_rerank]

    @staticmethod
    def _build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
        """
        Construct a strict RAG prompt designed to maximise EM and F1.

        Rules enforced:
          - Answer in 1–2 sentences ONLY
          - Use exact wording from context wherever possible
          - No extra explanation or preamble
          - If answer not in context → respond with "Not found in context."
        """
        if not chunks:
            return (
                "You are a precise research assistant.\n"
                "No context was retrieved for this question.\n\n"
                "STRICT RULES:\n"
                "- Answer in 1-2 sentences ONLY.\n"
                "- If you cannot answer from memory, say: Not found in context.\n"
                "- No extra explanation.\n\n"
                f"Question: {query}\n\nAnswer:"
            )

        passages = "\n\n---\n\n".join(
            f"[{i}] (source: {c.source})\n{c.text.strip()}"
            for i, c in enumerate(chunks, 1)
        )
        return (
            "You are a precise research assistant.\n\n"
            "STRICT RULES:\n"
            "- Answer using ONLY the context passages below.\n"
            "- Answer in 1-2 sentences ONLY.\n"
            "- Use exact wording from the context wherever possible.\n"
            "- Do NOT add information not present in the context.\n"
            "- Do NOT include preamble, explanation, or commentary.\n"
            "- If the answer is not in the context, respond with exactly: "
            "Not found in context.\n\n"
            f"Context:\n{passages}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    def _generate(self, prompt: str) -> tuple[str, Optional[str]]:
        """
        Call the local Ollama model via the python client.
        Returns (answer, error_message_or_None).

        All failure modes return a descriptive string instead of raising,
        so run() can always produce a PipelineResult.
        """
        import ollama
        log.debug("Sending prompt to Ollama (%s) ...", self.cfg.ollama_model)
        try:
            response = ollama.chat(
                model=self.cfg.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 400}
            )
            answer = response["message"]["content"].strip()
            
            if not answer:
                log.warning("Ollama returned an empty response.")
                return "[ERROR] Empty response from model.", "empty_response"

            return answer, None

        except ollama.ResponseError as exc:
            log.error("Ollama ResponseError: %s", exc)
            return f"[ERROR] Ollama failed: {exc}", str(exc)

        except Exception as exc:  # noqa: BLE001
            log.error("Unexpected generation error: %s", exc, exc_info=True)
            return f"[ERROR] {exc}", str(exc)

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save_result(self, result: PipelineResult) -> None:
        """Atomically write a single result JSON."""
        self.cfg.results_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self.cfg.config_name}_{int(time.time() * 1000)}.json"
        _atomic_json_write(self.cfg.results_dir / fname, result.to_dict())
        log.info("Result saved → %s", fname)

    def _save_batch(self, results: list[PipelineResult]) -> None:
        """Atomically write all batch results to a single JSON file."""
        self.cfg.results_dir.mkdir(parents=True, exist_ok=True)
        fpath = self.cfg.results_dir / f"{self.cfg.config_name}_results.json"
        _atomic_json_write(fpath, [r.to_dict() for r in results])
        log.info("Batch results saved → %s", fpath)

    def __repr__(self) -> str:
        return (
            f"AdaptiveRAGPipeline(config='{self.cfg.config_name}', "
            f"model='{self.cfg.ollama_model}', "
            f"top_k={self.cfg.top_k_retrieve}/{self.cfg.top_k_rerank})"
        )


# ── Utilities ──────────────────────────────────────────────────────────────────

def _atomic_json_write(path: Path | str, data: object) -> None:
    """Write JSON atomically via tmp → rename to avoid corrupt files on crash."""
    path = Path(path)
    tmp = path.with_suffix(".tmp.json")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── Module-level singleton ─────────────────────────────────────────────────────

_singleton: AdaptiveRAGPipeline | None = None


def get_pipeline(cfg: PipelineConfig = PipelineConfig()) -> AdaptiveRAGPipeline:
    """
    Return a module-level singleton AdaptiveRAGPipeline.

    Streamlit re-executes the script on every UI interaction.
    This singleton ensures FAISS, BM25, cross-encoder, and router
    are all loaded exactly once per process — not once per rerun.

    Note: the singleton is keyed on process lifetime, not cfg equality.
    If you need a different config, call AdaptiveRAGPipeline(cfg) directly.
    """
    global _singleton
    if _singleton is None:
        _singleton = AdaptiveRAGPipeline(cfg)
    return _singleton


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg      = PipelineConfig(use_router=True, use_reranker=True)
    pipeline = AdaptiveRAGPipeline(cfg)
    print(f"\n{pipeline}\n")

    test_queries = [
        ("What is BERT?",                                          1.0),
        ("How does self-attention work in transformers?",          0.5),
        ("Compare BERT and GPT in terms of training objectives.",  0.1),
    ]

    print(f"{'─'*70}")
    for query, budget in test_queries:
        result = pipeline.run(query, budget=budget)
        print(f"\nQuery    : {result.query}")
        print(f"Budget   : {budget:.1f}  →  Route: {result.route_label}  "
              f"(conf={result.route_confidence:.3f}, fallback={result.route_fallback})")
        print(f"Retriever: {result.retriever_used}  "
              f"chunks: {result.chunks_retrieved} → {result.chunks_final}")
        print(f"Latency  : {result.total_latency_ms:.1f} ms total  "
              f"| {json.dumps({k: f'{v:.1f}' for k, v in result.latency.items()})}")
        print(f"Answer   : {result.answer[:200]}...")
        print(f"{'─'*70}")

    # Edge case
    try:
        pipeline.run("")
    except ValueError as exc:
        print(f"\nEmpty query correctly raised ValueError: {exc}")

    print("\nSmoke-test complete.")
    sys.exit(0)
"""
app.py — Adaptive RAG Frontend
Owner: Samved Jain

Streamlit interface for the Budget-Aware Adaptive RAG pipeline.
Integrates AdaptiveRAGPipeline natively — no manual retriever calls,
no raw ollama.chat blocks, no dict-key chunk access.

Modes:
  - Adaptive RAG   : router + reranker fully active
  - Naive RAG      : no router, no reranker, always Hybrid
  - Compare        : side-by-side Adaptive vs Naive with delta summary

All results are typed PipelineResult objects; chunks are RetrievedChunk
dataclasses accessed via .attribute, not ["key"].

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Adaptive RAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f12;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

.rag-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.4rem 0 1rem; border-bottom: 1px solid #1e2530; margin-bottom: 2rem;
}
.rag-logo {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 600;
    letter-spacing: 0.12em; color: #64ffda;
    background: rgba(100,255,218,0.07); border: 1px solid rgba(100,255,218,0.2);
    padding: 4px 12px; border-radius: 4px;
}
.rag-title  { font-size: 1.35rem; font-weight: 500; color: #cbd5e1; letter-spacing: -0.01em; }
.rag-subtitle {
    font-size: 0.78rem; color: #475569;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.04em;
}

.card {
    background: #131820; border: 1px solid #1e2530;
    border-radius: 8px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.card-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; color: #475569; margin-bottom: 0.8rem;
}
.answer-text { font-size: 1.0rem; line-height: 1.75; color: #e2e8f0; font-weight: 300; }

.badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    padding: 3px 10px; border-radius: 3px; text-transform: uppercase;
}
.badge-single_hop_bm25 { background: rgba(59,130,246,0.15);  color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.badge-multi_hop_faiss { background: rgba(168,85,247,0.15);  color: #c084fc; border: 1px solid rgba(168,85,247,0.3); }
.badge-direct_llm      { background: rgba(245,158,11,0.15);  color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-naive           { background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }
.badge-fallback        { background: rgba(239,68,68,0.15);   color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

.chunk-card {
    background: #0d0f12; border: 1px solid #1e2530; border-left: 3px solid #1e2530;
    border-radius: 6px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; transition: border-color 0.2s;
}
.chunk-card:hover { border-left-color: #64ffda; }
.chunk-meta {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #475569;
    margin-bottom: 0.4rem; display: flex; gap: 12px; flex-wrap: wrap;
}
.chunk-text  { font-size: 0.85rem; color: #94a3b8; line-height: 1.6; }
.chunk-score { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #64ffda; }

.latency-row   { display: flex; align-items: center; gap: 12px; margin-bottom: 0.55rem; }
.latency-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    color: #64748b; width: 100px; flex-shrink: 0;
}
.latency-bar-bg  { flex: 1; height: 6px; background: #1e2530; border-radius: 3px; overflow: hidden; }
.latency-bar-fill { height: 100%; border-radius: 3px; }
.latency-ms {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    color: #94a3b8; width: 72px; text-align: right; flex-shrink: 0;
}

section[data-testid="stSidebar"] {
    background: #0a0c0f; border-right: 1px solid #1e2530;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    color: #64748b; letter-spacing: 0.06em; text-transform: uppercase;
}

.compare-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #475569;
    text-transform: uppercase; letter-spacing: 0.1em; text-align: center;
    padding: 0.5rem 0; border-bottom: 1px solid #1e2530; margin-bottom: 1rem;
}
.empty-state  { text-align: center; padding: 3rem 2rem; color: #334155; }
.empty-icon   { font-size: 2.5rem; margin-bottom: 0.8rem; }
.empty-text   { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 0.06em; }

.stTextArea textarea {
    background: #131820 !important; border: 1px solid #1e2530 !important;
    border-radius: 6px !important; color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #64ffda !important; box-shadow: 0 0 0 1px rgba(100,255,218,0.2) !important;
}
.stButton > button {
    background: #64ffda !important; color: #0d0f12 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important;
    font-weight: 600 !important; letter-spacing: 0.08em !important;
    border: none !important; border-radius: 5px !important;
    padding: 0.55rem 1.4rem !important; transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
div[data-testid="stSelectbox"] > div {
    background: #131820 !important; border: 1px solid #1e2530 !important;
    border-radius: 6px !important; color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_pipelines() -> dict:
    """
    Load AdaptiveRAGPipeline singletons for both modes, cached once per process.
    Returns a dict with keys: "adaptive", "naive", "status", "error".

    st.cache_resource ensures this runs exactly once even across Streamlit reruns.
    Heavy components (FAISS, BM25, cross-encoder, router) are loaded via
    module-level singletons inside AdaptiveRAGPipeline — no duplication.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    result: dict = {}

    try:
        from adaptive_pipeline import AdaptiveRAGPipeline, PipelineConfig

        adaptive_cfg = PipelineConfig(use_router=True,  use_reranker=True)
        naive_cfg    = PipelineConfig(use_router=False, use_reranker=False)

        # Adaptive loads first — its singletons are reused by naive
        result["adaptive"] = AdaptiveRAGPipeline(adaptive_cfg)
        result["naive"]    = AdaptiveRAGPipeline(naive_cfg)
        result["status"]   = "live"

    except Exception as exc:
        result["status"] = "demo"
        result["error"]  = str(exc)

    return result


# ── Demo data ─────────────────────────────────────────────────────────────────

def _demo_result(query: str, use_router: bool) -> "PipelineResult":
    """
    Return a plausible typed PipelineResult when the pipeline is unavailable.
    Imports are deferred so the module loads in demo mode without dependencies.
    """
    from faiss_retriever import RetrievedChunk
    from adaptive_pipeline import PipelineResult

    demo_chunks = [
        RetrievedChunk("doc1_chunk_004",
            "BERT is pre-trained using masked language modelling on large text corpora, "
            "enabling it to capture bidirectional context.",
            "devlin2018bert.pdf", 4, 0.912),
        RetrievedChunk("doc1_chunk_007",
            "The transformer architecture relies on self-attention to compute "
            "representations of sequences without recurrence.",
            "vaswani2017attention.pdf", 7, 0.874),
        RetrievedChunk("doc2_chunk_001",
            "Dense retrieval methods encode queries and documents into a shared "
            "embedding space for nearest-neighbour search.",
            "karpukhin2020dpr.pdf", 1, 0.831),
    ]

    return PipelineResult(
        query=query,
        answer="Demo mode — build your FAISS index and start Ollama to see live results.",
        config="full_adaptive" if use_router else "naive",
        route_label="Multi_Hop_FAISS",
        route_confidence=0.94,
        route_fallback=not use_router,
        retriever_used="Hybrid(FAISS+BM25)",
        chunks_retrieved=3,
        chunks_final=3,
        retrieved_chunks=demo_chunks,
        latency={"routing_ms": 1.8, "retrieval_ms": 12.4, "reranking_ms": 44.1, "generation_ms": 284.3},
    )


# ── Query executor ────────────────────────────────────────────────────────────

def _run_query(
    query:    str,
    budget:   float,
    mode:     str,
    pipelines: dict,
) -> dict[str, "PipelineResult"]:
    """
    Execute the query in the requested mode and return a dict of PipelineResults.
    Keys are "adaptive" and/or "naive" depending on mode.
    Never raises — errors are surfaced inside PipelineResult.error.
    """
    is_demo = pipelines.get("status") == "demo"

    if mode == "compare":
        return {
            "adaptive": (
                pipelines["adaptive"].run(query, budget=budget)
                if not is_demo else _demo_result(query, use_router=True)
            ),
            "naive": (
                pipelines["naive"].run(query, budget=budget)
                if not is_demo else _demo_result(query, use_router=False)
            ),
        }

    use_router = (mode == "adaptive")
    key = "adaptive" if use_router else "naive"

    if is_demo:
        return {key: _demo_result(query, use_router=use_router)}

    return {key: pipelines[key].run(query, budget=budget)}


# ── Rendering helpers ─────────────────────────────────────────────────────────

_LATENCY_COLORS = {
    "routing_ms":    "#64ffda",
    "retrieval_ms":  "#38bdf8",
    "reranking_ms":  "#818cf8",
    "generation_ms": "#fb7185",
}
_LATENCY_LABELS = {
    "routing_ms":    "routing",
    "retrieval_ms":  "retrieval",
    "reranking_ms":  "reranking",
    "generation_ms": "generation",
}


def _badge_html(route_label: str, fallback: bool = False) -> str:
    if fallback:
        css = "badge-fallback"
        label = "FALLBACK"
    else:
        css = f"badge-{route_label.lower()}"
        label = route_label
    return f'<span class="badge {css}">{label}</span>'


def render_answer_card(result: "PipelineResult", panel_label: str = "Answer") -> None:
    """Render the answer card for one PipelineResult."""
    badge = _badge_html(result.route_label, result.route_fallback)
    method_hint = (
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;'
        f'color:#475569;margin-left:8px;">conf={result.route_confidence:.3f}</span>'
    )
    error_banner = (
        f'<div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);'
        f'border-radius:4px;padding:6px 10px;margin-bottom:0.7rem;'
        f'font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;color:#f87171;">'
        f'⚠ {result.error}</div>'
        if result.error else ""
    )
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{panel_label}</div>
        <div style="margin-bottom:0.8rem;">{badge}{method_hint}</div>
        {error_banner}
        <div class="answer-text">{result.answer}</div>
    </div>
    """, unsafe_allow_html=True)


def render_chunks(chunks: list["RetrievedChunk"]) -> None:
    """
    Render retrieved chunks. Reads RetrievedChunk attributes (.score, .chunk_id etc.)
    — not dict keys. A missing attribute raises AttributeError immediately, not silently.
    """
    if not chunks:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">◻</div>'
            '<div class="empty-text">No chunks retrieved</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    for i, chunk in enumerate(chunks, 1):
        # Score bars: cosine sims are in [0,1]; CE logits can be negative or >1
        score_pct = int(min(max(chunk.score, 0.0), 1.0) * 100)
        preview   = chunk.text[:320] + ("…" if len(chunk.text) > 320 else "")
        st.markdown(f"""
        <div class="chunk-card">
            <div class="chunk-meta">
                <span>#{i}</span>
                <span>{chunk.chunk_id}</span>
                <span>{chunk.source}</span>
                <span>pos {chunk.position}</span>
                <span class="chunk-score">score {chunk.score:.4f}</span>
            </div>
            <div class="chunk-text">{preview}</div>
        </div>
        """, unsafe_allow_html=True)


def render_latency(result: "PipelineResult") -> None:
    """
    Render latency breakdown from PipelineResult.latency (dict[str, float])
    and PipelineResult.total_latency_ms (computed property).
    """
    latency   = result.latency
    total_ms  = result.total_latency_ms
    max_ms    = max(latency.values(), default=1.0)

    bars_html = ""
    for key in ("routing_ms", "retrieval_ms", "reranking_ms", "generation_ms"):
        ms    = latency.get(key, 0.0)
        pct   = int((ms / max_ms) * 100) if max_ms > 0 else 0
        color = _LATENCY_COLORS.get(key, "#64ffda")
        label = _LATENCY_LABELS.get(key, key)
        bars_html += f"""
        <div class="latency-row">
            <span class="latency-label">{label}</span>
            <div class="latency-bar-bg">
                <div class="latency-bar-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <span class="latency-ms">{ms:.1f} ms</span>
        </div>
        """

    st.markdown(f"""
    <div class="card">
        <div class="card-title">Latency Breakdown</div>
        {bars_html}
        <div style="border-top:1px solid #1e2530;margin-top:0.7rem;padding-top:0.7rem;
                    font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                    display:flex;justify-content:space-between;color:#64ffda;">
            <span>TOTAL</span><span>{total_ms:.1f} ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_router_card(result: "PipelineResult") -> None:
    """Render the router decision summary card."""
    badge = _badge_html(result.route_label, result.route_fallback)
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Router Decision</div>
        <div style="margin-bottom:0.6rem;">{badge}</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#475569;
                    display:flex;flex-direction:column;gap:4px;">
            <span>retriever : {result.retriever_used}</span>
            <span>confidence: {result.route_confidence:.4f}</span>
            <span>fallback  : {result.route_fallback}</span>
            <span>config    : {result.config}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_compare_delta(adaptive: "PipelineResult", naive: "PipelineResult") -> None:
    """Render the latency delta summary bar below the compare columns."""
    a_ms  = adaptive.total_latency_ms
    n_ms  = naive.total_latency_ms
    delta = n_ms - a_ms
    pct   = abs(delta / n_ms * 100) if n_ms else 0.0
    faster = "Adaptive" if delta > 0 else "Naive"

    st.markdown(f"""
    <div class="card" style="margin-top:0.5rem;display:flex;gap:2.5rem;align-items:center;flex-wrap:wrap;">
        <div>
            <div class="card-title">Latency Delta</div>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:1.05rem;color:#64ffda;">
                {faster} faster by {pct:.1f}%
            </span>
        </div>
        <div>
            <div class="card-title">Adaptive Total</div>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:1.0rem;color:#e2e8f0;">
                {a_ms:.1f} ms
            </span>
        </div>
        <div>
            <div class="card-title">Naive Total</div>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:1.0rem;color:#e2e8f0;">
                {n_ms:.1f} ms
            </span>
        </div>
        <div>
            <div class="card-title">Adaptive Route</div>
            {_badge_html(adaptive.route_label, adaptive.route_fallback)}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="rag-header">
        <span class="rag-logo">⬡ RAG</span>
        <div>
            <div class="rag-title">Adaptive Retrieval-Augmented Generation</div>
            <div class="rag-subtitle">LOCAL · OPEN-SOURCE · 8GB VRAM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load pipelines (cached) ───────────────────────────────────────────────
    with st.spinner("Loading pipeline components …"):
        pipelines = _load_pipelines()

    if pipelines["status"] == "demo":
        st.info(
            f"⚡ **Demo mode** — pipeline not fully loaded. "
            f"Build your FAISS index and start Ollama to go live.\n\n"
            f"`{pipelines.get('error', '')}`",
            icon="ℹ️",
        )
    else:
        st.success("Pipeline loaded and ready.", icon="✅")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Settings")

        mode = st.selectbox(
            "Mode",
            options=["adaptive", "naive", "compare"],
            format_func=lambda x: {
                "adaptive": "Adaptive RAG",
                "naive":    "Naive RAG",
                "compare":  "Compare Side-by-Side",
            }[x],
        )

        top_k  = st.slider("Top-K Chunks",    min_value=1,   max_value=10,  value=5)
        budget = st.slider("System Budget",   min_value=0.0, max_value=1.0, value=1.0, step=0.1,
                           help="1.0 = full resources (FAISS+reranker); 0.1 = failsafe (Direct LLM)")

        st.markdown("---")
        st.markdown("### Quick Queries")

        examples = {
            "Factual":    "What dataset was used to evaluate the model?",
            "Conceptual": "How does the attention mechanism work in transformers?",
            "Complex":    "Compare BM25 and dense retrieval across different query types.",
        }
        for label, q in examples.items():
            if st.button(label, key=f"ex_{label}"):
                st.session_state["query_input"] = q

        st.markdown("---")
        st.markdown(
            '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#334155;">'
            'BERT · MiniLM · FAISS · BM25<br>Ollama · phi3:mini · RAGAS'
            '</span>',
            unsafe_allow_html=True,
        )

    # ── Query input ───────────────────────────────────────────────────────────
    query = st.text_area(
        "Query",
        value=st.session_state.get("query_input", ""),
        placeholder="Ask anything about the document corpus …",
        height=90,
        label_visibility="collapsed",
    )

    col_run, col_clear, _ = st.columns([1, 1, 8])
    with col_run:
        run_clicked = st.button("Run Query", type="primary")
    with col_clear:
        if st.button("Clear"):
            st.session_state.pop("query_input", None)
            st.session_state.pop("last_results", None)
            st.session_state.pop("last_mode", None)
            st.rerun()

    # ── Execute ───────────────────────────────────────────────────────────────
    if run_clicked and query.strip():
        with st.spinner("Routing → Retrieving → Reranking → Generating …"):
            results = _run_query(query.strip(), budget, mode, pipelines)
        st.session_state["last_results"] = results
        st.session_state["last_mode"]    = mode

    # ── Render ────────────────────────────────────────────────────────────────
    results   = st.session_state.get("last_results")
    last_mode = st.session_state.get("last_mode", mode)

    if results is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⬡</div>
            <div class="empty-text">Enter a query above and press Run</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Compare layout ────────────────────────────────────────────────────────
    if last_mode == "compare" and "adaptive" in results and "naive" in results:
        adap  = results["adaptive"]
        naive = results["naive"]

        left, right = st.columns(2)

        with left:
            st.markdown('<div class="compare-label">Adaptive RAG</div>', unsafe_allow_html=True)
            render_answer_card(adap, "Answer")
            render_router_card(adap)
            render_latency(adap)
            with st.expander(f"Retrieved Chunks ({adap.chunks_final})"):
                render_chunks(adap.retrieved_chunks)

        with right:
            st.markdown('<div class="compare-label">Naive RAG</div>', unsafe_allow_html=True)
            render_answer_card(naive, "Answer")
            render_router_card(naive)
            render_latency(naive)
            with st.expander(f"Retrieved Chunks ({naive.chunks_final})"):
                render_chunks(naive.retrieved_chunks)

        render_compare_delta(adap, naive)

    # ── Single mode layout ────────────────────────────────────────────────────
    else:
        key    = "adaptive" if last_mode == "adaptive" else "naive"
        result = results.get(key)
        if result is None:
            st.error("No result available for the selected mode.")
            return

        left_col, right_col = st.columns([3, 2])

        with left_col:
            render_answer_card(result)
            with st.expander(f"Retrieved Chunks ({result.chunks_final})"):
                render_chunks(result.retrieved_chunks)

        with right_col:
            render_router_card(result)
            render_latency(result)

            with st.expander("Raw Result (JSON)"):
                export = {
                    "query":            result.query,
                    "config":           result.config,
                    "route_label":      result.route_label,
                    "route_confidence": result.route_confidence,
                    "route_fallback":   result.route_fallback,
                    "retriever_used":   result.retriever_used,
                    "chunks_retrieved": result.chunks_retrieved,
                    "chunks_final":     result.chunks_final,
                    "latency_ms":       result.latency,
                    "total_latency_ms": result.total_latency_ms,
                    "error":            result.error,
                }
                st.code(json.dumps(export, indent=2), language="json")


if __name__ == "__main__":
    main()
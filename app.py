"""
app.py — Adaptive RAG Frontend
Owner: Samved Jain

Streamlit interface for the Budget-Aware Adaptive RAG pipeline.
Run with:
    streamlit run app.py

FIXES APPLIED:
  1. RetrievedChunk import changed from faiss_retriever → base_retriever (correct canonical location)
  2. RetrievedChunk construction switched to kwargs (safe against field-order changes)
  3. _load_pipelines() uses get_pipeline() for adaptive path to avoid double LangGraph compilation
  4. Graceful import-error messages distinguish between missing packages clearly
  5. Added xgboost / sentence-transformers install hint in Demo Mode banner
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eco-RAG | Adaptive Retrieval System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 4rem; max-width: 1500px; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}
.app-header-left { display: flex; align-items: center; gap: 1.2rem; }
.app-logo {
    background: linear-gradient(135deg, #4f8ef7, #7c3aed);
    border-radius: 10px;
    width: 48px; height: 48px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; flex-shrink: 0;
}
.app-title { font-size: 1.4rem; font-weight: 700; color: #f0f4f8; line-height: 1.2; }
.app-subtitle { font-size: 0.78rem; color: #718096; margin-top: 2px; letter-spacing: 0.02em; }
.app-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 500;
    padding: 4px 10px; border-radius: 20px;
    border: 1px solid #4f8ef730;
    background: #4f8ef715;
    color: #4f8ef7;
    white-space: nowrap;
}
.status-pill {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.75rem; color: #a0aec0;
    font-family: 'JetBrains Mono', monospace;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #48bb78; box-shadow: 0 0 6px #48bb7880;
    animation: pulse 2s infinite;
}
.status-dot-warn { background: #f6ad55; box-shadow: 0 0 6px #f6ad5580; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ── Section heading ── */
.section-heading {
    font-size: 0.68rem; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #4a5568; margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2535;
}

/* ── Card ── */
.card {
    background: #141824;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.67rem; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #4a5568; margin-bottom: 0.85rem;
}
.answer-text {
    font-size: 0.97rem; line-height: 1.8;
    color: #cbd5e0; font-weight: 400;
}

/* ── Metric tiles ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.6rem; margin-bottom: 0.3rem;
}
.metric-tile {
    background: #0f1117;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
.metric-tile-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: #4a5568;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
}
.metric-tile-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem; font-weight: 600; color: #e2e8f0;
}

/* ── Route badges ── */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em;
    padding: 4px 10px; border-radius: 5px; text-transform: uppercase;
}
.badge-single_hop_bm25 { background: #2a4a8a20; color: #63b3ed; border: 1px solid #2a4a8a50; }
.badge-multi_hop_faiss { background: #44337a20; color: #b794f4; border: 1px solid #44337a50; }
.badge-direct_llm      { background: #7b341e20; color: #f6ad55; border: 1px solid #7b341e50; }
.badge-naive           { background: #2d374820; color: #a0aec0; border: 1px solid #4a556830; }
.badge-fallback        { background: #63171b20; color: #fc8181; border: 1px solid #63171b50; }

/* ── Chunk cards ── */
.chunk-card {
    background: #0f1117;
    border: 1px solid #2d3748;
    border-left: 3px solid #2d3748;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.55rem;
    transition: border-left-color 0.2s;
}
.chunk-card:hover { border-left-color: #4f8ef7; }
.chunk-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #4a5568;
    margin-bottom: 0.45rem;
    display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
}
.chunk-id   { color: #718096; }
.chunk-src  { color: #4a5568; }
.chunk-score { color: #4f8ef7; font-weight: 500; }
.chunk-text { font-size: 0.84rem; color: #a0aec0; line-height: 1.65; }

/* ── Latency bars ── */
.latency-row { display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem; }
.latency-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: #4a5568; width: 90px; flex-shrink: 0;
}
.latency-bar-bg  { flex: 1; height: 5px; background: #1e2535; border-radius: 3px; overflow: hidden; }
.latency-bar-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
.latency-ms {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: #718096; width: 68px; text-align: right; flex-shrink: 0;
}
.latency-total {
    border-top: 1px solid #2d3748;
    margin-top: 0.6rem; padding-top: 0.6rem;
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}
.latency-total-label { color: #4a5568; text-transform: uppercase; letter-spacing: 0.08em; }
.latency-total-value { color: #4f8ef7; font-weight: 600; }

/* ── Compare mode ── */
.compare-header {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #4a5568;
    padding: 0.6rem 0 1rem;
    border-bottom: 1px solid #2d3748;
    margin-bottom: 1.2rem;
}
.delta-card {
    background: #141824;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-top: 0.5rem;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
}
.delta-item-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: #4a5568;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
}
.delta-item-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.0rem; font-weight: 600; color: #e2e8f0;
}
.delta-item-value.highlight { color: #48bb78; }

/* ── Error banner ── */
.error-banner {
    background: #1a0a0a; border: 1px solid #e53e3e40;
    border-radius: 6px; padding: 8px 12px; margin-bottom: 0.8rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #fc8181;
}

/* ── Install hint box ── */
.install-hint {
    background: #0d1117; border: 1px solid #4f8ef730;
    border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 0.75rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #718096;
    line-height: 1.9;
}
.install-hint code {
    color: #4f8ef7; background: #1a1f2e;
    padding: 1px 6px; border-radius: 3px;
}

/* ── Empty state ── */
.empty-state { text-align: center; padding: 4rem 2rem; color: #2d3748; }
.empty-icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-title { font-size: 1.0rem; font-weight: 600; color: #4a5568; margin-bottom: 0.4rem; }
.empty-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: #2d3748; letter-spacing: 0.04em;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a0c12;
    border-right: 1px solid #1e2535;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important; color: #4a5568 !important;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.sidebar-section {
    padding: 1rem 0; border-bottom: 1px solid #1e2535; margin-bottom: 0.75rem;
}
.sidebar-section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; font-weight: 600;
    color: #2d3748; text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
}
.tech-tag {
    display: inline-block;
    background: #1a1f2e; border: 1px solid #2d3748;
    border-radius: 4px; padding: 2px 7px; margin: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: #4a5568;
}
.example-btn-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: #4a5568;
    text-transform: uppercase; letter-spacing: 0.08em;
}

/* ── Form elements ── */
.stTextArea textarea {
    background: #141824 !important; border: 1px solid #2d3748 !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.95rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #4f8ef7 !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #7c3aed) !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    border: none !important; border-radius: 7px !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
div[data-testid="stSelectbox"] > div {
    background: #141824 !important; border: 1px solid #2d3748 !important;
    border-radius: 7px !important; color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader ────────────────────────────────────────────────────────────

def _diagnose_import_error(exc: Exception) -> str:
    """Return a human-readable diagnosis and fix hint for common import failures."""
    msg = str(exc)
    hints = {
        "xgboost":             "pip install xgboost",
        "sentence_transformers": "pip install sentence-transformers",
        "sentence-transformers": "pip install sentence-transformers",
        "faiss":               "pip install faiss-cpu",
        "langchain":           "pip install langchain langchain-community",
        "langgraph":           "pip install langgraph",
        "rank_bm25":           "pip install rank-bm25",
        "ollama":              "pip install ollama  (and run: ollama serve)",
    }
    for pkg, fix in hints.items():
        if pkg.lower() in msg.lower():
            return f"Missing package — run: `{fix}`\n\nFull error: {msg}"
    return msg


@st.cache_resource(show_spinner=False)
def _load_pipelines() -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    result: dict = {}
    try:
        from adaptive_pipeline import AdaptiveRAGPipeline, PipelineConfig, get_pipeline  # noqa: F401

        # FIX 3: use get_pipeline() for the adaptive path so the LangGraph is only
        # compiled once and the heavy singletons (router, reranker, retriever) are
        # shared rather than instantiated twice.
        result["adaptive"] = get_pipeline(PipelineConfig(use_router=True,  use_reranker=True))
        result["naive"]    = AdaptiveRAGPipeline(PipelineConfig(use_router=False, use_reranker=False))
        result["status"]   = "live"
    except ImportError as exc:
        result["status"] = "demo"
        result["error"]  = _diagnose_import_error(exc)
    except Exception as exc:
        result["status"] = "demo"
        result["error"]  = str(exc)
    return result


# ── Demo data ─────────────────────────────────────────────────────────────────

def _demo_result(query: str, use_router: bool) -> "PipelineResult":
    # FIX 1: import RetrievedChunk from its canonical location (base_retriever),
    # not from faiss_retriever (which only re-exports it and could break).
    try:
        from base_retriever import RetrievedChunk
    except ImportError:
        # Fallback: faiss_retriever re-exports it; accept that if base_retriever
        # is unavailable (e.g. older checkout without the refactor).
        from faiss_retriever import RetrievedChunk  # type: ignore[no-redef]

    from adaptive_pipeline import PipelineResult  # type: ignore[import]

    # FIX 2: use kwargs so this stays correct even if the dataclass gains new
    # fields or changes field order in future refactors.
    demo_chunks = [
        RetrievedChunk(
            chunk_id="doc1_chunk_004",
            text=(
                "BERT is pre-trained using masked language modelling on large text corpora, "
                "enabling it to capture bidirectional context."
            ),
            source="devlin2018bert.pdf",
            score=0.912,
        ),
        RetrievedChunk(
            chunk_id="doc1_chunk_007",
            text=(
                "The transformer architecture relies on self-attention to compute "
                "representations of sequences without recurrence."
            ),
            source="vaswani2017attention.pdf",
            score=0.874,
        ),
        RetrievedChunk(
            chunk_id="doc2_chunk_001",
            text=(
                "Dense retrieval methods encode queries and documents into a shared "
                "embedding space for nearest-neighbour search."
            ),
            source="karpukhin2020dpr.pdf",
            score=0.831,
        ),
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
        latency={
            "routing_ms":    1.8,
            "retrieval_ms":  12.4,
            "reranking_ms":  44.1,
            "generation_ms": 284.3,
        },
    )


# ── Query executor ────────────────────────────────────────────────────────────

def _run_query(query: str, budget: float, mode: str, pipelines: dict) -> dict:
    is_demo = pipelines.get("status") == "demo"
    if mode == "compare":
        return {
            "adaptive": (
                pipelines["adaptive"].run(query, budget=budget)
                if not is_demo else _demo_result(query, True)
            ),
            "naive": (
                pipelines["naive"].run(query, budget=budget)
                if not is_demo else _demo_result(query, False)
            ),
        }
    use_router = (mode == "adaptive")
    key = "adaptive" if use_router else "naive"
    return {
        key: (
            pipelines[key].run(query, budget=budget)
            if not is_demo else _demo_result(query, use_router)
        )
    }


# ── Rendering helpers ─────────────────────────────────────────────────────────

_LATENCY_COLORS = {
    "routing_ms":    "#4f8ef7",
    "retrieval_ms":  "#9f7aea",
    "reranking_ms":  "#f6ad55",
    "generation_ms": "#68d391",
}
_LATENCY_LABELS = {
    "routing_ms":    "Routing",
    "retrieval_ms":  "Retrieval",
    "reranking_ms":  "Reranking",
    "generation_ms": "Generation",
}
_ROUTE_ICONS = {
    "Multi_Hop_FAISS": "🔷",
    "Single_Hop_BM25": "🔹",
    "Direct_LLM":      "🟡",
}


def _badge_html(route_label: str, fallback: bool = False) -> str:
    if fallback:
        return '<span class="badge badge-fallback">⚠ Fallback</span>'
    icon = _ROUTE_ICONS.get(route_label, "◆")
    css  = f"badge-{route_label.lower()}"
    return f'<span class="badge {css}">{icon} {route_label}</span>'


def render_answer_card(result: "PipelineResult", panel_label: str = "Answer") -> None:
    badge    = _badge_html(result.route_label, result.route_fallback)
    conf_html = (
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
        f'color:#4a5568;margin-left:10px;">conf = {result.route_confidence:.3f}</span>'
    )
    error_html = (
        f'<div class="error-banner">⚠ {result.error}</div>'
        if getattr(result, "error", None) else ""
    )
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{panel_label}</div>
        <div style="margin-bottom:0.9rem;">{badge}{conf_html}</div>
        {error_html}
        <div class="answer-text">{result.answer}</div>
    </div>
    """, unsafe_allow_html=True)


def render_router_card(result: "PipelineResult") -> None:
    badge = _badge_html(result.route_label, result.route_fallback)
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Router Decision</div>
        <div style="margin-bottom:0.9rem;">{badge}</div>
        <div class="metric-grid">
            <div class="metric-tile">
                <div class="metric-tile-label">Retriever</div>
                <div class="metric-tile-value" style="font-size:0.78rem;">{result.retriever_used}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Confidence</div>
                <div class="metric-tile-value">{result.route_confidence:.4f}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Config</div>
                <div class="metric-tile-value" style="font-size:0.78rem;">{result.config}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Fallback</div>
                <div class="metric-tile-value" style="color:{'#fc8181' if result.route_fallback else '#68d391'};">
                    {'Yes' if result.route_fallback else 'No'}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_latency(result: "PipelineResult") -> None:
    latency  = result.latency
    total_ms = result.total_latency_ms
    max_ms   = max(latency.values(), default=1.0)

    bars_html = ""
    for key in ("routing_ms", "retrieval_ms", "reranking_ms", "generation_ms"):
        ms    = latency.get(key, 0.0)
        pct   = int((ms / max_ms) * 100) if max_ms > 0 else 0
        color = _LATENCY_COLORS.get(key, "#4f8ef7")
        label = _LATENCY_LABELS.get(key, key)
        bars_html += f"""
        <div class="latency-row">
            <span class="latency-label">{label}</span>
            <div class="latency-bar-bg">
                <div class="latency-bar-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <span class="latency-ms">{ms:.1f} ms</span>
        </div>"""

    st.markdown(f"""
    <div class="card">
        <div class="card-title">Latency Breakdown</div>
        {bars_html}
        <div class="latency-total">
            <span class="latency-total-label">Total End-to-End</span>
            <span class="latency-total-value">{total_ms:.1f} ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_chunks(chunks: list) -> None:
    if not chunks:
        st.markdown(
            '<div class="empty-state" style="padding:1.5rem;">'
            '<div class="empty-text">No chunks retrieved for this query</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    for i, chunk in enumerate(chunks, 1):
        score_pct = int(min(max(chunk.score, 0.0), 1.0) * 100)
        preview   = chunk.text[:340] + ("…" if len(chunk.text) > 340 else "")
        st.markdown(f"""
        <div class="chunk-card">
            <div class="chunk-meta">
                <span style="color:#4f8ef7;font-weight:600;">#{i}</span>
                <span class="chunk-id">{chunk.chunk_id}</span>
                <span class="chunk-src">📄 {chunk.source}</span>
                <span class="chunk-score">score: {chunk.score:.4f}</span>
            </div>
            <div class="chunk-text">{preview}</div>
        </div>
        """, unsafe_allow_html=True)


def render_compare_delta(adaptive: "PipelineResult", naive: "PipelineResult") -> None:
    a_ms   = adaptive.total_latency_ms
    n_ms   = naive.total_latency_ms
    delta  = n_ms - a_ms
    pct    = abs(delta / n_ms * 100) if n_ms else 0.0
    winner = "Adaptive" if delta > 0 else "Naive"
    gain   = f"{winner} is {pct:.1f}% faster"

    st.markdown(f"""
    <div class="delta-card">
        <div>
            <div class="delta-item-label">Latency Advantage</div>
            <div class="delta-item-value highlight">{gain}</div>
        </div>
        <div>
            <div class="delta-item-label">Adaptive Total</div>
            <div class="delta-item-value">{a_ms:.0f} ms</div>
        </div>
        <div>
            <div class="delta-item-label">Naive Total</div>
            <div class="delta-item-value">{n_ms:.0f} ms</div>
        </div>
        <div>
            <div class="delta-item-label">Route Used</div>
            <div style="margin-top:2px;">{_badge_html(adaptive.route_label, adaptive.route_fallback)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── Load pipelines ────────────────────────────────────────────────────────
    with st.spinner("Initialising pipeline …"):
        pipelines = _load_pipelines()

    is_live = pipelines["status"] == "live"

    # ── Header ────────────────────────────────────────────────────────────────
    dot_class    = "status-dot" if is_live else "status-dot status-dot-warn"
    status_label = "Pipeline Live" if is_live else "Demo Mode"

    st.markdown(f"""
    <div class="app-header">
        <div class="app-header-left">
            <div class="app-logo">🔬</div>
            <div>
                <div class="app-title">Eco-RAG &nbsp;·&nbsp; Adaptive Retrieval-Augmented Generation</div>
                <div class="app-subtitle">
                    Budget-aware query routing &nbsp;·&nbsp; Hybrid retrieval (FAISS + BM25) &nbsp;·&nbsp;
                    Cross-encoder reranking &nbsp;·&nbsp; phi3:mini generation
                </div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:1rem;flex-shrink:0;">
            <span class="app-badge">QASPER Corpus</span>
            <div class="status-pill">
                <div class="{dot_class}"></div>
                <span>{status_label}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Demo Mode banner with actionable install hints ─────────────────────────
    if not is_live:
        error_msg = pipelines.get("error", "Unknown error")
        st.warning(
            f"**Demo Mode** — Pipeline components not fully loaded.\n\n`{error_msg}`"
        )
        st.markdown("""
        <div class="install-hint">
            <strong style="color:#e2e8f0;">To go live, run these in your venv:</strong><br>
            <code>pip install xgboost sentence-transformers faiss-cpu rank-bm25 langgraph</code><br>
            <code>python faiss_retriever.py --build-index</code>&nbsp;&nbsp;← builds the FAISS index<br>
            <code>ollama serve</code>&nbsp;&nbsp;&amp;&nbsp;&nbsp;<code>ollama pull phi3:mini</code>&nbsp;&nbsp;← starts the LLM<br>
            <code>streamlit run app.py</code>&nbsp;&nbsp;← restart after installing
        </div>
        """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0 0.5rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                        font-weight:600;color:#718096;letter-spacing:0.06em;">
                ⚙ Configuration
            </div>
        </div>
        """, unsafe_allow_html=True)

        mode = st.selectbox(
            "Query Mode",
            options=["adaptive", "naive", "compare"],
            format_func=lambda x: {
                "adaptive": "🔷  Adaptive RAG  (Router + Reranker)",
                "naive":    "🔹  Naive RAG  (Hybrid, no routing)",
                "compare":  "⚖  Side-by-Side Comparison",
            }[x],
        )

        budget = st.slider(
            "System Budget",
            min_value=0.0, max_value=1.0, value=1.0, step=0.1,
            help=(
                "Controls how much compute the pipeline may spend.\n\n"
                "**1.0** → Full FAISS + Reranker\n"
                "**0.5** → BM25 only\n"
                "**0.1** → Direct LLM (no retrieval)"
            ),
        )

        budget_desc = {
            1.0: "Full (FAISS + Reranker)",
            0.5: "Medium (BM25)",
            0.1: "Minimal (Direct LLM)",
        }
        b_label = budget_desc.get(round(budget, 1), f"Custom ({budget:.1f})")
        st.caption(f"Mode selected: **{b_label}**")

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-section-title">📋 Example Queries</div>
        """, unsafe_allow_html=True)

        examples = {
            "Factual":    "What dataset was used to evaluate the model?",
            "Conceptual": "How does the attention mechanism work in transformers?",
            "Complex":    "Compare BM25 and dense retrieval across different query types.",
        }
        for label, q in examples.items():
            if st.button(f"{label}", key=f"ex_{label}", use_container_width=True):
                st.session_state["query_input"] = q

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-section-title" style="margin-bottom:0.5rem;">🛠 Tech Stack</div>
        <div style="line-height:2.2;">
            <span class="tech-tag">FAISS</span>
            <span class="tech-tag">BM25</span>
            <span class="tech-tag">MiniLM-L6</span>
            <span class="tech-tag">phi3:mini</span>
            <span class="tech-tag">LangGraph</span>
            <span class="tech-tag">Ollama</span>
            <span class="tech-tag">RAGAS</span>
            <span class="tech-tag">SentenceTransformers</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Query input ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Query</div>', unsafe_allow_html=True)

    query = st.text_area(
        "Query",
        value=st.session_state.get("query_input", ""),
        placeholder="Enter a research question about the document corpus …",
        height=95,
        label_visibility="collapsed",
    )

    col_run, col_clear, _ = st.columns([1.2, 1, 7.8])
    with col_run:
        run_clicked = st.button("▶  Run Query", type="primary")
    with col_clear:
        if st.button("✕  Clear"):
            for k in ("query_input", "last_results", "last_mode"):
                st.session_state.pop(k, None)
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
            <div class="empty-icon">🔬</div>
            <div class="empty-title">Ready for a Query</div>
            <div class="empty-text">Enter a question above, or select an example from the sidebar</div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(
        '<div class="section-heading" style="margin-top:1rem;">Results</div>',
        unsafe_allow_html=True,
    )

    # ── Compare layout ────────────────────────────────────────────────────────
    if last_mode == "compare" and "adaptive" in results and "naive" in results:
        adap  = results["adaptive"]
        naive = results["naive"]

        left, right = st.columns(2)

        with left:
            st.markdown(
                '<div class="compare-header">🔷 Adaptive RAG</div>',
                unsafe_allow_html=True,
            )
            render_answer_card(adap, "Answer")
            render_router_card(adap)
            render_latency(adap)
            with st.expander(
                f"Retrieved Context Chunks ({adap.chunks_final})", expanded=False
            ):
                render_chunks(adap.retrieved_chunks)

        with right:
            st.markdown(
                '<div class="compare-header">🔹 Naive RAG (Baseline)</div>',
                unsafe_allow_html=True,
            )
            render_answer_card(naive, "Answer")
            render_router_card(naive)
            render_latency(naive)
            with st.expander(
                f"Retrieved Context Chunks ({naive.chunks_final})", expanded=False
            ):
                render_chunks(naive.retrieved_chunks)

        st.markdown(
            '<div class="section-heading" style="margin-top:0.5rem;">Comparative Summary</div>',
            unsafe_allow_html=True,
        )
        render_compare_delta(adap, naive)

    # ── Single mode layout ────────────────────────────────────────────────────
    else:
        key    = "adaptive" if last_mode == "adaptive" else "naive"
        result = results.get(key)
        if result is None:
            st.error("No result available for this mode.")
            return

        left_col, right_col = st.columns([3, 2])

        with left_col:
            render_answer_card(result)
            with st.expander(
                f"Retrieved Context Chunks ({result.chunks_final})", expanded=False
            ):
                render_chunks(result.retrieved_chunks)

        with right_col:
            render_router_card(result)
            render_latency(result)

            with st.expander("Export — Raw JSON Result", expanded=False):
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
                    "error":            getattr(result, "error", None),
                }
                st.code(json.dumps(export, indent=2), language="json")


if __name__ == "__main__":
    main()
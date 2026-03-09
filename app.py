"""
app.py — Adaptive RAG Frontend
Owner: Samved Jain

Streamlit interface for querying the Adaptive RAG pipeline.
Supports:
  - Single query inference (Adaptive or Naive mode)
  - Side-by-side Naive vs Adaptive comparison
  - Router decision visibility
  - Retrieved chunks with scores
  - Per-stage latency breakdown

Run with:
    streamlit run app.py
"""

import json
import time
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Adaptive RAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f12;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* ── Header bar ── */
.rag-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.4rem 0 1rem;
    border-bottom: 1px solid #1e2530;
    margin-bottom: 2rem;
}
.rag-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #64ffda;
    background: rgba(100,255,218,0.07);
    border: 1px solid rgba(100,255,218,0.2);
    padding: 4px 12px;
    border-radius: 4px;
}
.rag-title {
    font-size: 1.35rem;
    font-weight: 500;
    color: #cbd5e1;
    letter-spacing: -0.01em;
}
.rag-subtitle {
    font-size: 0.78rem;
    color: #475569;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.04em;
}

/* ── Cards ── */
.card {
    background: #131820;
    border: 1px solid #1e2530;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.8rem;
}

/* ── Answer block ── */
.answer-text {
    font-size: 1.0rem;
    line-height: 1.75;
    color: #e2e8f0;
    font-weight: 300;
}

/* ── Route badge ── */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 3px;
    text-transform: uppercase;
}
.badge-factual    { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.badge-conceptual { background: rgba(168,85,247,0.15); color: #c084fc; border: 1px solid rgba(168,85,247,0.3); }
.badge-complex    { background: rgba(245,158,11,0.15);  color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-naive      { background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }

/* ── Chunk cards ── */
.chunk-card {
    background: #0d0f12;
    border: 1px solid #1e2530;
    border-left: 3px solid #1e2530;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.chunk-card:hover { border-left-color: #64ffda; }
.chunk-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #475569;
    margin-bottom: 0.4rem;
    display: flex;
    gap: 12px;
}
.chunk-text {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.6;
}
.chunk-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #64ffda;
}

/* ── Latency bars ── */
.latency-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.55rem;
}
.latency-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    width: 90px;
    flex-shrink: 0;
}
.latency-bar-bg {
    flex: 1;
    height: 6px;
    background: #1e2530;
    border-radius: 3px;
    overflow: hidden;
}
.latency-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #64ffda, #38bdf8);
}
.latency-ms {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
    width: 68px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a0c0f;
    border-right: 1px solid #1e2530;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stToggle label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Compare divider ── */
.compare-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    text-align: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1e2530;
    margin-bottom: 1rem;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #334155;
}
.empty-icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
}
.empty-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
}

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: #131820 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #64ffda !important;
    box-shadow: 0 0 0 1px rgba(100,255,218,0.2) !important;
}
.stButton > button {
    background: #64ffda !important;
    color: #0d0f12 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
div[data-testid="stSelectbox"] > div {
    background: #131820 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader (cached so models load once) ──────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Load all pipeline components once and cache them in session.

    Returns a dict with keys: faiss, bm25, hybrid, router, chunks_loaded
    Falls back gracefully if pipeline files are not yet present (demo mode).
    """
    components = {}

    try:
        sys.path.insert(0, str(Path(__file__).parent))

        from faiss_retriever import FAISSRetriever
        from router import QueryRouter

        router = QueryRouter()
        components["router"] = router

        faiss = FAISSRetriever()
        index_path = "data/faiss_index/index.faiss"
        meta_path  = "data/faiss_index/metadata.pkl"

        if Path(index_path).exists():
            faiss.load(index_path, meta_path)
            components["faiss"] = faiss
            components["chunks_loaded"] = True
        else:
            components["faiss"] = faiss
            components["chunks_loaded"] = False

        # BM25 and Hybrid (optional — depend on Nivi's module)
        try:
            from bm25_retriever import BM25Retriever
            from hybrid_retriever import HybridRetriever
            components["bm25"] = None       # requires built index
            components["hybrid"] = None
        except ImportError:
            pass

        components["status"] = "live"

    except Exception as e:
        components["status"] = "demo"
        components["error"] = str(e)

    return components


def run_query(query: str, mode: str, top_k: int, pipeline: dict) -> dict:
    """
    Run a query through the pipeline and return structured results.

    Args:
        query:    User query string.
        mode:     "adaptive" | "naive" | "compare"
        top_k:    Number of chunks to retrieve.
        pipeline: Loaded pipeline components dict.

    Returns:
        Result dict with keys: answer, query_type, method, chunks, timings, total_ms
    """
    if pipeline.get("status") == "demo":
        return _demo_result(query, mode)

    import ollama

    results = {}

    def _run_single(pipeline_mode: str) -> dict:
        timings = {}
        chunks = []
        query_type = "conceptual"
        route_method = "—"

        # ── Routing ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        if pipeline_mode == "adaptive" and "router" in pipeline:
            query_type, confidence, route_method = pipeline["router"].classify(query)
        else:
            query_type = "conceptual"   # naive always uses dense
        timings["routing"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Retrieval ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        if pipeline.get("chunks_loaded"):
            if pipeline_mode == "naive" or query_type == "conceptual":
                chunks = pipeline["faiss"].retrieve(query, top_k=top_k)
            elif query_type == "factual" and pipeline.get("bm25"):
                chunks = pipeline["bm25"].retrieve(query, top_k=top_k)
            elif query_type == "complex" and pipeline.get("hybrid"):
                chunks = pipeline["hybrid"].retrieve(query, top_k=top_k)
            else:
                chunks = pipeline["faiss"].retrieve(query, top_k=top_k)
        timings["retrieval"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Reranking (adaptive only — stub if reranker not present) ─
        t0 = time.perf_counter()
        timings["reranking"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Generation ────────────────────────────────────────────────
        context = "\n\n".join(c["text"] for c in chunks) if chunks else "No context available."
        prompt = f"""Answer the following question based only on the provided context.
Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""

        t0 = time.perf_counter()
        try:
            resp = ollama.chat(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            answer = resp["message"]["content"].strip()
        except Exception as e:
            answer = f"[LLM error: {e}]"
        timings["generation"] = round((time.perf_counter() - t0) * 1000, 2)

        total_ms = round(sum(timings.values()), 2)

        return {
            "answer": answer,
            "query_type": query_type,
            "method": route_method,
            "chunks": chunks,
            "timings": timings,
            "total_ms": total_ms,
            "pipeline_mode": pipeline_mode,
        }

    if mode == "compare":
        results["adaptive"] = _run_single("adaptive")
        results["naive"]    = _run_single("naive")
        return results
    else:
        return _run_single(mode)


def _demo_result(query: str, mode: str) -> dict:
    """Return plausible-looking demo data when pipeline isn't loaded."""
    demo_chunks = [
        {"chunk_id": "doc1_chunk_004", "text": "BERT is pre-trained using masked language modelling on large text corpora, enabling it to capture bidirectional context.", "source": "devlin2018bert.pdf", "position": 4, "score": 0.912},
        {"chunk_id": "doc1_chunk_007", "text": "The transformer architecture relies on self-attention to compute representations of sequences without recurrence.", "source": "vaswani2017attention.pdf", "position": 7, "score": 0.874},
        {"chunk_id": "doc2_chunk_001", "text": "Dense retrieval methods encode queries and documents into a shared embedding space for nearest-neighbour search.", "source": "karpukhin2020dpr.pdf", "position": 1, "score": 0.831},
    ]

    single = {
        "answer": "This is a demo response. Load your FAISS index and start Ollama to see live results.",
        "query_type": "conceptual",
        "method": "embedding",
        "chunks": demo_chunks,
        "timings": {"routing": 1.8, "retrieval": 12.4, "reranking": 0.0, "generation": 284.3},
        "total_ms": 298.5,
        "pipeline_mode": mode,
    }

    if mode == "compare":
        naive = dict(single)
        naive["query_type"] = "conceptual"
        naive["method"] = "—"
        naive["timings"] = {"routing": 0.0, "retrieval": 18.2, "reranking": 0.0, "generation": 301.1}
        naive["total_ms"] = 319.3
        naive["pipeline_mode"] = "naive"
        return {"adaptive": single, "naive": naive}

    return single


# ── UI helpers ────────────────────────────────────────────────────────────

def render_badge(query_type: str) -> str:
    css_class = f"badge-{query_type}" if query_type in ("factual", "conceptual", "complex") else "badge-naive"
    return f'<span class="badge {css_class}">{query_type}</span>'


def render_answer_card(result: dict, label: str = "") -> None:
    badge_html = render_badge(result["query_type"])
    method_html = f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#475569;margin-left:8px;">via {result["method"]}</span>'

    st.markdown(f"""
    <div class="card">
        <div class="card-title">{label or "Answer"}</div>
        <div style="margin-bottom:0.8rem;">{badge_html}{method_html}</div>
        <div class="answer-text">{result["answer"]}</div>
    </div>
    """, unsafe_allow_html=True)


def render_chunks(chunks: list) -> None:
    if not chunks:
        st.markdown('<div class="empty-state"><div class="empty-icon">◻</div><div class="empty-text">No chunks retrieved</div></div>', unsafe_allow_html=True)
        return

    for i, chunk in enumerate(chunks):
        score_bar_pct = min(int(chunk["score"] * 100), 100) if chunk["score"] <= 1 else min(int(chunk["score"] / 20 * 100), 100)
        st.markdown(f"""
        <div class="chunk-card">
            <div class="chunk-meta">
                <span>#{i+1}</span>
                <span>{chunk["chunk_id"]}</span>
                <span>{chunk["source"]}</span>
                <span>pos {chunk["position"]}</span>
                <span class="chunk-score">score {chunk["score"]:.4f}</span>
            </div>
            <div class="chunk-text">{chunk["text"][:320]}{"…" if len(chunk["text"]) > 320 else ""}</div>
        </div>
        """, unsafe_allow_html=True)


def render_latency(timings: dict, total_ms: float) -> None:
    max_ms = max(timings.values()) if timings else 1
    stages = ["routing", "retrieval", "reranking", "generation"]
    colors = {
        "routing":    "#64ffda",
        "retrieval":  "#38bdf8",
        "reranking":  "#818cf8",
        "generation": "#fb7185",
    }

    bars_html = ""
    for stage in stages:
        ms = timings.get(stage, 0.0)
        pct = int((ms / max_ms) * 100) if max_ms > 0 else 0
        color = colors.get(stage, "#64ffda")
        bars_html += f"""
        <div class="latency-row">
            <span class="latency-label">{stage}</span>
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


# ── Main app ──────────────────────────────────────────────────────────────

def main():

    # ── Header ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="rag-header">
        <span class="rag-logo">⬡ RAG</span>
        <div>
            <div class="rag-title">Adaptive Retrieval-Augmented Generation</div>
            <div class="rag-subtitle">LOCAL · OPEN-SOURCE · 8GB VRAM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load pipeline ─────────────────────────────────────────────────
    with st.spinner("Loading pipeline components …"):
        pipeline = load_pipeline()

    status = pipeline.get("status", "demo")
    if status == "demo":
        st.info(f"⚡ **Demo mode** — pipeline not fully loaded. Build your FAISS index and start Ollama to go live.  \n`{pipeline.get('error','')}`", icon="ℹ️")
    else:
        st.success("Pipeline loaded.", icon="✅")

    # ── Sidebar ───────────────────────────────────────────────────────
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

        top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5)

        st.markdown("---")
        st.markdown("### Quick Test Queries")

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
            'BERT · MiniLM · FAISS · BM25<br>Ollama · llama3.2:3b · RAGAS'
            '</span>',
            unsafe_allow_html=True,
        )

    # ── Query input ───────────────────────────────────────────────────
    query = st.text_area(
        "Query",
        value=st.session_state.get("query_input", ""),
        placeholder="Ask anything about the document corpus …",
        height=90,
        label_visibility="collapsed",
    )

    col_btn, col_clear, _ = st.columns([1, 1, 8])
    with col_btn:
        run = st.button("Run Query", type="primary")
    with col_clear:
        if st.button("Clear"):
            st.session_state["query_input"] = ""
            st.session_state.pop("last_result", None)
            st.rerun()

    # ── Execute ───────────────────────────────────────────────────────
    if run and query.strip():
        with st.spinner("Retrieving and generating …"):
            result = run_query(query.strip(), mode, top_k, pipeline)
        st.session_state["last_result"] = result
        st.session_state["last_mode"] = mode

    # ── Results ───────────────────────────────────────────────────────
    result = st.session_state.get("last_result")
    last_mode = st.session_state.get("last_mode", mode)

    if result is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⬡</div>
            <div class="empty-text">Enter a query above and press Run</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Compare layout ────────────────────────────────────────────────
    if last_mode == "compare" and isinstance(result, dict) and "adaptive" in result:
        left, right = st.columns(2)

        with left:
            st.markdown('<div class="compare-label">Adaptive RAG</div>', unsafe_allow_html=True)
            render_answer_card(result["adaptive"], label="Answer")
            render_latency(result["adaptive"]["timings"], result["adaptive"]["total_ms"])

            with st.expander(f"Retrieved Chunks ({len(result['adaptive']['chunks'])})"):
                render_chunks(result["adaptive"]["chunks"])

        with right:
            st.markdown('<div class="compare-label">Naive RAG</div>', unsafe_allow_html=True)
            render_answer_card(result["naive"], label="Answer")
            render_latency(result["naive"]["timings"], result["naive"]["total_ms"])

            with st.expander(f"Retrieved Chunks ({len(result['naive']['chunks'])})"):
                render_chunks(result["naive"]["chunks"])

        # ── Delta summary ─────────────────────────────────────────────
        adap_total = result["adaptive"]["total_ms"]
        naive_total = result["naive"]["total_ms"]
        delta = naive_total - adap_total
        delta_pct = abs(delta / naive_total * 100) if naive_total else 0
        faster_label = "Adaptive" if delta > 0 else "Naive"

        st.markdown(f"""
        <div class="card" style="margin-top:0.5rem;display:flex;gap:2rem;align-items:center;">
            <div>
                <div class="card-title">Latency Delta</div>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;color:#64ffda;">
                    {faster_label} faster by {delta_pct:.1f}%
                </span>
            </div>
            <div>
                <div class="card-title">Adaptive Total</div>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:1.0rem;color:#e2e8f0;">
                    {adap_total:.1f} ms
                </span>
            </div>
            <div>
                <div class="card-title">Naive Total</div>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:1.0rem;color:#e2e8f0;">
                    {naive_total:.1f} ms
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Single mode layout ────────────────────────────────────────────
    else:
        left_col, right_col = st.columns([3, 2])

        with left_col:
            render_answer_card(result)

            with st.expander(f"Retrieved Chunks ({len(result.get('chunks', []))})"):
                render_chunks(result.get("chunks", []))

        with right_col:
            render_latency(result.get("timings", {}), result.get("total_ms", 0))

            st.markdown(f"""
            <div class="card">
                <div class="card-title">Router Decision</div>
                <div style="margin-bottom:0.5rem;">{render_badge(result.get("query_type","—"))}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#475569;">
                    method: {result.get("method","—")}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Raw JSON export
            with st.expander("Raw Result JSON"):
                export = {k: v for k, v in result.items() if k != "chunks"}
                st.code(json.dumps(export, indent=2), language="json")


if __name__ == "__main__":
    main()
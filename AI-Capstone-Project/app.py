# app.py
"""
University Handbook Search System
──────────────────────────────────
Full pipeline: Upload → Index → Search → Explain → Evaluate → RAG (Phi-3)
UI: Dark black + purple theme
"""

import os
import sys
import re
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Internal modules ──────────────────────────────────────────────────────────
from loaders.document_loader    import (
    load_documents_from_uploaded,
    load_documents_from_folder,
)
from preprocessing.text_preprocessor import preprocess, preprocess_query
from indexing.inverted_index          import InvertedIndex
from retrieval.retriever              import KeywordRetriever, LangChainKeywordRetriever
from evaluation.evaluator             import run_evaluation, EVALUATION_QUERIES
from agent.search_coach               import SearchCoach
from rag.answer_generator             import AnswerGenerator, check_ollama_available
from utils.session_state              import (
    init_session_state, get, set as ss_set,
    DOCS_KEY, INDEX_KEY, RETRIEVER_KEY, LC_RETRIEVER_KEY,
    RESULTS_KEY, QUERY_KEY, COACH_KEY, EVAL_KEY, ANSWER_KEY,
    LEMMATIZE_KEY, INDEXED_KEY,
)
from utils.logger                      import log_query

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="UniSearch — Handbook IR System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — Black + Purple Theme ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font import ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:   #0a0a0f;
    --bg-secondary: #111118;
    --bg-card:      #16161f;
    --bg-card-alt:  #1c1c28;
    --accent:       #7c3aed;
    --accent-light: #9d5af0;
    --accent-dim:   #3d1e7a;
    --accent-glow:  rgba(124,58,237,0.25);
    --text-primary: #f0eeff;
    --text-secondary: #a89ec8;
    --text-muted:   #6b6485;
    --border:       #2a2a3a;
    --border-accent:#7c3aed44;
    --green:        #10b981;
    --red:          #ef4444;
    --yellow:       #f59e0b;
    --blue:         #3b82f6;
    --radius:       12px;
    --radius-sm:    8px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0a1a 50%, #0a0a0f 100%) !important;
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.2s ease;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.4rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    background: var(--accent-light) !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-accent) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── Selectbox / Radio ── */
.stSelectbox > div > div,
.stRadio > div {
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stSelectbox"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* ── Sliders ── */
.stSlider > div > div > div {
    background: var(--accent) !important;
}

/* ── Toggles ── */
.stToggle > div {
    color: var(--text-primary) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-light) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card-alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── Dataframe / Tables ── */
.stDataFrame, [data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

/* ── Code blocks ── */
code {
    background: var(--bg-card-alt) !important;
    color: var(--accent-light) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em !important;
}
pre {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 1rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--accent-dim); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Alert boxes ── */
.stAlert { border-radius: var(--radius-sm) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Custom card classes (used via st.markdown) ── */
.uni-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.uni-card-accent {
    background: var(--bg-card);
    border: 1px solid var(--accent);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 0 20px var(--accent-glow);
}
.uni-card-green {
    background: var(--bg-card);
    border: 1px solid #10b98144;
    border-left: 4px solid #10b981;
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.uni-card-yellow {
    background: var(--bg-card);
    border: 1px solid #f59e0b44;
    border-left: 4px solid #f59e0b;
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.uni-card-red {
    background: var(--bg-card);
    border: 1px solid #ef444444;
    border-left: 4px solid #ef4444;
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-light);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.badge {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent-light);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}
.badge-green {
    background: #10b98120;
    color: #10b981;
}
.badge-red {
    background: #ef444420;
    color: #ef4444;
}
.badge-yellow {
    background: #f59e0b20;
    color: #f59e0b;
}
.score-pill {
    display: inline-block;
    background: var(--accent);
    color: white;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.82rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--accent-dim);
    color: var(--accent-light);
    border-radius: 50%;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 8px;
}
.highlight-term {
    background: rgba(124,58,237,0.3);
    color: #c4b5fd;
    border-radius: 3px;
    padding: 1px 5px;
    font-weight: 600;
}
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 0.5rem 0;
}
.metric-box {
    background: var(--bg-card-alt);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.7rem 1.2rem;
    text-align: center;
    min-width: 100px;
    flex: 1;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent-light);
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.coach-card {
    background: linear-gradient(135deg, #1a1028 0%, #16161f 100%);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px var(--accent-glow);
}
.answer-card {
    background: linear-gradient(135deg, #12101e 0%, #1a1028 100%);
    border: 2px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(124,58,237,0.3);
}
.answer-text {
    font-size: 1rem;
    line-height: 1.75;
    color: var(--text-primary);
}
.source-chip {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent-light);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 3px;
    border: 1px solid var(--border-accent);
}
.pipeline-step {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--bg-card-alt);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin: 3px;
}
.pipeline-step.active {
    background: var(--accent-dim);
    border-color: var(--accent);
    color: var(--accent-light);
}
.sidebar-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

SAMPLE_DOCS_PATH = os.path.join(ROOT, "sample_docs")


@st.cache_data(ttl=30)
def _cached_ollama_check():
    """Cached check for Ollama availability (TTL: 30 seconds)."""
    return check_ollama_available()


def _build_index(documents, use_lemmatization: bool):
    """Pre-process documents and build inverted index."""
    tokens_per_doc = [
        preprocess(doc["text"], use_lemmatization=use_lemmatization)
        for doc in documents
    ]
    index = InvertedIndex()
    index.build(documents, tokens_per_doc)
    retriever    = KeywordRetriever(index, use_lemmatization=use_lemmatization)
    lc_retriever = LangChainKeywordRetriever(retriever, top_k=5)
    return index, retriever, lc_retriever


def _card(content: str, style: str = "") -> None:
    css_class = {
        "accent":  "uni-card-accent",
        "green":   "uni-card-green",
        "yellow":  "uni-card-yellow",
        "red":     "uni-card-red",
        "coach":   "coach-card",
        "answer":  "answer-card",
    }.get(style, "uni-card")
    st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)


def _section_title(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-title">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


# ── Page header ───────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div style="
        padding: 2rem 0 1.5rem;
        border-bottom: 1px solid #2a2a3a;
        margin-bottom: 2rem;
    ">
        <div style="display:flex; align-items:center; gap:1rem;">
            <div style="
                width:52px; height:52px;
                background: linear-gradient(135deg,#7c3aed,#4f46e5);
                border-radius:14px;
                display:flex; align-items:center; justify-content:center;
                font-size:1.6rem;
                box-shadow: 0 4px 20px rgba(124,58,237,0.4);
            ">🎓</div>
            <div>
                <h1 style="margin:0; font-size:1.7rem; font-weight:800;
                           background:linear-gradient(135deg,#c4b5fd,#7c3aed);
                           -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                    University Handbook Search
                </h1>
                <p style="margin:0; color:#6b6485; font-size:0.85rem; letter-spacing:0.06em; text-transform:uppercase;">
                    IR · TF-IDF · LangChain Retriever · Phi-3 RAG
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:0.5rem 0 1.5rem;">
            <div style="font-size:2rem;">🎓</div>
            <div style="font-weight:700; font-size:1rem; color:#c4b5fd;">UniSearch</div>
            <div style="font-size:0.72rem; color:#6b6485; letter-spacing:0.08em; text-transform:uppercase;">
                Handbook IR System
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Active mode selector ──────────────────────────────────────────
        mode = st.radio(
            "Navigation",
            ["📤 Upload & Index", "🔍 Search", "📊 Evaluation"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ── Preprocessing toggles ─────────────────────────────────────────
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**⚙️ Preprocessing**")
        lemmatize = st.toggle(
            "Lemmatization",
            value=get(LEMMATIZE_KEY),
            key="toggle_lemmatize",
            help="Apply WordNet lemmatization during indexing and query processing.",
        )
        
        # Warn if lemmatize setting changes after index is built
        if get(INDEXED_KEY) and lemmatize != get(LEMMATIZE_KEY):
            st.warning("Lemmatization setting changed — please rebuild the index for consistent results.")
        
        ss_set(LEMMATIZE_KEY, lemmatize)

        st.toggle("Stopword Removal", value=True, disabled=True,
                  help="Always enabled — removes common English stopwords.")
        st.toggle("Tokenization", value=True, disabled=True,
                  help="Always enabled — splits text into tokens.")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Index stats ───────────────────────────────────────────────────
        index: InvertedIndex = get(INDEX_KEY)
        if index and get(INDEXED_KEY):
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**📈 Index Stats**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Docs", index.num_docs)
            with col2:
                st.metric("Vocab", f"{index.vocab_size:,}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Ollama status ─────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**🤖 Ollama / Phi-3**")
        ollama_ok, ollama_model = _cached_ollama_check()
        if ollama_ok:
            st.markdown(f'<span class="badge badge-green">● Online</span> '
                        f'<code style="font-size:0.75rem;">{ollama_model}</code>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-red">● Offline</span>', unsafe_allow_html=True)
            st.caption("Run `ollama serve` to enable RAG generation.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.72rem; color:#6b6485; text-align:center;">'
            'IR Pipeline · TF-IDF · Phi-3 RAG<br>Built with Streamlit</div>',
            unsafe_allow_html=True,
        )

    return mode


# ── Tab 1: Upload & Index ─────────────────────────────────────────────────────

def render_upload_tab():
    render_header()
    _section_title("📤", "Document Upload & Indexing")

    col_up, col_info = st.columns([3, 2], gap="large")

    with col_up:
        _card("""
        <div class="section-title">📂 Upload Documents</div>
        <p style="color:#a89ec8; font-size:0.88rem; margin:0 0 0.8rem;">
          Supported: <strong>PDF · DOCX · TXT · HTML</strong> — Upload 5–10 university documents.
        </p>
        """, "accent")

        uploaded = st.file_uploader(
            "Drop files here",
            type=["pdf", "docx", "txt", "html", "htm"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        st.markdown("**— or —**")
        use_sample = st.button("📚 Load Sample Dataset (7 documents)", use_container_width=True)

        if use_sample:
            if os.path.exists(SAMPLE_DOCS_PATH):
                with st.spinner("Loading sample documents..."):
                    docs = load_documents_from_folder(SAMPLE_DOCS_PATH)
                if docs:
                    ss_set(DOCS_KEY, docs)
                    ss_set(INDEXED_KEY, False)
                    st.success(f"✅ Loaded {len(docs)} sample documents.")
                else:
                    st.error("No sample documents found.")
            else:
                st.error(f"Sample docs folder not found: {SAMPLE_DOCS_PATH}")

        if uploaded:
            with st.spinner(f"Extracting text from {len(uploaded)} file(s)..."):
                docs = load_documents_from_uploaded(uploaded)
            if docs:
                ss_set(DOCS_KEY, docs)
                ss_set(INDEXED_KEY, False)
                st.success(f"✅ Loaded {len(docs)} document(s).")

    with col_info:
        docs: list = get(DOCS_KEY, [])
        if docs:
            _section_title("📋", "Loaded Documents")
            for doc in docs:
                cat_colors = {
                    "Academic": "#7c3aed", "Fees & Finance": "#10b981",
                    "Hostel": "#3b82f6", "Library": "#f59e0b",
                    "Admissions": "#ec4899", "Syllabus": "#06b6d4",
                    "Student Welfare": "#8b5cf6", "General": "#6b7280",
                }
                color = cat_colors.get(doc.get("category", "General"), "#6b7280")
                st.markdown(f"""
                <div style="background:#16161f; border:1px solid #2a2a3a; border-left:3px solid {color};
                            border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.5rem;">
                    <div style="font-weight:600; font-size:0.88rem; color:#f0eeff;">
                        📄 {doc['filename']}
                    </div>
                    <div style="margin-top:3px; display:flex; gap:0.5rem; flex-wrap:wrap;">
                        <span style="background:{color}22; color:{color}; border-radius:4px;
                                     padding:1px 8px; font-size:0.72rem; font-weight:600;">
                            {doc['category']}
                        </span>
                        <span style="color:#6b6485; font-size:0.72rem;">
                            {doc['char_count']:,} chars
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Build Index ───────────────────────────────────────────────────────────
    docs: list = get(DOCS_KEY, [])
    if docs:
        st.markdown("---")
        col_btn, col_status = st.columns([2, 3])
        with col_btn:
            if st.button("⚡ Build Inverted Index", use_container_width=True, type="primary"):
                use_lemmatize = get(LEMMATIZE_KEY)
                with st.spinner("Building index..."):
                    index, retriever, lc_retriever = _build_index(docs, use_lemmatize)
                ss_set(INDEX_KEY,        index)
                ss_set(RETRIEVER_KEY,    retriever)
                ss_set(LC_RETRIEVER_KEY, lc_retriever)
                ss_set(INDEXED_KEY,      True)
                st.success("✅ Index built successfully!")

        with col_status:
            if get(INDEXED_KEY):
                index: InvertedIndex = get(INDEX_KEY)
                st.markdown(f"""
                <div style="display:flex; gap:1rem; flex-wrap:wrap; padding-top:0.3rem;">
                    <div class="metric-box">
                        <div class="metric-value">{index.num_docs}</div>
                        <div class="metric-label">Documents</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{index.vocab_size:,}</div>
                        <div class="metric-label">Vocab Terms</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">TF-IDF</div>
                        <div class="metric-label">Scoring</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Timing Comparison ─────────────────────────────────────────────────────
    if get(INDEXED_KEY):
        st.markdown("---")
        _section_title("⏱️", "Performance Timing")
        
        if st.button("Run timing comparison", use_container_width=True):
            test_query = st.session_state.get("last_query", "student fees")
            import time
            retriever: KeywordRetriever = get(RETRIEVER_KEY)
            
            # Index-based retrieval timing
            t0 = time.perf_counter()
            _ = retriever.retrieve(test_query)
            index_time = time.perf_counter() - t0
            
            st.success(f"Index lookup: {index_time*1000:.2f}ms")

    # ── Pipeline diagram ──────────────────────────────────────────────────────
    st.markdown("---")
    _section_title("🔄", "System Pipeline")
    steps = [
        ("📂", "Upload"),
        ("📝", "Extract"),
        ("⚙️",  "Preprocess"),
        ("🗂️",  "Index"),
        ("🔍", "Search"),
        ("📊", "Rank"),
        ("💡", "Explain"),
        ("📏", "Evaluate"),
        ("🔗", "LangChain"),
        ("🤖", "Phi-3 RAG"),
    ]
    is_indexed = get(INDEXED_KEY)
    steps_html = ""
    for i, (icon, name) in enumerate(steps):
        active = "active" if is_indexed or i < 3 else ""
        steps_html += f'<span class="pipeline-step {active}">{icon} {name}</span>'
        if i < len(steps) - 1:
            steps_html += '<span style="color:#3d1e7a; margin:0 2px;">→</span>'
    st.markdown(f'<div style="display:flex; flex-wrap:wrap; align-items:center; gap:4px;">{steps_html}</div>',
                unsafe_allow_html=True)


# ── Tab 2: Search ─────────────────────────────────────────────────────────────

def render_search_tab():
    render_header()

    if not get(INDEXED_KEY):
        st.warning("⚠️ Please upload documents and build the index first (Upload & Index tab).")
        return

    retriever: KeywordRetriever = get(RETRIEVER_KEY)
    
    # Initialize and cache SearchCoach and AnswerGenerator in session state
    if "search_coach" not in st.session_state:
        st.session_state["search_coach"] = SearchCoach()
    if "answer_generator" not in st.session_state:
        st.session_state["answer_generator"] = AnswerGenerator()
    
    coach = st.session_state["search_coach"]
    generator = st.session_state["answer_generator"]

    # ── Query Input ───────────────────────────────────────────────────────────
    _section_title("🔍", "Query Input")
    _card("""
    <div style="color:#a89ec8; font-size:0.88rem; margin-bottom:0.5rem;">
        Enter your question or keywords about the university handbook.
        The IR engine uses <strong>TF-IDF ranking</strong> — no embeddings.
    </div>
    """, "accent")

    col_q, col_k = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Search query",
            value=get(QUERY_KEY, ""),
            placeholder="e.g., examination attendance grading system",
            label_visibility="collapsed",
            key="query_input",
        )
    with col_k:
        top_k = st.selectbox("Top K", [3, 5, 7, 10], index=1, label_visibility="collapsed")

    col_search, col_clear, col_generate = st.columns([2, 1, 2])
    with col_search:
        do_search = st.button("🔍 Search Documents", use_container_width=True, type="primary")
    with col_clear:
        if st.button("✕ Clear", use_container_width=True):
            from utils.session_state import clear_results
            clear_results()
            st.rerun()
    with col_generate:
        do_generate = st.button("🤖 Generate Answer (Phi-3)", use_container_width=True)

    # ── Run Search ────────────────────────────────────────────────────────────
    if do_search and query.strip():
        ss_set(QUERY_KEY, query)
        with st.spinner("Searching..."):
            results = retriever.retrieve(query.strip(), top_k=top_k)
            advice  = coach.analyze(query.strip(), results, top_k=top_k)
        ss_set(RESULTS_KEY, results)
        ss_set(COACH_KEY,   advice)
        ss_set(ANSWER_KEY,  None)
        # Log the search query, results, and coach advice
        log_query(query.strip(), results, advice)

    results: list = get(RESULTS_KEY, [])
    current_query: str = get(QUERY_KEY, "")

    if not results and not current_query:
        st.markdown("""
        <div style="text-align:center; padding:3rem 0; color:#3d1e7a;">
            <div style="font-size:3rem;">🔍</div>
            <div style="font-size:1rem; margin-top:0.5rem; color:#6b6485;">
                Enter a query above and click Search to begin
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("---")

    # ── Search Results ────────────────────────────────────────────────────────
    _section_title("📋", f"Search Results ({len(results)} found)")

    if not results:
        _card("""
        <div style="text-align:center; padding:1rem; color:#ef4444;">
            <div style="font-size:2rem;">😕</div>
            <div style="margin-top:0.5rem;">No documents matched your query.</div>
            <div style="font-size:0.85rem; color:#6b6485; margin-top:0.3rem;">
                See the Search Coach below for suggestions.
            </div>
        </div>
        """, "red")
    else:
        for res in results:
            _render_result_card(res, retriever)

    st.markdown("---")

    # ── Search Coach ──────────────────────────────────────────────────────────
    coach_advice = get(COACH_KEY)
    if coach_advice:
        _render_coach_card(coach_advice, query)

    st.markdown("---")

    # ── Explanation Panel ─────────────────────────────────────────────────────
    if results:
        _section_title("💡", "Explanation Panel — Score Breakdown")
        _render_explanation_panel(results, current_query)

    st.markdown("---")

    # ── RAG Answer Generation ─────────────────────────────────────────────────
    _section_title("🤖", "Generated Answer — Phi-3 via Ollama")

    if do_generate and results and current_query:
        with st.spinner("Generating answer with Phi-3... (may take 30–90 seconds)"):
            answer_data = generator.generate(current_query, results, top_k=4)
        ss_set(ANSWER_KEY, answer_data)

    answer_data = get(ANSWER_KEY)
    if answer_data:
        _render_answer_card(answer_data)
    elif not do_generate:
        st.markdown("""
        <div style="background:#16161f; border:1px dashed #3d1e7a; border-radius:12px;
                    padding:2rem; text-align:center; color:#6b6485;">
            🤖 Click <strong style="color:#9d5af0;">Generate Answer (Phi-3)</strong>
            after searching to get an AI-synthesized answer from retrieved documents.
        </div>
        """, unsafe_allow_html=True)


def _render_result_card(res: dict, retriever: KeywordRetriever):
    rank         = res["rank"]
    filename     = res["filename"]
    category     = res["category"]
    score        = res["score"]
    matched_terms= res["matched_terms"]
    snippet      = res["snippet"]

    # Highlight matched terms in snippet
    highlighted_snippet = snippet
    for term in matched_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted_snippet = pattern.sub(
            lambda m: f'<span class="highlight-term">{m.group(0)}</span>',
            highlighted_snippet,
        )

    terms_html = " ".join(
        f'<span class="badge">{t}</span>' for t in matched_terms
    ) if matched_terms else '<span style="color:#6b6485; font-size:0.8rem;">— no matched terms —</span>'

    cat_colors = {
        "Academic": "#7c3aed", "Fees & Finance": "#10b981", "Hostel": "#3b82f6",
        "Library": "#f59e0b", "Admissions": "#ec4899", "Syllabus": "#06b6d4",
        "Student Welfare": "#8b5cf6", "General": "#6b7280",
    }
    cat_color = cat_colors.get(category, "#6b7280")

    st.markdown(f"""
    <div class="uni-card" style="border-left: 3px solid {cat_color};">
        <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:1rem;">
            <div style="flex:1;">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
                    <span class="rank-badge">#{rank}</span>
                    <strong style="color:#f0eeff; font-size:0.95rem;">📄 {filename}</strong>
                    <span style="background:{cat_color}22; color:{cat_color}; border-radius:4px;
                                 padding:2px 10px; font-size:0.72rem; font-weight:600;">{category}</span>
                </div>
                <div style="color:#a89ec8; font-size:0.85rem; line-height:1.6; margin-bottom:0.7rem;">
                    {highlighted_snippet}
                </div>
                <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;">
                    <span style="color:#6b6485; font-size:0.75rem; text-transform:uppercase;
                                 letter-spacing:0.05em;">Matched:</span>
                    {terms_html}
                </div>
            </div>
            <div style="text-align:center; min-width:80px;">
                <div class="score-pill">{score:.4f}</div>
                <div style="font-size:0.7rem; color:#6b6485; margin-top:3px;">TF-IDF</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_coach_card(advice: dict, query: str):
    _section_title("🧭", "Search Coach")
    status  = advice["status"]
    message = advice["message"]
    suggestions = advice.get("suggestions", [])
    clarifying  = advice.get("clarifying_question")
    suggested_queries = advice.get("suggested_queries", [])

    style_map = {
        "no_results":   "red",
        "short_query":  "yellow",
        "ambiguous":    "yellow",
        "broad":        "yellow",
        "few_results":  "yellow",
        "good":         "green",
    }
    card_style = style_map.get(status, "accent")

    sugg_html = "".join(
        f'<li style="color:#a89ec8; margin-bottom:4px;">{s}</li>'
        for s in suggestions
    )
    clarify_html = ""
    if clarifying:
        clarify_html = f"""
        <div style="background:#1c1c28; border:1px solid #7c3aed44; border-radius:8px;
                    padding:0.8rem 1rem; margin-top:0.8rem;">
            <strong style="color:#c4b5fd;">❓ Clarification needed:</strong>
            <div style="color:#a89ec8; margin-top:4px;">{clarifying}</div>
        </div>"""

    sq_html = ""
    if suggested_queries:
        chips = "".join(f'<span class="source-chip" style="cursor:pointer;">🔍 {q}</span>' for q in suggested_queries)
        sq_html = f"""
        <div style="margin-top:0.8rem;">
            <div style="font-size:0.78rem; color:#6b6485; text-transform:uppercase;
                        letter-spacing:0.05em; margin-bottom:5px;">Suggested queries:</div>
            {chips}
        </div>"""

    st.markdown(f"""
    <div class="{'coach-card' if card_style == 'accent' else 'uni-card-' + card_style}" 
         style="border-radius:12px; padding:1.4rem; margin-bottom:1.2rem;">
        <div style="font-size:1rem; font-weight:600; margin-bottom:0.6rem;">{message}</div>
        <ul style="margin:0; padding-left:1.2rem;">{sugg_html}</ul>
        {clarify_html}
        {sq_html}
    </div>
    """, unsafe_allow_html=True)


def _render_explanation_panel(results: list, query: str):
    tabs = st.tabs([f"#{r['rank']} {r['filename'][:25]}…" if len(r['filename']) > 25
                    else f"#{r['rank']} {r['filename']}" for r in results])
    for tab, res in zip(tabs, results):
        with tab:
            breakdown = res.get("breakdown", [])
            if not breakdown:
                st.info("No breakdown available.")
                continue

            col_why, col_table = st.columns([2, 3])
            with col_why:
                matched = res.get("matched_terms", [])
                st.markdown(f"""
                <div class="uni-card-accent">
                    <div style="font-weight:700; color:#c4b5fd; margin-bottom:0.5rem;">
                        Why did this rank #{res['rank']}?
                    </div>
                    <div style="color:#a89ec8; font-size:0.88rem; line-height:1.7;">
                        Document <strong>{res['filename']}</strong> ranked 
                        <strong style="color:#7c3aed;">#{res['rank']}</strong> because 
                        it contains <strong>{len(matched)}</strong> of your query terms 
                        with a combined TF-IDF score of 
                        <strong style="color:#9d5af0;">{res['score']:.6f}</strong>.
                    </div>
                    <div style="margin-top:0.8rem; font-size:0.8rem; color:#6b6485;">
                        TF-IDF = (Term Frequency / Doc Length) × log((1 + N) / (1 + DF) + 1)
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_table:
                st.markdown("**Score Breakdown per Term:**")
                rows = []
                for b in breakdown:
                    hit = "✅" if b["in_doc"] else "❌"
                    rows.append({
                        "Term":     b["term"],
                        "In Doc?":  hit,
                        "Raw Freq": b["raw_freq"],
                        "TF":       f"{b['tf']:.5f}",
                        "IDF":      f"{b['idf']:.4f}",
                        "TF-IDF":   f"{b['tfidf']:.5f}",
                    })
                df = pd.DataFrame(rows)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Term":     st.column_config.TextColumn(width="medium"),
                        "TF-IDF":   st.column_config.TextColumn(width="small"),
                    },
                )


def _render_answer_card(answer_data: dict):
    if answer_data.get("success"):
        model   = answer_data.get("model", "phi3")
        answer  = answer_data.get("answer", "")
        sources = answer_data.get("sources", [])
        sources_html = "".join(f'<span class="source-chip">📄 {s}</span>' for s in sources)

        st.markdown(f"""
        <div class="answer-card">
            <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:1rem;">
                <span style="font-size:1.5rem;">🤖</span>
                <div>
                    <div style="font-weight:700; color:#c4b5fd; font-size:1rem;">
                        Phi-3 Generated Answer
                    </div>
                    <div style="font-size:0.75rem; color:#6b6485;">
                        Model: <code>{model}</code> · Context: retrieved docs only
                    </div>
                </div>
            </div>
            <div class="answer-text">{answer.replace(chr(10), '<br>')}</div>
            <div style="margin-top:1rem; padding-top:0.8rem; border-top:1px solid #2a2a3a;">
                <span style="font-size:0.75rem; color:#6b6485; text-transform:uppercase;
                             letter-spacing:0.05em;">Sources used:</span><br>
                <div style="margin-top:4px;">{sources_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📝 View Full Prompt (transparency)"):
            st.code(answer_data.get("prompt", ""), language="text")
    else:
        error = answer_data.get("error", "Unknown error.")
        st.markdown(f"""
        <div class="uni-card-red">
            <div style="font-weight:700; color:#ef4444; margin-bottom:0.5rem;">
                ❌ Answer Generation Failed
            </div>
            <pre style="background:transparent; border:none; padding:0;
                        color:#fca5a5; font-size:0.85rem; white-space:pre-wrap;">{error}</pre>
        </div>
        """, unsafe_allow_html=True)


# ── Tab 3: Evaluation ─────────────────────────────────────────────────────────

def render_evaluation_tab():
    render_header()
    _section_title("📊", "Retrieval Evaluation — Precision & Recall")
    
    st.info("Evaluation uses the bundled sample dataset. Please load it first (Upload tab > Load sample docs) for meaningful precision/recall results.")

    if not get(INDEXED_KEY):
        st.warning("⚠️ Please build the index first (Upload & Index tab).")
        return

    retriever: KeywordRetriever = get(RETRIEVER_KEY)

    col_run, _ = st.columns([2, 4])
    with col_run:
        top_k_eval = st.selectbox("Evaluate at Top-K", [3, 5, 7], index=1)
        run_eval   = st.button("▶ Run Evaluation", use_container_width=True, type="primary")

    if run_eval:
        with st.spinner("Running evaluation on 5 predefined queries..."):
            eval_results = run_evaluation(retriever, top_k=top_k_eval)
        ss_set(EVAL_KEY, eval_results)

    eval_data = get(EVAL_KEY)
    if not eval_data:
        _card("""
        <div style="text-align:center; padding:1.5rem; color:#6b6485;">
            <div style="font-size:2rem;">📏</div>
            <div style="margin-top:0.5rem;">
                Click <strong style="color:#9d5af0;">Run Evaluation</strong> to compute
                Precision, Recall, F1, and MAP for 5 ground-truth queries.
            </div>
        </div>
        """)

        # Show the evaluation queries
        st.markdown("---")
        _section_title("📋", "Predefined Evaluation Queries")
        for q in EVALUATION_QUERIES:
            rel_docs_html = " ".join(
                f'<span class="source-chip">📄 {d}</span>' for d in q["relevant_docs"]
            )
            st.markdown(f"""
            <div class="uni-card">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.3rem;">
                    <span class="badge">{q['query_id']}</span>
                    <strong style="color:#f0eeff;">{q['query']}</strong>
                </div>
                <div style="color:#a89ec8; font-size:0.8rem; margin-bottom:0.5rem;">
                    {q['description']}
                </div>
                <div style="font-size:0.75rem; color:#6b6485;">Relevant docs: {rel_docs_html}</div>
            </div>
            """, unsafe_allow_html=True)
        return

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    agg = eval_data["aggregate"]
    k   = eval_data["top_k"]

    st.markdown(f"""
    <div class="uni-card-accent" style="margin-bottom:1.5rem;">
        <div class="section-title">🏆 Aggregate Metrics @ Top-{k}</div>
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-value">{agg['mean_precision']:.2%}</div>
                <div class="metric-label">Mean Precision</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{agg['mean_recall']:.2%}</div>
                <div class="metric-label">Mean Recall</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{agg['mean_f1']:.2%}</div>
                <div class="metric-label">Mean F1</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{agg['map']:.2%}</div>
                <div class="metric-label">MAP</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-query results ─────────────────────────────────────────────────────
    _section_title("🔎", "Per-Query Results")

    for q_res in eval_data["per_query"]:
        ea = q_res["error_analysis"]
        tp_html = " ".join(f'<span class="badge badge-green">✅ {d}</span>' for d in ea["true_positives"]) or "<span style='color:#6b6485'>—</span>"
        fp_html = " ".join(f'<span class="badge badge-red">❌ {d}</span>' for d in ea["false_positives"]) or "<span style='color:#6b6485'>—</span>"
        fn_html = " ".join(f'<span class="badge badge-yellow">⚠️ {d}</span>' for d in ea["false_negatives"]) or "<span style='color:#6b6485'>—</span>"

        # Color based on F1
        f1 = q_res["f1"]
        border_color = "#10b981" if f1 >= 0.7 else "#f59e0b" if f1 >= 0.4 else "#ef4444"

        st.markdown(f"""
        <div class="uni-card" style="border-left: 3px solid {border_color}; margin-bottom:1.2rem;">
            <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:1rem;">
                <div style="flex:1;">
                    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
                        <span class="badge">{q_res['query_id']}</span>
                        <strong style="color:#f0eeff;">{q_res['query']}</strong>
                    </div>
                    <div style="color:#a89ec8; font-size:0.8rem; margin-bottom:0.8rem;">
                        {q_res['description']}
                    </div>
                    <div style="font-size:0.8rem; margin-bottom:4px;">
                        <strong style="color:#c4b5fd;">True Positives:</strong> {tp_html}
                    </div>
                    <div style="font-size:0.8rem; margin-bottom:4px;">
                        <strong style="color:#ef4444;">False Positives:</strong> {fp_html}
                    </div>
                    <div style="font-size:0.8rem;">
                        <strong style="color:#f59e0b;">Missed (FN):</strong> {fn_html}
                    </div>
                </div>
                <div style="display:flex; flex-direction:column; gap:0.5rem; min-width:140px;">
                    <div class="metric-box" style="min-width:unset;">
                        <div class="metric-value" style="font-size:1.3rem;">{q_res['precision']:.0%}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-box" style="min-width:unset;">
                        <div class="metric-value" style="font-size:1.3rem;">{q_res['recall']:.0%}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-box" style="min-width:unset;">
                        <div class="metric-value" style="font-size:1.3rem; color:{border_color};">{q_res['f1']:.0%}</div>
                        <div class="metric-label">F1 Score</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Error analysis summary ────────────────────────────────────────────────
    st.markdown("---")
    _section_title("🔬", "Error Analysis Summary")

    all_fp = sum(len(q["error_analysis"]["false_positives"]) for q in eval_data["per_query"])
    all_fn = sum(len(q["error_analysis"]["false_negatives"]) for q in eval_data["per_query"])
    all_tp = sum(len(q["error_analysis"]["true_positives"]) for q in eval_data["per_query"])

    st.markdown(f"""
    <div class="uni-card">
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-value" style="color:#10b981;">{all_tp}</div>
                <div class="metric-label">Total True Positives</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color:#ef4444;">{all_fp}</div>
                <div class="metric-label">Total False Positives</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color:#f59e0b;">{all_fn}</div>
                <div class="metric-label">Total False Negatives</div>
            </div>
        </div>
        <div style="color:#a89ec8; font-size:0.85rem; margin-top:1rem; line-height:1.7;">
            <strong style="color:#c4b5fd;">Interpretation:</strong>
            False positives indicate documents retrieved but not relevant — 
            often caused by term overlap on common university vocabulary.
            False negatives indicate missed relevant documents — 
            often due to vocabulary mismatch (query uses different terms than the document).
            Increasing <em>top-k</em> typically improves recall at the cost of precision.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main entrypoint ───────────────────────────────────────────────────────────

def main():
    init_session_state()
    mode = render_sidebar()

    if "Upload" in mode:
        render_upload_tab()
    elif "Search" in mode:
        render_search_tab()
    elif "Evaluation" in mode:
        render_evaluation_tab()


if __name__ == "__main__":
    main()

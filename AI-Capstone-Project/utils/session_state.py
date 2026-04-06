# utils/session_state.py
"""
Streamlit session-state helpers.
Centralizes all st.session_state keys so every module uses the same names.
"""

import streamlit as st
from typing import Any, Optional


# ── Key names (single source of truth) ───────────────────────────────────────
DOCS_KEY         = "documents"          # List[Dict] — loaded documents
INDEX_KEY        = "inverted_index"     # InvertedIndex instance
RETRIEVER_KEY    = "retriever"          # KeywordRetriever instance
LC_RETRIEVER_KEY = "lc_retriever"       # LangChainKeywordRetriever instance
RESULTS_KEY      = "search_results"     # Last search results
QUERY_KEY        = "last_query"         # Last query string
COACH_KEY        = "coach_advice"       # Last coach advice dict
EVAL_KEY         = "eval_results"       # Last evaluation results
ANSWER_KEY       = "generated_answer"   # Last RAG answer dict
LEMMATIZE_KEY    = "use_lemmatization"  # bool toggle
INDEXED_KEY      = "is_indexed"         # bool — index built?


def init_session_state() -> None:
    """Initialize all session state keys with default values."""
    defaults = {
        DOCS_KEY:         [],
        INDEX_KEY:        None,
        RETRIEVER_KEY:    None,
        LC_RETRIEVER_KEY: None,
        RESULTS_KEY:      [],
        QUERY_KEY:        "",
        COACH_KEY:        None,
        EVAL_KEY:         None,
        ANSWER_KEY:       None,
        LEMMATIZE_KEY:    True,
        INDEXED_KEY:      False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set(key: str, value: Any) -> None:
    st.session_state[key] = value


def clear_results() -> None:
    """Clear search results and answers (not the index)."""
    st.session_state[RESULTS_KEY]  = []
    st.session_state[QUERY_KEY]    = ""
    st.session_state[COACH_KEY]    = None
    st.session_state[ANSWER_KEY]   = None

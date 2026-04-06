# retrieval/retriever.py
"""
Custom retriever module.
Implements:
  - KeywordRetriever: main search class wrapping InvertedIndex
  - LangChainRetriever: LangChain-compatible custom retriever (NO embeddings)
  - Explainability: per-result score breakdowns and matched-term highlights
"""

import re
from typing import List, Dict, Any, Tuple, Optional

from indexing.inverted_index import InvertedIndex
from preprocessing.text_preprocessor import preprocess_query

# Try to import LangChain core components
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False
    BaseRetriever = object
    Document = None
    CallbackManagerForRetrieverRun = None


# ── Core Retriever ────────────────────────────────────────────────────────────

class KeywordRetriever:
    """
    Keyword-based retriever backed by an inverted index.
    Returns ranked results with explainability data.
    """

    def __init__(self, index: InvertedIndex, use_lemmatization: bool = True):
        self.index = index
        self.use_lemmatization = use_lemmatization

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for the query.

        Returns:
            List of result dicts with document, score, matched terms,
            and explainability data.
        """
        query_tokens = preprocess_query(query, use_lemmatization=self.use_lemmatization)

        if not query_tokens:
            return []

        raw_results = self.index.search(query_tokens, top_k=top_k)

        results = []
        for rank, (doc_id, total_score, term_scores) in enumerate(raw_results, start=1):
            doc = self.index.get_document(doc_id)
            matched_terms = self.index.get_matched_terms(query_tokens, doc_id)

            # Build explainability breakdown
            breakdown = []
            for term in query_tokens:
                tf_val = self.index.tf(term, doc_id)
                idf_val = self.index.idf(term)
                tfidf_val = tf_val * idf_val
                raw_freq = self.index.get_term_frequency_in_doc(term, doc_id)
                breakdown.append({
                    "term": term,
                    "raw_freq": raw_freq,
                    "tf": round(tf_val, 5),
                    "idf": round(idf_val, 4),
                    "tfidf": round(tfidf_val, 5),
                    "in_doc": term in matched_terms,
                })

            # Generate a snippet from the document
            snippet = self._extract_snippet(doc.get("text", ""), matched_terms)

            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "filename": doc.get("filename", doc_id),
                "category": doc.get("category", "General"),
                "score": round(total_score, 6),
                "matched_terms": matched_terms,
                "query_tokens": query_tokens,
                "breakdown": breakdown,
                "snippet": snippet,
                "text": doc.get("text", ""),
                "char_count": doc.get("char_count", 0),
            })

        return results

    def _extract_snippet(
        self,
        text: str,
        matched_terms: List[str],
        window: int = 300,
    ) -> str:
        """Extract a relevant snippet from document text around matched terms."""
        if not matched_terms or not text:
            return text[:window] + "..." if len(text) > window else text

        text_lower = text.lower()
        best_pos = -1
        for term in matched_terms:
            pos = text_lower.find(term)
            if pos != -1:
                best_pos = pos
                break

        if best_pos == -1:
            return text[:window] + "..." if len(text) > window else text

        start = max(0, best_pos - 100)
        end = min(len(text), best_pos + window)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

    def highlight_text(self, text: str, terms: List[str]) -> str:
        """
        Return text with matched terms wrapped in **bold** markers
        (for Streamlit markdown rendering).
        """
        if not terms:
            return text
        for term in terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"**{m.group(0)}**", text)
        return text


# ── LangChain-Compatible Retriever ────────────────────────────────────────────

class LangChainDocument:
    """Minimal LangChain Document-like object (avoids langchain dependency)."""

    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"LangChainDocument(source={self.metadata.get('source', '')}, chars={len(self.page_content)})"


if LANGCHAIN_CORE_AVAILABLE:
    class LangChainKeywordRetriever(BaseRetriever):
        """
        LangChain-compatible custom retriever extending BaseRetriever.
        Implements _get_relevant_documents(query) → List[Document]
        Uses ONLY keyword IR — zero embeddings.
        """

        def __init__(self, keyword_retriever: KeywordRetriever, top_k: int = 5):
            super().__init__()
            self._retriever = keyword_retriever
            self._top_k = top_k

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
            """Main LangChain retriever method."""
            results = self._retriever.retrieve(query, top_k=self._top_k)
            docs = []
            for r in results:
                docs.append(
                    Document(
                        page_content=r["text"],
                        metadata={
                            "filename": r["filename"],
                            "category": r["category"],
                            "score": r["score"],
                            "matched_terms": r["matched_terms"],
                            "rank": r["rank"],
                        },
                    )
                )
            return docs

        # Async stub (LangChain compatibility)
        async def aget_relevant_documents(self, query: str) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=None)

        def invoke(self, query: str) -> List[Document]:
            """New LangChain LCEL interface."""
            return self._get_relevant_documents(query, run_manager=None)
else:
    # Fallback implementation when langchain_core is not available
    class LangChainKeywordRetriever:
        """
        LangChain-compatible custom retriever (fallback without BaseRetriever).
        Implements get_relevant_documents(query) → List[LangChainDocument]
        Uses ONLY keyword IR — zero embeddings.
        """

        def __init__(self, keyword_retriever: KeywordRetriever, top_k: int = 5):
            self._retriever = keyword_retriever
            self._top_k = top_k

        # LangChain interface
        def get_relevant_documents(self, query: str) -> List[LangChainDocument]:
            """Main LangChain retriever method."""
            results = self._retriever.retrieve(query, top_k=self._top_k)
            docs = []
            for r in results:
                docs.append(
                    LangChainDocument(
                        page_content=r["text"],
                        metadata={
                            "source": r["filename"],
                            "category": r["category"],
                            "score": r["score"],
                            "matched_terms": r["matched_terms"],
                            "rank": r["rank"],
                        },
                    )
                )
            return docs

        # Async stub (LangChain compatibility)
        async def aget_relevant_documents(self, query: str) -> List[LangChainDocument]:
            return self.get_relevant_documents(query)

        def invoke(self, query: str) -> List[LangChainDocument]:
            """New LangChain LCEL interface."""
            return self.get_relevant_documents(query)

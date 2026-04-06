# indexing/inverted_index.py
"""
Inverted Index builder.
Structure: { term: { doc_id: term_frequency } }
Also stores document lengths and total document count for TF-IDF.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple, Any


class InvertedIndex:
    """
    Inverted index with TF-IDF scoring capability.

    Attributes:
        index: term → {doc_id: raw_frequency}
        doc_lengths: doc_id → total token count in document
        documents: doc_id → original document dict
        term_doc_freq: term → number of documents containing it (DF)
    """

    def __init__(self):
        self.index: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.doc_lengths: Dict[str, int] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.term_doc_freq: Dict[str, int] = {}
        self._num_docs: int = 0

    def add_document(self, doc: Dict[str, Any], tokens: List[str]) -> None:
        """
        Add a document to the index.

        Args:
            doc: Document dict with at least 'doc_id' key
            tokens: Pre-processed list of tokens for this document
        """
        doc_id = doc["doc_id"]
        self.documents[doc_id] = doc
        self.doc_lengths[doc_id] = len(tokens)
        self._num_docs += 1

        # Count token frequencies for this doc
        term_counts: Dict[str, int] = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1

        # Update inverted index and incrementally update term_doc_freq
        for term, freq in term_counts.items():
            self.index[term][doc_id] = freq
            self.term_doc_freq[term] = len(self.index[term])

    def build(self, documents: List[Dict[str, Any]], tokens_per_doc: List[List[str]]) -> None:
        """
        Build the entire index from scratch.

        Args:
            documents: List of document dicts
            tokens_per_doc: Parallel list of token lists
        """
        self.index.clear()
        self.doc_lengths.clear()
        self.documents.clear()
        self.term_doc_freq.clear()
        self._num_docs = 0

        for doc, tokens in zip(documents, tokens_per_doc):
            doc_id = doc["doc_id"]
            self.documents[doc_id] = doc
            self.doc_lengths[doc_id] = len(tokens)
            self._num_docs += 1

            term_counts: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1
            for term, freq in term_counts.items():
                self.index[term][doc_id] = freq
                self.term_doc_freq[term] = len(self.index[term])

    # ── Scoring ──────────────────────────────────────────────────────────────

    def tf(self, term: str, doc_id: str) -> float:
        """Term frequency (raw count / doc length)."""
        raw = self.index.get(term, {}).get(doc_id, 0)
        length = self.doc_lengths.get(doc_id, 1)
        return raw / length if length > 0 else 0.0

    def idf(self, term: str) -> float:
        """Inverse document frequency with smoothing."""
        df = self.term_doc_freq.get(term, 0)
        if df == 0 or self._num_docs == 0:
            return 0.0
        return math.log((1 + self._num_docs) / (1 + df)) + 1

    def tf_idf(self, term: str, doc_id: str) -> float:
        """TF-IDF score."""
        return self.tf(term, doc_id) * self.idf(term)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_tokens: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Search the index using TF-IDF scoring.

        Args:
            query_tokens: Pre-processed query tokens
            top_k: Maximum number of results to return

        Returns:
            List of (doc_id, total_score, term_scores_dict) sorted descending
        """
        if not query_tokens or self._num_docs == 0:
            return []

        # Accumulate TF-IDF scores per doc
        scores: Dict[str, float] = defaultdict(float)
        term_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

        for token in query_tokens:
            if token not in self.index:
                continue
            idf_score = self.idf(token)
            for doc_id in self.index[token]:
                tf_score = self.tf(token, doc_id)
                contribution = tf_score * idf_score
                scores[doc_id] += contribution
                term_scores[doc_id][token] = contribution

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, total_score in ranked[:top_k]:
            results.append((doc_id, total_score, term_scores[doc_id]))

        return results

    # ── Utilities ────────────────────────────────────────────────────────────

    def get_matched_terms(self, query_tokens: List[str], doc_id: str) -> List[str]:
        """Return which query tokens actually appear in the given document."""
        return [
            t for t in query_tokens
            if t in self.index and doc_id in self.index[t]
        ]

    def get_term_frequency_in_doc(self, term: str, doc_id: str) -> int:
        """Raw term frequency."""
        return self.index.get(term, {}).get(doc_id, 0)

    @property
    def num_docs(self) -> int:
        return self._num_docs

    @property
    def vocab_size(self) -> int:
        return len(self.index)

    def get_all_doc_ids(self) -> List[str]:
        return list(self.documents.keys())

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        return self.documents.get(doc_id, {})

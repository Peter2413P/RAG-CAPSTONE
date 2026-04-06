# preprocessing/text_preprocessor.py
"""
Text preprocessing pipeline:
  - Tokenization
  - Stopword removal
  - Lemmatization (toggleable)
"""

import re
import string
from typing import List

# ── optional NLTK ────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Download necessary NLTK data (silent)
    for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    NLTK_AVAILABLE = True
    _STOP_WORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
except ImportError:
    NLTK_AVAILABLE = False
    _STOP_WORDS = set()
    _LEMMATIZER = None

# Built-in fallback stopwords (subset of common English stopwords)
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "that", "this", "these",
    "those", "it", "its", "not", "no", "nor", "as", "if", "so", "yet",
    "both", "either", "neither", "each", "than", "then", "when", "where",
    "which", "who", "whom", "what", "how", "all", "any", "some", "such",
    "more", "most", "other", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "about", "up", "down",
    "they", "them", "their", "we", "our", "you", "your", "he", "she",
    "his", "her", "i", "my", "me", "also", "only", "just", "per",
}

if not NLTK_AVAILABLE:
    _STOP_WORDS = _FALLBACK_STOPWORDS
else:
    _STOP_WORDS = _STOP_WORDS | _FALLBACK_STOPWORDS


def tokenize(text: str) -> List[str]:
    """Split text into lowercase tokens, removing punctuation but preserving meaningful numbers like "2024", "21CS001", "3.2"."""
    text = text.lower()
    # Replace only punctuation with space (but preserve digits)
    text = re.sub(r"[^\w\s]", " ", text)

    if NLTK_AVAILABLE:
        try:
            raw_tokens = word_tokenize(text)
        except Exception:
            raw_tokens = text.split()
    else:
        raw_tokens = text.split()

    # Keep tokens with meaningful length, drop purely numeric tokens, but preserve alphanumeric tokens like "2024", "21CS001", "3.2"
    tokens = [t for t in raw_tokens if len(t) > 1 and not t.isdigit()]
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove stopwords from token list."""
    return [t for t in tokens if t not in _STOP_WORDS]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens using NLTK WordNetLemmatizer if available."""
    if NLTK_AVAILABLE and _LEMMATIZER:
        return [_LEMMATIZER.lemmatize(t) for t in tokens]
    # Naive fallback: strip common suffixes
    result = []
    for t in tokens:
        if t.endswith("ing") and len(t) > 5:
            t = t[:-3]
        elif t.endswith("tion") and len(t) > 6:
            t = t[:-4]
        elif t.endswith("ies") and len(t) > 4:
            t = t[:-3] + "y"
        elif t.endswith("es") and len(t) > 3:
            t = t[:-2]
        elif t.endswith("s") and len(t) > 3:
            t = t[:-1]
        result.append(t)
    return result


def preprocess(
    text: str,
    use_lemmatization: bool = True,
    remove_stops: bool = True,
) -> List[str]:
    """
    Full preprocessing pipeline.

    Args:
        text: Raw document/query text
        use_lemmatization: If True, apply lemmatization
        remove_stops: If True, remove stopwords

    Returns:
        List of processed tokens
    """
    tokens = tokenize(text)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    if use_lemmatization:
        tokens = lemmatize_tokens(tokens)
    return tokens


def preprocess_query(
    query: str,
    use_lemmatization: bool = True,
) -> List[str]:
    """Preprocess a search query (always removes stopwords)."""
    return preprocess(query, use_lemmatization=use_lemmatization, remove_stops=True)

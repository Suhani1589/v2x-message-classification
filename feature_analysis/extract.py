"""Interpretable linguistic features and TF-IDF, independent of Keras tokenization."""

from __future__ import annotations

import string
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from feature_analysis.load_messages import TEXT_COLUMN

LINGUISTIC_FEATURES = [
    "n_words",
    "n_chars",
    "avg_word_length",
    "n_digits",
    "n_uppercase",
    "n_punctuation",
    "n_special_chars",
    "n_unique_words",
    "unique_word_ratio",
]

_PUNCT = set(string.punctuation)


def _words(text: str) -> List[str]:
    return [token for token in text.split() if token]


def linguistic_features(text: str) -> dict:
    """Count surface features on the stored message string (whitespace words)."""
    words = _words(text)
    n_words = len(words)
    word_chars = sum(len(word) for word in words)
    unique = len(set(word.lower() for word in words))
    n_digits = sum(ch.isdigit() for ch in text)
    n_upper = sum(ch.isupper() for ch in text)
    n_punct = sum(ch in _PUNCT for ch in text)
    # Special = not letter, digit, whitespace, or ASCII punctuation.
    n_special = sum(
        (not ch.isalnum()) and (not ch.isspace()) and (ch not in _PUNCT)
        for ch in text
    )
    return {
        "n_words": n_words,
        "n_chars": len(text),
        "avg_word_length": (word_chars / n_words) if n_words else 0.0,
        "n_digits": n_digits,
        "n_uppercase": n_upper,
        "n_punctuation": n_punct,
        "n_special_chars": n_special,
        "n_unique_words": unique,
        "unique_word_ratio": (unique / n_words) if n_words else 0.0,
    }


def linguistic_matrix(texts: Iterable[str]) -> pd.DataFrame:
    rows = [linguistic_features(str(text)) for text in texts]
    return pd.DataFrame(rows, columns=LINGUISTIC_FEATURES)


def fit_tfidf(texts: Iterable[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Corpus TF-IDF. Lowercasing is inside this analyzer only."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z0-9]+\b",
        min_df=1,
    )
    matrix = vectorizer.fit_transform(list(texts))
    return vectorizer, matrix


def build_feature_matrix(messages: pd.DataFrame) -> pd.DataFrame:
    """Message id, label, and the nine linguistic features."""
    features = linguistic_matrix(messages[TEXT_COLUMN].tolist())
    meta = messages[["Message ID", "Message Type", "split"]].reset_index(drop=True)
    return pd.concat([meta, features], axis=1)

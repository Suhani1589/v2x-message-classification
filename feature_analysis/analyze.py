"""Class-wise statistics, distinctive TF-IDF terms, and a written summary."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from feature_analysis.extract import LINGUISTIC_FEATURES
from feature_analysis.load_messages import LABEL_ORDER

TOP_K = 8


def class_feature_stats(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation, and median of each linguistic feature per class."""
    rows = []
    for label in LABEL_ORDER:
        group = feature_frame.loc[feature_frame["Message Type"] == label]
        if group.empty:
            continue
        for feature in LINGUISTIC_FEATURES:
            series = group[feature]
            rows.append(
                {
                    "class": label,
                    "feature": feature,
                    "n": int(len(series)),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                    "median": float(series.median()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )
    return pd.DataFrame(rows)


def important_tfidf_terms(
    texts: List[str],
    labels: List[str],
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Terms with the highest mean TF-IDF inside each class, plus a contrast score.

    contrast = mean TF-IDF in the class minus mean TF-IDF in the other classes.
    """
    terms = np.array(vectorizer.get_feature_names_out())
    label_array = np.array(labels)
    rows = []
    for label in LABEL_ORDER:
        mask = label_array == label
        if not mask.any():
            continue
        class_mean = np.asarray(matrix[mask].mean(axis=0)).ravel()
        rest_mean = np.asarray(matrix[~mask].mean(axis=0)).ravel()
        contrast = class_mean - rest_mean
        # Rank by contrast so the listed terms are the ones most associated
        # with the class relative to the other five classes.
        order = np.argsort(contrast)[::-1][:top_k]
        support = int(mask.sum())
        for rank, index in enumerate(order, start=1):
            rows.append(
                {
                    "class": label,
                    "rank": rank,
                    "term": terms[index],
                    "mean_tfidf_in_class": float(class_mean[index]),
                    "mean_tfidf_in_other_classes": float(rest_mean[index]),
                    "contrast": float(contrast[index]),
                    "n_messages": support,
                }
            )
    return pd.DataFrame(rows)


def write_findings(
    path,
    n_messages: int,
    feature_frame: pd.DataFrame,
    stats: pd.DataFrame,
    terms: pd.DataFrame,
    n_tfidf_terms: int,
) -> None:
    """Summarize only quantities computed from this corpus."""
    lines: List[str] = []
    lines.append("V2X textual feature analysis")
    lines.append("Independent of the CNN-BRNN classifiers.")
    lines.append("")
    lines.append(
        f"Corpus: {n_messages} messages from data/processed/train.csv, "
        "val.csv, and test.csv (Message Content + Message Type)."
    )
    lines.append(f"TF-IDF vocabulary size: {n_tfidf_terms} terms.")
    lines.append("")
    lines.append("Linguistic features are computed on the stored message text:")
    lines.append(
        "word count (whitespace tokens), character count, average word length, "
        "digit count, uppercase count, punctuation count (string.punctuation), "
        "special-character count (not letter, digit, whitespace, or ASCII punctuation), "
        "unique word count, and unique-word ratio."
    )
    lines.append("")

    zero_features = []
    for feature in ("n_digits", "n_uppercase", "n_punctuation", "n_special_chars"):
        if float(feature_frame[feature].sum()) == 0.0:
            zero_features.append(feature)
    if zero_features:
        lines.append(
            "The following features are zero for every message in this corpus, "
            "so they do not separate the six classes: "
            + ", ".join(zero_features)
            + "."
        )
        lines.append(
            "Messages in the current synthetic CSVs are lowercase word sequences "
            "without digits or punctuation."
        )
        lines.append("")

    varying = [
        feature
        for feature in LINGUISTIC_FEATURES
        if feature not in zero_features
    ]
    lines.append("Class means for features that vary:")
    for feature in varying:
        subset = stats.loc[stats["feature"] == feature].sort_values("mean")
        if subset.empty:
            continue
        low = subset.iloc[0]
        high = subset.iloc[-1]
        spread = float(high["mean"] - low["mean"])
        lines.append(
            f"- {feature}: lowest mean {low['class']} ({low['mean']:.3f}), "
            f"highest mean {high['class']} ({high['mean']:.3f}), "
            f"spread {spread:.3f}."
        )
    lines.append("")

    if not varying or all(
        float(stats.loc[stats["feature"] == feature, "mean"].max()
              - stats.loc[stats["feature"] == feature, "mean"].min()) < 0.15
        for feature in ("n_words", "avg_word_length", "unique_word_ratio")
        if feature in varying
    ):
        lines.append(
            "Surface counts (length, unique-word ratio, average word length) "
            "differ only slightly across Emergency, Hazard, Roadwork, Spam, "
            "Traffic, and Weather. Class identity is not well described by "
            "those length features alone."
        )
    lines.append("")
    lines.append(
        "TF-IDF terms below are ranked by contrast "
        "(mean TF-IDF inside the class minus mean TF-IDF in the other classes). "
        "Positive contrast means the term is more characteristic of that class "
        "in this corpus."
    )
    lines.append("")
    for label in LABEL_ORDER:
        class_terms = terms.loc[terms["class"] == label].sort_values("rank")
        if class_terms.empty:
            continue
        shown = []
        for _, row in class_terms.head(5).iterrows():
            shown.append(f"{row['term']} ({row['contrast']:+.3f})")
        lines.append(f"- {label}: " + "; ".join(shown))
    lines.append("")
    lines.append(
        "These associations describe the synthetic message texts already in "
        "the project. They are not model accuracy, F1, or loss, and they do "
        "not change the classification results."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Distribution plots for the independent feature analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from feature_analysis.load_messages import LABEL_ORDER

VARYING_FEATURES = [
    "n_words",
    "n_chars",
    "avg_word_length",
    "n_unique_words",
    "unique_word_ratio",
]


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def plot_feature_boxplots(feature_frame: pd.DataFrame, output_dir: Path) -> Path:
    """One panel per linguistic feature that is not identically zero."""
    _style()
    present = [
        column
        for column in VARYING_FEATURES
        if column in feature_frame.columns and float(feature_frame[column].std(ddof=0)) > 0
    ]
    if not present:
        present = VARYING_FEATURES
    fig, axes = plt.subplots(1, len(present), figsize=(3.2 * len(present), 4.6), sharey=False)
    if len(present) == 1:
        axes = [axes]
    order = [label for label in LABEL_ORDER if label in set(feature_frame["Message Type"])]
    for axis, column in zip(axes, present):
        sns.boxplot(
            data=feature_frame,
            x="Message Type",
            y=column,
            order=order,
            ax=axis,
            color="#4C78A8",
            fliersize=2,
        )
        axis.set_xlabel("")
        axis.set_title(column.replace("_", " "))
        axis.tick_params(axis="x", rotation=40)
    fig.suptitle("Linguistic feature distributions by V2X class", y=1.02)
    fig.tight_layout()
    path = output_dir / "feature_distributions_by_class.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_mean_heatmap(stats: pd.DataFrame, output_dir: Path) -> Path:
    """Z-scored class means so features on different scales can be compared."""
    _style()
    table = stats.pivot(index="feature", columns="class", values="mean")
    table = table.reindex(columns=[c for c in LABEL_ORDER if c in table.columns])
    # Drop all-zero rows so the heatmap is not dominated by unused counts.
    table = table.loc[table.abs().sum(axis=1) > 0]
    centered = table.sub(table.mean(axis=1), axis=0)
    scale = table.std(axis=1, ddof=0).replace(0, np.nan)
    z = centered.div(scale, axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.heatmap(z, cmap="RdBu_r", center=0, ax=ax, linewidths=0.4)
    ax.set_title("Class means relative to other classes (z-score across classes)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    path = output_dir / "class_mean_heatmap.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tfidf_terms(terms: pd.DataFrame, output_dir: Path) -> Path:
    """Top contrastive TF-IDF terms for each class."""
    _style()
    classes = [label for label in LABEL_ORDER if label in set(terms["class"])]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=False)
    for axis, label in zip(axes.ravel(), classes):
        subset = terms.loc[terms["class"] == label].sort_values("contrast")
        axis.barh(subset["term"], subset["contrast"], color="#1F4E79")
        axis.set_title(label)
        axis.set_xlabel("TF-IDF contrast")
    fig.suptitle("Terms more associated with each V2X class (TF-IDF contrast)")
    fig.tight_layout()
    path = output_dir / "tfidf_class_terms.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path

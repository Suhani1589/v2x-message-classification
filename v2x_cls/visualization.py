"""Plotting helpers for training curves, confusion matrices, and attention."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_history(history: Dict[str, List[float]], out_dir: Path, model_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("loss", [])) + 1)

    def _save(keys, title, ylabel, filename):
        plt.figure(figsize=(7, 4))
        for key, label in keys:
            if key in history:
                plt.plot(epochs, history[key], label=label)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=150)
        plt.close()

    _save(
        [("loss", "Train"), ("val_loss", "Validation")],
        f"{model_name} — Loss",
        "Loss",
        f"{model_name}_loss_curves.png",
    )
    _save(
        [("accuracy", "Train"), ("val_accuracy", "Validation")],
        f"{model_name} — Accuracy",
        "Accuracy",
        f"{model_name}_accuracy_curves.png",
    )


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    labels: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(cm)
    plt.figure(figsize=(7, 6))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_attention_bars(
    tokens: Sequence[str],
    weights: Sequence[float],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop padding tokens for readability but keep alignment if all pad.
    pairs = [(t, float(w)) for t, w in zip(tokens, weights) if t != "<PAD>"]
    if not pairs:
        pairs = list(zip(tokens, [float(w) for w in weights]))
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]

    plt.figure(figsize=(8, 4))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, vals, color="#2F6F8F")
    plt.yticks(y_pos, labels)
    plt.xlabel("Attention weight")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison(table_csv: Path, out_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(table_csv)
    plt.figure(figsize=(8, 4))
    x = np.arange(len(df))
    width = 0.2
    for i, col in enumerate(["Accuracy", "Precision_macro", "Recall_macro", "F1_macro"]):
        plt.bar(x + i * width, df[col], width=width, label=col)
    plt.xticks(x + 1.5 * width, df["Model"], rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model comparison (test set)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

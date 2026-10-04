"""Evaluation metrics, comparison tables, and artifact saving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(
    y_true_idx: np.ndarray,
    y_prob: np.ndarray,
    label_order: Sequence[str],
    loss_value: float | None = None,
) -> Dict[str, Any]:
    y_pred = np.argmax(y_prob, axis=1)
    labels = list(range(len(label_order)))
    report = classification_report(
        y_true_idx,
        y_pred,
        labels=labels,
        target_names=list(label_order),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_idx, y_pred, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_true_idx, y_pred)),
        "precision_macro": float(precision_score(y_true_idx, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_idx, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_idx, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true_idx, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true_idx, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true_idx, y_pred, average="weighted", zero_division=0)),
        "loss": None if loss_value is None else float(loss_value),
        "per_class": {
            lab: {
                "precision": report[lab]["precision"],
                "recall": report[lab]["recall"],
                "f1": report[lab]["f1-score"],
                "support": report[lab]["support"],
            }
            for lab in label_order
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return metrics


def metrics_row(model_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Model": model_name,
        "Accuracy": metrics["accuracy"],
        "Precision_macro": metrics["precision_macro"],
        "Recall_macro": metrics["recall_macro"],
        "F1_macro": metrics["f1_macro"],
        "Precision_weighted": metrics["precision_weighted"],
        "Recall_weighted": metrics["recall_weighted"],
        "F1_weighted": metrics["f1_weighted"],
        "Loss": metrics.get("loss"),
    }


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def write_comparison_table(rows: List[Dict[str, Any]], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    # Markdown without floatfmt (mixed str/float columns break tabulate floatfmt).
    md_path = path.with_suffix(".md")
    numeric = [c for c in df.columns if c != "Model"]
    df_fmt = df.copy()
    for c in numeric:
        df_fmt[c] = df_fmt[c].map(lambda x: f"{float(x):.4f}" if x is not None else "")
    with md_path.open("w", encoding="utf-8") as f:
        f.write(df_fmt.to_markdown(index=False))
        f.write("\n")
    return df

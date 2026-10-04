#!/usr/bin/env python3
"""Evaluate one or more trained models on the held-out test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2x_cls.config import load_config, resolve_path
from v2x_cls.evaluation import evaluate_predictions, metrics_row, save_metrics, write_comparison_table
from v2x_cls.models.attention import AdditiveAttention, PaddingMask
from v2x_cls.preprocessing import TextPreprocessor
from v2x_cls.seeds import set_global_seed
from v2x_cls.visualization import plot_attention_bars, plot_confusion_matrix, plot_model_comparison


def _load_model(path: Path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "AdditiveAttention": AdditiveAttention,
            "PaddingMask": PaddingMask,
        },
        compile=True,
        safe_mode=False,
    )


def evaluate_one(model_name: str, cfg: dict) -> dict:
    processed_dir = resolve_path(cfg, cfg["paths"]["processed_data_dir"])
    test_df = pd.read_csv(processed_dir / "test.csv")
    pre = TextPreprocessor.load(processed_dir / "artifacts")
    x_test, y_test, y_idx = pre.transform(test_df["Message Content"], test_df["Message Type"])

    ckpt_dir = resolve_path(cfg, cfg["paths"]["checkpoints_dir"], model_name)
    model_path = ckpt_dir / "best.keras"
    if not model_path.exists():
        model_path = ckpt_dir / "final.keras"
    model = _load_model(model_path)

    loss_acc = model.evaluate(x_test, y_test, verbose=0)
    # model.evaluate returns [loss, accuracy, precision, recall] with our compile setup
    loss_value = float(loss_acc[0]) if isinstance(loss_acc, (list, tuple, np.ndarray)) else float(loss_acc)
    y_prob = model.predict(x_test, verbose=0)
    metrics = evaluate_predictions(y_idx, y_prob, pre.label_order, loss_value=loss_value)

    results_dir = resolve_path(cfg, cfg["paths"]["results_dir"], model_name)
    save_metrics(metrics, results_dir / "test_metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        pre.label_order,
        results_dir / "confusion_matrix.png",
        title=f"{model_name} confusion matrix (test)",
    )

    # Attention examples for the proposed model.
    if model_name == "cnn_brnn_attention":
        companion = ckpt_dir / "attention_companion.keras"
        if companion.exists():
            attn_model = _load_model(companion)
            sample_texts = test_df["Message Content"].head(5).tolist()
            sample_x = pre.texts_to_sequences(sample_texts)
            probs, weights = attn_model.predict(sample_x, verbose=0)
            attn_dir = results_dir / "attention_examples"
            attn_dir.mkdir(parents=True, exist_ok=True)
            examples = []
            for i, text in enumerate(sample_texts):
                tokens = pre.sequence_to_tokens(sample_x[i])
                w = weights[i].tolist()
                pred = pre.index_to_label[int(np.argmax(probs[i]))]
                conf = float(np.max(probs[i]))
                pairs = [
                    {"token": t, "weight": float(ww)}
                    for t, ww in zip(tokens, w)
                    if t != "<PAD>"
                ]
                pairs_sorted = sorted(pairs, key=lambda d: d["weight"], reverse=True)
                examples.append(
                    {
                        "text": text,
                        "predicted": pred,
                        "confidence": conf,
                        "token_weights": pairs,
                        "top_tokens": pairs_sorted[:5],
                    }
                )
                plot_attention_bars(
                    tokens,
                    w,
                    attn_dir / f"sample_{i+1}.png",
                    title=f"Attention — pred={pred} ({conf:.2%})",
                )
            with (attn_dir / "examples.json").open("w", encoding="utf-8") as f:
                json.dump(examples, f, indent=2)

    return metrics_row(model_name, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained V2X models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["brnn", "cnn_brnn", "cnn_brnn_attention"],
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    rows = []
    for name in args.models:
        print(f"Evaluating {name} ...")
        rows.append(evaluate_one(name, cfg))

    results_root = resolve_path(cfg, cfg["paths"]["results_dir"])
    table = write_comparison_table(rows, results_root / "comparison_table.csv")
    plot_model_comparison(results_root / "comparison_table.csv", results_root / "comparison_bar.png")
    print(table.to_string(index=False))
    print(f"\nSaved comparison to {results_root / 'comparison_table.csv'}")


if __name__ == "__main__":
    main()

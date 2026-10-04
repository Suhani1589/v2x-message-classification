#!/usr/bin/env python3
"""Interactive / CLI inference for V2X message classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2x_cls.config import load_config, resolve_path
from v2x_cls.models.attention import AdditiveAttention, PaddingMask
from v2x_cls.preprocessing import TextPreprocessor
from v2x_cls.visualization import plot_attention_bars


def load_classifier(cfg, model_name: str):
    ckpt_dir = resolve_path(cfg, cfg["paths"]["checkpoints_dir"], model_name)
    path = ckpt_dir / "best.keras"
    if not path.exists():
        path = ckpt_dir / "final.keras"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    custom = {
        "AdditiveAttention": AdditiveAttention,
        "PaddingMask": PaddingMask,
    }
    model = tf.keras.models.load_model(
        path,
        custom_objects=custom,
        compile=False,
        safe_mode=False,
    )
    attn_model = None
    companion = ckpt_dir / "attention_companion.keras"
    if companion.exists():
        attn_model = tf.keras.models.load_model(
            companion,
            custom_objects=custom,
            compile=False,
            safe_mode=False,
        )
    return model, attn_model


def predict_text(text: str, cfg, model_name: str, save_plot: Path | None = None) -> dict:
    processed_dir = resolve_path(cfg, cfg["paths"]["processed_data_dir"])
    pre = TextPreprocessor.load(processed_dir / "artifacts")
    model, attn_model = load_classifier(cfg, model_name)

    x = pre.texts_to_sequences([text])
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = pre.index_to_label[pred_idx]
    confidence = float(probs[pred_idx])

    result = {
        "message": text,
        "model": model_name,
        "predicted_class": pred_label,
        "confidence": confidence,
        "confidence_percent": round(100.0 * confidence, 2),
        "class_probabilities": {
            pre.index_to_label[i]: float(p) for i, p in enumerate(probs)
        },
    }

    if attn_model is not None:
        _, weights = attn_model.predict(x, verbose=0)
        tokens = pre.sequence_to_tokens(x[0])
        w = weights[0].tolist()
        pairs = [
            {"token": t, "weight": float(ww)}
            for t, ww in zip(tokens, w)
            if t != "<PAD>"
        ]
        pairs_sorted = sorted(pairs, key=lambda d: d["weight"], reverse=True)
        top_k = cfg["inference"]["top_k_attention_tokens"]
        result["important_tokens"] = [p["token"] for p in pairs_sorted[:top_k]]
        result["attention_weights"] = pairs
        result["attention_note"] = (
            "Attention weights are an interpretability aid, not a perfect causal "
            "explanation of model reasoning."
        )
        if save_plot is not None:
            plot_attention_bars(
                tokens,
                w,
                save_plot,
                title=f"Attention — {pred_label} ({confidence:.2%})",
            )
            result["attention_plot"] = str(save_plot)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict V2X message class")
    parser.add_argument(
        "--text",
        default=None,
        help='Message text, e.g. "Emergency vehicle approaching intersection"',
    )
    parser.add_argument(
        "--model",
        default="cnn_brnn_attention",
        choices=["brnn", "cnn_brnn", "cnn_brnn_attention"],
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--save-attention-plot", default=None)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.interactive or args.text is None:
        print("Enter a V2X message (empty line to quit):")
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                break
            out = predict_text(text, cfg, args.model, Path(args.save_attention_plot) if args.save_attention_plot else None)
            print(f"\nPredicted Class: {out['predicted_class']}")
            print(f"Confidence: {out['confidence_percent']}%")
            if "important_tokens" in out:
                print("Important tokens:", ", ".join(out["important_tokens"]))
            print()
    else:
        plot_path = Path(args.save_attention_plot) if args.save_attention_plot else None
        if plot_path is None and args.model == "cnn_brnn_attention":
            plot_path = resolve_path(cfg, cfg["paths"]["results_dir"], "inference_attention.png")
        out = predict_text(args.text, cfg, args.model, plot_path)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

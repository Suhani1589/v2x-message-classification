#!/usr/bin/env python3
"""Train a single V2X classifier (brnn | cnn_brnn | cnn_brnn_attention)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2x_cls.config import load_config, resolve_path
from v2x_cls.models import build_model_by_name, compile_model
from v2x_cls.preprocessing import TextPreprocessor
from v2x_cls.seeds import set_global_seed
from v2x_cls.training import detect_device, make_callbacks, save_history, save_run_config
from v2x_cls.visualization import plot_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a V2X message classifier")
    parser.add_argument(
        "--model",
        required=True,
        choices=["brnn", "cnn_brnn", "cnn_brnn_attention"],
        help="Model architecture",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate

    set_global_seed(cfg["project"]["seed"])
    device = detect_device()
    print(f"Device: {device}")

    processed_dir = resolve_path(cfg, cfg["paths"]["processed_data_dir"])
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    pre = TextPreprocessor.load(processed_dir / "artifacts")

    x_train, y_train, _ = pre.transform(train_df["Message Content"], train_df["Message Type"])
    x_val, y_val, _ = pre.transform(val_df["Message Content"], val_df["Message Type"])

    model, attn_model = build_model_by_name(
        args.model, pre.vocab_size, pre.num_classes, cfg
    )
    compile_model(model, cfg)
    model.summary()

    ckpt_dir = resolve_path(cfg, cfg["paths"]["checkpoints_dir"], args.model)
    callbacks = make_callbacks(ckpt_dir, cfg)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    # Save final and ensure best exists.
    model.save(ckpt_dir / "final.keras")
    best_path = ckpt_dir / "best.keras"
    if not best_path.exists():
        model.save(best_path)

    # Persist attention companion graph weights via classifier weights.
    if attn_model is not None:
        attn_model.set_weights(model.get_weights())
        attn_model.save(ckpt_dir / "attention_companion.keras")

    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    save_history(hist, ckpt_dir / "history.json")
    results_dir = resolve_path(cfg, cfg["paths"]["results_dir"], args.model)
    plot_history(hist, results_dir / "curves", args.model)
    save_run_config(
        cfg,
        ckpt_dir / "run_config.json",
        extra={
            "model": args.model,
            "device": device,
            "vocab_size": pre.vocab_size,
            "num_classes": pre.num_classes,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
        },
    )

    print(json.dumps({"model": args.model, "checkpoint_dir": str(ckpt_dir)}, indent=2))


if __name__ == "__main__":
    main()

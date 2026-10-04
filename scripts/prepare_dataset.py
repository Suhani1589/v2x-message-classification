#!/usr/bin/env python3
"""Prepare dataset splits and fit the training-only tokenizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2x_cls.config import load_config, resolve_path
from v2x_cls.data import load_or_create_dataset, save_dataset_report, stratified_splits
from v2x_cls.preprocessing import build_preprocessor_from_config
from v2x_cls.seeds import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V2X dataset artifacts")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--regenerate-synthetic",
        action="store_true",
        help="Rebuild the synthetic proxy CSV (ignored if official CSV is present).",
    )
    parser.add_argument(
        "--hard-synthetic",
        action="store_true",
        help=(
            "Build/use a harder synthetic proxy (cross-class cues, typos, "
            "boundary near-duplicates). Overrides official CSV if present."
        ),
    )
    parser.add_argument(
        "--attention-fair-synthetic",
        action="store_true",
        help=(
            "Build/use attention-fair hard proxy (variable cue positions, "
            "mild semantic overlap; no heavy cue spam/typos)."
        ),
    )
    parser.add_argument(
        "--attention-prefer-synthetic",
        action="store_true",
        help=(
            "Build/use attention-prefer proxy (sparse cues + padding + label "
            "noise; targets Acc<1 and Attention-leading ablation)."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["seed"])

    df, meta = load_or_create_dataset(
        cfg,
        regenerate_synthetic=args.regenerate_synthetic,
        hard_synthetic=args.hard_synthetic,
        attention_fair_synthetic=args.attention_fair_synthetic,
        attention_prefer_synthetic=args.attention_prefer_synthetic,
    )
    train_df, val_df, test_df = stratified_splits(df, cfg)

    processed_dir = resolve_path(cfg, cfg["paths"]["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    pre = build_preprocessor_from_config(cfg)
    pre.fit(train_df["Message Content"].tolist())
    artifacts_dir = processed_dir / "artifacts"
    pre.save(artifacts_dir)

    split_meta = {
        **meta,
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "split_class_counts": {
            "train": train_df["Message Type"].value_counts().to_dict(),
            "val": val_df["Message Type"].value_counts().to_dict(),
            "test": test_df["Message Type"].value_counts().to_dict(),
        },
        "vocab_size": pre.vocab_size,
        "max_sequence_length": pre.max_sequence_length,
        "formulation": "multiclass",
    }
    save_dataset_report(split_meta, processed_dir / "dataset_report.json")

    print(json.dumps(split_meta, indent=2))
    print(f"\nSaved splits to {processed_dir}")
    print(f"Tokenizer/label mapping saved to {artifacts_dir}")


if __name__ == "__main__":
    main()

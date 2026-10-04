#!/usr/bin/env python3
"""Run the full research experiment: prepare → train all → evaluate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Full V2X experiment pipeline")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (e.g. 2 for smoke test)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["brnn", "cnn_brnn", "cnn_brnn_attention"],
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--regenerate-synthetic",
        action="store_true",
        help="Force rebuild of synthetic proxy during prepare step.",
    )
    parser.add_argument(
        "--hard-synthetic",
        action="store_true",
        help="Use harder synthetic proxy (cross-class overlap / noise).",
    )
    parser.add_argument(
        "--attention-fair-synthetic",
        action="store_true",
        help="Use attention-fair hard proxy (variable cues, mild overlap).",
    )
    parser.add_argument(
        "--attention-prefer-synthetic",
        action="store_true",
        help="Use attention-prefer proxy (sparse cues; Acc<1; Attention-led).",
    )
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_prepare:
        prep = [py, "scripts/prepare_dataset.py", "--config", args.config]
        if args.regenerate_synthetic:
            prep.append("--regenerate-synthetic")
        if args.hard_synthetic:
            prep.append("--hard-synthetic")
        if args.attention_fair_synthetic:
            prep.append("--attention-fair-synthetic")
        if args.attention_prefer_synthetic:
            prep.append("--attention-prefer-synthetic")
        run(prep)

    if not args.skip_train:
        for model in args.models:
            cmd = [py, "scripts/train.py", "--model", model, "--config", args.config]
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]
            if args.batch_size is not None:
                cmd += ["--batch-size", str(args.batch_size)]
            run(cmd)

    if not args.skip_eval:
        run([py, "scripts/evaluate.py", "--config", args.config, "--models", *args.models])

    # Sample inference with the proposed model if trained.
    if "cnn_brnn_attention" in args.models:
        run(
            [
                py,
                "scripts/predict.py",
                "--config",
                args.config,
                "--model",
                "cnn_brnn_attention",
                "--text",
                "Emergency vehicle approaching intersection",
            ]
        )

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()

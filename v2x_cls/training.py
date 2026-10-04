"""Device detection and training utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


def detect_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return "GPU"
    # Apple Metal plugin appears as GPU in recent TF-macos builds; also check MPS via torch if present.
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
        if "Apple" in out:
            return "CPU (Apple Silicon; TensorFlow CPU build)"
    except Exception:
        pass
    return "CPU"


def make_callbacks(checkpoint_dir: Path, cfg: Dict[str, Any]) -> List[tf.keras.callbacks.Callback]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor = cfg["training"]["checkpoint_monitor"]
    mode = cfg["training"]["checkpoint_mode"]
    callbacks: List[tf.keras.callbacks.Callback] = [
        ModelCheckpoint(
            filepath=str(checkpoint_dir / "best.keras"),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        CSVLogger(str(checkpoint_dir / "history.csv")),
        ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.5, patience=2, verbose=1),
    ]
    patience = cfg["training"].get("early_stopping_patience")
    if patience:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=int(patience),
                restore_best_weights=True,
                verbose=1,
            )
        )
    return callbacks


def save_history(history: Dict[str, List[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_run_config(cfg: Dict[str, Any], path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = dict(cfg)
    if extra:
        payload = {**payload, "run_extra": extra}
    # Drop non-serializable private keys carefully
    clean = {k: v for k, v in payload.items() if not str(k).startswith("_") or k in {"_project_root"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, default=str)

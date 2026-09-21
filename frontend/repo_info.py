"""Helpers to load repository artifacts for the dashboard (read-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]


def load_dataset_report() -> Optional[Dict[str, Any]]:
    path = ROOT / "data" / "processed" / "dataset_report.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def checkpoint_status() -> Dict[str, bool]:
    status = {}
    for name in ("brnn", "cnn_brnn", "cnn_brnn_attention"):
        ckpt = ROOT / "checkpoints" / name
        status[name] = (ckpt / "best.keras").exists() or (ckpt / "final.keras").exists()
    return status


def artifacts_ready() -> bool:
    arts = ROOT / "data" / "processed" / "artifacts"
    return (arts / "tokenizer.pkl").exists() and (arts / "label_mapping.json").exists()

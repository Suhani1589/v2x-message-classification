"""Read-only loaders for repository artifacts used by the command center."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

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


def load_comparison_table() -> Optional[pd.DataFrame]:
    path = ROOT / "experiments" / "results" / "comparison_table.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_test_metrics(model_key: str) -> Optional[Dict[str, Any]]:
    path = ROOT / "experiments" / "results" / model_key / "test_metrics.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_label_order() -> List[str]:
    path = ROOT / "data" / "processed" / "artifacts" / "label_mapping.json"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        order = data.get("label_order")
        if isinstance(order, list) and order:
            return list(order)
    return ["Emergency", "Hazard", "Roadwork", "Spam", "Traffic", "Weather"]


def load_attention_examples() -> List[Dict[str, Any]]:
    path = (
        ROOT
        / "experiments"
        / "results"
        / "cnn_brnn_attention"
        / "attention_examples"
        / "examples.json"
    )
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_message_corpus() -> Optional[pd.DataFrame]:
    """Held-out test split used as the message history corpus."""
    path = ROOT / "data" / "processed" / "test.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def system_ready() -> bool:
    return artifacts_ready() and any(checkpoint_status().values())

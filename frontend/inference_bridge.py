"""Thin bridge to existing CLI inference (scripts/predict.py). No model reimplementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2x_cls.config import load_config


def _load_predict_module():
    path = ROOT / "scripts" / "predict.py"
    spec = importlib.util.spec_from_file_location("v2x_cls_predict_cli", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load prediction module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_prediction(text: str, model_name: str) -> Dict[str, Any]:
    predict_mod = _load_predict_module()
    cfg = load_config(str(ROOT / "config.yaml"))
    return predict_mod.predict_text(text.strip(), cfg, model_name, save_plot=None)


def try_prediction(text: str, model_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text.strip():
        return None, "Empty message text."
    try:
        return run_prediction(text, model_name), None
    except FileNotFoundError as exc:
        return None, f"Checkpoint or artifacts missing: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"Inference failed: {exc}"


def try_batch_predictions(
    texts: List[str],
    model_name: str = "cnn_brnn_attention",
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Run existing predict_text on a list of corpus messages (no user input)."""
    results: List[Dict[str, Any]] = []
    for text in texts:
        result, err = try_prediction(text, model_name)
        if err:
            return results, err
        if result:
            results.append(result)
    return results, None

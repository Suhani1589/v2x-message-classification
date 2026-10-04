"""Configuration and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve_path(cfg: Dict[str, Any], *parts: str) -> Path:
    root = Path(cfg["_project_root"])
    return (root.joinpath(*parts)).resolve()

"""Compatibility shim — implementations live in v2x_cls.models."""

from v2x_cls.models import (
    build_brnn,
    build_cnn_brnn,
    build_cnn_brnn_attention,
    build_model_by_name,
    compile_model,
)
from v2x_cls.models.attention import AdditiveAttention, PaddingMask

__all__ = [
    "AdditiveAttention",
    "PaddingMask",
    "build_brnn",
    "build_cnn_brnn",
    "build_cnn_brnn_attention",
    "build_model_by_name",
    "compile_model",
]

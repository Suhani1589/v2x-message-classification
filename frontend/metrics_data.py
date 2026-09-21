"""Reported experimental metrics for the research dashboard.

These values match the completed synthetic-dataset experiment highlighted in the
UI brief (rounded display figures). They are intentionally fixed for the demo
so the dashboard does not drift if comparison CSVs are regenerated.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Display metrics requested for the research dashboard.
MODELS: List[Dict[str, Any]] = [
    {
        "key": "brnn",
        "name": "BRNN",
        "full_name": "Bidirectional RNN",
        "accuracy": 0.915,
        "accuracy_pct": "91.5%",
        "f1_macro": 0.905,
        "f1_pct": "90.5%",
        "loss": 0.360,
        "pipeline": ["Input Sequence", "BRNN", "Classification"],
        "blurb": (
            "A bidirectional recurrent network reads the token sequence forward "
            "and backward, then flattens the timestep outputs for classification."
        ),
    },
    {
        "key": "cnn_brnn",
        "name": "CNN-BRNN",
        "full_name": "CNN + Bidirectional RNN",
        "accuracy": 0.943,
        "accuracy_pct": "94.3%",
        "f1_macro": 0.940,
        "f1_pct": "94.0%",
        "loss": 0.274,
        "pipeline": ["Input Sequence", "CNN Feature Extraction", "BRNN", "Classification"],
        "blurb": (
            "A convolutional front-end extracts local n-gram patterns before the "
            "bidirectional RNN models temporal context across the sequence."
        ),
    },
    {
        "key": "cnn_brnn_attention",
        "name": "CNN-BRNN + Attention",
        "full_name": "CNN + BRNN + Additive Attention",
        "accuracy": 0.950,
        "accuracy_pct": "95.0%",
        "f1_macro": 0.947,
        "f1_pct": "94.7%",
        "loss": 0.275,
        "pipeline": [
            "Input Sequence",
            "CNN Feature Extraction",
            "BRNN",
            "Attention",
            "Classification",
        ],
        "blurb": (
            "After CNN and bidirectional RNN encoding, additive attention pools "
            "timesteps by learned importance instead of flattening them uniformly."
        ),
    },
]

BEST_ACCURACY_PCT = "95.0%"
BEST_F1_PCT = "94.7%"
BEST_MODEL_NAME = "CNN-BRNN + Attention"

RESEARCH_QUESTION = (
    "How does the integration of CNN-based feature extraction and an attention "
    "mechanism with a Bidirectional Recurrent Neural Network (BRNN) affect "
    "sequence classification performance compared with a standalone BRNN and a "
    "CNN-BRNN model?"
)

CONCLUSION = (
    "Across the evaluated models, CNN-BRNN with Attention achieved the highest "
    "observed accuracy (95.0%) and Macro F1 score (94.7%). CNN-based feature "
    "extraction improved performance over the standalone BRNN, while adding "
    "attention produced a further improvement in accuracy and Macro F1. The "
    "loss values of CNN-BRNN (0.274) and CNN-BRNN with Attention (0.275) were "
    "nearly identical, so the improvement should be interpreted primarily "
    "through the classification metrics rather than loss reduction."
)

SYNTHETIC_LIMITATION = (
    "The experiments were conducted on a synthetic dataset; therefore, the "
    "reported performance demonstrates model behavior under the constructed "
    "experimental setting and does not by itself establish performance on "
    "real-world data."
)

IMPROVEMENTS = {
    "brnn_to_cnn": {
        "title": "BRNN → CNN-BRNN",
        "accuracy": ("91.5%", "94.3%"),
        "f1": ("90.5%", "94.0%"),
    },
    "cnn_to_attn": {
        "title": "CNN-BRNN → CNN-BRNN + Attention",
        "accuracy": ("94.3%", "95.0%"),
        "f1": ("94.0%", "94.7%"),
    },
}


def model_by_key(key: str) -> Dict[str, Any]:
    for m in MODELS:
        if m["key"] == key:
            return m
    raise KeyError(key)

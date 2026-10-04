"""System Architecture — research pipeline visualization."""

from __future__ import annotations

import streamlit as st

from components.flows import architecture_models_html, system_pipeline_html
from components.ui import footer_html, section_header
from metrics_data import RESEARCH_QUESTION
from repo_info import artifacts_ready, checkpoint_status


def render() -> None:
    st.markdown(
        section_header("System Architecture", "Actual project pipeline"),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="panel">
  <div class="panel-title">Research question</div>
  <p style="margin:0;line-height:1.55;color:#E5E7EB;">“{RESEARCH_QUESTION}”</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        section_header("End-to-end Pipeline", "From V2X-style data to monitoring"),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="panel">{system_pipeline_html()}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        section_header("Model Family", "BRNN → CNN-BRNN → CNN-BRNN + Attention"),
        unsafe_allow_html=True,
    )
    st.markdown(architecture_models_html(), unsafe_allow_html=True)

    st.markdown(section_header("Runtime Components", "Repository mapping"), unsafe_allow_html=True)
    ckpts = checkpoint_status()
    rows = [
        ("Data preparation", "`v2x_cls/data.py`, `scripts/prepare_dataset.py`"),
        ("Preprocessing", "`v2x_cls/preprocessing.py` (tokenizer, maxlen=12)"),
        ("Models", "`v2x_cls/models/` — BRNN, CNN-BRNN, Additive Attention"),
        ("Training / evaluation", "`scripts/train.py`, `scripts/evaluate.py`, `v2x_cls/training.py`"),
        ("Inference", "`scripts/predict.py` (reused by this frontend)"),
        ("Config", "`config.yaml`, `requirements.txt`"),
        ("Command center UI", "`frontend/` (this application)"),
    ]
    for title, detail in rows:
        st.markdown(
            f"""
<div class="panel" style="padding:0.85rem 1rem;">
  <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:0.5rem;">
    <span style="color:#22D3EE;font-family:Orbitron,sans-serif;font-size:0.75rem;letter-spacing:0.08em;text-transform:uppercase;">{title}</span>
    <span class="muted">{detail}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(section_header("Artifact Readiness", ""), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tokenizer", "Ready" if artifacts_ready() else "Missing")
    c2.metric("BRNN ckpt", "Ready" if ckpts.get("brnn") else "Missing")
    c3.metric("CNN-BRNN ckpt", "Ready" if ckpts.get("cnn_brnn") else "Missing")
    c4.metric("Attention ckpt", "Ready" if ckpts.get("cnn_brnn_attention") else "Missing")

    st.markdown(footer_html(), unsafe_allow_html=True)

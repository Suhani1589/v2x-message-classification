"""Message Classification — auto-displayed real model outputs (no user input)."""

from __future__ import annotations

from html import escape

import streamlit as st

from charts import real_attention_figure
from components.ui import (
    confidence_bar_html,
    empty_state,
    footer_html,
    section_header,
)
from repo_info import load_attention_examples


def _show_chart(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render() -> None:
    st.markdown(section_header("Message Classification", "Automatic monitoring view"), unsafe_allow_html=True)
    st.markdown(
        """
<div class="panel">
  <p class="muted" style="margin:0;line-height:1.5;">
    This page does not accept operator text input. Classifications below come from
    saved attention-example inference artifacts produced by the evaluation run
    (<code>experiments/results/cnn_brnn_attention/attention_examples/examples.json</code>).
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        section_header("Saved Attention Examples", "experiments/results/.../attention_examples/examples.json"),
        unsafe_allow_html=True,
    )
    examples = load_attention_examples()
    if not examples:
        st.markdown(
            empty_state("No attention-example classifications are available on disk."),
            unsafe_allow_html=True,
        )
    else:
        for ex in examples:
            pred = str(ex.get("predicted", "—"))
            conf = float(ex.get("confidence", 0.0)) * 100.0
            text = str(ex.get("text", ""))
            warn = pred == "Spam"
            top = ex.get("top_tokens") or []
            top_str = ", ".join(
                f"{t.get('token')} ({float(t.get('weight', 0)):.2f})" for t in top[:5]
            )
            st.markdown(
                f"""
<div class="class-card{" spam" if warn else ""}">
  <div class="class-meta">
    <span class="class-label">{escape(pred)}</span>
    <span class="badge{" warn" if warn else " ok"}">{"Spam category" if warn else "Classified"}</span>
    <span class="muted">Saved model output</span>
  </div>
  <p class="msg-text">{escape(text)}</p>
  {confidence_bar_html(conf, warn=warn)}
  <div class="muted" style="margin-top:0.35rem;">Top attended tokens: {escape(top_str) if top_str else "—"}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            weights = ex.get("token_weights") or []
            toks = [w["token"] for w in weights if w.get("token") and w.get("token") != "<PAD>"]
            vals = [float(w["weight"]) for w in weights if w.get("token") and w.get("token") != "<PAD>"]
            if toks and vals:
                with st.expander(f"Attention weights — {pred}", expanded=False):
                    _show_chart(real_attention_figure(toks, vals, f"Attention — predicted {pred}"))

    st.markdown(footer_html(), unsafe_allow_html=True)

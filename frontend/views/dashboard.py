"""Dashboard — system status overview from real artifacts only."""

from __future__ import annotations

from html import escape

import streamlit as st

from charts import accuracy_chart, f1_chart, loss_chart
from components.flows import pipeline_flow_html, v2x_comm_html
from components.ui import (
    confidence_bar_html,
    empty_state,
    footer_html,
    metric_card_html,
    section_header,
    status_pill_html,
)
from metrics_data import BEST_ACCURACY_PCT, BEST_F1_PCT, BEST_MODEL_NAME, MODELS
from repo_info import (
    checkpoint_status,
    load_attention_examples,
    load_dataset_report,
    system_ready,
)


def _show_chart(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render() -> None:
    report = load_dataset_report()
    ckpts = checkpoint_status()
    ready = system_ready()
    examples = load_attention_examples()
    n_models_ready = sum(1 for ok in ckpts.values() if ok)
    n_rows = report.get("n_rows") if report else None
    latest = examples[0] if examples else None

    st.markdown(
        f"""
<div class="hero-panel">
  <div class="hero-kicker">V2X Command Center</div>
  <h1 class="hero-title">Intelligent V2X Communication &amp; Message Classification</h1>
  <p class="hero-lead">
    Research monitoring surface for a connected-vehicle message classifier.
    The dashboard visualizes the existing pipeline, held-out evaluation metrics,
    and model outputs — without requiring operator message entry.
  </p>
  <div class="badge-row">
    <span class="badge">Connected-vehicle research</span>
    <span class="badge">6-class multiclass</span>
    <span class="badge">CNN-BRNN + Attention</span>
    <span class="badge warn">Synthetic experimental dataset</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(section_header("System Status", "Live artifact readiness"), unsafe_allow_html=True)

    status_label = "NORMAL — artifacts online" if ready else "ATTENTION — checkpoints/artifacts incomplete"
    st.markdown(
        status_pill_html(status_label, attention=not ready),
        unsafe_allow_html=True,
    )
    st.write("")

    cards = [
        metric_card_html(
            "System Status",
            "Artifacts ready" if ready else "Degraded",
            "Checkpoints + tokenizer on disk" if ready else "Missing artifacts",
        ),
        metric_card_html(
            "Classification Engine",
            BEST_MODEL_NAME,
            f"{n_models_ready}/3 model checkpoints present",
        ),
        metric_card_html(
            "Messages in Corpus",
            str(n_rows) if n_rows is not None else "—",
            "From dataset_report.json" if n_rows is not None else "Report unavailable",
        ),
        metric_card_html(
            "Best Test Accuracy",
            BEST_ACCURACY_PCT,
            f"Macro F1 {BEST_F1_PCT}",
        ),
    ]
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    st.markdown(section_header("V2X Message Overview", "End-to-end flow"), unsafe_allow_html=True)
    st.markdown(f'<div class="panel">{pipeline_flow_html()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel"><div class="panel-title">Communication path</div>{v2x_comm_html()}</div>', unsafe_allow_html=True)

    st.markdown(
        section_header("Reported Model Snapshot", "Fixed display figures from completed experiment"),
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        _show_chart(accuracy_chart())
    with c2:
        _show_chart(f1_chart())
    with c3:
        _show_chart(loss_chart())

    st.caption(
        "Highest Accuracy / Macro F1: CNN-BRNN + Attention. "
        "Lowest Loss: CNN-BRNN (0.274); Attention loss is 0.275 (nearly identical)."
    )

    st.markdown(
        section_header("Latest Classification", "From saved attention-example inference outputs"),
        unsafe_allow_html=True,
    )
    if latest:
        pred = str(latest.get("predicted", "—"))
        conf = float(latest.get("confidence", 0.0)) * 100.0
        text = str(latest.get("text", ""))
        warn = pred == "Spam"
        st.markdown(
            f"""
<div class="class-card{" spam" if warn else ""}">
  <div class="class-meta">
    <span class="class-label">{escape(pred)}</span>
    <span class="badge{" warn" if warn else " ok"}">{"Spam / attention category" if warn else "Classified"}</span>
    <span class="muted">Model: CNN-BRNN + Attention · artifact: attention_examples</span>
  </div>
  <p class="msg-text">{escape(text)}</p>
  {confidence_bar_html(conf, warn=warn)}
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            empty_state(
                "No saved attention-example classifications found under "
                "experiments/results/cnn_brnn_attention/attention_examples/."
            ),
            unsafe_allow_html=True,
        )

    st.markdown(section_header("Model Roster", ""), unsafe_allow_html=True)
    cols = st.columns(3)
    for col, m in zip(cols, MODELS):
        ok = ckpts.get(m["key"], False)
        with col:
            st.markdown(
                f"""
<div class="panel">
  <div class="panel-title">{escape(m["name"])}</div>
  <div class="muted">{escape(m["full_name"])}</div>
  <p style="margin:0.65rem 0 0.25rem;font-family:Orbitron,sans-serif;font-size:1.2rem;color:#F8FAFC;">
    {escape(m["accuracy_pct"])}
  </p>
  <div class="muted">Macro F1 {escape(m["f1_pct"])} · Loss {m["loss"]:.3f}</div>
  <div style="margin-top:0.65rem;">{status_pill_html("Checkpoint ready" if ok else "Checkpoint missing", attention=not ok)}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown(footer_html(), unsafe_allow_html=True)

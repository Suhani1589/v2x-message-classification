"""Streamlit research dashboard for V2X CNN-BRNN + Attention.

Run from the repository root:
  streamlit run frontend/app.py

This UI is intentionally separate from training/evaluation code. Inference
reuses scripts/predict.py and existing checkpoints when available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

from charts import (  # noqa: E402
    accuracy_chart,
    conceptual_attention_figure,
    f1_chart,
    loss_chart,
    real_attention_figure,
)
from inference_bridge import try_prediction  # noqa: E402
from metrics_data import (  # noqa: E402
    BEST_ACCURACY_PCT,
    BEST_F1_PCT,
    BEST_MODEL_NAME,
    CONCLUSION,
    IMPROVEMENTS,
    MODELS,
    RESEARCH_QUESTION,
    SYNTHETIC_LIMITATION,
    model_by_key,
)
from repo_info import artifacts_ready, checkpoint_status, load_dataset_report  # noqa: E402
from styles import APP_CSS, architecture_html  # noqa: E402

st.set_page_config(
    page_title="CNN-BRNN + Attention | Research Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f"## {title}")


# ---------------------------------------------------------------------------
# 1. Hero
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  <div class="badge-row">
    <span class="badge">Research Project</span>
    <span class="badge">Synthetic Dataset</span>
    <span class="badge">Sequence Classification</span>
  </div>
  <h1>Sequence Classification with CNN-BRNN and Attention</h1>
  <div class="subtitle">
    Comparative Analysis of BRNN, CNN-BRNN, and CNN-BRNN with Attention
  </div>
  <p style="margin:0;max-width:48rem;color:rgba(248,250,252,0.9);line-height:1.55;">
    This dashboard presents a controlled comparison of three deep sequence models
    for multiclass V2X message classification. The study asks whether adding
    convolutional feature extraction and attention improves over a standalone
    bidirectional RNN — under a clearly labeled synthetic experimental setting.
  </p>
  <div class="metric-pills">
    <div class="metric-pill">
      <div class="label">Best observed Accuracy</div>
      <div class="value">{BEST_ACCURACY_PCT}</div>
      <div style="font-size:0.82rem;opacity:0.8;">{BEST_MODEL_NAME}</div>
    </div>
    <div class="metric-pill">
      <div class="label">Best observed Macro F1</div>
      <div class="value">{BEST_F1_PCT}</div>
      <div style="font-size:0.82rem;opacity:0.8;">{BEST_MODEL_NAME}</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 2. Research question
# ---------------------------------------------------------------------------
section("Research Question")
st.markdown(
    f"""
<div class="section-card rq-card">
  <div style="font-size:0.78rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;color:#0F766E;">
    Central question
  </div>
  <p class="rq-text">“{RESEARCH_QUESTION}”</p>
  <div style="font-size:0.9rem;color:#334155;margin-bottom:0.35rem;"><b>Experimental progression</b></div>
  <div class="flow">
    <span class="flow-step">BRNN</span><span class="flow-arrow">→</span>
    <span class="flow-step">Bidirectional temporal modeling</span>
  </div>
  <div class="flow">
    <span class="flow-step">CNN + BRNN</span><span class="flow-arrow">→</span>
    <span class="flow-step">Spatial/local feature extraction + bidirectional temporal modeling</span>
  </div>
  <div class="flow">
    <span class="flow-step">CNN + BRNN + Attention</span><span class="flow-arrow">→</span>
    <span class="flow-step">Spatial features + bidirectional temporal modeling + focus on informative parts of the sequence</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 3. Architectures
# ---------------------------------------------------------------------------
section("Model Architectures")
st.markdown(architecture_html(), unsafe_allow_html=True)
with st.expander("Beginner-friendly component guide", expanded=False):
    st.markdown(
        """
- **Input sequence** — tokenized message text padded/truncated to a fixed length.
- **CNN** — scans short windows of tokens to detect local patterns.
- **BRNN** — processes the sequence in both directions for surrounding context.
- **Flatten** — concatenates all timestep vectors (used by BRNN / CNN-BRNN).
- **Attention** — learns a weight for each timestep and forms a weighted summary.
- **Classification head** — maps the summary vector to one of six message classes.
"""
    )

# ---------------------------------------------------------------------------
# 4. Dataset
# ---------------------------------------------------------------------------
section("Dataset")
report = load_dataset_report()
st.markdown(
    f"""
<div class="section-card">
  <div class="badge-row" style="margin-bottom:0.75rem;">
    <span class="badge" style="background:#0F766E;border:none;">Dataset Type: Synthetic</span>
  </div>
  <p style="color:#334155;line-height:1.55;margin-top:0;">
    Experiments were run on a <b>synthetic</b> V2X-style message corpus constructed
    for controlled model comparison. The official IEEE DataPort CSV is not required
    for this dashboard; the synthetic setting remains intentional and is stated
    throughout the results.
  </p>
  <div class="limit-box">{SYNTHETIC_LIMITATION}</div>
</div>
""",
    unsafe_allow_html=True,
)

if report:
    cols = st.columns(4)
    cols[0].metric("Rows", f"{report.get('n_rows', '—')}")
    cols[1].metric("Vocab size", f"{report.get('vocab_size', '—')}")
    cols[2].metric("Max sequence length", f"{report.get('max_sequence_length', '—')}")
    cols[3].metric("Formulation", str(report.get("formulation", "—")))
    if report.get("class_counts"):
        st.caption("Class counts from `data/processed/dataset_report.json` (current prepared split source).")
        st.dataframe(
            pd.DataFrame(
                [{"Class": k, "Count": v} for k, v in report["class_counts"].items()]
            ),
            hide_index=True,
            width="stretch",
        )
    src = report.get("source")
    if src:
        st.caption(f"Dataset report source key: `{src}`")
else:
    st.info("Dataset report not found at `data/processed/dataset_report.json`. Metrics below still use the reported experimental figures.")

# ---------------------------------------------------------------------------
# 5. Performance dashboard
# ---------------------------------------------------------------------------
section("Model Performance Dashboard")
st.caption("Reported experimental results on the synthetic dataset (display figures).")

c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(accuracy_chart(), width="stretch")
with c2:
    st.plotly_chart(f1_chart(), width="stretch")
with c3:
    st.plotly_chart(loss_chart(), width="stretch")

table = pd.DataFrame(
    [
        {
            "Model": m["name"],
            "Accuracy": m["accuracy_pct"],
            "Macro F1": m["f1_pct"],
            "Loss": f"{m['loss']:.3f}",
        }
        for m in MODELS
    ]
)
st.dataframe(table, hide_index=True, width="stretch")
st.markdown(
    """
- Highest **Accuracy** and **Macro F1**: <span class="highlight">CNN-BRNN + Attention</span>
- Lowest **Loss**: <span class="highlight">CNN-BRNN (0.274)</span> — Attention loss is **0.275** (nearly identical; not lower)
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 6. Improvements
# ---------------------------------------------------------------------------
section("Performance Improvement")
st.markdown('<div class="delta-grid">', unsafe_allow_html=True)
left, right = st.columns(2)
for col, key in ((left, "brnn_to_cnn"), (right, "cnn_to_attn")):
    block = IMPROVEMENTS[key]
    with col:
        st.markdown(
            f"""
<div class="delta-card">
  <h4>{block["title"]}</h4>
  <div class="delta-row"><span>Accuracy</span><span><span class="delta-from">{block["accuracy"][0]}</span> → <span class="delta-to">{block["accuracy"][1]}</span></span></div>
  <div class="delta-row"><span>Macro F1</span><span><span class="delta-from">{block["f1"][0]}</span> → <span class="delta-to">{block["f1"][1]}</span></span></div>
</div>
""",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# 7. Attention explanation
# ---------------------------------------------------------------------------
section("Attention Mechanism")
st.markdown(
    """
<div class="section-card">
  <p style="color:#334155;line-height:1.55;">
    Attention allows the model to assign different importance to different parts
    of the sequence, helping it focus on the most informative representations.
  </p>
  <div class="flow">
    <span class="flow-step">t1</span><span class="flow-arrow">→</span>
    <span class="flow-step">t2</span><span class="flow-arrow">→</span>
    <span class="flow-step">t3</span><span class="flow-arrow">→</span>
    <span class="flow-step">t4</span><span class="flow-arrow">→</span>
    <span class="flow-step">t5</span>
  </div>
  <div class="flow">
    <span class="flow-step">low</span><span class="flow-arrow">→</span>
    <span class="flow-step">low</span><span class="flow-arrow">→</span>
    <span class="flow-step">high</span><span class="flow-arrow">→</span>
    <span class="flow-step">high</span><span class="flow-arrow">→</span>
    <span class="flow-step">low</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.plotly_chart(conceptual_attention_figure(), width="stretch")
st.caption("Conceptual illustration — not measured weights from a specific prediction.")

# ---------------------------------------------------------------------------
# 8. Interactive model comparison
# ---------------------------------------------------------------------------
section("Interactive Model Comparison")
choice = st.selectbox(
    "Select a model",
    options=[m["key"] for m in MODELS],
    format_func=lambda k: model_by_key(k)["name"],
)
selected = model_by_key(choice)
m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", selected["accuracy_pct"])
m2.metric("Macro F1", selected["f1_pct"])
m3.metric("Loss", f"{selected['loss']:.3f}")
st.markdown("**Architecture**")
st.code(" → ".join(selected["pipeline"]))
st.write(selected["blurb"])

# ---------------------------------------------------------------------------
# 9. Inference demo
# ---------------------------------------------------------------------------
section("Inference / Demo")
ckpts = checkpoint_status()
ready = artifacts_ready() and any(ckpts.values())

if not ready:
    st.warning(
        "Trained-model inference is available in this repository via "
        "`scripts/predict.py`, but checkpoints and/or preprocessing artifacts "
        "were not found in the expected locations. No fake predictions are shown."
    )
    st.markdown(
        """
**Model Demo (offline)**  
Once `checkpoints/*/best.keras` and `data/processed/artifacts/` are present, this
section loads the real models. Until then, use the CLI:

```bash
python scripts/predict.py --model cnn_brnn_attention --text "Emergency vehicle approaching intersection"
```
"""
    )
else:
    st.success("Checkpoints and tokenizer artifacts detected — live inference enabled.")
    missing = [k for k, ok in ckpts.items() if not ok]
    if missing:
        st.info("Missing checkpoints for: " + ", ".join(missing))

    demo_options = [k for k, ok in ckpts.items() if ok]
    default_infer = (
        "cnn_brnn_attention"
        if "cnn_brnn_attention" in demo_options
        else demo_options[0]
    )
    demo_model = st.selectbox(
        "Inference model",
        options=demo_options,
        index=demo_options.index(default_infer),
        format_func=lambda k: model_by_key(k)["name"],
        key="infer_model",
    )
    text = st.text_area(
        "V2X message text",
        value="Emergency vehicle approaching intersection",
        height=90,
    )
    if st.button("Run prediction", type="primary"):
        with st.spinner("Running model…"):
            result, err = try_prediction(text, demo_model)
        if err:
            st.error(err)
        elif result:
            st.markdown(f"**Predicted class:** `{result['predicted_class']}`")
            st.markdown(f"**Confidence:** {result['confidence_percent']}%")
            prob_df = pd.DataFrame(
                [
                    {"Class": k, "Probability": v}
                    for k, v in result["class_probabilities"].items()
                ]
            ).sort_values("Probability", ascending=False)
            st.dataframe(prob_df, hide_index=True, width="stretch")
            if "attention_weights" in result:
                pairs = [p for p in result["attention_weights"] if p["token"] != "<PAD>"]
                if pairs:
                    st.markdown("**Real attention weights** from the companion attention model")
                    st.caption(result.get("attention_note", ""))
                    st.plotly_chart(
                        real_attention_figure(
                            [p["token"] for p in pairs],
                            [p["weight"] for p in pairs],
                            f"Attention weights — {result['predicted_class']}",
                        ),
                        width="stretch",
                    )
                    if result.get("important_tokens"):
                        st.write("Top tokens:", ", ".join(result["important_tokens"]))

# ---------------------------------------------------------------------------
# 10. Conclusion
# ---------------------------------------------------------------------------
section("Research Conclusion")
st.markdown(
    f"""
<div class="section-card rq-card">
  <p class="rq-text" style="font-size:1.02rem;">{CONCLUSION}</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 11. Limitations
# ---------------------------------------------------------------------------
section("Limitations")
st.markdown(
    f"""
<div class="section-card">
  <ul style="color:#334155;line-height:1.6;">
    <li><b>Synthetic dataset</b> — {SYNTHETIC_LIMITATION}</li>
    <li>Results may not directly generalize to real-world V2X traffic.</li>
    <li>Further validation on real-world datasets would be required before operational claims.</li>
    <li>Additional experiments could evaluate robustness and generalization more thoroughly.</li>
  </ul>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 12. Future work
# ---------------------------------------------------------------------------
section("Future Work")
st.markdown(
    """
<div class="section-card">
  <ul style="color:#334155;line-height:1.6;">
    <li>Evaluation on real-world datasets</li>
    <li>Larger and more diverse datasets</li>
    <li>Cross-dataset validation</li>
    <li>Hyperparameter tuning</li>
    <li>More extensive ablation studies</li>
    <li>Broader attention visualization studies using real model weights</li>
  </ul>
  <p class="footnote">These items are proposed next steps — they have not been completed in this dashboard.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")
st.caption(
    "UI module: `frontend/` · Research code unchanged under `v2x_cls/` and `scripts/` · "
    "Synthetic-dataset experiment demo"
)

"""Model Performance — real evaluation metrics and confusion matrices."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import confusion_matrix_figure, model_metrics_radar
from components.ui import empty_state, footer_html, metric_card_html, section_header
from metrics_data import IMPROVEMENTS, MODELS, model_by_key
from repo_info import load_comparison_table, load_label_order, load_test_metrics


def _show_chart(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def render() -> None:
    st.markdown(
        section_header("Model Performance", "Held-out test evaluation from project artifacts"),
        unsafe_allow_html=True,
    )

    keys = [m["key"] for m in MODELS]
    choice = st.selectbox(
        "Model",
        options=keys,
        index=keys.index("cnn_brnn_attention") if "cnn_brnn_attention" in keys else 0,
        format_func=lambda k: model_by_key(k)["name"],
    )
    selected = model_by_key(choice)
    metrics = load_test_metrics(choice)

    st.markdown(
        f"""
<div class="panel">
  <div class="panel-title">{selected["name"]}</div>
  <p class="muted" style="margin:0;line-height:1.5;">{selected["blurb"]}</p>
  <div class="muted" style="margin-top:0.55rem;">Pipeline: {" → ".join(selected["pipeline"])}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Prefer detailed JSON metrics when present; fall back to display constants
    if metrics:
        cards = [
            metric_card_html("Accuracy", _fmt_pct(metrics["accuracy"]), "test_metrics.json"),
            metric_card_html("Precision (macro)", _fmt_pct(metrics["precision_macro"]), "macro average"),
            metric_card_html("Recall (macro)", _fmt_pct(metrics["recall_macro"]), "macro average"),
            metric_card_html("F1 (macro)", _fmt_pct(metrics["f1_macro"]), "macro average"),
        ]
        st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Loss", f"{float(metrics['loss']):.3f}")
        with c2:
            st.metric("Weighted F1", _fmt_pct(metrics.get("f1_weighted", 0)))
        st.caption(
            "Display dashboard figures (research brief): "
            f"Acc {selected['accuracy_pct']} · F1 {selected['f1_pct']} · Loss {selected['loss']:.3f}. "
            "JSON values above are the exact evaluation dump."
        )
    else:
        cards = [
            metric_card_html("Accuracy", selected["accuracy_pct"], "display figure"),
            metric_card_html("Macro F1", selected["f1_pct"], "display figure"),
            metric_card_html("Loss", f"{selected['loss']:.3f}", "display figure"),
            metric_card_html("Status", "JSON metrics missing", "using display constants"),
        ]
        st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    # Confusion matrix
    st.markdown(
        section_header("Confusion Matrix", "Actual (rows) vs Predicted (columns)"),
        unsafe_allow_html=True,
    )
    labels = load_label_order()
    if metrics and metrics.get("confusion_matrix"):
        _show_chart(
            confusion_matrix_figure(
                metrics["confusion_matrix"],
                labels,
                title=f"Confusion matrix — {selected['name']} (test set)",
            )
        )
        st.markdown(
            """
<div class="panel">
  <p class="muted" style="margin:0;line-height:1.5;">
    Each cell is the count of test examples with a given true label (row) that the
    model predicted as a given class (column). The diagonal shows correct predictions.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(empty_state("Confusion matrix not available for this model."), unsafe_allow_html=True)

    # Per-class table
    if metrics and metrics.get("per_class"):
        st.markdown(section_header("Per-class Metrics", ""), unsafe_allow_html=True)
        rows = []
        for lab, vals in metrics["per_class"].items():
            rows.append(
                {
                    "Class": lab,
                    "Precision": _fmt_pct(vals["precision"]),
                    "Recall": _fmt_pct(vals["recall"]),
                    "F1": _fmt_pct(vals["f1"]),
                    "Support": int(vals["support"]),
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")

    # Comparison table + improvements
    st.markdown(section_header("Cross-model Comparison", ""), unsafe_allow_html=True)
    cmp_df = load_comparison_table()
    if cmp_df is not None:
        pretty = cmp_df.copy()
        for col in [
            "Accuracy",
            "Precision_macro",
            "Recall_macro",
            "F1_macro",
            "Precision_weighted",
            "Recall_weighted",
            "F1_weighted",
        ]:
            if col in pretty.columns:
                pretty[col] = pretty[col].map(lambda x: f"{100 * float(x):.1f}%")
        if "Loss" in pretty.columns:
            pretty["Loss"] = pretty["Loss"].map(lambda x: f"{float(x):.3f}")
        st.dataframe(pretty, hide_index=True, width="stretch")

        attn = cmp_df[cmp_df["Model"] == "cnn_brnn_attention"]
        if not attn.empty:
            row = attn.iloc[0].to_dict()
            _show_chart(model_metrics_radar(row))
    else:
        st.dataframe(
            [
                {
                    "Model": m["name"],
                    "Accuracy": m["accuracy_pct"],
                    "Macro F1": m["f1_pct"],
                    "Loss": f"{m['loss']:.3f}",
                }
                for m in MODELS
            ],
            hide_index=True,
            width="stretch",
        )

    left, right = st.columns(2)
    for col, key in ((left, "brnn_to_cnn"), (right, "cnn_to_attn")):
        block = IMPROVEMENTS[key]
        with col:
            st.markdown(
                f"""
<div class="panel">
  <div class="panel-title">{block["title"]}</div>
  <div class="muted">Accuracy: {block["accuracy"][0]} → <b style="color:#22D3EE">{block["accuracy"][1]}</b></div>
  <div class="muted">Macro F1: {block["f1"][0]} → <b style="color:#22D3EE">{block["f1"][1]}</b></div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
<div class="panel">
  <div class="panel-title">Loss note</div>
  <p class="muted" style="margin:0;line-height:1.5;">
    CNN-BRNN loss is 0.274 and CNN-BRNN + Attention loss is 0.275 — nearly identical.
    Attention is not claimed to reduce loss; gains are reported on Accuracy and Macro F1.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(footer_html(), unsafe_allow_html=True)

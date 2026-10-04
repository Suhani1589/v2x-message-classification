"""Analytics — charts from real dataset report and evaluation artifacts."""

from __future__ import annotations

import streamlit as st

from charts import (
    accuracy_chart,
    class_distribution_chart,
    f1_chart,
    loss_chart,
    per_class_f1_chart,
    split_distribution_chart,
)
from components.ui import empty_state, footer_html, section_header
from metrics_data import MODELS
from repo_info import load_dataset_report, load_test_metrics


def _show_chart(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render() -> None:
    st.markdown(
        section_header("Analytics", "Dataset composition and comparative model metrics"),
        unsafe_allow_html=True,
    )

    report = load_dataset_report()
    if report and report.get("class_counts"):
        c1, c2 = st.columns(2)
        with c1:
            _show_chart(class_distribution_chart(report["class_counts"]))
        with c2:
            if report.get("split_sizes"):
                _show_chart(split_distribution_chart(report["split_sizes"]))
            else:
                st.markdown(empty_state("Split sizes unavailable."), unsafe_allow_html=True)
        st.caption(
            f"Dataset source key: `{report.get('source', '—')}` · "
            f"rows={report.get('n_rows', '—')} · vocab={report.get('vocab_size', '—')} · "
            f"maxlen={report.get('max_sequence_length', '—')}"
        )
    else:
        st.markdown(
            empty_state("dataset_report.json not found — category analytics unavailable."),
            unsafe_allow_html=True,
        )

    st.markdown(
        section_header("Comparative Performance", "Reported display metrics"),
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    with a:
        _show_chart(accuracy_chart())
    with b:
        _show_chart(f1_chart())
    with c:
        _show_chart(loss_chart())

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

    st.markdown(
        section_header("Per-class F1", "CNN-BRNN + Attention test_metrics.json"),
        unsafe_allow_html=True,
    )
    metrics = load_test_metrics("cnn_brnn_attention")
    if metrics and metrics.get("per_class"):
        fig = per_class_f1_chart(metrics["per_class"])
        if fig is not None:
            _show_chart(fig)
    else:
        st.markdown(
            empty_state("Per-class metrics file not available."),
            unsafe_allow_html=True,
        )

    st.markdown(footer_html(), unsafe_allow_html=True)

"""Message History — held-out test corpus browser (real rows only)."""

from __future__ import annotations

import streamlit as st

from components.ui import empty_state, footer_html, section_header
from repo_info import load_label_order, load_message_corpus


def render() -> None:
    st.markdown(
        section_header("Message History", "Held-out test split corpus (data/processed/test.csv)"),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="panel">
  <p class="muted" style="margin:0;line-height:1.5;">
    These records are the evaluation messages already prepared by the project pipeline.
    Columns are ground-truth labels from the corpus — not fabricated traffic.
    Filter by category using the control below (no free-text message entry).
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    df = load_message_corpus()
    if df is None or df.empty:
        st.markdown(
            empty_state("Test corpus not found at data/processed/test.csv."),
            unsafe_allow_html=True,
        )
        st.markdown(footer_html(), unsafe_allow_html=True)
        return

    labels = ["All"] + load_label_order()
    # Ensure any unexpected labels still appear
    for lab in sorted(df["Message Type"].dropna().unique().tolist()):
        if lab not in labels:
            labels.append(lab)

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        selected = st.selectbox("Category filter", options=labels, index=0)
    with c2:
        sort_by = st.selectbox(
            "Sort by",
            options=["Message ID", "Message Type", "Priority", "Source Vehicle"],
            index=0,
        )
    with c3:
        page_size = st.selectbox("Rows per page", options=[10, 20, 50, 100], index=1)

    view = df.copy()
    if selected != "All":
        view = view[view["Message Type"] == selected]

    if sort_by in view.columns:
        view = view.sort_values(sort_by)

    total = len(view)
    n_pages = max(1, (total + page_size - 1) // page_size)
    page = st.selectbox("Page", options=list(range(1, n_pages + 1)), index=0)
    start = (page - 1) * page_size
    page_df = view.iloc[start : start + page_size]

    show_cols = [
        c
        for c in [
            "Message ID",
            "Source Vehicle",
            "Destination Vehicle",
            "Message Type",
            "Message Content",
            "Priority",
            "Spam",
        ]
        if c in page_df.columns
    ]

    st.caption(f"Showing {len(page_df)} of {total} filtered records (test split).")
    st.dataframe(page_df[show_cols], hide_index=True, width="stretch")

    # Category security panel — Spam is a real project class
    if "Message Type" in df.columns:
        spam_n = int((df["Message Type"] == "Spam").sum())
        st.markdown(
            section_header("Corpus Attention Panel", "Based on ground-truth labels in the test split"),
            unsafe_allow_html=True,
        )
        if spam_n > 0:
            st.markdown(
                f"""
<div class="panel">
  <span class="status-pill attention"><span class="dot"></span>SPAM LABELS PRESENT</span>
  <p class="muted" style="margin:0.75rem 0 0;line-height:1.5;">
    The held-out test split contains <b>{spam_n}</b> messages labeled <b>Spam</b>
    (of {len(df)} total). Spam is one of the six trained categories — this panel
    highlights that class for monitoring, not a fabricated threat score.
  </p>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="panel">
  <span class="status-pill"><span class="dot"></span>NO SPAM LABELS IN TEST SPLIT</span>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown(footer_html(), unsafe_allow_html=True)

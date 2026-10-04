"""V2X Command Center — Streamlit frontend entrypoint.

Run from the repository root:
  streamlit run frontend/app.py

No operator message entry. Visualizes existing project artifacts and reuses
scripts/predict.py for automatic classification of corpus samples.
Does not modify training/evaluation logic under v2x_cls/ or scripts/.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import streamlit as st

warnings.filterwarnings(
    "ignore",
    message=".*keyword arguments have been deprecated.*",
)

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

from styles import APP_CSS  # noqa: E402
from views import (  # noqa: E402
    about,
    analytics,
    architecture,
    classification,
    dashboard,
    history,
    performance,
)

st.set_page_config(
    page_title="V2X Command Center",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

PAGES = {
    "Dashboard": dashboard.render,
    "Message Classification": classification.render,
    "Analytics": analytics.render,
    "Message History": history.render,
    "Model Performance": performance.render,
    "System Architecture": architecture.render,
    "About": about.render,
}

with st.sidebar:
    st.markdown('<div class="v2x-brand">V2X Command Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="v2x-brand-sub">Intelligent transportation · message classification · research monitoring</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        options=list(PAGES.keys()),
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("No manual message entry · real project artifacts only")
    st.caption("GitHub: Suhani1589/v2x-message-classification")

PAGES[page]()

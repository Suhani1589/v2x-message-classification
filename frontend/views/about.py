"""About — structured project briefing for demos / viva."""

from __future__ import annotations

import streamlit as st

from components.ui import footer_html, section_header
from metrics_data import CONCLUSION, RESEARCH_QUESTION, SYNTHETIC_LIMITATION


def render() -> None:
    st.markdown(
        section_header("About", "V2X Message Classification with Attention"),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="hero-panel">
  <div class="hero-kicker">Project briefing</div>
  <h1 class="hero-title">AI-Powered V2X Communication &amp; Message Classification Platform</h1>
  <p class="hero-lead">
    A research system that classifies short vehicle-to-everything style messages into
    operational categories using bidirectional RNNs, convolutional features, and attention.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    cards = [
        (
            "What is V2X?",
            "Vehicle-to-Everything (V2X) covers communication among vehicles, infrastructure, "
            "pedestrians, and networks. This project focuses on classifying message text content "
            "into traffic-relevant categories.",
        ),
        (
            "Why it matters",
            "Connected vehicles exchange safety and status information at high volume. "
            "Automatic categorization helps prioritize emergency/hazard content and filter spam "
            "in research and monitoring settings.",
        ),
        (
            "Problem solved",
            "Compare three sequence models — BRNN, CNN-BRNN, and CNN-BRNN with Attention — "
            "for multiclass V2X-style message classification under a controlled synthetic setting.",
        ),
        (
            "How ML is used",
            "Tokenized message sequences (maxlen 12) are embedded and classified with Softmax "
            "over six classes: Emergency, Hazard, Roadwork, Spam, Traffic, Weather.",
        ),
        (
            "Classification system",
            "CNN extracts local patterns; BRNN models bidirectional context; Attention pools "
            "informative timesteps. Training/evaluation use the existing TensorFlow pipeline.",
        ),
        (
            "Technologies",
            "Python · TensorFlow/Keras · scikit-learn · pandas · Streamlit · Plotly · YAML config.",
        ),
    ]

    html = ['<div class="about-grid">']
    for title, body in cards:
        html.append(
            f'<div class="about-card"><h4>{title}</h4><p>{body}</p></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    st.markdown(section_header("Research Question", ""), unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="panel">
  <p style="margin:0;line-height:1.55;color:#E5E7EB;">“{RESEARCH_QUESTION}”</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(section_header("Conclusion", "Consistent with reported metrics"), unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="panel">
  <p style="margin:0;line-height:1.55;color:#E5E7EB;">{CONCLUSION}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(section_header("Limitations", ""), unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="panel">
  <ul class="muted" style="margin:0;padding-left:1.1rem;line-height:1.6;">
    <li>{SYNTHETIC_LIMITATION}</li>
    <li>Results may not generalize directly to real-world V2X traffic.</li>
    <li>Operational deployment would require real-world validation beyond this dashboard.</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(section_header("Use cases", "Intended research / demo contexts"), unsafe_allow_html=True)
    st.markdown(
        """
<div class="about-grid">
  <div class="about-card"><h4>College demonstration</h4><p>Walk evaluators through architecture, metrics, and live corpus monitoring.</p></div>
  <div class="about-card"><h4>Viva / review</h4><p>Show comparative evidence for CNN + Attention vs baseline BRNN.</p></div>
  <div class="about-card"><h4>Portfolio / GitHub</h4><p>Present a polished command-center UI over a real ML research codebase.</p></div>
  <div class="about-card"><h4>Research presentation</h4><p>Visualize pipeline, confusion matrices, and attention examples without fabricating traffic.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(footer_html(), unsafe_allow_html=True)

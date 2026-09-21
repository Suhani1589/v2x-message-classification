"""Plotly chart builders for the research dashboard."""

from __future__ import annotations

from typing import List

import plotly.graph_objects as go

from metrics_data import MODELS

# Academic palette: ink / slate / teal — no purple-forward theme.
_COLORS = {
    "brnn": "#64748B",
    "cnn_brnn": "#0F766E",
    "cnn_brnn_attention": "#0B3A53",
}


def _base_layout(title: str, y_title: str, y_range=None) -> dict:
    layout = dict(
        title=dict(text=title, font=dict(size=16, color="#0F172A", family="IBM Plex Sans")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.9)",
        font=dict(family="IBM Plex Sans", color="#334155", size=13),
        margin=dict(l=48, r=24, t=56, b=48),
        yaxis=dict(title=y_title, gridcolor="#E2E8F0", zeroline=False),
        xaxis=dict(title="", gridcolor="#E2E8F0"),
        showlegend=False,
        height=340,
    )
    if y_range is not None:
        layout["yaxis"]["range"] = y_range
    return layout


def accuracy_chart() -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[m["name"] for m in MODELS],
            y=[m["accuracy"] * 100 for m in MODELS],
            marker_color=[_COLORS[m["key"]] for m in MODELS],
            text=[m["accuracy_pct"] for m in MODELS],
            textposition="outside",
            hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Accuracy (%)", "Accuracy (%)", y_range=[85, 100]))
    return fig


def f1_chart() -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[m["name"] for m in MODELS],
            y=[m["f1_macro"] * 100 for m in MODELS],
            marker_color=[_COLORS[m["key"]] for m in MODELS],
            text=[m["f1_pct"] for m in MODELS],
            textposition="outside",
            hovertemplate="%{x}<br>Macro F1: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Macro F1 (%)", "Macro F1 (%)", y_range=[85, 100]))
    return fig


def loss_chart() -> go.Figure:
    # Do not specially highlight Attention as "best" on loss — CNN-BRNN is slightly lower.
    colors = [_COLORS[m["key"]] for m in MODELS]
    fig = go.Figure(
        go.Bar(
            x=[m["name"] for m in MODELS],
            y=[m["loss"] for m in MODELS],
            marker_color=colors,
            text=[f"{m['loss']:.3f}" for m in MODELS],
            textposition="outside",
            hovertemplate="%{x}<br>Loss: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Loss (lower is better)", "Loss", y_range=[0, 0.45]))
    return fig


def conceptual_attention_figure() -> go.Figure:
    """Clearly labeled conceptual illustration — not real model weights."""
    tokens = ["t1", "t2", "t3", "t4", "t5"]
    weights = [0.08, 0.10, 0.34, 0.32, 0.16]
    fig = go.Figure(
        go.Bar(
            x=tokens,
            y=weights,
            marker_color=["#94A3B8", "#94A3B8", "#0F766E", "#0F766E", "#94A3B8"],
            text=["low", "low", "high", "high", "low"],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_base_layout(
            "Conceptual illustration — example attention weight pattern",
            "Relative weight",
            y_range=[0, 0.5],
        )
    )
    return fig


def real_attention_figure(tokens: List[str], weights: List[float], title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=tokens,
            y=weights,
            marker_color="#0B3A53",
            text=[f"{w:.2f}" for w in weights],
            textposition="outside",
            hovertemplate="Token: %{x}<br>Weight: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(title, "Attention weight"))
    return fig

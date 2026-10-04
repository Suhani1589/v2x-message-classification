"""Plotly chart builders — dark V2X command-center theme. Real metrics only."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import plotly.graph_objects as go

from metrics_data import MODELS

_COLORS = {
    "brnn": "#64748B",
    "cnn_brnn": "#3B82F6",
    "cnn_brnn_attention": "#22D3EE",
}

_CLASS_COLORS = {
    "Emergency": "#F87171",
    "Hazard": "#FB923C",
    "Roadwork": "#FBBF24",
    "Spam": "#A78BFA",
    "Traffic": "#60A5FA",
    "Weather": "#34D399",
}


def _layout(title: str, y_title: str = "", y_range=None, height: int = 340) -> dict:
    layout = dict(
        title=dict(text=title, font=dict(size=14, color="#E5E7EB", family="Source Sans 3")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        font=dict(family="Source Sans 3", color="#94A3B8", size=12),
        margin=dict(l=48, r=20, t=52, b=48),
        yaxis=dict(
            title=y_title,
            gridcolor="rgba(30,42,60,0.9)",
            zeroline=False,
            color="#94A3B8",
        ),
        xaxis=dict(title="", gridcolor="rgba(30,42,60,0.9)", color="#94A3B8"),
        showlegend=False,
        height=height,
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
            textfont=dict(color="#E5E7EB"),
            hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(**_layout("Accuracy (%)", "Accuracy (%)", y_range=[85, 100]))
    return fig


def f1_chart() -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[m["name"] for m in MODELS],
            y=[m["f1_macro"] * 100 for m in MODELS],
            marker_color=[_COLORS[m["key"]] for m in MODELS],
            text=[m["f1_pct"] for m in MODELS],
            textposition="outside",
            textfont=dict(color="#E5E7EB"),
            hovertemplate="%{x}<br>Macro F1: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(**_layout("Macro F1 (%)", "Macro F1 (%)", y_range=[85, 100]))
    return fig


def loss_chart() -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[m["name"] for m in MODELS],
            y=[m["loss"] for m in MODELS],
            marker_color=[_COLORS[m["key"]] for m in MODELS],
            text=[f"{m['loss']:.3f}" for m in MODELS],
            textposition="outside",
            textfont=dict(color="#E5E7EB"),
            hovertemplate="%{x}<br>Loss: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**_layout("Loss (lower is better)", "Loss", y_range=[0, 0.45]))
    return fig


def class_distribution_chart(class_counts: Dict[str, int]) -> go.Figure:
    labels = list(class_counts.keys())
    values = [class_counts[k] for k in labels]
    colors = [_CLASS_COLORS.get(k, "#22D3EE") for k in labels]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=values,
            textposition="outside",
            textfont=dict(color="#E5E7EB"),
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(**_layout("Message category distribution (dataset)", "Count", height=360))
    return fig


def split_distribution_chart(split_sizes: Dict[str, int]) -> go.Figure:
    labels = list(split_sizes.keys())
    values = [split_sizes[k] for k in labels]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=["#22D3EE", "#3B82F6", "#64748B"]),
            textinfo="label+value",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    layout = _layout("Train / validation / test split sizes", height=360)
    layout["showlegend"] = True
    layout["legend"] = dict(font=dict(color="#94A3B8"))
    fig.update_layout(**layout)
    return fig


def confusion_matrix_figure(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    title: str = "Confusion matrix (test set)",
) -> go.Figure:
    z = [list(row) for row in matrix]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(labels),
            y=list(labels),
            colorscale=[
                [0.0, "#0B1220"],
                [0.35, "#0E7490"],
                [0.7, "#22D3EE"],
                [1.0, "#ECFEFF"],
            ],
            text=z,
            texttemplate="%{text}",
            textfont=dict(color="#E5E7EB", size=11),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#E5E7EB")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        font=dict(family="Source Sans 3", color="#94A3B8", size=12),
        margin=dict(l=80, r=24, t=56, b=80),
        xaxis=dict(title="Predicted", color="#94A3B8"),
        yaxis=dict(title="Actual", color="#94A3B8", autorange="reversed"),
        height=480,
    )
    return fig


def per_class_f1_chart(per_class: Dict[str, Dict[str, Any]]) -> Optional[go.Figure]:
    if not per_class:
        return None
    labels = list(per_class.keys())
    f1s = [float(per_class[k].get("f1", 0.0)) * 100 for k in labels]
    colors = [_CLASS_COLORS.get(k, "#22D3EE") for k in labels]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=f1s,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in f1s],
            textposition="outside",
            textfont=dict(color="#E5E7EB"),
            hovertemplate="%{x}<br>F1: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(**_layout("Per-class F1 (best model test metrics)", "F1 (%)", y_range=[0, 105], height=360))
    return fig


def real_attention_figure(tokens: List[str], weights: List[float], title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=tokens,
            y=weights,
            marker_color="#22D3EE",
            text=[f"{w:.2f}" for w in weights],
            textposition="outside",
            textfont=dict(color="#E5E7EB"),
            hovertemplate="Token: %{x}<br>Weight: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(**_layout(title, "Attention weight", height=320))
    return fig


def model_metrics_radar(metrics_row: Dict[str, float]) -> go.Figure:
    """Optional radar from real comparison-table fields only."""
    cats = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    vals = [
        metrics_row.get("Accuracy", 0) * 100,
        metrics_row.get("Precision_macro", 0) * 100,
        metrics_row.get("Recall_macro", 0) * 100,
        metrics_row.get("F1_macro", 0) * 100,
    ]
    fig = go.Figure(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(34, 211, 238, 0.18)",
            line=dict(color="#22D3EE"),
            name="Metrics",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(15,23,42,0.55)",
            radialaxis=dict(visible=True, range=[80, 100], gridcolor="rgba(30,42,60,0.9)"),
            angularaxis=dict(gridcolor="rgba(30,42,60,0.9)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", family="Source Sans 3"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=360,
        showlegend=False,
        title=dict(text="Held-out test metrics profile", font=dict(size=14, color="#E5E7EB")),
    )
    return fig

"""HTML helpers for cards, badges, and system states."""

from __future__ import annotations

from html import escape


def metric_card_html(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<div class="hint">{escape(hint)}</div>' if hint else ""
    return f"""
<div class="metric-card">
  <div class="label">{escape(label)}</div>
  <div class="value">{escape(value)}</div>
  {hint_html}
</div>
"""


def status_pill_html(label: str, *, attention: bool = False) -> str:
    cls = "status-pill attention" if attention else "status-pill"
    return f'<span class="{cls}"><span class="dot"></span>{escape(label)}</span>'


def confidence_bar_html(percent: float, *, warn: bool = False) -> str:
    pct = max(0.0, min(100.0, float(percent)))
    fill = "conf-fill warn" if warn else "conf-fill"
    return f"""
<div class="conf-track"><div class="{fill}" style="width:{pct:.1f}%;"></div></div>
<div class="muted">Confidence {pct:.1f}%</div>
"""


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<span class="sub">{escape(subtitle)}</span>' if subtitle else ""
    return f"""
<div class="section-head">
  <h2>{escape(title)}</h2>
  {sub}
</div>
"""


def empty_state(message: str) -> str:
    return f'<div class="state-box">{escape(message)}</div>'


def error_state(message: str) -> str:
    return f'<div class="state-box error">{escape(message)}</div>'


def footer_html() -> str:
    return """
<div class="footer-bar">
  <span>AI-Powered V2X Communication &amp; Message Classification Platform</span>
  <span>Frontend: <code>frontend/</code> · ML: <code>v2x_cls/</code> · Synthetic experimental setting</span>
</div>
"""

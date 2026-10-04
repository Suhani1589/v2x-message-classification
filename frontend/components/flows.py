"""Pipeline / architecture HTML visualizations."""

from __future__ import annotations


def pipeline_flow_html() -> str:
    nodes = [
        ("VEHICLE", False),
        ("V2X MESSAGE", False),
        ("PREPROCESS", False),
        ("ML CLASSIFIER", True),
        ("CATEGORY", False),
        ("MONITORING", False),
    ]
    parts = ['<div class="flow-rail">']
    for i, (label, accent) in enumerate(nodes):
        cls = "flow-node accent" if accent else "flow-node"
        parts.append(f'<div class="{cls}">{label}</div>')
        if i < len(nodes) - 1:
            parts.append('<div class="flow-arrow">→</div>')
    parts.append("</div>")
    return "".join(parts)


def v2x_comm_html() -> str:
    return """
<div class="v2x-comm">
  <div class="v2x-box">
    <h5>CONNECTED NODE</h5>
    <p>Vehicles &amp; roadside units exchange short V2X-style text messages
    (broadcast / directed).</p>
  </div>
  <div class="v2x-box core">
    <h5>AI CLASSIFIER</h5>
    <p>CNN-BRNN + Attention maps each message to one of six operational
    categories used in this project.</p>
  </div>
  <div class="v2x-box">
    <h5>MONITOR OUTPUT</h5>
    <p>Category + confidence surface on this command center for research
    monitoring and model comparison.</p>
  </div>
</div>
<div class="badge-row" style="margin-top:0.85rem;">
  <span class="badge">Message corpus · V2X-style text</span>
  <span class="badge">6-class multiclass</span>
  <span class="badge">Spam flagged as attention category</span>
</div>
"""


def architecture_models_html() -> str:
    return """
<div class="arch-grid">
  <div class="arch-card">
    <h4>1 · BRNN</h4>
    <div class="arch-pipe">
      <div class="arch-node">Input Sequence</div>
      <div class="arch-node hi">Bidirectional RNN</div>
      <div class="arch-node">Classification Head</div>
    </div>
    <p>Baseline temporal model reading each token sequence forward and backward.</p>
  </div>
  <div class="arch-card">
    <h4>2 · CNN-BRNN</h4>
    <div class="arch-pipe">
      <div class="arch-node">Input Sequence</div>
      <div class="arch-node hi">CNN Feature Extraction</div>
      <div class="arch-node hi">Bidirectional RNN</div>
      <div class="arch-node">Classification Head</div>
    </div>
    <p>Local convolutional features feed a bidirectional recurrent encoder.</p>
  </div>
  <div class="arch-card">
    <h4>3 · CNN-BRNN + Attention</h4>
    <div class="arch-pipe">
      <div class="arch-node">Input Sequence</div>
      <div class="arch-node hi">CNN Feature Extraction</div>
      <div class="arch-node hi">Bidirectional RNN</div>
      <div class="arch-node hi">Additive Attention</div>
      <div class="arch-node">Classification Head</div>
    </div>
    <p>Attention pools timesteps by learned importance before Softmax classification.</p>
  </div>
</div>
"""


def system_pipeline_html() -> str:
    steps = [
        "V2X DATA",
        "PREPROCESSING",
        "FEATURE / SEQUENCE ENCODING",
        "ML MODEL (BRNN · CNN-BRNN · ATTENTION)",
        "MESSAGE CLASSIFICATION",
        "RESULT ANALYSIS",
        "COMMAND CENTER DASHBOARD",
    ]
    parts = ['<div class="arch-pipe">']
    for i, step in enumerate(steps):
        cls = "arch-node hi" if i in (3, 4) else "arch-node"
        parts.append(f'<div class="{cls}">{step}</div>')
    parts.append("</div>")
    return "".join(parts)

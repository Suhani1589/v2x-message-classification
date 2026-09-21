"""Shared CSS / HTML snippets for a research-lab aesthetic."""

from __future__ import annotations

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

:root {
  --ink: #0B1220;
  --slate: #334155;
  --muted: #64748B;
  --line: #E2E8F0;
  --panel: #FFFFFF;
  --soft: #F8FAFC;
  --teal: #0F766E;
  --navy: #0B3A53;
  --amber: #B45309;
}

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", sans-serif;
  color: var(--ink);
}

.block-container {
  padding-top: 1.5rem;
  padding-bottom: 3rem;
  max-width: 1120px;
}

h1, h2, h3 {
  font-family: "IBM Plex Serif", Georgia, serif !important;
  letter-spacing: -0.02em;
  color: var(--ink) !important;
}

.hero {
  background: linear-gradient(160deg, #0B3A53 0%, #0F766E 100%);
  color: #F8FAFC;
  border-radius: 18px;
  padding: 2.4rem 2.2rem 2.2rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 18px 40px rgba(11, 58, 83, 0.18);
}

.hero h1 {
  color: #FFFFFF !important;
  font-size: 2.15rem;
  margin: 0.35rem 0 0.55rem 0;
  line-height: 1.2;
}

.hero .subtitle {
  color: rgba(248, 250, 252, 0.88);
  font-size: 1.05rem;
  margin-bottom: 1.1rem;
  max-width: 46rem;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.9rem;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.22);
  color: #fff;
}

.metric-pills {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.75rem;
  margin-top: 1.1rem;
}

.metric-pill {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 12px;
  padding: 0.85rem 1rem;
}

.metric-pill .label {
  font-size: 0.78rem;
  color: rgba(248,250,252,0.75);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.metric-pill .value {
  font-size: 1.55rem;
  font-weight: 700;
  margin-top: 0.15rem;
}

.section-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1.35rem 1.4rem;
  margin-bottom: 1rem;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}

.rq-card {
  border-left: 4px solid var(--teal);
  background: #F0FDFA;
}

.rq-text {
  font-family: "IBM Plex Serif", Georgia, serif;
  font-size: 1.12rem;
  line-height: 1.55;
  color: var(--ink);
  margin: 0.4rem 0 0.8rem 0;
}

.flow {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem;
  margin: 0.65rem 0;
}

.flow-step {
  background: #F1F5F9;
  border: 1px solid var(--line);
  color: var(--slate);
  padding: 0.4rem 0.7rem;
  border-radius: 8px;
  font-size: 0.86rem;
  font-weight: 600;
}

.flow-arrow {
  color: var(--muted);
  font-weight: 700;
}

.arch-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.9rem;
}

.arch-card {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1rem 1rem 1.1rem;
  background: var(--soft);
  min-height: 220px;
}

.arch-card h4 {
  margin: 0 0 0.55rem 0;
  font-family: "IBM Plex Serif", Georgia, serif;
  color: var(--navy);
}

.arch-card p {
  color: var(--slate);
  font-size: 0.9rem;
  line-height: 1.45;
}

.arch-pipe {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  margin: 0.7rem 0 0.85rem;
}

.arch-node {
  background: white;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.45rem 0.6rem;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--ink);
  text-align: center;
}

.arch-node.accent {
  background: #0F766E;
  color: white;
  border-color: #0F766E;
}

.delta-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.9rem;
}

.delta-card {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  background: white;
}

.delta-card h4 {
  margin: 0 0 0.75rem 0;
  color: var(--navy);
  font-family: "IBM Plex Serif", Georgia, serif;
}

.delta-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.45rem 0;
  border-top: 1px solid var(--line);
  font-size: 0.92rem;
}

.delta-from { color: var(--muted); }
.delta-to { color: var(--teal); font-weight: 700; }

.highlight {
  background: #ECFDF5;
  border: 1px solid #A7F3D0;
  color: #065F46;
  font-weight: 700;
  padding: 0.1rem 0.35rem;
  border-radius: 6px;
}

.footnote {
  color: var(--muted);
  font-size: 0.86rem;
  line-height: 1.45;
}

.limit-box {
  background: #FFFBEB;
  border: 1px solid #FDE68A;
  border-radius: 12px;
  padding: 1rem 1.1rem;
  color: #78350F;
}

@media (max-width: 900px) {
  .arch-grid, .delta-grid, .metric-pills {
    grid-template-columns: 1fr;
  }
  .hero h1 { font-size: 1.7rem; }
}
"""


def architecture_html() -> str:
    return """
<div class="arch-grid">
  <div class="arch-card">
    <h4>1. BRNN</h4>
    <div class="arch-pipe">
      <div class="arch-node">Input Sequence</div>
      <div class="arch-node accent">BRNN</div>
      <div class="arch-node">Classification</div>
    </div>
    <p><b>BRNN</b> reads the message left-to-right and right-to-left so each token has context from both directions, then classifies the flattened sequence.</p>
  </div>
  <div class="arch-card">
    <h4>2. CNN-BRNN</h4>
    <div class="arch-pipe">
      <div class="arch-node">Input Sequence</div>
      <div class="arch-node accent">CNN Feature Extraction</div>
      <div class="arch-node accent">BRNN</div>
      <div class="arch-node">Classification</div>
    </div>
    <p><b>CNN</b> finds short local patterns (like phrase fragments). <b>BRNN</b> then models longer temporal structure over those features.</p>
  </div>
  <div class="arch-card">
    <h4>3. CNN-BRNN + Attention</h4>
    <div class="arch-pipe">
      <div class="arch-node">Input Sequence</div>
      <div class="arch-node accent">CNN Feature Extraction</div>
      <div class="arch-node accent">BRNN</div>
      <div class="arch-node accent">Attention</div>
      <div class="arch-node">Classification</div>
    </div>
    <p><b>Attention</b> learns which timesteps matter most, pooling the BRNN outputs by importance instead of treating every position equally.</p>
  </div>
</div>
"""

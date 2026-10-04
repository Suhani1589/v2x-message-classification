"""V2X Command Center design system — dark automotive / AI aesthetic."""

from __future__ import annotations

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
  --bg: #070B14;
  --bg-elevated: #0B1220;
  --surface: #111827;
  --surface-2: #162032;
  --border: #1E2A3C;
  --border-glow: rgba(34, 211, 238, 0.28);
  --primary: #22D3EE;
  --primary-dim: #0891B2;
  --secondary: #3B82F6;
  --accent: #67E8F9;
  --text: #E5E7EB;
  --text-muted: #94A3B8;
  --success: #34D399;
  --warning: #FBBF24;
  --danger: #F87171;
  --spam: #FB923C;
  --shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
  --radius: 14px;
  --font-ui: "Source Sans 3", "Segoe UI", sans-serif;
  --font-display: "Orbitron", sans-serif;
}

html, body, [class*="css"], .stApp {
  font-family: var(--font-ui) !important;
  color: var(--text);
  background: var(--bg) !important;
}

.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 10% -10%, rgba(34, 211, 238, 0.08), transparent 55%),
    radial-gradient(ellipse 60% 40% at 100% 0%, rgba(59, 130, 246, 0.07), transparent 50%),
    linear-gradient(180deg, #070B14 0%, #0A101C 100%) !important;
}

.stApp::before {
  content: "";
  pointer-events: none;
  position: fixed;
  inset: 0;
  z-index: 0;
  opacity: 0.22;
  background-image:
    linear-gradient(rgba(34, 211, 238, 0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(34, 211, 238, 0.04) 1px, transparent 1px);
  background-size: 48px 48px;
  mask-image: radial-gradient(ellipse at center, black 30%, transparent 75%);
}

.block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1280px !important;
  position: relative;
  z-index: 1;
}

/* Hide default Streamlit chrome noise */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0B1220 0%, #070B14 100%) !important;
  border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .block-container {
  padding-top: 1.5rem !important;
}

h1, h2, h3, .v2x-display {
  font-family: var(--font-display) !important;
  letter-spacing: 0.04em;
  color: var(--text) !important;
}

.v2x-brand {
  font-family: var(--font-display);
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--primary);
  text-transform: uppercase;
  margin-bottom: 0.15rem;
}

.v2x-brand-sub {
  font-size: 0.78rem;
  color: var(--text-muted);
  margin-bottom: 1.25rem;
  line-height: 1.4;
}

.hero-panel {
  position: relative;
  overflow: hidden;
  border-radius: calc(var(--radius) + 4px);
  border: 1px solid var(--border);
  background:
    linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 32, 50, 0.92) 55%, rgba(8, 145, 178, 0.18) 100%);
  box-shadow: var(--shadow);
  padding: 1.75rem 1.85rem 1.6rem;
  margin-bottom: 1.25rem;
}

.hero-panel::after {
  content: "";
  position: absolute;
  right: -40px;
  top: -40px;
  width: 220px;
  height: 220px;
  border-radius: 50%;
  border: 1px solid rgba(34, 211, 238, 0.2);
  box-shadow: inset 0 0 0 28px rgba(34, 211, 238, 0.03);
  pointer-events: none;
}

.hero-kicker {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--primary);
  margin-bottom: 0.55rem;
}

.hero-title {
  font-family: var(--font-display) !important;
  font-size: clamp(1.35rem, 2.4vw, 1.85rem) !important;
  margin: 0 0 0.55rem 0 !important;
  line-height: 1.25 !important;
  color: #F8FAFC !important;
}

.hero-lead {
  max-width: 46rem;
  color: var(--text-muted);
  font-size: 0.98rem;
  line-height: 1.55;
  margin: 0;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 1rem 0 0;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.28rem 0.65rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  border: 1px solid var(--border-glow);
  background: rgba(34, 211, 238, 0.08);
  color: var(--accent);
}

.badge.warn {
  border-color: rgba(251, 191, 36, 0.35);
  background: rgba(251, 191, 36, 0.1);
  color: var(--warning);
}

.badge.ok {
  border-color: rgba(52, 211, 153, 0.35);
  background: rgba(52, 211, 153, 0.1);
  color: var(--success);
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.85rem;
  margin: 0.25rem 0 1.1rem;
}

.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.05rem;
  box-shadow: var(--shadow);
  transition: border-color 0.2s ease, transform 0.2s ease;
}

.metric-card:hover {
  border-color: var(--border-glow);
  transform: translateY(-2px);
}

.metric-card .label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  margin-bottom: 0.4rem;
}

.metric-card .value {
  font-family: var(--font-display);
  font-size: 1.35rem;
  color: #F8FAFC;
  letter-spacing: 0.02em;
}

.metric-card .hint {
  margin-top: 0.35rem;
  font-size: 0.78rem;
  color: var(--text-muted);
}

.section-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin: 1.35rem 0 0.75rem;
  padding-bottom: 0.45rem;
  border-bottom: 1px solid var(--border);
}

.section-head h2 {
  margin: 0 !important;
  font-size: 1.05rem !important;
  font-family: var(--font-display) !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--primary) !important;
}

.section-head .sub {
  font-size: 0.82rem;
  color: var(--text-muted);
}

.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.15rem 1.2rem;
  margin-bottom: 0.9rem;
  box-shadow: var(--shadow);
}

.panel-title {
  font-family: var(--font-display);
  font-size: 0.82rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 0.75rem 0;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.4rem 0.8rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  border: 1px solid rgba(52, 211, 153, 0.35);
  background: rgba(52, 211, 153, 0.1);
  color: var(--success);
}

.status-pill .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 8px var(--success);
  animation: pulse 2s ease-in-out infinite;
}

.status-pill.attention {
  border-color: rgba(251, 191, 36, 0.4);
  background: rgba(251, 191, 36, 0.1);
  color: var(--warning);
}

.status-pill.attention .dot {
  background: var(--warning);
  box-shadow: 0 0 8px var(--warning);
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}

.flow-rail {
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  gap: 0.45rem;
  margin: 0.4rem 0;
}

.flow-node {
  flex: 1 1 110px;
  min-width: 100px;
  text-align: center;
  padding: 0.7rem 0.55rem;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  color: var(--text);
}

.flow-node.accent {
  border-color: var(--border-glow);
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent);
}

.flow-arrow {
  display: flex;
  align-items: center;
  color: var(--primary-dim);
  font-weight: 700;
  font-size: 0.95rem;
}

.arch-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.85rem;
}

.arch-card {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem;
  transition: border-color 0.2s ease;
}

.arch-card:hover { border-color: var(--border-glow); }

.arch-card h4 {
  margin: 0 0 0.65rem 0;
  font-family: var(--font-display);
  font-size: 0.85rem;
  letter-spacing: 0.06em;
  color: var(--primary);
}

.arch-pipe {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  margin-bottom: 0.75rem;
}

.arch-node {
  text-align: center;
  padding: 0.45rem 0.5rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--bg-elevated);
  font-size: 0.78rem;
  font-weight: 600;
}

.arch-node.hi {
  border-color: var(--border-glow);
  background: rgba(34, 211, 238, 0.12);
  color: var(--accent);
}

.arch-card p {
  margin: 0;
  font-size: 0.86rem;
  line-height: 1.45;
  color: var(--text-muted);
}

.class-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.1rem;
  margin-bottom: 0.75rem;
  transition: border-color 0.2s ease, transform 0.2s ease;
}

.class-card:hover {
  border-color: var(--border-glow);
  transform: translateY(-1px);
}

.class-card.spam {
  border-color: rgba(251, 146, 60, 0.45);
}

.class-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  align-items: center;
  margin-bottom: 0.55rem;
}

.class-label {
  font-family: var(--font-display);
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent);
}

.conf-track {
  height: 8px;
  border-radius: 999px;
  background: #1E293B;
  overflow: hidden;
  margin: 0.45rem 0 0.2rem;
}

.conf-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--primary-dim), var(--primary));
}

.conf-fill.warn {
  background: linear-gradient(90deg, #EA580C, var(--spam));
}

.msg-text {
  font-size: 0.92rem;
  color: var(--text);
  line-height: 1.45;
  margin: 0.35rem 0;
  word-break: break-word;
}

.muted { color: var(--text-muted); font-size: 0.82rem; }

.state-box {
  border: 1px dashed var(--border);
  border-radius: var(--radius);
  padding: 1.4rem 1.2rem;
  text-align: center;
  color: var(--text-muted);
  background: rgba(15, 23, 42, 0.5);
}

.state-box.error {
  border-color: rgba(248, 113, 113, 0.4);
  color: #FECACA;
}

.v2x-comm {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 0.75rem;
  align-items: center;
}

.v2x-box {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem;
  background: var(--surface-2);
  text-align: center;
}

.v2x-box.core {
  border-color: var(--border-glow);
  background: rgba(34, 211, 238, 0.08);
}

.v2x-box h5 {
  margin: 0 0 0.35rem 0;
  font-family: var(--font-display);
  font-size: 0.75rem;
  letter-spacing: 0.1em;
  color: var(--primary);
}

.v2x-box p { margin: 0; font-size: 0.82rem; color: var(--text-muted); }

.about-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.85rem;
}

.about-card {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.1rem;
}

.about-card h4 {
  margin: 0 0 0.45rem 0;
  font-family: var(--font-display);
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--primary);
}

.about-card p {
  margin: 0;
  font-size: 0.9rem;
  line-height: 1.5;
  color: var(--text-muted);
}

.footer-bar {
  margin-top: 1.75rem;
  padding-top: 0.85rem;
  border-top: 1px solid var(--border);
  font-size: 0.78rem;
  color: var(--text-muted);
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1.25rem;
  justify-content: space-between;
}

div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.75rem 0.9rem;
}

div[data-testid="stMetric"] label { color: var(--text-muted) !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-family: var(--font-display);
}

.stDataFrame, [data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}

@media (max-width: 960px) {
  .metric-grid, .arch-grid, .about-grid, .v2x-comm {
    grid-template-columns: 1fr;
  }
  .flow-arrow { display: none; }
  .hero-title { font-size: 1.25rem !important; }
}
"""

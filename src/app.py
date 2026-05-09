import streamlit as st
import requests
import json
import csv
import io
import os


try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# ================================================================
# CONFIG
# ================================================================
st.set_page_config(
    page_title="DocuLens — Intelligence Suite",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ================================================================
# GLOBAL CSS — Cyberpunk Noir + Glassmorphism
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&family=Outfit:wght@300;400;500;600&display=swap');

:root {
  --bg-void:       #030308;
  --bg-deep:       #07070f;
  --bg-card:       rgba(12,12,24,0.85);
  --bg-glass:      rgba(255,255,255,0.03);
  --bg-glass-h:    rgba(255,255,255,0.06);
  --border:        rgba(100,100,200,0.12);
  --border-glow:   rgba(120,100,255,0.35);
  --accent:        #7c6fff;
  --accent-2:      #ff6b9d;
  --accent-3:      #00f5c4;
  --accent-4:      #ffb347;
  --text-pri:      #f0f0ff;
  --text-sec:      #8888aa;
  --text-muted:    #44445a;
  --cyan:          #00e5ff;
  --violet:        #c77dff;
}

html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif !important;
  background: var(--bg-void) !important;
}

/* Animated grid background */
.stApp {
  background:
    linear-gradient(rgba(124,111,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(124,111,255,0.03) 1px, transparent 1px),
    radial-gradient(ellipse 80% 60% at 20% 0%, rgba(124,111,255,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 100%, rgba(255,107,157,0.06) 0%, transparent 60%),
    var(--bg-void) !important;
  background-size: 40px 40px, 40px 40px, 100% 100%, 100% 100%, 100% 100% !important;
}

.main .block-container {
  padding: 1.5rem 2.5rem 3rem !important;
  max-width: 1280px !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"] {
  background: rgba(5,5,12,0.95) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(20px);
}
[data-testid="stSidebar"]::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 200px;
  background: linear-gradient(180deg, rgba(124,111,255,0.08) 0%, transparent 100%);
  pointer-events: none;
}
[data-testid="stSidebar"] * { color: var(--text-sec) !important; }
[data-testid="stSidebar"] .stRadio label {
  color: var(--text-sec) !important;
  font-weight: 500;
  font-size: 0.88rem;
  letter-spacing: 0.3px;
  transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--text-pri) !important; }

/* ============ TYPOGRAPHY ============ */
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 3rem;
  font-weight: 800;
  background: linear-gradient(135deg, #ffffff 0%, var(--accent) 50%, var(--accent-2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -2px;
  line-height: 1;
  margin-bottom: 0.4rem;
}
.hero-sub {
  font-size: 0.92rem;
  color: var(--text-muted);
  font-weight: 400;
  letter-spacing: 0.5px;
  margin-bottom: 2.5rem;
}
.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  font-weight: 500;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 3px;
  margin-bottom: 0.8rem;
  margin-top: 1.8rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-label::before {
  content: '';
  display: inline-block;
  width: 18px; height: 1px;
  background: var(--accent);
  opacity: 0.6;
}

/* ============ GLASS CARDS ============ */
.g-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.4rem;
  margin-bottom: 1rem;
  backdrop-filter: blur(12px);
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.g-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(124,111,255,0.4), transparent);
}
.g-card:hover {
  border-color: var(--border-glow);
  box-shadow: 0 0 30px rgba(124,111,255,0.08), 0 8px 40px rgba(0,0,0,0.4);
}

/* ============ KPI METRICS ============ */
.kpi-wrap {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.4rem 1rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: all 0.3s;
  backdrop-filter: blur(10px);
}
.kpi-wrap::after {
  content: '';
  position: absolute;
  bottom: 0; left: 50%;
  transform: translateX(-50%);
  width: 40%; height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity: 0.6;
}
.kpi-wrap:hover {
  border-color: var(--border-glow);
  transform: translateY(-2px);
  box-shadow: 0 12px 40px rgba(124,111,255,0.1);
}
.kpi-num {
  font-family: 'Syne', sans-serif;
  font-size: 2.2rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent) 0%, var(--cyan) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1;
  margin-bottom: 0.4rem;
}
.kpi-lbl {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 2px;
  font-weight: 500;
}

/* ============ BADGES ============ */
.doc-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 12px;
  border-radius: 100px;
  font-size: 0.75rem;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.3px;
}
.b-invoice  { background: rgba(76,175,80,0.1);  color: #69f0ae; border: 1px solid rgba(76,175,80,0.25); }
.b-email    { background: rgba(124,111,255,0.1); color: #a89aff; border: 1px solid rgba(124,111,255,0.25); }
.b-ticket   { background: rgba(255,107,157,0.1); color: #ff8fb3; border: 1px solid rgba(255,107,157,0.25); }
.b-business { background: rgba(255,179,71,0.1);  color: #ffd080; border: 1px solid rgba(255,179,71,0.25); }
.b-world    { background: rgba(0,229,255,0.1);   color: #40e0ff; border: 1px solid rgba(0,229,255,0.25); }
.b-sports   { background: rgba(199,125,255,0.1); color: #d09bff; border: 1px solid rgba(199,125,255,0.25); }
.b-scitech  { background: rgba(0,245,196,0.1);   color: #40f5d0; border: 1px solid rgba(0,245,196,0.25); }
.b-general  { background: rgba(150,150,150,0.1); color: #aaaacc; border: 1px solid rgba(150,150,150,0.2); }

/* ============ ENTITY CHIPS ============ */
.ent-wrap { display: flex; flex-wrap: wrap; gap: 7px; margin: 0.6rem 0; }
.ent-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 8px;
  font-size: 0.78rem;
  font-weight: 500;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.2px;
  transition: transform 0.2s;
}
.ent-chip:hover { transform: scale(1.04); }
.e-PER  { background: rgba(255,213,79,0.08);  color: #ffe082; border: 1px solid rgba(255,213,79,0.2); }
.e-ORG  { background: rgba(105,240,174,0.08); color: #69f0ae; border: 1px solid rgba(105,240,174,0.2); }
.e-LOC  { background: rgba(100,181,246,0.08); color: #90caf9; border: 1px solid rgba(100,181,246,0.2); }
.e-MISC { background: rgba(206,147,216,0.08); color: #ce93d8; border: 1px solid rgba(206,147,216,0.2); }
.ent-type {
  font-size: 0.58rem;
  opacity: 0.55;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  border-left: 1px solid currentColor;
  padding-left: 6px;
  margin-left: 2px;
}

/* ============ FIELD GRID ============ */
.f-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.f-item {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.75rem 1rem;
}
.f-key {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.62rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 3px;
}
.f-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.9rem;
  color: var(--text-pri);
  font-weight: 500;
}

/* ============ SUMMARY ============ */
.summary-box {
  background: rgba(124,111,255,0.04);
  border-left: 3px solid var(--accent);
  border-radius: 0 12px 12px 0;
  padding: 1rem 1.3rem;
  font-size: 0.9rem;
  color: #c8c8e8;
  line-height: 1.75;
  font-style: italic;
  font-weight: 300;
  position: relative;
}
.summary-box::before {
  content: '"';
  font-size: 3rem;
  color: var(--accent);
  opacity: 0.15;
  position: absolute;
  top: -8px; left: 12px;
  font-family: Georgia, serif;
  line-height: 1;
}

/* ============ SEARCH BAR ============ */
.stTextInput input {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-pri) !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.95rem !important;
  padding: 0.65rem 1rem !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(124,111,255,0.12) !important;
}
.stTextArea textarea {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text-pri) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.85rem !important;
  line-height: 1.7 !important;
}
.stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(124,111,255,0.12) !important;
}

/* ============ BUTTONS ============ */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #9b7cff) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.9rem !important;
  padding: 0.6rem 1.6rem !important;
  letter-spacing: 0.3px !important;
  transition: all 0.2s !important;
  box-shadow: 0 4px 16px rgba(124,111,255,0.25) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 28px rgba(124,111,255,0.4) !important;
}
.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.04) !important;
  color: var(--text-sec) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
  background: rgba(124,111,255,0.08) !important;
  border-color: var(--border-glow) !important;
  color: var(--text-pri) !important;
  transform: none !important;
  box-shadow: none !important;
}

/* ============ FILE UPLOADER ============ */
div[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px dashed rgba(124,111,255,0.25) !important;
  border-radius: 12px !important;
  transition: border-color 0.2s !important;
}
div[data-testid="stFileUploader"]:hover {
  border-color: rgba(124,111,255,0.5) !important;
}

/* ============ PROGRESS ============ */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--cyan)) !important; }

/* ============ EXPANDER ============ */
.stExpander {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
.stExpander:hover { border-color: var(--border-glow) !important; }

/* ============ DOWNLOAD BTN ============ */
.stDownloadButton > button {
  background: rgba(255,255,255,0.03) !important;
  color: var(--accent) !important;
  border: 1px solid rgba(124,111,255,0.3) !important;
  border-radius: 10px !important;
  font-size: 0.85rem !important;
  transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
  background: rgba(124,111,255,0.08) !important;
  box-shadow: 0 0 20px rgba(124,111,255,0.2) !important;
}

/* ============ DIVIDER ============ */
.slim-div {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
  margin: 1.5rem 0;
}

/* ============ PILL TABS ============ */
.pill-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 1rem; }

/* ============ RESULT CARD ============ */
.result-row {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 0.8rem;
  position: relative;
  overflow: hidden;
  transition: all 0.25s;
}
.result-row::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, var(--accent), var(--accent-2));
  border-radius: 3px 0 0 3px;
}
.result-row:hover {
  border-color: var(--border-glow);
  transform: translateX(3px);
  box-shadow: 0 4px 30px rgba(124,111,255,0.08);
}

/* ============ RECENT CARD ============ */
.recent-row {
  background: rgba(255,255,255,0.015);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.9rem 1rem;
  margin-bottom: 0.5rem;
  transition: all 0.2s;
  cursor: pointer;
}
.recent-row:hover {
  background: rgba(124,111,255,0.04);
  border-color: var(--border-glow);
}

/* Activity badge */
.activity-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.7rem 0;
  border-bottom: 1px solid rgba(100,100,200,0.06);
}

/* ============ INPUT TABS ============ */
.stRadio > div { gap: 4px !important; }
.stRadio > div > label {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 0.4rem 1rem !important;
  font-size: 0.85rem !important;
  transition: all 0.2s !important;
}
.stRadio > div > label:hover {
  border-color: var(--border-glow) !important;
  background: rgba(124,111,255,0.06) !important;
}

/* Alert / success overrides */
div[data-testid="stAlert"] { border-radius: 10px !important; }

/* Stat bar */
.stat-bar-wrap { margin-bottom: 1rem; }
.stat-bar-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.stat-bar-name { font-size: 0.85rem; color: var(--text-pri); font-weight: 500; }
.stat-bar-count { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--text-muted); }
.stat-bar-track { height: 5px; background: rgba(255,255,255,0.04); border-radius: 3px; overflow: hidden; }
.stat-bar-fill { height: 100%; border-radius: 3px; }

/* Model card */
.model-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.8rem;
  transition: border-color 0.2s;
}
.model-card:hover { border-color: var(--border-glow); }
.model-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: var(--accent); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 3px; }
.model-name { font-size: 0.95rem; color: var(--text-pri); font-weight: 600; margin-bottom: 2px; }
.model-desc { font-size: 0.78rem; color: var(--text-muted); }

/* Badge for doc ID */
.doc-id-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--text-muted);
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px 10px;
}

/* Sidebar logo */
.sidebar-logo {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  font-weight: 800;
  background: linear-gradient(135deg, #fff 0%, var(--accent) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# HELPERS
# ================================================================
DOC_CFG = {
    "Invoice":        {"icon": "🧾", "badge": "b-invoice",  "color": "#69f0ae"},
    "Email":          {"icon": "📧", "badge": "b-email",    "color": "#a89aff"},
    "Support Ticket": {"icon": "🎫", "badge": "b-ticket",   "color": "#ff8fb3"},
    "Business":       {"icon": "💼", "badge": "b-business", "color": "#ffd080"},
    "World":          {"icon": "🌍", "badge": "b-world",    "color": "#40e0ff"},
    "Sports":         {"icon": "⚽", "badge": "b-sports",   "color": "#d09bff"},
    "Sci/Tech":       {"icon": "🔬", "badge": "b-scitech",  "color": "#40f5d0"},
}

BAR_PALETTE = ["#7c6fff","#ff6b9d","#00f5c4","#ffb347","#00e5ff","#c77dff","#ff9e7a","#64ffda"]

def bar_chart_html(data_dict):
    """Styled horizontal bar chart with glow effect."""
    if not data_dict:
        return '<div style="color:var(--text-muted);font-size:0.85rem;padding:1rem;">No data yet.</div>'
    total = sum(data_dict.values()) or 1
    max_val = max(data_dict.values()) or 1
    items = list(data_dict.items())
    rows = []
    for i, (label, val) in enumerate(items):
        color     = BAR_PALETTE[i % len(BAR_PALETTE)]
        color_border = color + "33"
        color_fade   = color + "99"
        color_glow1  = color + "55"
        color_glow2  = color + "88"
        pct_of_max   = round(val / max_val * 100, 2)
        pct_of_total = round(val / total * 100)
        icon = DOC_CFG.get(label, {}).get("icon", "📄")
        row = (
            '<div style="margin-bottom:1.1rem;">'
              '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                '<span style="font-size:0.88rem;color:#f0f0ff;font-weight:500;display:flex;align-items:center;gap:7px;">'
                  f'<span style="font-size:1rem;">{icon}</span>{label}'
                '</span>'
                '<div style="display:flex;align-items:center;gap:10px;">'
                  f'<span style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:{color};'
                  f'background:rgba(0,0,0,0.3);padding:2px 9px;border-radius:20px;'
                  f'border:1px solid {color_border};">{val}</span>'
                  f'<span style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#44445a;">{pct_of_total}%</span>'
                '</div>'
              '</div>'
              '<div style="height:8px;background:rgba(255,255,255,0.04);border-radius:6px;overflow:hidden;">'
                f'<div style="height:100%;width:{pct_of_max}%;'
                f'background:linear-gradient(90deg,{color},{color_fade});'
                f'border-radius:6px;box-shadow:0 0 10px {color_glow1},0 0 3px {color_glow2};"></div>'
              '</div>'
            '</div>'
        )
        rows.append(row)
    return '<div style="padding:0.2rem 0;">' + "".join(rows) + '</div>'

def badge(doc_type):
    c = DOC_CFG.get(doc_type, {"icon": "📄", "badge": "b-general"})
    return f'<span class="doc-badge {c["badge"]}">{c["icon"]} {doc_type}</span>'

def entity_html(entities):
    if not entities:
        return '<span style="color:var(--text-muted);font-size:0.82rem;">No entities detected</span>'
    chips = [
        f'<span class="ent-chip e-{e["type"]}">{e["text"].title()}<span class="ent-type">{e["type"]}</span></span>'
        for e in entities
    ]
    return f'<div class="ent-wrap">{"".join(chips)}</div>'

def fields_html(fields):
    if not fields:
        return ""
    items = [
        f'<div class="f-item"><div class="f-key">{k.replace("_"," ").title()}</div><div class="f-val">{v}</div></div>'
        for k, v in fields.items()
    ]
    return f'<div class="f-grid">{"".join(items)}</div>'

def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=90)
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", "API error")
    except Exception as e:
        return None, str(e)

def api_get(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=30)
        if r.status_code == 200:
            return r.json(), None
        return None, "API error"
    except Exception as e:
        return None, str(e)

def extract_text(uploaded_file):
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if ext == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore"), None
    elif ext == "pdf":
        if pdfplumber is None:
            return None, "pdfplumber not installed."
        try:
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
            if not text:
                return None, "Could not extract text — PDF may be image-based."
            return text, None
        except Exception as e:
            return None, str(e)
    return None, "Unsupported file type"

def render_result(result, text_input):
    st.success("✅ Analysis complete")

    # Header row
    hc1, hc2 = st.columns([4, 1])
    with hc1:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin:0.5rem 0 1.2rem;">'
            f'<span style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:700;color:var(--text-pri);">Analysis Result</span>'
            f'{badge(result["doc_type"])}'
            f'</div>',
            unsafe_allow_html=True
        )
    with hc2:
        st.markdown(
            f'<div style="text-align:right;padding-top:0.5rem;"><span class="doc-id-badge">{result["doc_id"][:12]}…</span></div>',
            unsafe_allow_html=True
        )

    # KPIs
    conf_pct = f'{result["confidence"]*100:.1f}%'
    wc = len(text_input.split())
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in [
        (k1, result["doc_type"].split("/")[0], "Type"),
        (k2, conf_pct, "Confidence"),
        (k3, str(len(result["entities"])), "Entities"),
        (k4, f'{wc:,}', "Words"),
    ]:
        with col:
            st.markdown(f'<div class="kpi-wrap"><div class="kpi-num">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="slim-div"></div>', unsafe_allow_html=True)

    if result.get("extracted_fields"):
        st.markdown('<div class="section-label">Extracted Fields</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="g-card">{fields_html(result["extracted_fields"])}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">AI Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-box">{result["summary"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Named Entities</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="g-card">{entity_html(result["entities"])}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
    ex1, ex2 = st.columns(2)
    with ex1:
        st.download_button("⬇ Download JSON", data=json.dumps(result, indent=2),
                           file_name=f"docuLens_{result['doc_id']}.json",
                           mime="application/json", use_container_width=True)
    with ex2:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Field", "Value"])
        w.writerow(["Document ID", result["doc_id"]])
        w.writerow(["Document Type", result["doc_type"]])
        w.writerow(["Confidence", conf_pct])
        w.writerow(["Summary", result["summary"]])
        for ent in result["entities"]:
            w.writerow([f"Entity ({ent['type']})", ent["text"]])
        for k, v in result.get("extracted_fields", {}).items():
            w.writerow([k.replace("_", " ").title(), v])
        st.download_button("⬇ Download CSV", data=buf.getvalue(),
                           file_name=f"docuLens_{result['doc_id']}.csv",
                           mime="text/csv", use_container_width=True)


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:1.8rem 0 1rem;">
      <div class="sidebar-logo">⬡ DocuLens</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:var(--text-muted);
                  margin-top:5px;letter-spacing:2.5px;text-transform:uppercase;">Intelligence Suite</div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin-bottom:1.5rem;"></div>
    """, unsafe_allow_html=True)

    page = st.radio("Nav", ["Analyze", "History", "Search", "Dashboard"], label_visibility="collapsed")

    st.markdown("""
    <div style="height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:1.5rem 0;"></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--text-muted);
                text-transform:uppercase;letter-spacing:2px;margin-bottom:1rem;">Active Models</div>
    """, unsafe_allow_html=True)

    models = [
        ("NER", "DistilBERT", "F1 92.6%"),
        ("CLF", "BERT-base", "F1 92.8%"),
        ("SUM", "DistilBART", "CNN-12-6"),
        ("EMB", "MiniLM-L6", "Semantic"),
    ]
    for tag, name, detail in models:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:0.45rem 0;border-bottom:1px solid rgba(100,100,200,0.05);">
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                       background:rgba(124,111,255,0.12);color:var(--accent);
                       padding:2px 7px;border-radius:4px;letter-spacing:0.5px;">{tag}</span>
          <span style="font-size:0.82rem;color:var(--text-sec);">{name}</span>
          <span style="font-size:0.72rem;color:var(--text-muted);margin-left:auto;">{detail}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem;padding:0.8rem 1rem;background:rgba(124,111,255,0.04);
                border:1px solid rgba(124,111,255,0.1);border-radius:10px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--text-muted);
                  text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Accepts</div>
      <div style="font-size:0.8rem;color:var(--text-sec);">PDF · TXT · Plain Text</div>
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# PAGE: ANALYZE
# ================================================================
if page == "Analyze":
    st.markdown('<div class="hero-title">Document Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload or paste any business document — invoices, emails, tickets, articles</div>', unsafe_allow_html=True)

    input_mode = st.radio("Input", ["✏️ Paste Text", "📎 Upload File"], horizontal=True, label_visibility="collapsed")
    text_input = ""

    if input_mode == "✏️ Paste Text":
        text_input = st.text_area("Content", height=210,
            placeholder="Paste your invoice, email, support ticket, or any document here…",
            label_visibility="collapsed")
    else:
        uploaded = st.file_uploader("Upload", type=["pdf", "txt"], label_visibility="collapsed")
        if uploaded:
            with st.spinner("Extracting text…"):
                text_input, err = extract_text(uploaded)
            if err:
                st.error(f"❌ {err}")
            else:
                st.success(f"✅ {uploaded.name} — {len(text_input):,} characters extracted")

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
    analyze_btn = st.button("⬡ Analyze Document", use_container_width=False)

    if analyze_btn:
        if not text_input or not text_input.strip():
            st.error("Please provide document content first.")
        else:
            with st.spinner("Running NLP pipeline…"):
                result, error = api_post("/analyze", {"text": text_input})
            if error:
                st.error(f"❌ Pipeline error: {error}")
            else:
                st.markdown('<div class="slim-div"></div>', unsafe_allow_html=True)
                render_result(result, text_input)

    # Recent
    st.markdown('<div class="slim-div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Recently Analyzed</div>', unsafe_allow_html=True)
    recent_data, _ = api_get("/documents")
    if recent_data and recent_data["total"] > 0:
        for doc in recent_data["documents"][:3]:
            cfg = DOC_CFG.get(doc["doc_type"], {"icon": "📄", "color": "#888"})
            st.markdown(f"""
            <div class="recent-row">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;">
                {badge(doc["doc_type"])}
                <span style="font-size:0.82rem;color:var(--text-muted);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                  {doc["text_preview"][:65]}…
                </span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--text-muted);">{doc["created_at"][11:16]}</span>
              </div>
              <div style="font-size:0.8rem;color:var(--text-muted);line-height:1.5;font-style:italic;">{doc["summary"][:110]}…</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:var(--text-muted);font-size:0.85rem;padding:1rem;">No documents yet — analyze one above to get started.</div>', unsafe_allow_html=True)


# ================================================================
# PAGE: HISTORY
# ================================================================
elif page == "History":
    st.markdown('<div class="hero-title">History</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">All previously analyzed documents</div>', unsafe_allow_html=True)

    rc1, rc2 = st.columns([6, 1])
    with rc2:
        if st.button("↻ Refresh"):
            st.rerun()

    data, error = api_get("/documents")
    if error:
        st.error(f"❌ {error}")
    elif not data or data["total"] == 0:
        st.markdown('<div class="g-card" style="text-align:center;color:var(--text-muted);padding:3rem;">No documents yet. Head to Analyze to get started.</div>', unsafe_allow_html=True)
    else:
        all_types = list(set(d["doc_type"] for d in data["documents"]))
        filter_opts = ["All"] + sorted(all_types)
        if "hfilter" not in st.session_state:
            st.session_state.hfilter = "All"

        cols = st.columns(len(filter_opts))
        for i, t in enumerate(filter_opts):
            with cols[i]:
                icon = DOC_CFG.get(t, {}).get("icon", "🗂️")
                label = icon if t == "All" else f'{icon}'
                is_active = st.session_state.hfilter == t
                if st.button(f'{icon} {t}', key=f"hf_{t}", use_container_width=True,
                              type="primary" if is_active else "secondary"):
                    st.session_state.hfilter = t
                    st.rerun()

        st.markdown('<div class="slim-div"></div>', unsafe_allow_html=True)
        sel = st.session_state.hfilter
        docs = data["documents"] if sel == "All" else [d for d in data["documents"] if d["doc_type"] == sel]
        st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:var(--text-muted);margin-bottom:1rem;">{len(docs)} document{"s" if len(docs)!=1 else ""} · {sel}</div>', unsafe_allow_html=True)

        if not docs:
            st.markdown('<div class="g-card" style="text-align:center;color:var(--text-muted);padding:2rem;">No documents match this filter.</div>', unsafe_allow_html=True)
        else:
            for doc in docs:
                cfg = DOC_CFG.get(doc["doc_type"], {"icon": "📄"})
                with st.expander(f'{cfg["icon"]}  {doc["doc_type"]}  ·  {doc["text_preview"][:55]}…'):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'<div class="f-item"><div class="f-key">Type</div><div class="f-val">{doc["doc_type"]}</div></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="f-item"><div class="f-key">Confidence</div><div class="f-val">{doc["confidence"]*100:.1f}%</div></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="f-item"><div class="f-key">Analyzed At</div><div class="f-val">{doc["created_at"]}</div></div>', unsafe_allow_html=True)
                    st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
                    if doc.get("extracted_fields"):
                        st.markdown(fields_html(doc["extracted_fields"]), unsafe_allow_html=True)
                        st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="summary-box">{doc["summary"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
                    st.markdown(entity_html(doc["entities"]), unsafe_allow_html=True)


# ================================================================
# PAGE: SEARCH
# ================================================================
elif page == "Search":
    st.markdown('<div class="hero-title">Semantic Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Filter and search across all analyzed documents</div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns([5, 1])
    with sc1:
        query = st.text_input("Query", placeholder="e.g. unpaid invoices, login issues, quarterly earnings…", label_visibility="collapsed")
    with sc2:
        search_btn = st.button("⬡ Search", use_container_width=True)

    st.markdown('<div class="section-label">Filter by Type</div>', unsafe_allow_html=True)
    all_types = ["All", "Invoice", "Email", "Support Ticket", "Business", "World", "Sports", "Sci/Tech"]
    if "stype" not in st.session_state:
        st.session_state.stype = "All"

    fcols = st.columns(len(all_types))
    for i, t in enumerate(all_types):
        with fcols[i]:
            icon = DOC_CFG.get(t, {}).get("icon", "🗂️")
            is_active = st.session_state.stype == t
            if st.button(f'{icon} {t}', key=f"sf_{t}", use_container_width=True,
                          type="primary" if is_active else "secondary"):
                st.session_state.stype = t
                st.rerun()

    st.markdown('<div class="slim-div"></div>', unsafe_allow_html=True)
    data, error = api_get("/documents")
    if error:
        st.error(f"❌ {error}")
    elif not data or data["total"] == 0:
        st.markdown('<div class="g-card" style="text-align:center;color:var(--text-muted);padding:3rem;">No documents yet.</div>', unsafe_allow_html=True)
    else:
        sel_type = st.session_state.stype
        filtered = data["documents"] if sel_type == "All" else [d for d in data["documents"] if d["doc_type"] == sel_type]

        if search_btn and query.strip():
            with st.spinner("Searching…"):
                results, serr = api_post("/search", {"query": query, "n_results": 20})
            if serr:
                st.error(f"❌ {serr}")
            else:
                previews = [r["text_preview"][:80] for r in results.get("results", [])]
                matched = [d for d in filtered if any(d["text_preview"][:80] in sp or sp in d["text_preview"][:80] for sp in previews)]
                display = matched if matched else filtered
                st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:var(--text-muted);margin-bottom:1rem;">{len(display)} result{"s" if len(display)!=1 else ""} for &ldquo;{query}&rdquo;</div>', unsafe_allow_html=True)
        else:
            display = filtered
            st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:var(--text-muted);margin-bottom:1rem;">{len(display)} document{"s" if len(display)!=1 else ""} · {sel_type}</div>', unsafe_allow_html=True)

        if not display:
            st.markdown('<div class="g-card" style="text-align:center;color:var(--text-muted);padding:2rem;">No documents match.</div>', unsafe_allow_html=True)
        else:
            for i, doc in enumerate(display, 1):
                st.markdown(f"""
                <div class="result-row">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.8rem;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                 color:var(--text-muted);background:rgba(255,255,255,0.03);
                                 border:1px solid var(--border);border-radius:6px;padding:2px 8px;">#{i:02d}</span>
                    {badge(doc["doc_type"])}
                    <span style="margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--text-muted);">{doc["created_at"][:16]}</span>
                  </div>
                  <div class="summary-box" style="margin-bottom:0.8rem;">{doc["summary"]}</div>
                  <div style="font-size:0.8rem;color:var(--text-muted);line-height:1.65;">{doc["text_preview"][:240]}…</div>
                </div>
                """, unsafe_allow_html=True)


# ================================================================
# PAGE: DASHBOARD
# ================================================================
elif page == "Dashboard":
    st.markdown('<div class="hero-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">System performance and document intelligence</div>', unsafe_allow_html=True)

    dc1, dc2 = st.columns([6, 1])
    with dc2:
        if st.button("↻ Refresh"):
            st.rerun()

    stats, error = api_get("/stats")
    data, _ = api_get("/documents")

    if error:
        st.error(f"❌ {error}")
    else:
        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-wrap"><div class="kpi-num">{stats["total_documents"]}</div><div class="kpi-lbl">Documents</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-wrap"><div class="kpi-num">{stats["vector_store_count"]}</div><div class="kpi-lbl">Vectors</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-wrap"><div class="kpi-num">{len(stats["documents_by_type"])}</div><div class="kpi-lbl">Doc Types</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-wrap"><div class="kpi-num">4</div><div class="kpi-lbl">Models Active</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="slim-div"></div>', unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            # HORIZONTAL BAR CHART
            if stats["documents_by_type"]:
                st.markdown('<div class="section-label">Distribution</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="g-card">{bar_chart_html(stats["documents_by_type"])}</div>',
                    unsafe_allow_html=True
                )

            # Breakdown bars
            st.markdown('<div class="section-label">Volume Breakdown</div>', unsafe_allow_html=True)
            total_docs = stats["total_documents"] or 1
            bar_colors = ["#7c6fff","#ff6b9d","#00f5c4","#ffb347","#00e5ff","#c77dff","#ff9e7a"]
            breakdown_html = '<div style="padding:0.1rem 0;">'
            for i, (dt, cnt) in enumerate(stats["documents_by_type"].items()):
                pct = round(cnt / total_docs * 100)
                color = bar_colors[i % len(bar_colors)]
                c88 = color + "88"
                c44 = color + "44"
                icon = DOC_CFG.get(dt, {}).get("icon", "📄")
                breakdown_html += (
                    '<div class="stat-bar-wrap">'
                      '<div class="stat-bar-header">'
                        f'<span class="stat-bar-name">{icon} {dt}</span>'
                        f'<span class="stat-bar-count">{cnt} · {pct}%</span>'
                      '</div>'
                      '<div class="stat-bar-track">'
                        f'<div class="stat-bar-fill" style="width:{pct}%;'
                        f'background:linear-gradient(90deg,{color},{c88});'
                        f'box-shadow:0 0 8px {c44};"></div>'
                      '</div>'
                    '</div>'
                )
            breakdown_html += '</div>'
            st.markdown(breakdown_html, unsafe_allow_html=True)

        with col_r:
            # Model cards
            st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)

            model_data = [
                ("NER · DistilBERT", "WikiANN English", "7 entity types · 20K samples", 0.9263),
                ("Classifier · BERT-base", "AG News · 4 Classes", "World · Sports · Business · Sci/Tech", 0.9277),
                ("Summarizer · DistilBART", "sshleifer/distilbart-cnn-12-6", "News & General documents", None),
                ("Embeddings · MiniLM-L6", "Semantic Similarity", "384-dim · cosine search", None),
            ]
            for tag, name, desc, score in model_data:
                st.markdown(f"""
                <div class="model-card">
                  <div class="model-tag">{tag}</div>
                  <div class="model-name">{name}</div>
                  <div class="model-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
                if score:
                    st.progress(score, text=f"F1: {score*100:.2f}%")
                    st.markdown('<div style="height:0.2rem;"></div>', unsafe_allow_html=True)

            # Recent activity
            if data and data["total"] > 0:
                st.markdown('<div class="section-label">Recent Activity</div>', unsafe_allow_html=True)
                st.markdown('<div class="g-card" style="padding:0.4rem 1rem;">', unsafe_allow_html=True)
                for doc in data["documents"][:6]:
                    cfg = DOC_CFG.get(doc["doc_type"], {"icon": "📄", "color": "#888"})
                    color = cfg.get("color", "#888")
                    st.markdown(f"""
                    <div class="activity-row">
                      <div style="display:flex;align-items:center;gap:10px;">
                        <div style="width:8px;height:8px;border-radius:50%;background:{color};box-shadow:0 0 6px {color};flex-shrink:0;"></div>
                        <span style="font-size:0.85rem;color:var(--text-pri);">{doc["doc_type"]}</span>
                        <span style="font-size:0.78rem;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:120px;">{doc["text_preview"][:35]}…</span>
                      </div>
                      <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--text-muted);">{doc["created_at"][11:16]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
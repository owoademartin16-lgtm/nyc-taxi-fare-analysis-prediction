"""Visual system for the NYC taxi fare intelligence dashboard.

Asphalt, taxi yellow, receipt paper and a dispatch-board typography stack. The
dashboard is deliberately brighter and more street-level than the bank churn
case study while keeping the same chart and spacing discipline.
"""

from __future__ import annotations

import base64
import html

import streamlit as st

# ------------------------------------------------------------------ palette ---
BG = "#F3EADB"
SURFACE = "#FFFCF4"
SURFACE_ALT = "#EDE1CB"
BORDER = "#D8C7A9"
BORDER_SOFT = "#E8DCC8"

INK = "#14181D"
ASPHALT = "#14181D"
ASPHALT_2 = "#222A32"
TITLE = "#14181D"
SUBTITLE = "#3B3B36"
MUTED = "#6B6A62"
FAINT = "#9A9589"

YELLOW = "#F5C518"
YELLOW_BRIGHT = "#FFD84D"
YELLOW_SOFT = "#FAE9A8"
ORANGE = "#D95D39"
ORANGE_BRIGHT = "#F0744E"
ORANGE_SOFT = "#F6D7CC"
TEAL = "#237A6B"
TEAL_BRIGHT = "#2FA394"
TEAL_SOFT = "#D8EAE3"
BLUE = "#416B8A"
BLUE_SOFT = "#DCE7EF"
CREAM = "#FBF6EA"

ACCENT = YELLOW
SECONDARY = ORANGE
TERTIARY = TEAL

FARE_SCALE = [
    [0.0, "#FBF3D8"],
    [0.25, "#F5D86B"],
    [0.50, "#F0B43B"],
    [0.75, "#D96A2F"],
    [1.00, "#8E2F1C"],
]
ASPHALT_SCALE = [
    [0.0, "#F4EBDD"],
    [0.35, "#CDBB9D"],
    [0.70, "#6D6A62"],
    [1.00, "#14181D"],
]
TEAL_SCALE = [
    [0.0, "#F1F7F3"],
    [0.45, "#BFDCD4"],
    [0.75, "#4E9C8C"],
    [1.00, "#1B5B51"],
]

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "displaylogo": False,
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700;800&display=swap');

:root{
  --bg:#F3EADB; --surface:#FFFCF4; --surface-alt:#EDE1CB;
  --border:#D8C7A9; --border-soft:#E8DCC8;
  --field:#F6EDDC; --field-hover:#F0E4CE;
  --ink:#14181D; --title:#14181D; --subtitle:#3B3B36;
  --muted:#6B6A62; --faint:#9A9589;
  --asphalt:#14181D; --asphalt-2:#222A32;
  --yellow:#F5C518; --yellow-bright:#FFD84D; --yellow-soft:#FAE9A8;
  --orange:#D95D39; --orange-bright:#F0744E; --orange-soft:#F6D7CC;
  --teal:#237A6B; --teal-bright:#2FA394; --teal-soft:#D8EAE3;
  --blue:#416B8A; --blue-soft:#DCE7EF;
  --display:'Bebas Neue',Impact,sans-serif;
  --sans:'Manrope',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --mono:'IBM Plex Mono','SFMono-Regular',Menlo,monospace;
}

html, body, .stApp, [data-testid="stAppViewContainer"]{
  background-color:var(--bg) !important; color:var(--ink);
}
.stApp{
  background-image:
    radial-gradient(880px 430px at 0% -10%, rgba(245,197,24,.16), transparent 62%),
    radial-gradient(760px 430px at 100% -8%, rgba(217,93,57,.08), transparent 60%),
    repeating-linear-gradient(0deg, rgba(20,24,29,.018) 0 1px, transparent 1px 5px);
  background-attachment:fixed;
}
[data-testid="stHeader"]{ background:transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{
  visibility:hidden !important; height:0 !important;
}
.block-container{ padding-top:1.55rem; padding-bottom:3.6rem; max-width:1380px; }
[data-testid="stSidebar"]{ width:306px !important; }

h1,h2,h3,h4{ font-family:var(--display) !important; color:var(--title); letter-spacing:.01em; }
p,li,label,span,div,input,button,textarea,td,th{ font-family:var(--sans); }
a{ color:var(--teal); }

/* ---------------------------------------------------------------- sidebar --- */
[data-testid="stSidebar"]{
  background:#FBF5E9 !important;
  border-right:1px solid var(--border);
}
[data-testid="stSidebar"] .block-container{ padding-top:.8rem; }
[data-testid="stSidebar"] hr{ border-color:var(--border-soft); margin:.85rem 0; }
[data-testid="stSidebarNav"] a{
  border-radius:8px !important; padding:7px 10px !important;
  transition:background .15s ease;
}
[data-testid="stSidebarNav"] a:hover{ background:var(--surface-alt) !important; }
[data-testid="stSidebarNav"] a[aria-current="page"]{
  background:linear-gradient(90deg, rgba(245,197,24,.28), rgba(245,197,24,.04)) !important;
  box-shadow:inset 3px 0 0 var(--asphalt);
}
[data-testid="stSidebarNav"] a[aria-current="page"] span{
  color:var(--asphalt) !important; font-weight:800 !important;
}
[data-testid="stSidebarNav"] span{ font-size:12.5px !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label{
  font-size:12px !important; color:var(--subtitle) !important;
}

.tx-side-label{
  font-family:var(--mono); font-size:9.5px; letter-spacing:.17em;
  text-transform:uppercase; color:var(--muted); font-weight:600;
  margin:2px 0 7px 2px;
}
.tx-side-note{ font-size:11.5px; color:#5E5D56; line-height:1.55; }
.tx-side-stat{
  display:flex; justify-content:space-between; gap:10px; font-size:11.5px;
  color:var(--subtitle); padding:5px 0; border-bottom:1px dashed var(--border);
}
.tx-side-stat b{ font-family:var(--mono); color:var(--asphalt); font-weight:600; }
.tx-side-plate{
  border:1px solid var(--border); background:var(--surface); border-radius:10px;
  padding:8px 10px; margin-top:8px;
  font-family:var(--mono); font-size:9px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); line-height:1.55;
}
.tx-side-plate b{ color:var(--asphalt); }

/* ------------------------------------------------------------------- hero --- */
.tx-hero{
  position:relative; overflow:hidden; border-radius:20px;
  background:
    radial-gradient(520px 260px at 92% -20%, rgba(245,197,24,.18), transparent 68%),
    linear-gradient(118deg,#111418 0%,#1B2026 54%,#252C33 100%);
  color:#FFF8E7; padding:1.5rem 1.8rem 1.7rem;
  box-shadow:0 18px 38px -28px rgba(20,24,29,.72);
  margin-bottom:1.1rem;
}
.tx-hero::before{
  content:""; position:absolute; right:-50px; bottom:-120px; width:370px; height:370px;
  border-radius:50%; border:1px solid rgba(245,197,24,.14);
  box-shadow:0 0 0 28px rgba(245,197,24,.035), 0 0 0 56px rgba(245,197,24,.02);
  pointer-events:none;
}
.tx-hero::after{
  content:""; position:absolute; left:0; right:0; bottom:0; height:9px;
  background:conic-gradient(var(--yellow) 25%, #101317 0 50%, var(--yellow) 0 75%, #101317 0);
  background-size:18px 18px; opacity:.95;
}
.tx-hero-top{
  display:flex; justify-content:space-between; align-items:center;
  gap:12px; margin-bottom:13px; position:relative; z-index:1;
}
.tx-eyebrow{
  display:inline-flex; align-items:center; gap:8px;
  font-family:var(--mono); font-size:9.5px; letter-spacing:.17em;
  text-transform:uppercase; color:#FBE8A3;
  background:rgba(245,197,24,.12); border:1px solid rgba(245,197,24,.32);
  padding:5px 11px; border-radius:999px;
}
.tx-eyebrow .dot{ width:6px; height:6px; border-radius:50%; background:var(--yellow); }
.tx-hero-meta{
  font-family:var(--mono); font-size:9.5px; letter-spacing:.1em;
  color:#AEB7BE; text-transform:uppercase; white-space:nowrap;
}
.tx-hero-main{ display:flex; align-items:center; gap:18px; position:relative; z-index:1; }
.tx-hero-mark{
  flex:0 0 auto; width:64px; height:64px; border-radius:15px;
  display:grid; place-items:center;
  background:var(--yellow); border:1px solid rgba(255,255,255,.25);
  box-shadow:0 10px 24px -14px rgba(0,0,0,.8);
}
.tx-hero-mark svg, .tx-hero-mark img{ width:43px; height:43px; display:block; }
.tx-title{
  font-family:var(--display) !important; font-weight:400;
  font-size:clamp(2.35rem,4.6vw,4rem); line-height:.94;
  letter-spacing:.012em; text-transform:uppercase; margin:0 0 7px; color:#FFF8E7;
}
.tx-title em{ font-style:normal; color:var(--yellow); }
.tx-lede{
  margin:0; max-width:920px; color:#CDD5DA; font-size:13.5px; line-height:1.65;
}
.tx-hero-pills{ display:flex; flex-wrap:wrap; gap:7px; margin-top:14px; position:relative; z-index:1; }
.tx-hero-pill{
  font-family:var(--mono); font-size:10px; letter-spacing:.04em;
  color:#E8EDF0; background:rgba(255,255,255,.07);
  border:1px solid rgba(255,255,255,.14); border-radius:999px; padding:4px 10px;
}
.tx-hero-pill b{ color:var(--yellow); font-weight:600; }

/* ------------------------------------------------------------------- kpis --- */
.tx-kpis{
  display:grid; grid-template-columns:repeat(var(--cols,6),minmax(0,1fr));
  gap:10px; margin:16px 0 2px;
}
@media (max-width:1220px){ .tx-kpis{ grid-template-columns:repeat(3,minmax(0,1fr)) !important; } }
@media (max-width:700px){ .tx-kpis{ grid-template-columns:repeat(2,minmax(0,1fr)) !important; } }
.tx-kpi{
  background:var(--surface); border:1px solid var(--border);
  border-radius:13px; padding:12px 14px 11px; position:relative; overflow:hidden;
  box-shadow:0 1px 2px rgba(20,24,29,.03), 0 14px 26px -28px rgba(20,24,29,.65);
}
.tx-kpi::before{
  content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
  background:var(--accent,var(--yellow));
}
.tx-kpi::after{
  content:""; position:absolute; right:10px; top:9px; width:18px; height:5px;
  background:conic-gradient(var(--accent,var(--yellow)) 25%, transparent 0 50%, var(--accent,var(--yellow)) 0 75%, transparent 0);
  background-size:6px 5px; opacity:.5;
}
.tx-kpi-label{
  font-family:var(--mono); font-size:8.8px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); font-weight:500; margin-bottom:5px;
  padding-right:18px;
}
.tx-kpi-value{
  font-family:var(--display); font-size:clamp(28px,2.8vw,40px);
  font-weight:400; color:var(--asphalt); line-height:.96; letter-spacing:.015em;
}
.tx-kpi-value small{ font-family:var(--sans); font-size:11px; color:var(--muted); font-weight:600; margin-left:3px; }
.tx-kpi-sub{ font-size:10.5px; color:var(--muted); margin-top:6px; line-height:1.45; }

/* ---------------------------------------------------------------- section --- */
.tx-section{ margin:26px 0 8px; }
.tx-kicker{
  font-family:var(--mono); font-size:9.5px; letter-spacing:.18em;
  text-transform:uppercase; color:var(--orange); font-weight:600; margin-bottom:3px;
}
.tx-section h3{
  font-family:var(--display) !important; font-size:30px; font-weight:400;
  margin:0 0 3px; color:var(--title); letter-spacing:.025em; text-transform:uppercase;
}
.tx-section p{ font-size:13px; color:var(--subtitle); margin:0; max-width:950px; line-height:1.6; }

/* --------------------------------------------------------------- briefing --- */
.tx-brief{ margin:6px 0 2px; }
.tx-brief-item{
  display:grid; grid-template-columns:52px minmax(0,1fr); gap:16px;
  padding:17px 0 18px; border-bottom:1px dashed var(--border); position:relative;
}
.tx-brief-item:first-child{ border-top:1px dashed var(--border); }
.tx-brief-item::before{
  content:""; position:absolute; left:0; top:18px; bottom:18px; width:3px;
  background:var(--accent,var(--yellow)); border-radius:2px;
}
.tx-brief-index{
  font-family:var(--display); font-size:30px; line-height:1; color:var(--faint);
  padding-left:11px; letter-spacing:.02em;
}
.tx-brief-body{ min-width:0; }
.tx-brief-meta{ display:flex; flex-wrap:wrap; align-items:center; gap:9px; margin-bottom:3px; }
.tx-brief-tag{
  font-family:var(--mono); font-size:8.8px; letter-spacing:.15em; text-transform:uppercase;
  color:var(--orange); font-weight:600;
}
.tx-brief-stat{
  font-family:var(--display); font-size:31px; font-weight:400; color:var(--asphalt);
  line-height:1; letter-spacing:.015em;
}
.tx-brief-stat small{ font-family:var(--sans); font-size:11px; color:var(--muted); font-weight:600; margin-left:2px; }
.tx-brief-title{
  font-size:13px; color:var(--asphalt); font-weight:800; margin:3px 0 3px; line-height:1.4;
}
.tx-brief-copy{ font-size:12px; color:var(--subtitle); margin:0; line-height:1.62; max-width:980px; }
@media (max-width:700px){
  .tx-brief-item{ grid-template-columns:38px minmax(0,1fr); gap:10px; }
  .tx-brief-index{ font-size:24px; padding-left:8px; }
}

/* ------------------------------------------------------------------ cards --- */
.tx-card{
  background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:15px 17px; box-shadow:0 1px 2px rgba(20,24,29,.03);
}
.tx-card h4{
  font-family:var(--display) !important; font-size:23px; font-weight:400;
  margin:0 0 2px; color:var(--asphalt); letter-spacing:.03em; text-transform:uppercase;
}
.tx-card p{ font-size:12px; color:var(--muted); margin:0; line-height:1.55; }

/* ------------------------------------------------------------------ table --- */
.tx-table{ width:100%; border-collapse:collapse; font-size:12.5px; }
.tx-table thead th{
  font-family:var(--mono); font-size:8.8px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); text-align:left; font-weight:600; padding:7px 9px;
  border-bottom:1px solid var(--border);
}
.tx-table tbody td{
  padding:9px 9px; border-bottom:1px dashed var(--border-soft); color:var(--subtitle);
  vertical-align:middle;
}
.tx-table tbody tr:last-child td{ border-bottom:none; }
.tx-table tbody tr:hover{ background:#FCF7EB; }
.tx-table .num{ text-align:right; font-family:var(--mono); color:var(--asphalt); }
.tx-table .name{ font-weight:700; color:var(--asphalt); }
.tx-table .subtle{ font-size:11px; color:var(--muted); }

.tx-chip{
  display:inline-block; font-family:var(--mono); font-size:8.8px; letter-spacing:.08em;
  text-transform:uppercase; padding:3px 8px; border-radius:999px; font-weight:600;
  white-space:nowrap;
}
.tx-chip.yellow{ background:var(--yellow-soft); color:#6D5300; }
.tx-chip.orange{ background:var(--orange-soft); color:#8C3419; }
.tx-chip.teal{ background:var(--teal-soft); color:#1B5B51; }
.tx-chip.dark{ background:var(--asphalt); color:#FFF8E7; }
.tx-chip.info{ background:var(--blue-soft); color:#274A6B; }

/* ------------------------------------------------------------------ meter --- */
.tx-meter{
  position:relative; overflow:hidden; border-radius:18px;
  background:
    radial-gradient(420px 220px at 100% 0%, rgba(245,197,24,.18), transparent 65%),
    linear-gradient(118deg,#12161A,#1D232A 60%,#293139);
  color:#FFF8E7; padding:1.45rem 1.55rem 1.6rem;
  box-shadow:0 18px 34px -24px rgba(20,24,29,.72);
}
.tx-meter::after{
  content:""; position:absolute; left:0; right:0; bottom:0; height:8px;
  background:conic-gradient(var(--yellow) 25%, #101317 0 50%, var(--yellow) 0 75%, #101317 0);
  background-size:16px 16px;
}
.tx-meter .plate{
  display:inline-block; font-family:var(--mono); font-size:8.5px; letter-spacing:.15em;
  text-transform:uppercase; color:#FBE8A3; border:1px solid rgba(245,197,24,.35);
  background:rgba(245,197,24,.10); border-radius:4px; padding:3px 7px;
}
.tx-meter .kicker{
  font-family:var(--mono); font-size:9px; letter-spacing:.16em; text-transform:uppercase;
  color:#AEB7BE; margin-top:12px;
}
.tx-meter .fare{
  font-family:var(--display); font-size:clamp(56px,7vw,82px); line-height:.88;
  letter-spacing:.02em; color:var(--yellow); margin:2px 0 4px;
}
.tx-meter .range{ font-size:12.5px; color:#D8DEE2; }
.tx-meter .range b{ color:#FFF8E7; }
.tx-meter .route{
  margin:14px 0 10px; height:10px; border-radius:999px;
  background:linear-gradient(90deg,var(--yellow),var(--orange));
  position:relative;
}
.tx-meter .route::before, .tx-meter .route::after{
  content:""; position:absolute; top:50%; width:12px; height:12px; border-radius:50%;
  background:#FFF8E7; border:3px solid #14181D; transform:translateY(-50%);
}
.tx-meter .route::before{ left:-2px; }
.tx-meter .route::after{ right:-2px; }
.tx-meter .meta{
  display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px;
  margin-top:14px; padding-top:12px; border-top:1px solid rgba(255,255,255,.16);
}
.tx-meter .meta div span{
  display:block; font-family:var(--mono); font-size:8.5px; letter-spacing:.1em;
  text-transform:uppercase; color:#9FAAB1; margin-bottom:3px;
}
.tx-meter .meta div b{ font-size:13px; color:#FFF8E7; font-weight:700; }
.tx-meter .fine{ font-size:10.5px; color:#9FAAB1; line-height:1.5; margin-top:10px; }

/* ------------------------------------------------------------------ note --- */
.tx-mono{ font-family:var(--mono); font-size:11px; color:var(--muted); }
.tx-empty{
  border:1px dashed var(--border); border-radius:16px; padding:2.2rem 1.5rem;
  text-align:center; background:rgba(255,252,244,.55);
}
.tx-empty .mark{ font-family:var(--display); font-size:40px; color:var(--yellow); line-height:1; }
.tx-empty h4{ margin:.3rem 0 .2rem; font-size:25px; font-weight:400; }
.tx-empty p{ margin:0 auto; max-width:360px; font-size:12px; color:var(--muted); line-height:1.55; }

/* --------------------------------------------------------------- streamlit --- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--surface); border-radius:14px;
}
[data-testid="stMetric"]{
  background:var(--surface); border:1px solid var(--border); border-radius:12px;
  padding:10px 12px;
}
[data-testid="stMetricLabel"] p{ font-size:11px !important; color:var(--muted) !important; }
[data-testid="stMetricValue"]{ font-family:var(--display) !important; color:var(--asphalt) !important; }

.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button{
  border-radius:8px !important; font-weight:700 !important;
  border:1px solid var(--border) !important;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"]{
  background:var(--asphalt) !important;
  border:1px solid var(--asphalt) !important; color:var(--yellow) !important;
  box-shadow:0 8px 18px -12px rgba(20,24,29,.7);
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover{
  background:var(--yellow) !important; color:var(--asphalt) !important;
  border-color:var(--yellow) !important;
}
[data-baseweb="tab-list"]{
  gap:4px; background:var(--surface-alt); padding:4px; border-radius:9px;
  border:1px solid var(--border);
}
[data-baseweb="tab"]{
  border-radius:6px !important; padding:7px 14px !important;
  font-size:12px !important; color:var(--muted) !important;
}
[data-baseweb="tab"][aria-selected="true"]{
  background:var(--surface) !important; color:var(--asphalt) !important;
  box-shadow:0 1px 3px rgba(20,24,29,.10);
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{ display:none; }

[data-testid="stExpander"]{
  background:var(--surface); border:1px solid var(--border) !important;
  border-radius:11px !important;
}
[data-testid="stDataFrame"]{
  border:1px solid var(--border); border-radius:10px; overflow:hidden;
}
[data-testid="stAlert"]{ border-radius:10px; }
[data-testid="stForm"]{
  background:var(--surface); border:1px solid var(--border); border-radius:14px;
}
/* Dropdown fields sit on a warmer shade so they read as editable controls. */
[data-testid="stForm"] [data-testid="stSelectbox"] [role="group"],
[data-testid="stForm"] [data-testid="stSelectbox"] [data-baseweb="select"] > div{
  background:var(--field) !important;
  border-color:var(--border) !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.55);
  transition:background .12s ease, border-color .12s ease;
}
[data-testid="stForm"] [data-testid="stSelectbox"] [role="group"]:hover,
[data-testid="stForm"] [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover{
  background:var(--field-hover) !important;
}
[data-testid="stForm"] [data-testid="stSelectbox"] [role="group"]:focus-within,
[data-testid="stForm"] [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within{
  background:var(--field-hover) !important;
  border-color:var(--asphalt) !important;
}

.tx-footer{
  margin-top:34px; padding-top:13px; border-top:1px solid var(--border);
  display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;
  font-family:var(--mono); font-size:9px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--faint);
}

/* ------------------------------------------------------------- plotly maps --- */
[data-testid="stPlotlyChart"] .maplibregl-control-container{
  position:absolute !important;
  inset:0 !important;
  width:100% !important;
  height:100% !important;
  pointer-events:none;
}
[data-testid="stPlotlyChart"] .maplibregl-ctrl-bottom-right{
  position:absolute !important;
  right:6px !important;
  bottom:6px !important;
}
[data-testid="stPlotlyChart"] .maplibregl-ctrl{
  pointer-events:auto;
}
[data-testid="stPlotlyChart"] .maplibregl-ctrl-attrib{
  background:rgba(255,252,244,.92) !important;
  border-radius:4px;
}
[data-testid="stPlotlyChart"] .maplibregl-ctrl-attrib summary{
  display:none !important;
}
[data-testid="stPlotlyChart"] .maplibregl-ctrl-attrib a{
  font-family:var(--mono) !important;
  font-size:9px !important;
  color:var(--muted) !important;
}

/* -------------------------------------------------------------- responsive --- */
@media (max-width:1100px){
  [data-testid="stHorizontalBlock"]{ flex-wrap:wrap !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
    flex:1 1 100% !important; width:100% !important; min-width:100% !important;
  }
}
@media (max-width:640px){
  .tx-hero{ padding:1.2rem 1.2rem 1.45rem; }
  .tx-hero-main{ gap:12px; align-items:flex-start; }
  .tx-hero-mark{ width:48px; height:48px; }
  .tx-hero-mark svg, .tx-hero-mark img{ width:32px; height:32px; }
  .tx-hero-meta{ display:none; }
  .tx-title{ font-size:2.1rem; }
}
</style>
"""


def inject() -> None:
    st.html(CSS)


_ICON_SVGS = {
    "dashboard": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none" '
        'stroke="#14181D" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M11 40h42v9H11z"/>'
        '<path d="M16 40l7-14h18l7 14"/>'
        '<path d="M23 26h18"/>'
        '<rect x="27" y="19" width="10" height="5" rx="1.5"/>'
        '<circle cx="21" cy="49" r="5"/>'
        '<circle cx="43" cy="49" r="5"/>'
        '<path d="M15 44h34"/>'
        "</svg>"
    ),
    "demand": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none" '
        'stroke="#14181D" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="32" cy="32" r="20"/>'
        '<path d="M32 19v14l9 6"/>'
        '<path d="M25 8h14M32 8v5"/>'
        "</svg>"
    ),
    "fares": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none" '
        'stroke="#14181D" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="10" y="16" width="44" height="32" rx="4"/>'
        '<path d="M18 26h20M18 34h12"/>'
        '<path d="M44 26v12M40 30h8"/>'
        '</svg>'
    ),
    "model": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none" '
        'stroke="#14181D" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="17" y="17" width="30" height="30" rx="3"/>'
        '<path d="M25 9v8M39 9v8M25 47v8M39 47v8M9 25h8M9 39h8M47 25h8M47 39h8"/>'
        '<path d="M27 27h10v10H27z"/>'
        "</svg>"
    ),
    "estimator": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none" '
        'stroke="#14181D" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M32 8c9 0 16 7 16 16 0 11-16 28-16 28S16 35 16 24c0-9 7-16 16-16z"/>'
        '<circle cx="32" cy="24" r="6"/>'
        "</svg>"
    ),
    "insights": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none" '
        'stroke="#14181D" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M24 43h16"/>'
        '<path d="M26 37c-4-3.4-6-8-6-12.5A12 12 0 0 1 44 24.5c0 4.5-2 9.1-6 12.5"/>'
        '<path d="M27 43v5h10v-5M29 54h6"/>'
        "</svg>"
    ),
}


def header(
    title: str,
    lede: str,
    *,
    eyebrow: str = "NYC TLC · YELLOW TAXI · JANUARY 2025",
    meta: str = "3.3M TRIPS · 5 BOROUGHS · 24 HOURS",
    pills: list[str] | None = None,
    icon: str = "dashboard",
) -> None:
    inject()
    pill_html = "".join(f'<span class="tx-hero-pill">{p}</span>' for p in (pills or []))
    icon_svg = _ICON_SVGS.get(icon, _ICON_SVGS["dashboard"])
    icon_uri = "data:image/svg+xml;base64," + base64.b64encode(
        icon_svg.encode("utf-8")
    ).decode("ascii")
    st.html(
f"""
        <div class="tx-hero">
          <div class="tx-hero-top">
            <span class="tx-eyebrow"><span class="dot"></span>{html.escape(eyebrow)}</span>
            <span class="tx-hero-meta">{html.escape(meta)}</span>
          </div>
          <div class="tx-hero-main">
            <div class="tx-hero-mark">
              <img src="{icon_uri}" alt="" aria-hidden="true" />
            </div>
            <div>
              <h1 class="tx-title">{title}</h1>
              <p class="tx-lede">{lede}</p>
            </div>
          </div>
          <div class="tx-hero-pills">{pill_html}</div>
        </div>
        """
    )


def section(title: str, description: str = "", kicker: str = "") -> None:
    st.html(
f"""
        <div class="tx-section">
          <div class="tx-kicker">{html.escape(kicker)}</div>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
        """
    )


def kpis(items: list[dict], columns: int | None = None) -> None:
    cols = columns or len(items)
    cards = []
    for item in items:
        accent = item.get("accent", YELLOW)
        cards.append(
            f"""
            <div class="tx-kpi" style="--accent:{accent}">
              <div class="tx-kpi-label">{html.escape(str(item['label']))}</div>
              <div class="tx-kpi-value">{item['value']}</div>
              <div class="tx-kpi-sub">{item.get('sub', '')}</div>
            </div>
            """
        )
    st.html(f'<div class="tx-kpis" style="--cols:{cols}">' + "".join(cards) + "</div>")


def chips(items: list[tuple[str, str]]) -> str:
    return "".join(
        f'<span class="tx-chip {tone}">{html.escape(text)}</span>' for text, tone in items
    )


def briefing(items: list[dict]) -> None:
    rows = []
    for index, item in enumerate(items, start=1):
        accent = item.get("accent", YELLOW)
        rows.append(
            f"""
            <div class="tx-brief-item" style="--accent:{accent}">
              <div class="tx-brief-index">{index:02d}</div>
              <div class="tx-brief-body">
                <div class="tx-brief-meta">
                  <span class="tx-brief-tag">{html.escape(item['tag'])}</span>
                  <span class="tx-brief-stat">{item['stat']}</span>
                </div>
                <div class="tx-brief-title">{item['title']}</div>
                <p class="tx-brief-copy">{item['copy']}</p>
              </div>
            </div>
            """
        )
    st.html('<div class="tx-brief">' + "".join(rows) + "</div>")


def footer(text: str = "NYC Yellow Taxi · January 2025 · fare intelligence") -> None:
    st.html(
f"""
        <div class="tx-footer">
          <span>{html.escape(text)}</span>
          <span>LightGBM · 14 features · holdout R² 0.948</span>
        </div>
        """
    )


def meter_card(
    fare: float,
    low: float,
    high: float,
    per_mile: float,
    tier: str,
    rate_label: str,
    distance: float,
    hour_label: str,
    day: str,
    comps: dict,
    *,
    model_label: str = "LightGBM",
    fallback: bool = False,
) -> None:
    comp_text = (
        f"<b>{comps['count']:,}</b> similar trips · observed median "
        f"<b>${comps['median']:.2f}</b>"
        if comps.get("count")
        else "No close comparison group in the sample"
    )
    fallback_note = (
        "<div class='fine'>Portable scikit-learn fallback active · "
        "the deployed app uses the saved LightGBM model.</div>"
        if fallback
        else ""
    )
    st.html(
f"""
        <div class="tx-meter">
          <span class="plate">ESTIMATE · TOTAL AMOUNT</span>
          <div class="kicker">Meter reading</div>
          <div class="fare">${fare:,.2f}</div>
          <div class="range">Typical range <b>${low:,.2f}–${high:,.2f}</b> · model error ±$2.61 MAE</div>
          <div class="route"></div>
          <div class="meta">
            <div><span>Per mile</span><b>${per_mile:,.2f}</b></div>
            <div><span>Trip tier</span><b>{html.escape(tier)}</b></div>
            <div><span>Rate</span><b>{html.escape(rate_label)}</b></div>
          </div>
          <div class="meta" style="margin-top:10px;padding-top:10px;">
            <div><span>Distance</span><b>{distance:,.1f} mi</b></div>
            <div><span>Pickup</span><b>{html.escape(hour_label)} · {html.escape(day[:3])}</b></div>
            <div><span>Model</span><b>{html.escape(model_label)}</b></div>
          </div>
          <div class="fine">{comp_text}. Estimate includes tips, tolls and fees; it is not a quote.</div>
          {fallback_note}
        </div>
        """
    )


def empty_state() -> None:
    st.html(
"""
        <div class="tx-empty">
          <div class="mark">$ --.--</div>
          <h4>Your meter is waiting</h4>
          <p>Set a pickup, a drop-off and a time. The saved fare model will price the ride and show how similar trips actually settled.</p>
        </div>
        """
    )


def side_snapshot(summary: dict, model_metrics: dict) -> None:
    """Sidebar dataset card shared by every page."""
    m = model_metrics["metrics"]
    with st.sidebar:
        st.html('<div class="tx-side-label">Snapshot</div>')
        st.html(
f"""
        <div class="tx-side-stat"><span>Valid trips</span><b>{summary['total_trips'] / 1e6:.2f}M</b></div>
        <div class="tx-side-stat"><span>Meter revenue</span><b>${summary['revenue'] / 1e6:.1f}M</b></div>
        <div class="tx-side-stat"><span>Peak hour</span><b>6 PM</b></div>
        <div class="tx-side-stat"><span>Model R²</span><b>{m['r2']:.4f}</b></div>
        <div class="tx-side-stat"><span>Holdout</span><b>{model_metrics['split']['test']:,}</b></div>
        <div class="tx-side-plate">
          <b>NYC TLC</b> Yellow Taxi · January 2025<br>
          Source: official trip-record parquet
        </div>
        """
        )

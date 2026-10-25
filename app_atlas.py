"""Station Atlas: dashboard for the WeatherAPI > Kafka > Spark > MongoDB pipeline.

Run:  streamlit run app_atlas.py     (http://localhost:8501)
"""
import io
import os
from datetime import datetime, timezone
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

import models as M

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB = os.getenv("MONGO_DB", "weather_db")
TZ = "Asia/Manila"
REFRESH_SECONDS = 15
RAW_LIMIT = 60000
AGG_LIMIT = 20000
FONT = "IBM Plex Sans, system-ui, sans-serif"

# palette (black and deep green)
C_PANEL, C_LINE, C_LINE2 = "#07150D", "#173625", "#2A5A40"
C_INK, C_INK2, C_INK3 = "#E4EEE7", "#A6BBAE", "#7A9484"
C_SAGE, C_AMBER = "#7DBE9A", "#E2B65C"
STATION_COLORS = ["#7DBE9A", "#4FA3A5", "#B6C46A", "#D9C9A0", "#9DB4D6", "#C99A7A"]

st.set_page_config(page_title="Station Atlas", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ styling --
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{
  --black:#000000; --panel:#07150D; --panel-hover:#0D2418; --raised:#123222;
  --line:#173625; --line-strong:#2A5A40;
  --ink:#E4EEE7; --ink-2:#A6BBAE; --ink-3:#7A9484;
  --sage:#7DBE9A; --amber:#E2B65C;
  --serif:'Fraunces', Georgia, 'Times New Roman', serif;
  --sans:'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp{background:var(--black); color:var(--ink); font-family:var(--sans);
  font-size:16px; line-height:24px;}
.stApp p, .stApp label, .stApp button, .stApp input, .stApp li, .stApp td, .stApp th,
.stApp [data-baseweb], .stApp [data-testid="stMarkdownContainer"]{font-family:var(--sans);}
.stApp [data-testid="stIconMaterial"]{font-family:"Material Symbols Rounded" !important;}
[data-testid="stHeader"]{background:transparent; height:0;}
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer{display:none !important;}
.block-container{max-width:1200px; padding:48px 32px 96px;}
.sr-only{position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap;}

/* masthead */
.masthead{display:flex; justify-content:space-between; align-items:flex-end; gap:32px;
  padding-bottom:24px; border-bottom:1px solid var(--line); margin-bottom:8px; flex-wrap:wrap;}
.wordmark{font-family:var(--serif); font-size:40px; line-height:48px; font-weight:600;
  letter-spacing:-0.01em; margin:0; color:var(--ink);}
.tagline{font-size:16px; line-height:24px; color:var(--ink-2); margin:8px 0 0; max-width:56ch;}
.facts{display:flex; gap:32px; margin:0; flex-wrap:wrap;}
.facts div{min-width:72px;}
.facts dt{font-size:13px; line-height:20px; color:var(--ink-3);}
.facts dd{margin:0; font-size:16px; line-height:24px; font-weight:500;
  font-variant-numeric:tabular-nums; color:var(--ink);}
.pulse{display:inline-flex; align-items:center; gap:8px;}
.pulse::before{content:""; width:8px; height:8px; border-radius:50%; background:var(--sage);
  animation:pulse 1.8s ease-in-out infinite;}
.pulse.stale::before{background:var(--amber); animation:none;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* sections */
.section-head{margin:48px 0 16px;}
.section-head h2{font-family:var(--serif); font-size:24px; line-height:32px; font-weight:500;
  margin:0; color:var(--ink);}
.section-head p{font-size:14px; line-height:20px; color:var(--ink-3); margin:8px 0 0; max-width:72ch;}

/* ledger */
.ledger-wrap{overflow-x:auto;}
.ledger{width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; min-width:880px;}
.ledger th{font-weight:500; font-size:13px; line-height:20px; color:var(--ink-3); text-align:left;
  padding:8px 16px; border-bottom:1px solid var(--line-strong); white-space:nowrap;}
.ledger td{padding:8px 16px; height:64px; border-bottom:1px solid var(--line); vertical-align:middle;
  font-size:16px; line-height:24px;}
.ledger .num{text-align:right;}
.ledger tbody tr{transition:background-color .12s ease;}
.ledger tbody tr:hover{background:var(--panel-hover);}
.ledger tr.is-selected{background:var(--panel-hover); box-shadow:inset 2px 0 0 var(--sage);}
.rank{color:var(--ink-3); width:32px;}
.station-name{font-family:var(--serif); font-size:20px; line-height:24px; font-weight:500; display:block;}
.station-sub{font-size:13px; line-height:20px; color:var(--ink-3); display:block;}
.z-cell{font-size:24px; line-height:32px; font-weight:500; text-align:right;}
.z-cell.hot{color:var(--amber);}
.dim{color:var(--ink-3);}
.spark{display:block; overflow:visible;}
.spark polyline{fill:none; stroke:var(--sage); stroke-width:1.5; stroke-linejoin:round;}
.spark circle{fill:var(--ink);}

/* status chips */
.chip{display:inline-flex; align-items:center; gap:8px; height:24px; padding:0 8px;
  border:1px solid var(--line-strong); border-radius:4px; font-size:13px; line-height:20px;
  color:var(--ink-2); white-space:nowrap;}
.chip::before{content:""; width:8px; height:8px; border-radius:50%; background:var(--ink-3);}
.chip-ok{color:var(--ink);} .chip-ok::before{background:var(--sage);}
.chip-watch{border-color:var(--amber); color:var(--amber);}
.chip-watch::before{background:transparent; border:2px solid var(--amber);}
.chip-anomaly{background:var(--amber); border-color:var(--amber); color:#000; font-weight:600;}
.chip-anomaly::before{background:#000;}
.chip-pending{border-style:dashed;}

/* readouts */
.readouts{display:grid; grid-template-columns:1fr 1fr; column-gap:24px; margin:0;}
.readouts div{border-top:1px solid var(--line); padding:12px 0 12px;}
.readouts dt{font-size:13px; line-height:20px; color:var(--ink-3);}
.readouts dd{margin:0; font-size:20px; line-height:24px; font-weight:500;
  font-variant-numeric:tabular-nums; color:var(--ink);}
.readouts dd small{font-size:13px; line-height:20px; color:var(--ink-3); font-weight:400; display:block;}
.readouts .wide{grid-column:1 / -1;}

/* small table */
.mini{width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums;}
.mini th{font-weight:500; font-size:13px; line-height:20px; color:var(--ink-3); text-align:right;
  padding:8px 8px; border-bottom:1px solid var(--line-strong);}
.mini th:first-child, .mini td:first-child{text-align:left; padding-left:0;}
.mini td{padding:8px; border-bottom:1px solid var(--line); text-align:right; font-size:14px; line-height:24px;}
.mini caption{caption-side:top; text-align:left; font-size:14px; line-height:20px; color:var(--ink-3);
  padding-bottom:8px;}

/* states */
.skeleton{background:linear-gradient(90deg,var(--panel) 25%,var(--panel-hover) 50%,var(--panel) 75%);
  background-size:200% 100%; animation:shimmer 1.6s linear infinite;
  border:1px solid var(--line); border-radius:4px;}
@keyframes shimmer{to{background-position:-200% 0}}
.notice{border:1px solid var(--line-strong); border-radius:4px; padding:24px; background:var(--panel);}
.notice strong{display:block; font-family:var(--serif); font-size:20px; line-height:24px; font-weight:500;
  margin-bottom:8px;}
.notice p{margin:0; color:var(--ink-2); font-size:14px; line-height:20px; max-width:64ch;}
.notice code{background:var(--raised); padding:0 4px; border-radius:2px; color:var(--ink);}
@media (prefers-reduced-motion:reduce){.skeleton,.pulse::before{animation:none;}
  .ledger tbody tr{transition:none;}}

/* tabs */
[data-baseweb="tab-list"]{gap:8px; border-bottom:1px solid var(--line);}
button[data-baseweb="tab"]{height:48px; padding:0 16px; background:transparent; color:var(--ink-2);
  font-size:16px; border-radius:0;}
button[data-baseweb="tab"]:hover{color:var(--ink); background:var(--panel);}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--ink); font-weight:600;}
button[data-baseweb="tab"]:focus-visible{outline:2px solid var(--sage); outline-offset:-2px;}
[data-baseweb="tab-highlight"]{background:var(--sage) !important; height:2px;}
[data-baseweb="tab-border"]{background:transparent !important;}

/* station picker (radio as segmented control) */
[data-testid="stRadio"] [role="radiogroup"]{gap:8px; flex-wrap:wrap;}
[data-testid="stRadio"] label[data-baseweb="radio"]{margin:0; padding:8px 16px; min-height:40px;
  border:1px solid var(--line-strong); border-radius:4px; background:transparent; cursor:pointer;
  transition:background-color .12s ease, border-color .12s ease;}
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child{display:none;}
[data-testid="stRadio"] label[data-baseweb="radio"]:hover{background:var(--panel-hover);}
[data-testid="stRadio"] label[data-baseweb="radio"]:active{background:var(--raised);}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked){background:var(--raised);
  border-color:var(--sage);}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:focus-visible){outline:2px solid var(--sage);
  outline-offset:2px;}
[data-testid="stRadio"] label p{font-size:15px; line-height:24px; margin:0; color:var(--ink);}

/* buttons */
[data-testid^="stBaseButton"], .stDownloadButton button{background:var(--raised); color:var(--ink);
  border:1px solid var(--line-strong); border-radius:4px; min-height:40px; padding:0 24px; font-weight:500;
  transition:background-color .12s ease, border-color .12s ease;}
[data-testid^="stBaseButton"]:hover, .stDownloadButton button:hover{background:var(--panel-hover);
  border-color:var(--sage); color:var(--ink);}
[data-testid^="stBaseButton"]:active, .stDownloadButton button:active{transform:translateY(1px);
  background:var(--panel);}
[data-testid^="stBaseButton"]:focus-visible, .stDownloadButton button:focus-visible{
  outline:2px solid var(--sage); outline-offset:2px;}
[data-testid^="stBaseButton"]:disabled, .stDownloadButton button:disabled{opacity:.4; cursor:not-allowed;
  border-color:var(--line); transform:none;}

/* inputs */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"]{
  background:var(--panel) !important; border-color:var(--line-strong) !important; border-radius:4px;}
[data-baseweb="select"] > div:hover, [data-baseweb="input"]:hover{border-color:var(--sage) !important;}
[data-baseweb="select"]:focus-within > div, [data-baseweb="input"]:focus-within{
  border-color:var(--sage) !important; box-shadow:0 0 0 1px var(--sage);}
[data-testid="stWidgetLabel"] p{font-size:13px; line-height:20px; color:var(--ink-3);}
[data-testid="stDataFrame"]{border:1px solid var(--line); border-radius:4px;}
[data-testid="stExpander"] details{border:1px solid var(--line); border-radius:4px; background:transparent;}
@media (max-width:720px){.block-container{padding:24px 16px 64px;} .wordmark{font-size:32px; line-height:40px;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------- data --
@st.cache_resource
def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    db = client[DB]
    db["raw_readings"].create_index([("event_time", -1)])
    db["aggregates"].create_index([("window_start", -1)])
    return db


@st.cache_data(ttl=10, show_spinner=False)
def load_raw() -> pd.DataFrame:
    cur = (get_db()["raw_readings"].find({}, {"_id": 0, "event_time": 0})
           .sort("event_time", -1).limit(RAW_LIMIT))
    df = pd.DataFrame(list(cur))
    if df.empty:
        return df
    df["time"] = (pd.to_datetime(df["ingested_at"], utc=True, format="ISO8601")
                  .dt.tz_convert(TZ))
    return (df.drop(columns=["ingested_at"]).sort_values("time")
            .reset_index(drop=True))


@st.cache_data(ttl=10, show_spinner=False)
def load_agg() -> pd.DataFrame:
    cur = (get_db()["aggregates"].find({}, {"_id": 0})
           .sort("window_start", -1).limit(AGG_LIMIT))
    df = pd.DataFrame(list(cur))
    if df.empty:
        return df
    for c in ("window_start", "window_end"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(TZ)
    return df.sort_values("window_start").reset_index(drop=True)


@st.cache_data(ttl=10, show_spinner=False)
def atlas_state():
    raw = load_raw()
    if raw.empty:
        return None
    atlas = M.build_atlas(raw)
    atlas["raw_count"] = len(raw)
    return atlas


@st.cache_data(ttl=10, show_spinner=False)
def pipeline_stats() -> dict:
    coll = get_db()["raw_readings"]
    last = coll.find_one({}, {"_id": 0, "ingested_at": 1}, sort=[("event_time", -1)])
    last_ts = None
    if last and last.get("ingested_at"):
        last_ts = pd.to_datetime(last["ingested_at"], utc=True, format="ISO8601")
    return {"stored": coll.estimated_document_count(),
            "windows": get_db()["aggregates"].estimated_document_count(),
            "stations": len(coll.distinct("city")), "last": last_ts}


# ------------------------------------------------------------------ helpers --
def fnum(x, d=1, sign=False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&mdash;"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def loc(series) -> pd.Series:
    """tz-aware series -> naive local clock time for plotting."""
    return pd.to_datetime(series).dt.tz_convert(TZ).dt.tz_localize(None)


def skeleton(height: int) -> str:
    return (f'<div class="skeleton" role="status" aria-label="Loading" '
            f'style="height:{height}px"></div>')


def notice(title: str, body: str) -> str:
    return f'<div class="notice"><strong>{title}</strong><p>{body}</p></div>'


def section(title: str, note: str = "") -> None:
    p = f"<p>{note}</p>" if note else ""
    st.markdown(f'<div class="section-head"><h2>{title}</h2>{p}</div>',
                unsafe_allow_html=True)


def spark_svg(vals, w=120, h=32) -> str:
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return '<span class="dim">&mdash;</span>'
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = len(vals)
    pts = [(3 + i * (w - 6) / (n - 1), h - 4 - (v - lo) / span * (h - 8))
           for i, v in enumerate(vals)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    lx, ly = pts[-1]
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="Last {n} observations, {lo:.1f} to {hi:.1f} degrees">'
            f'<polyline points="{poly}"/><circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.5"/></svg>')


def ledger_html(ranking: pd.DataFrame, selected: str) -> str:
    head = ("<thead><tr>"
            '<th scope="col" class="num">Rank</th><th scope="col">Station</th>'
            '<th scope="col" class="num">Now (°C)</th>'
            '<th scope="col" class="num">Expected (°C)</th>'
            '<th scope="col" class="num">Residual z</th>'
            '<th scope="col" class="num">In 3 hours (°C)</th>'
            '<th scope="col" class="num">Fit R²</th>'
            '<th scope="col">Last 24 observations</th>'
            '<th scope="col">Status</th></tr></thead>')
    rows = []
    for i, r in enumerate(ranking.itertuples(), 1):
        cls = "is-selected" if r.city == selected else ""
        cur = ' aria-current="true"' if r.city == selected else ""
        hot = " hot" if r.status_key in ("watch", "anomaly") else ""
        observation_label = "observation" if r.n == 1 else "observations"
        rows.append(
            f'<tr class="{cls}"{cur}>'
            f'<td class="num rank">{i}</td>'
            f'<td><span class="station-name">{escape(r.city)}</span>'
            f'<span class="station-sub">{escape(str(r.country))}, {r.n} {observation_label}</span></td>'
            f'<td class="num">{fnum(r.now)}</td>'
            f'<td class="num">{fnum(r.expected)}</td>'
            f'<td class="z-cell{hot}">{fnum(r.z, 2, sign=True)}</td>'
            f'<td class="num">{fnum(r.f3)}</td>'
            f'<td class="num">{fnum(r.r2, 2)}</td>'
            f'<td>{spark_svg(r.spark)}</td>'
            f'<td><span class="chip chip-{r.status_key}">{escape(r.status)}</span></td></tr>')
    return ('<div class="ledger-wrap"><table class="ledger">'
            '<caption class="sr-only">Stations ranked by residual z-score</caption>'
            f'{head}<tbody>{"".join(rows)}</tbody></table></div>')


def readouts_html(t: dict, n_polls: int) -> str:
    f = t["forecast"]
    f3, f12 = f.iloc[2], f.iloc[-1]
    form = "Trend and daily cycle" if t["harmonic"] else "Linear trend"
    items = [
        ("Latest observation", f"{fnum(t['now'])} °C", "", False),
        ("Model expected", f"{fnum(t['expected'])} °C", "", False),
        ("Residual z-score", fnum(t["z"], 2, sign=True), t["status"], False),
        ("Drift per day", f"{fnum(t['slope_per_day'], 2, sign=True)} °C", "", False),
        ("In 3 hours", f"{fnum(f3['mean'])} °C",
         f"{fnum(f3['lo'])} to {fnum(f3['hi'])}", False),
        ("In 12 hours", f"{fnum(f12['mean'])} °C",
         f"{fnum(f12['lo'])} to {fnum(f12['hi'])}", False),
        ("R²", fnum(t["r2"], 3), f"adjusted {fnum(t['adj_r2'], 3)}", False),
        ("RMSE", f"{fnum(t['rmse'], 2)} °C", "", False),
        ("Observations", f"{t['n']}", f"from {n_polls:,} polls", False),
        ("Model form", form, "", False),
    ]
    out = []
    for label, val, sub, wide in items:
        s = f"<small>{sub}</small>" if sub else ""
        out.append(f'<div><dt>{label}</dt><dd>{val}{s}</dd></div>')
    return f'<dl class="readouts">{"".join(out)}</dl>'


def drivers_html(d: dict, city: str) -> str:
    rows = []
    for r in d["table"].itertuples():
        rows.append(f"<tr><td>{escape(r.term)}</td><td>{fnum(r.coef, 3, sign=True)}</td>"
                    f"<td>{fnum(r.se, 3)}</td><td>{fnum(r.t, 1, sign=True)}</td>"
                    f"<td>{fnum(r.std_beta, 2, sign=True)}</td></tr>")
    cap = (f"Temperature explained by humidity, wind and pressure at {escape(city)}: "
           f"R² {fnum(d['r2'], 3)}, RMSE {fnum(d['rmse'], 2)} °C, n = {d['n']}.")
    return (f'<table class="mini"><caption>{cap}</caption><thead><tr><th scope="col">Term</th>'
            '<th scope="col">Coefficient</th><th scope="col">Std. error</th>'
            '<th scope="col">t</th><th scope="col">Std. beta</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# ------------------------------------------------------------------- charts --
def style(fig: go.Figure, height: int, hover="x unified") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=C_INK2, size=13), hovermode=hover,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=C_INK2)),
        hoverlabel=dict(bgcolor=C_PANEL, bordercolor=C_LINE2,
                        font=dict(family=FONT, color=C_INK, size=13)))
    fig.update_xaxes(gridcolor=C_LINE, zeroline=False, linecolor=C_LINE2,
                     tickfont=dict(color=C_INK3))
    fig.update_yaxes(gridcolor=C_LINE, zeroline=False, linecolor=C_LINE2,
                     tickfont=dict(color=C_INK3))
    return fig


def show(fig: go.Figure) -> None:
    cfg = {"displaylogo": False, "displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", theme=None, config=cfg)
    except TypeError:  # older Streamlit without the width argument
        st.plotly_chart(fig, use_container_width=True, theme=None, config=cfg)


def forecast_fig(t: dict) -> go.Figure:
    times = loc(t["times"]).reset_index(drop=True)
    f = t["forecast"].copy()
    ft = loc(f["time"]).reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ft, y=f["hi"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ft, y=f["lo"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(125,190,154,0.18)",
                             name="95% prediction interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=times, y=t["y"], mode="lines+markers", name="Observed",
                             line=dict(color=C_INK3, width=1),
                             marker=dict(size=4, color=C_INK),
                             hovertemplate="%{y:.1f} °C"))
    fig.add_trace(go.Scatter(x=times, y=t["fitted"], mode="lines", name="Model fit",
                             line=dict(color=C_SAGE, width=2),
                             hovertemplate="%{y:.1f} °C"))
    fx = pd.concat([times.iloc[[-1]], ft], ignore_index=True)
    fy = np.concatenate([[t["fitted"][-1]], f["mean"].to_numpy()])
    fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", name="Forecast",
                             line=dict(color=C_SAGE, width=2, dash="dash"),
                             hovertemplate="%{y:.1f} °C"))
    flag = np.abs(t["stud"]) >= M.Z_ANOMALY
    if flag.any():
        fig.add_trace(go.Scatter(x=times[flag], y=np.asarray(t["y"])[flag], mode="markers",
                                 name="Outside 2 sigma",
                                 marker=dict(size=10, color=C_AMBER, symbol="diamond",
                                             line=dict(color="#000", width=1)),
                                 hovertemplate="%{y:.1f} °C"))
    last = times.iloc[-1]
    fig.add_shape(type="line", x0=last, x1=last, y0=0, y1=1, yref="paper",
                  line=dict(color=C_LINE2, width=1, dash="dot"))
    fig.add_annotation(x=last, y=1, yref="paper", text="Latest observation", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(color=C_INK3, size=12))
    style(fig, 440)
    fig.update_yaxes(title_text="Temperature (°C)", title_font=dict(color=C_INK3))
    if (last - times.iloc[0]) > pd.Timedelta(hours=72):
        fig.update_xaxes(range=[last - pd.Timedelta(hours=72), ft.iloc[-1]])
    fig.update_xaxes(rangeselector=dict(
        bgcolor=C_PANEL, activecolor=C_LINE2, bordercolor=C_LINE2, borderwidth=1,
        font=dict(color=C_INK2),
        buttons=[dict(count=24, label="24 h", step="hour", stepmode="backward"),
                 dict(count=3, label="3 days", step="day", stepmode="backward"),
                 dict(step="all", label="All")]))
    return fig


def residual_fig(t: dict) -> go.Figure:
    times = loc(t["times"]).reset_index(drop=True)
    z = np.asarray(t["stud"])
    colors = np.where(np.abs(z) >= M.Z_ANOMALY, C_AMBER, C_SAGE)
    fig = go.Figure()
    fig.add_hrect(y0=-2, y1=2, fillcolor="rgba(125,190,154,0.08)", line_width=0)
    fig.add_hline(y=0, line=dict(color=C_LINE2, width=1))
    fig.add_trace(go.Scatter(x=times, y=z, mode="markers", showlegend=False,
                             marker=dict(size=6, color=colors),
                             hovertemplate="z = %{y:+.2f}"))
    style(fig, 280, hover="closest")
    fig.update_yaxes(title_text="Studentized residual", title_font=dict(color=C_INK3))
    return fig


def heat_fig(corr: pd.DataFrame) -> go.Figure:
    names = {"temp_c": "Temperature", "humidity": "Humidity",
             "wind_kph": "Wind", "pressure_mb": "Pressure"}
    labels = [names[c] for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale=[[0, "#B8873A"], [0.5, "#0A1F16"], [1, "#7DBE9A"]],
        texttemplate="%{z:.2f}", textfont=dict(color=C_INK, size=13),
        showscale=False, hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>"))
    style(fig, 280, hover="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def pooled_fig(stations: dict, pooled: dict) -> go.Figure:
    fig = go.Figure()
    for i, (city, s) in enumerate(sorted(stations.items())):
        o = s["obs"]
        fig.add_trace(go.Scatter(
            x=o["humidity"], y=o["temp_c"], mode="markers", name=city,
            marker=dict(size=6, opacity=0.7, color=STATION_COLORS[i % len(STATION_COLORS)]),
            hovertemplate=f"{city}: %{{x:.0f}}% humidity, %{{y:.1f}} °C<extra></extra>"))
    fig.add_trace(go.Scatter(x=pooled["line_x"], y=pooled["line_y"], mode="lines",
                             name="Pooled regression line",
                             line=dict(color=C_INK, width=2), hoverinfo="skip"))
    style(fig, 420, hover="closest")
    fig.update_xaxes(title_text="Relative humidity (%)", title_font=dict(color=C_INK3))
    fig.update_yaxes(title_text="Temperature (°C)", title_font=dict(color=C_INK3))
    return fig


def agg_fig(a: pd.DataFrame) -> go.Figure:
    x = loc(a["window_start"]).reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=a["max_temp"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=a["min_temp"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(125,190,154,0.18)",
                             name="Min to max", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=a["avg_temp"], mode="lines+markers", name="Window average",
                             line=dict(color=C_SAGE, width=2), marker=dict(size=4, color=C_INK),
                             customdata=np.stack([a["min_temp"], a["max_temp"], a["count"]], axis=-1),
                             hovertemplate=("%{y:.1f} °C average<br>%{customdata[0]:.1f} to "
                                            "%{customdata[1]:.1f} °C, %{customdata[2]} readings")))
    style(fig, 380)
    fig.update_yaxes(title_text="Temperature (°C)", title_font=dict(color=C_INK3))
    return fig


def compare_fig(agg: pd.DataFrame) -> go.Figure:
    g = (agg.groupby("city").agg(avg=("avg_temp", "mean"), lo=("min_temp", "min"),
                                 hi=("max_temp", "max")).sort_values("avg"))
    fig = go.Figure(go.Bar(
        x=g["avg"], y=g.index, orientation="h", marker=dict(color=C_SAGE),
        customdata=np.stack([g["lo"], g["hi"]], axis=-1),
        hovertemplate="%{y}: %{x:.1f} °C average, range %{customdata[0]:.1f} to %{customdata[1]:.1f}<extra></extra>"))
    style(fig, 280, hover="closest")
    fig.update_xaxes(title_text="Mean of window averages (°C)", title_font=dict(color=C_INK3))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------------------------------------------- export --
def naive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.DatetimeTZDtype):
            out[c] = out[c].dt.tz_localize(None)
    return out


def summary_table(ranking: pd.DataFrame) -> pd.DataFrame:
    s = ranking.drop(columns=["spark", "status_key"]).copy()
    s = s[["city", "country", "n", "now", "expected", "z", "f3", "r2", "status"]]
    s.columns = ["Station", "Country", "Observations", "Latest (C)", "Expected (C)",
                 "Residual z", "In 3 hours (C)", "Fit R2", "Status"]
    return s.round(3)


@st.cache_data(ttl=30, show_spinner=False)
def make_xlsx(history: pd.DataFrame, agg: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        naive(history).to_excel(xw, sheet_name="Readings", index=False)
        if not agg.empty:
            naive(agg).to_excel(xw, sheet_name="Aggregates", index=False)
        summary.to_excel(xw, sheet_name="Model summary", index=False)
        for ws in xw.sheets.values():
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0
                            for c in col[:60]) + 2
                ws.column_dimensions[col[0].column_letter].width = min(width, 32)
    return buf.getvalue()


@st.cache_data(ttl=30, show_spinner=False)
def make_pdf(history: pd.DataFrame, summary: pd.DataFrame, window_label: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    green = colors.HexColor("#0D2518")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=28, rightMargin=28,
                            topMargin=28, bottomMargin=28)
    ss = getSampleStyleSheet()

    def table(data):
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), green), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white])]))
        return t

    per_city = (history.groupby("city")
                .agg(readings=("temp_c", "size"), mean=("temp_c", "mean"),
                     low=("temp_c", "min"), high=("temp_c", "max"),
                     humidity=("humidity", "mean"), wind=("wind_kph", "mean"))
                .round(1).reset_index())
    per_city.columns = ["Station", "Readings", "Mean (C)", "Min (C)", "Max (C)",
                        "Mean humidity (%)", "Mean wind (kph)"]
    cols = ["time", "city", "temp_c", "humidity", "wind_kph", "pressure_mb", "condition"]
    recent = naive(history[cols].tail(150).iloc[::-1])
    recent["time"] = recent["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    story = [
        Paragraph("Station Atlas report", ss["Title"]),
        Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M}, window: {window_label}, "
                  f"{len(history):,} readings, pipeline WeatherAPI to Kafka to Spark to MongoDB.",
                  ss["Normal"]),
        Spacer(1, 12), Paragraph("Regression summary by station", ss["Heading2"]),
        table([list(summary.columns)] + summary.astype(str).values.tolist()), Spacer(1, 12),
        Paragraph("Descriptive statistics by station", ss["Heading2"]),
        table([list(per_city.columns)] + per_city.astype(str).values.tolist()), Spacer(1, 12),
        Paragraph("Most recent readings (up to 150 rows)", ss["Heading2"]),
        table([cols] + recent.astype(str).values.tolist()),
    ]
    doc.build(story)
    return buf.getvalue()


# --------------------------------------------------------------------- views --
@st.fragment(run_every=REFRESH_SECONDS)
def masthead():
    s = pipeline_stats()
    if s["last"] is None:
        status, cls = "Waiting for the first reading", " stale"
        last_txt = "none yet"
    else:
        age = max(0, int((datetime.now(timezone.utc) - s["last"].to_pydatetime()).total_seconds()))
        live = age <= 90
        status = f"Streaming, last reading {age} s ago" if live else f"No new readings for {age // 60} min"
        cls = "" if live else " stale"
        last_txt = s["last"].tz_convert(TZ).strftime("%H:%M:%S")
    st.markdown(
        '<header class="masthead"><div>'
        '<h1 class="wordmark">Station Atlas</h1>'
        '<p class="tagline">Weather readings from five cities, fitted with a regression per station '
        'and ranked by how far the latest reading strays from the model.</p></div>'
        '<dl class="facts">'
        f'<div><dt>Stations</dt><dd>{s["stations"]}</dd></div>'
        f'<div><dt>Readings stored</dt><dd>{s["stored"]:,}</dd></div>'
        f'<div><dt>Spark windows</dt><dd>{s["windows"]:,}</dd></div>'
        f'<div><dt>Last reading (Manila)</dt><dd>{last_txt}</dd></div>'
        f'<div><dt>Pipeline</dt><dd><span class="pulse{cls}">{status}</span></dd></div>'
        '</dl></header>', unsafe_allow_html=True)


@st.fragment(run_every=REFRESH_SECONDS)
def atlas_view():
    holder = st.empty()
    if "atlas_ready" not in st.session_state:
        holder.markdown(skeleton(360), unsafe_allow_html=True)
    state = atlas_state()
    holder.empty()
    if state is None:
        st.markdown(notice(
            "No readings in MongoDB yet",
            "Start <code>weather_producer.py</code> and <code>spark_processor.py</code>. "
            "This page fills in within one refresh (15 seconds) of the first batch."),
            unsafe_allow_html=True)
        return
    st.session_state["atlas_ready"] = True

    ranking, stations, pooled = state["ranking"], state["stations"], state["pooled"]
    order = ranking["city"].tolist()
    if st.session_state.get("inspect") not in order:
        st.session_state["inspect"] = order[0]
    selected = st.session_state["inspect"]

    section("Stations ranked by how unusual they look now",
            "Each station has its own regression. The residual z-score compares the latest observation "
            "with what the model expected: beyond 1.5 is a watch, beyond 2 is an anomaly.")
    st.markdown(ledger_html(ranking, selected), unsafe_allow_html=True)

    st.markdown('<div class="section-head" style="margin-top:24px"><p>Open a station</p></div>',
                unsafe_allow_html=True)
    selected = st.radio("Station", order, key="inspect", horizontal=True,
                        label_visibility="collapsed")

    s = stations[selected]
    t = s["trend"]
    section(f"{escape(selected)}: observed, fitted and the next 12 hours")
    if not t["ok"]:
        need = t["need"]
        st.markdown(notice(
            f"{escape(selected)} has {t['n']} of {need} observations needed",
            "The trend model is fitted once a station has 8 distinct observations. WeatherAPI "
            "refreshes every 10 to 15 minutes, so this takes a while live. To fill the history now, "
            "run <code>python backfill_history.py</code> while Spark is running."),
            unsafe_allow_html=True)
        st.markdown(skeleton(320), unsafe_allow_html=True)
    else:
        form = ("a linear trend plus a 24-hour cycle" if t["harmonic"]
                else "a linear trend (the daily cycle is added after 24 hours of data)")
        st.caption(f"Ordinary least squares with {form}. {t['n']} distinct observations from "
                   f"{s['polls']:,} polls; the shaded band is the 95% prediction interval.")
        left, right = st.columns([8, 4], gap="large")
        with left:
            show(forecast_fig(t))
        with right:
            st.markdown(readouts_html(t, s["polls"]), unsafe_allow_html=True)

        section("What explains the residuals",
                "Left: residuals over time, amber where the reading is more than 2 standard deviations "
                "from the fit. Right: a second regression of temperature on humidity, wind and pressure.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            show(residual_fig(t))
        with c2:
            d = M.fit_drivers(s["obs"])
            if d["ok"]:
                st.markdown(drivers_html(d, selected), unsafe_allow_html=True)
                show(heat_fig(d["corr"]))
            else:
                st.markdown(notice(
                    "Driver model not ready",
                    f"It needs {M.DRIVER_MIN_OBS} observations with some variation in humidity, "
                    f"wind or pressure; {selected} has {d['n']}."), unsafe_allow_html=True)

    section("Humidity against temperature, all stations",
            "One regression pooled across every station's distinct observations. Compare it with each "
            "station's own driver model above: pooling mixes climates, so the slope can differ.")
    if pooled["ok"]:
        c1, c2 = st.columns([8, 4], gap="large")
        with c1:
            show(pooled_fig(stations, pooled))
        with c2:
            items = [("Slope", f"{fnum(pooled['slope'], 3, sign=True)} °C per % humidity"),
                     ("Intercept", f"{fnum(pooled['intercept'], 1)} °C"),
                     ("Pearson r", fnum(pooled["r"], 3, sign=True)),
                     ("R²", fnum(pooled["r2"], 3)),
                     ("Observations", f"{pooled['n']:,}")]
            st.markdown('<dl class="readouts">' + "".join(
                f'<div class="wide"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in items) + "</dl>",
                unsafe_allow_html=True)
    else:
        st.markdown(notice("Pooled model not ready",
                           "It needs at least 8 distinct observations across stations."),
                    unsafe_allow_html=True)


@st.fragment(run_every=REFRESH_SECONDS)
def aggregates_view():
    agg = load_agg()
    section("Spark window aggregates",
            "Structured Streaming groups readings into 5-minute windows per city and stores the "
            "average, minimum and maximum. Shown in Manila time.")
    if agg.empty:
        st.markdown(notice("No windows yet",
                           "Windows appear after Spark has processed a few minutes of readings."),
                    unsafe_allow_html=True)
        return
    cities = sorted(agg["city"].unique())
    city = st.selectbox("Station", cities, key="agg_city")
    a = agg[agg["city"] == city]
    c1, c2 = st.columns([8, 4], gap="large")
    with c1:
        show(agg_fig(a))
    with c2:
        show(compare_fig(agg))
    st.dataframe(
        naive(agg.sort_values("window_start", ascending=False)), hide_index=True, width="stretch",
        height=360, column_config={
            "window_start": st.column_config.DatetimeColumn("Window start", format="MMM D, HH:mm"),
            "window_end": st.column_config.DatetimeColumn("Window end", format="MMM D, HH:mm"),
            "city": "Station",
            "avg_temp": st.column_config.NumberColumn("Average (C)", format="%.1f"),
            "min_temp": st.column_config.NumberColumn("Min (C)", format="%.1f"),
            "max_temp": st.column_config.NumberColumn("Max (C)", format="%.1f"),
            "avg_humidity": st.column_config.NumberColumn("Humidity (%)", format="%.0f"),
            "avg_wind": st.column_config.NumberColumn("Wind (kph)", format="%.1f"),
            "count": st.column_config.NumberColumn("Readings", format="%d")})


def history_view():
    raw = load_raw()
    section("Every reading collected",
            "Raw records as stored in MongoDB, one row per poll. Filter, search or open the JSON.")
    if raw.empty:
        st.markdown(notice("History is empty", "Nothing has been stored yet."), unsafe_allow_html=True)
        return
    c1, c2, c3 = st.columns([3, 3, 4])
    cities = c1.multiselect("Stations", sorted(raw["city"].unique()), key="h_cities")
    conds = c2.multiselect("Conditions", sorted(raw["condition"].unique()), key="h_conds")
    text = c3.text_input("Search station or condition", key="h_text")
    v = raw
    if cities:
        v = v[v["city"].isin(cities)]
    if conds:
        v = v[v["condition"].isin(conds)]
    if text:
        v = v[v["city"].str.contains(text, case=False) | v["condition"].str.contains(text, case=False)]
    v = v.sort_values("time", ascending=False)
    st.caption(f"{len(v):,} of {len(raw):,} readings loaded (newest {RAW_LIMIT:,} at most).")
    cols = ["time", "city", "country", "temp_c", "humidity", "wind_kph", "pressure_mb",
            "condition", "lat", "lon"]
    st.dataframe(naive(v[[c for c in cols if c in v.columns]]), hide_index=True, width="stretch",
                 height=480, column_config={
                     "time": st.column_config.DatetimeColumn("Time (Manila)", format="MMM D, HH:mm:ss"),
                     "city": "Station", "country": "Country",
                     "temp_c": st.column_config.NumberColumn("Temp (C)", format="%.1f"),
                     "humidity": st.column_config.NumberColumn("Humidity (%)", format="%d"),
                     "wind_kph": st.column_config.NumberColumn("Wind (kph)", format="%.1f"),
                     "pressure_mb": st.column_config.NumberColumn("Pressure (mb)", format="%.0f"),
                     "condition": "Condition", "lat": None, "lon": None})
    with st.expander("Show the newest 5 records as JSON"):
        st.json(naive(v.head(5)).astype(str).to_dict("records"))


def export_view():
    raw = load_raw()
    section("Export", "Download the readings you select, with the Spark aggregates and the model "
                      "summary included in the Excel file and the PDF.")
    state = atlas_state()
    if raw.empty or state is None:
        st.markdown(notice("Nothing to export yet", "Downloads unlock after the first readings arrive."),
                    unsafe_allow_html=True)
        st.download_button("Download CSV", data=b"", disabled=True, key="x_csv_off")
        return
    c1, c2 = st.columns([6, 4])
    cities = c1.multiselect("Stations", sorted(raw["city"].unique()),
                            default=sorted(raw["city"].unique()), key="x_cities")
    windows = {"Last 6 hours": 6, "Last 24 hours": 24, "Last 7 days": 168, "All collected": None}
    label = c2.selectbox("Time window", list(windows), index=1, key="x_window")
    v = raw[raw["city"].isin(cities)]
    if windows[label]:
        v = v[v["time"] >= v["time"].max() - pd.Timedelta(hours=windows[label])]
    v = v.sort_values("time", ascending=False)
    st.caption(f"{len(v):,} readings selected.")
    agg = load_agg()
    summary = summary_table(state["ranking"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    empty = v.empty
    b1, b2, b3 = st.columns(3)
    b1.download_button("Download CSV", naive(v).to_csv(index=False).encode() if not empty else b"",
                       f"station_atlas_{stamp}.csv", "text/csv", disabled=empty, key="x_csv")
    b2.download_button("Download Excel", make_xlsx(v, agg, summary) if not empty else b"",
                       f"station_atlas_{stamp}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       disabled=empty, key="x_xlsx")
    b3.download_button("Download PDF report", make_pdf(v, summary, label) if not empty else b"",
                       f"station_atlas_{stamp}.pdf", "application/pdf", disabled=empty, key="x_pdf")


# ---------------------------------------------------------------------- page --
try:
    get_db()
except Exception as exc:  # noqa: BLE001
    st.markdown(notice(
        "MongoDB is not reachable",
        f"Could not connect to <code>{escape(MONGO_URI)}</code>. Start it with "
        f"<code>docker compose up -d</code> and reload. Details: {escape(str(exc)[:160])}"),
        unsafe_allow_html=True)
    st.stop()

masthead()
tab_atlas, tab_agg, tab_hist, tab_export = st.tabs(["Atlas", "Aggregates", "History", "Export"])
with tab_atlas:
    atlas_view()
with tab_agg:
    aggregates_view()
with tab_hist:
    history_view()
with tab_export:
    export_view()
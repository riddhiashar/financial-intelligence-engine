"""
app.py — Financial Intelligence & Alerting Engine
Week 7: Enhanced Streamlit Dashboard

Changes in this version:
- Fixed API latency measurement using cached timing
- RAG summary and sources clearly separated with distinct sections
"""

import streamlit as st
import requests
import re
import json
import csv
import io
import time
import os
from datetime import datetime


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
APP_TITLE     = "Financial Intelligence & Alerting Engine"
APP_ICON      = "📈"
SWAGGER_URL   = "http://localhost:8000/docs"

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title            = APP_TITLE,
    page_icon             = APP_ICON,
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# ─── Session State Initialization ─────────────────────────
if "dark_mode"       not in st.session_state:
    st.session_state["dark_mode"]       = False
if "search_history"  not in st.session_state:
    st.session_state["search_history"]  = []
if "api_latency"     not in st.session_state:
    st.session_state["api_latency"]     = None
if "api_healthy"     not in st.session_state:
    st.session_state["api_healthy"]     = None
if "api_health_data" not in st.session_state:
    st.session_state["api_health_data"] = None
if "api_last_check"  not in st.session_state:
    st.session_state["api_last_check"]  = 0

DARK   = st.session_state["dark_mode"]
BG     = "#0e1117" if DARK else "#ffffff"
CARD   = "#1e2130" if DARK else "#f8f9fa"
TEXT   = "#fafafa"  if DARK else "#1a1a2e"
MUTED  = "#aaaaaa"  if DARK else "#666666"
BORDER = "#333355"  if DARK else "#dee2e6"
ACCENT = "#4a90e2"  if DARK else "#1f4e79"
SUCCESS= "#2ecc71"  if DARK else "#28a745"
DANGER = "#e74c3c"  if DARK else "#dc3545"
WARN   = "#f39c12"  if DARK else "#ffc107"

# ─── CSS ──────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {BG};
        color: {TEXT};
    }}
    .main-header {{
        font-size: 2.4rem;
        font-weight: 700;
        color: {ACCENT};
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }}
    .sub-header {{
        font-size: 1rem;
        color: {MUTED};
        margin-bottom: 2rem;
        font-weight: 400;
    }}
    .section-header {{
        font-size: 1.3rem;
        font-weight: 600;
        color: {TEXT};
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 6px;
        border-bottom: 2px solid {ACCENT};
        display: inline-block;
    }}
    .card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        transition: box-shadow 0.2s ease;
    }}
    .card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.12); }}
    .result-card {{
        background: {CARD};
        border-left: 4px solid {ACCENT};
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: transform 0.15s ease;
    }}
    .result-card:hover {{ transform: translateX(2px); }}
    .result-card-ticker {{
        background: {CARD};
        border-left: 4px solid #0d47a1;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 10px;
    }}
    .result-card-boosted {{
        background: {CARD};
        border-left: 4px solid #6a1b9a;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 10px;
    }}
    .badge-bullish {{
        background: #d4edda; color: #155724;
        padding: 3px 10px; border-radius: 20px;
        font-weight: 600; font-size: 0.8rem;
        display: inline-block;
    }}
    .badge-bearish {{
        background: #f8d7da; color: #721c24;
        padding: 3px 10px; border-radius: 20px;
        font-weight: 600; font-size: 0.8rem;
        display: inline-block;
    }}
    .badge-neutral {{
        background: #fff3cd; color: #856404;
        padding: 3px 10px; border-radius: 20px;
        font-weight: 600; font-size: 0.8rem;
        display: inline-block;
    }}
    .badge-ticker {{
        background: #e3f2fd; color: #0d47a1;
        padding: 2px 8px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600;
        display: inline-block;
    }}
    .badge-hybrid {{
        background: #f3e5f5; color: #6a1b9a;
        padding: 2px 8px; border-radius: 12px;
        font-size: 0.75rem; display: inline-block;
    }}
    .badge-semantic {{
        background: #e8f5e9; color: #1b5e20;
        padding: 2px 8px; border-radius: 12px;
        font-size: 0.75rem; display: inline-block;
    }}
    .kw-highlight {{
        background: #fff176; color: #000;
        border-radius: 3px; padding: 0 2px;
        font-weight: 500;
    }}
    .status-ok    {{ color: {SUCCESS}; font-weight: 600; }}
    .status-error {{ color: {DANGER};  font-weight: 600; }}
    .status-warn  {{ color: {WARN};    font-weight: 600; }}

    /* ── RAG Summary box ── */
    .summary-box {{
        background: linear-gradient(135deg, #e8f0fe, #dce8f7);
        border: 2px solid #90b8e8;
        border-radius: 12px;
        padding: 28px 32px;
        font-size: 1.12rem;
        line-height: 1.9;
        color: #1a1a2e;
        margin: 8px 0 16px 0;
        position: relative;
    }}
    .summary-label {{
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #1f4e79;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}
    .sources-divider {{
        border: none;
        border-top: 2px dashed {BORDER};
        margin: 20px 0;
    }}
    .sources-label {{
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: {MUTED};
        text-transform: uppercase;
        margin-bottom: 12px;
    }}

    .pipeline-step {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 500;
        color: {TEXT};
    }}
    .pipeline-arrow {{
        text-align: center;
        font-size: 1.5rem;
        color: {MUTED};
        padding: 4px 0;
    }}
    .history-item {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }}
    .example-chip {{
        display: inline-block;
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        color: {ACCENT};
        margin: 3px;
    }}
    .tech-badge {{
        display: inline-block;
        background: {ACCENT};
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin: 3px;
        font-weight: 500;
    }}
    .error-box {{
        background: #fff5f5;
        border: 1px solid #fed7d7;
        border-radius: 8px;
        padding: 16px;
        color: #c53030;
    }}
    .conf-bar-bg {{
        background: {BORDER};
        border-radius: 6px;
        height: 10px;
        width: 100%;
        margin-top: 4px;
    }}
    .conf-bar-fill {{
        height: 10px;
        border-radius: 6px;
        transition: width 0.5s ease;
    }}
    #MainMenu  {{ visibility: hidden; }}
    footer     {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ─── Cached API Health Check ──────────────────────────────

# persistent session for accurate latency measurement
_health_session = requests.Session()

def get_api_health(force_refresh: bool = False):
    """
    Check API health with 30-second caching.
    Uses persistent session to eliminate TCP overhead.
    """
    now        = time.time()
    last_check = st.session_state["api_last_check"]
    cache_valid = (now - last_check) < 30

    if cache_valid and not force_refresh:
        return (
            st.session_state["api_healthy"],
            st.session_state["api_health_data"],
            st.session_state["api_latency"]
        )

    try:
        # first call establishes TCP connection — discard
        try:
            _health_session.get(
                f"{API_BASE_URL}/health", timeout=3
            )
        except Exception:
            pass

        # second call reuses connection — this is the real latency
        start = time.perf_counter()
        r = _health_session.get(
            f"{API_BASE_URL}/health", timeout=5
        )
        latency = (time.perf_counter() - start) * 1000

        if r.status_code == 200:
            st.session_state["api_healthy"]     = True
            st.session_state["api_health_data"] = r.json()
            st.session_state["api_latency"]     = round(latency, 1)
        else:
            st.session_state["api_healthy"]     = False
            st.session_state["api_health_data"] = None
            st.session_state["api_latency"]     = None

    except Exception:
        st.session_state["api_healthy"]     = False
        st.session_state["api_health_data"] = None
        st.session_state["api_latency"]     = None

    st.session_state["api_last_check"] = now
    return (
        st.session_state["api_healthy"],
        st.session_state["api_health_data"],
        st.session_state["api_latency"]
    )


# ─── Helper Functions ─────────────────────────────────────

def clean_headline(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_tickers(query: str) -> list:
    return list(set(re.findall(r"\$[A-Z]{1,6}\b", query)))


def highlight_keywords(text: str, keywords: list) -> str:
    if not keywords:
        return text
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(
            f'<span class="kw-highlight">{kw}</span>', text
        )
    return text


def get_sentiment_badge(sentiment: str) -> str:
    icons = {"Bullish": "📈", "Bearish": "📉", "Neutral": "➡️"}
    cls   = {"Bullish": "bullish", "Bearish": "bearish", "Neutral": "neutral"}
    icon  = icons.get(sentiment, "")
    c     = cls.get(sentiment, "neutral")
    return f'<span class="badge-{c}">{icon} {sentiment}</span>'


def get_sentiment_color(sentiment: str) -> str:
    return {
        "Bullish": "#28a745",
        "Bearish": "#dc3545",
        "Neutral": "#ffc107",
    }.get(sentiment, "#666")


def format_similarity(score: float) -> str:
    return f"{score * 100:.1f}%"


def add_to_history(query: str, page: str, result_count: int):
    entry = {
        "query":     query,
        "page":      page,
        "results":   result_count,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    history = st.session_state.get("search_history", [])
    if not history or history[0]["query"] != query:
        st.session_state["search_history"] = [entry] + history[:9]


def call_analyze(text, include_similar=True, similar_limit=5):
    try:
        r = requests.post(
            f"{API_BASE_URL}/analyze",
            json={
                "text":            text,
                "include_similar": include_similar,
                "similar_limit":   similar_limit
            },
            timeout=60
        )
        return (True, r.json()) if r.status_code == 200 \
            else (False, r.json())
    except requests.exceptions.ConnectionError:
        return False, {"detail": "Cannot connect to API. Make sure FastAPI is running on port 8000."}
    except requests.exceptions.Timeout:
        return False, {"detail": "Request timed out. The model may be loading — try again."}
    except Exception as e:
        return False, {"detail": str(e)}


def call_search(query, limit=5, sentiment=None, min_similarity=0.3):
    try:
        params = {"query": query, "limit": limit, "min_similarity": min_similarity}
        if sentiment:
            params["sentiment"] = sentiment
        r = requests.get(f"{API_BASE_URL}/search", params=params, timeout=30)
        return (True, r.json()) if r.status_code == 200 else (False, r.json())
    except requests.exceptions.ConnectionError:
        return False, {"detail": "Cannot connect to API."}
    except Exception as e:
        return False, {"detail": str(e)}


def call_hybrid_search(query, limit=5, sentiment=None, min_similarity=0.3):
    try:
        params = {"query": query, "limit": limit, "min_similarity": min_similarity}
        if sentiment:
            params["sentiment"] = sentiment
        r = requests.get(f"{API_BASE_URL}/hybrid-search", params=params, timeout=30)
        return (True, r.json()) if r.status_code == 200 else (False, r.json())
    except requests.exceptions.ConnectionError:
        return False, {"detail": "Cannot connect to API."}
    except Exception as e:
        return False, {"detail": str(e)}


def call_summarize(query, top_k=5, sentiment_filter=None):
    try:
        payload = {"query": query, "top_k": top_k}
        if sentiment_filter:
            payload["sentiment_filter"] = sentiment_filter
        r = requests.post(
            f"{API_BASE_URL}/summarize", json=payload, timeout=120
        )
        return (True, r.json()) if r.status_code == 200 else (False, r.json())
    except requests.exceptions.ConnectionError:
        return False, {"detail": "Cannot connect to API. Make sure FastAPI is running."}
    except requests.exceptions.Timeout:
        return False, {"detail": "LLM timed out. Ollama may not be running. Check that llama3 is pulled."}
    except Exception as e:
        return False, {"detail": str(e)}


def call_stats():
    try:
        r = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        return (True, r.json()) if r.status_code == 200 else (False, r.json())
    except Exception as e:
        return False, {"detail": str(e)}


def show_error(msg: str, suggestion: str = ""):
    st.markdown(
        f'<div class="error-box">❌ <b>Error:</b> {msg}'
        + (f"<br>💡 <b>Suggestion:</b> {suggestion}" if suggestion else "")
        + "</div>",
        unsafe_allow_html=True
    )


def render_confidence_bar(label: str, pct_str: str, color: str):
    val = float(pct_str.replace("%", ""))
    st.markdown(
        f"<div style='margin-bottom:8px'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:0.85rem;color:{MUTED}'>"
        f"<span>{label}</span><span>{pct_str}</span></div>"
        f"<div class='conf-bar-bg'>"
        f"<div class='conf-bar-fill' "
        f"style='width:{val}%;background:{color}'></div>"
        f"</div></div>",
        unsafe_allow_html=True
    )


def render_headline_card(rank, sentiment, similarity, headline,
                          ticker=None, retrieval_type="semantic",
                          hybrid_score=None, keyword_boost=None,
                          keywords_to_highlight=None):
    badge      = get_sentiment_badge(sentiment)
    sim_pct    = format_similarity(similarity)
    clean_text = clean_headline(headline)

    if keywords_to_highlight:
        clean_text = highlight_keywords(clean_text, keywords_to_highlight)

    ticker_str = (
        f'<span class="badge-ticker">🎯 {ticker}</span> '
        if ticker and ticker != "None" else ""
    )

    if retrieval_type == "ticker_match":
        type_badge = '<span class="badge-hybrid">📌 Ticker Match</span> '
        card_class = "result-card-ticker"
    elif keyword_boost and keyword_boost > 0:
        type_badge = (
            f'<span class="badge-hybrid">⚡ +{keyword_boost:.3f}</span> '
        )
        card_class = "result-card-boosted"
    else:
        type_badge = '<span class="badge-semantic">🔎 Semantic</span> '
        card_class = "result-card"

    score_str = (
        f"Hybrid: <b>{format_similarity(hybrid_score)}</b>"
        if hybrid_score else f"Similarity: <b>{sim_pct}</b>"
    )

    st.markdown(
        f'<div class="{card_class}">'
        f'<div style="margin-bottom:6px">'
        f'<b>#{rank}</b> {badge} {ticker_str}{type_badge}'
        f'<span style="float:right;font-size:0.82rem;color:{MUTED}">'
        f'{score_str}</span></div>'
        f'<div style="font-size:0.92rem;color:{TEXT};line-height:1.5">'
        f'{clean_text}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ─── Download Helpers ─────────────────────────────────────

def results_to_csv(results: list) -> str:
    buf = io.StringIO()
    if not results:
        return ""
    writer = csv.DictWriter(
        buf,
        fieldnames=["rank", "sentiment", "similarity", "headline", "ticker"]
    )
    writer.writeheader()
    for r in results:
        writer.writerow({
            "rank":       r.get("rank", ""),
            "sentiment":  r.get("sentiment", ""),
            "similarity": r.get("similarity", r.get("hybrid_score", "")),
            "headline":   clean_headline(r.get("headline", "")),
            "ticker":     r.get("ticker", "")
        })
    return buf.getvalue()


def results_to_json(results: list) -> str:
    cleaned = []
    for r in results:
        cleaned.append({
            "rank":       r.get("rank"),
            "sentiment":  r.get("sentiment"),
            "similarity": r.get("similarity", r.get("hybrid_score")),
            "headline":   clean_headline(r.get("headline", "")),
            "ticker":     r.get("ticker", "")
        })
    return json.dumps(cleaned, indent=2)


def results_to_txt(results: list, query: str) -> str:
    lines = [f"Query: {query}", "=" * 50, ""]
    for r in results:
        lines.append(
            f"#{r.get('rank')} [{r.get('sentiment')}] "
            f"(sim={r.get('similarity', r.get('hybrid_score', 0)):.3f})"
        )
        lines.append(clean_headline(r.get("headline", "")))
        lines.append("")
    return "\n".join(lines)


def render_download_buttons(results: list, query: str, prefix: str):
    if not results:
        return
    st.markdown("**📥 Download Results**")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "⬇️ CSV",
            data      = results_to_csv(results),
            file_name = f"{prefix}_results.csv",
            mime      = "text/csv",
            key       = f"{prefix}_csv_{int(time.time())}"
        )
    with d2:
        st.download_button(
            "⬇️ JSON",
            data      = results_to_json(results),
            file_name = f"{prefix}_results.json",
            mime      = "application/json",
            key       = f"{prefix}_json_{int(time.time())}"
        )
    with d3:
        st.download_button(
            "⬇️ TXT",
            data      = results_to_txt(results, query),
            file_name = f"{prefix}_results.txt",
            mime      = "text/plain",
            key       = f"{prefix}_txt_{int(time.time())}"
        )


# ─── Sidebar ──────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        col_logo, col_theme = st.columns([3, 1])
        with col_logo:
            st.markdown(f"## {APP_ICON} FinIntel")
        with col_theme:
            if st.button(
                "🌙" if not DARK else "☀️",
                key="theme_toggle",
                help="Toggle dark/light mode"
            ):
                st.session_state["dark_mode"] = not DARK
                st.rerun()

        st.markdown("---")

        page = st.radio(
            "Navigate",
            options=[
                "🏠 Home",
                "🔍 Sentiment Analysis",
                "🔎 Semantic Search",
                "⚡ Hybrid Search",
                "🤖 RAG Summary",
                "📊 System Status",
                "ℹ️ About"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # ── Cached latency display ──
        is_healthy, _, latency = get_api_health()

        if is_healthy and latency is not None:
            st.markdown(
                f'<p class="status-ok">● API Online '
                f'<span style="font-weight:400;font-size:0.8rem">'
                f'({latency}ms)</span></p>',
                unsafe_allow_html=True
            )
        elif is_healthy:
            st.markdown(
                '<p class="status-ok">● API Online</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p class="status-error">● API Offline</p>',
                unsafe_allow_html=True
            )
            st.caption("Start: `uvicorn main:app --reload`")

        # manual refresh button for latency
        if st.button("↻ Refresh status", key="refresh_health"):
            get_api_health(force_refresh=True)
            st.rerun()

        st.markdown("---")
        st.markdown("**🤖 Models**")
        st.caption("Sentiment: DistilBERT")
        st.caption("Embeddings: all-MiniLM-L6-v2")
        st.caption("LLM: Llama3 (Ollama)")
        st.caption("Vector DB: Qdrant")

        st.markdown("---")
        st.markdown(f'🔗 [API Docs ↗]({SWAGGER_URL})')

        history = st.session_state.get("search_history", [])
        if history:
            st.markdown("---")
            st.markdown("**🕐 Recent Searches**")
            for h in history[:5]:
                icon = {
                    "🔍 Sentiment Analysis": "🔍",
                    "🔎 Semantic Search":    "🔎",
                    "⚡ Hybrid Search":      "⚡",
                    "🤖 RAG Summary":        "🤖"
                }.get(h["page"], "🔎")
                st.markdown(
                    f'<div class="history-item">'
                    f'{icon} {h["query"][:28]}'
                    f'{"..." if len(h["query"]) > 28 else ""}'
                    f'<br><span style="color:{MUTED};font-size:0.75rem">'
                    f'{h["page"].split()[-1]} · {h["timestamp"]} '
                    f'· {h["results"]} results</span></div>',
                    unsafe_allow_html=True
                )
            if st.button("Clear History", key="clear_hist"):
                st.session_state["search_history"] = []
                st.rerun()

        st.markdown("---")
        st.caption(f"v1.0.0 | {datetime.now().strftime('%d %b %Y')}")

    return page


# ─── Home ─────────────────────────────────────────────────

def render_home():
    st.markdown(
        '<p class="main-header">📈 Financial Intelligence Engine</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Production-grade financial news '
        'sentiment analysis · Semantic search · RAG summarization</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    features = [
        ("🔍", "Sentiment",      "DistilBERT classifies headlines as Bearish, Bullish or Neutral."),
        ("🔎", "Semantic Search","Find related headlines by meaning across 9,380 records."),
        ("⚡", "Hybrid Search",  "Semantic + keyword boost + ticker-aware retrieval."),
        ("🤖", "RAG Summary",    "Llama3 generates 2-sentence market briefings from context."),
        ("📊", "Status",         "Live monitoring of all pipeline components."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4, col5], features):
        with col:
            st.markdown(
                f'<div class="card" style="height:180px">'
                f'<div style="font-size:1.8rem">{icon}</div>'
                f'<div style="font-weight:600;margin:6px 0 4px">{title}</div>'
                f'<div style="font-size:0.8rem;color:{MUTED}">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total",   "9,380")
        m2.metric("Bearish", "1,428")
        m3.metric("Bullish", "1,904")
        m4.metric("Neutral", "6,048")

    with right:
        st.markdown('<p class="section-header">Sentiment Distribution</p>', unsafe_allow_html=True)
        try:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Pie(
                labels   = ["Bearish", "Bullish", "Neutral"],
                values   = [1428, 1904, 6048],
                hole     = 0.55,
                marker   = dict(colors=["#dc3545", "#28a745", "#ffc107"]),
                textinfo = "label+percent",
                hovertemplate = "%{label}: %{value}<extra></extra>"
            )])
            fig.update_layout(
                showlegend=False, height=200,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT)
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install plotly: `pip install plotly`")

    st.markdown("---")
    st.markdown('<p class="section-header">💡 Example Queries</p>', unsafe_allow_html=True)
    examples = {
        "🔍 Sentiment": [
            "$AAPL beats earnings estimates revenue up 12%",
            "$TSLA misses revenue target stock falls",
            "Federal Reserve holds interest rates steady",
        ],
        "🔎 / ⚡ Search": [
            "$AAPL iPhone revenue growth",
            "recession market crash oil prices",
            "analyst raises price target strong buy",
        ],
        "🤖 RAG": [
            "Tesla earnings revenue performance",
            "Federal Reserve interest rate inflation",
            "technology stocks earnings growth",
        ]
    }
    for category, queries in examples.items():
        st.markdown(f"**{category}**")
        st.markdown(
            " ".join(f'<span class="example-chip">{q}</span>' for q in queries),
            unsafe_allow_html=True
        )
        st.markdown("")

    st.markdown("---")
    st.markdown('<p class="section-header">🔄 Pipeline Overview</p>', unsafe_allow_html=True)
    steps = [
        ("📥", "Input",      "Raw headline"),
        ("🧹", "Clean",      "URL removal"),
        ("🧠", "DistilBERT", "Sentiment"),
        ("🔢", "Embed",      "384-dim vector"),
        ("🗄️","Qdrant",     "Vector search"),
        ("⚡", "FastAPI",    "REST layer"),
        ("🤖", "Llama3",     "RAG summary"),
        ("🖥️","Streamlit",  "Dashboard"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="pipeline-step">'
                f'<div style="font-size:1.4rem">{icon}</div>'
                f'<div style="font-weight:600;font-size:0.82rem;margin:4px 0 2px">{title}</div>'
                f'<div style="font-size:0.72rem;color:{MUTED}">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ─── Sentiment Analysis ───────────────────────────────────

def render_sentiment():
    st.markdown("## 🔍 Sentiment Analysis")
    st.write("Analyze a financial headline using fine-tuned DistilBERT.")

    text = st.text_area(
        "Financial Headline",
        placeholder="e.g. $AAPL beats earnings estimates revenue up 12%",
        height=100, key="sentiment_text"
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        include_similar = st.checkbox(
            "Show similar headlines", value=True,
            key="sentiment_include_similar"
        )
    with col2:
        similar_limit = st.slider(
            "Similar headlines count",
            min_value=1, max_value=10, value=5,
            key="sentiment_limit"
        )

    if st.button("🔍 Analyze Sentiment", type="primary",
                 key="sentiment_btn"):
        if not text.strip():
            show_error("No headline entered.",
                       "Type or paste a financial headline above.")
            return
        bar = st.progress(0, text="🧹 Cleaning text...")
        time.sleep(0.2)
        bar.progress(30, text="🧠 Running DistilBERT model...")
        success, data = call_analyze(text, include_similar, similar_limit)
        bar.progress(80, text="🔎 Fetching similar headlines...")
        time.sleep(0.2)
        bar.progress(100, text="✅ Done!")
        time.sleep(0.3)
        bar.empty()

        if not success:
            show_error(data.get("detail", "Unknown error"),
                       "Check that FastAPI is running on port 8000.")
            return
        st.session_state["sentiment_result"] = data
        add_to_history(text[:50], "🔍 Sentiment Analysis", 1)

    if "sentiment_result" in st.session_state:
        data      = st.session_state["sentiment_result"]
        sentiment = data["sentiment"]
        color     = get_sentiment_color(sentiment)

        st.markdown("---")
        st.markdown(
            f'<div class="card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div>'
            f'<div style="font-size:0.85rem;color:{MUTED};margin-bottom:4px">Sentiment</div>'
            f'<div style="font-size:1.6rem">{get_sentiment_badge(sentiment)}</div>'
            f'</div>'
            f'<div style="text-align:center">'
            f'<div style="font-size:0.85rem;color:{MUTED}">Confidence</div>'
            f'<div style="font-size:2rem;font-weight:700;color:{color}">{data["confidence"]}</div>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<div style="font-size:0.85rem;color:{MUTED}">Processing</div>'
            f'<div style="font-size:1.2rem;font-weight:600">{data["processing_time_ms"]:.0f}ms</div>'
            f'</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

        st.markdown('<p class="section-header">Score Breakdown</p>', unsafe_allow_html=True)
        scores = data["scores"]
        render_confidence_bar("📉 Bearish", scores.get("Bearish", "0%"), "#dc3545")
        render_confidence_bar("📈 Bullish", scores.get("Bullish", "0%"), "#28a745")
        render_confidence_bar("➡️ Neutral", scores.get("Neutral", "0%"), "#ffc107")

        with st.expander("🧹 View cleaned text"):
            st.code(data["cleaned_text"])

        if include_similar and data.get("similar_headlines"):
            st.markdown('<p class="section-header">Similar Historical Headlines</p>',
                        unsafe_allow_html=True)
            for h in data["similar_headlines"]:
                render_headline_card(
                    h["rank"], h["sentiment"],
                    h["similarity"], h["headline"],
                    h.get("ticker")
                )
            render_download_buttons(
                data["similar_headlines"], text[:30], "sentiment_similar"
            )

        with st.expander("📋 Response metadata"):
            st.json({
                "request_id": data["request_id"],
                "timestamp":  data["timestamp"],
                "label":      data["label"]
            })


# ─── Semantic Search ──────────────────────────────────────

def render_search():
    st.markdown("## 🔎 Semantic Search")
    st.write("Search 9,380 historical headlines by meaning.")
    st.info(
        "💡 For ticker-specific queries like $AAPL — try ⚡ Hybrid Search.",
        icon="💡"
    )

    query = st.text_input(
        "Search Query",
        placeholder="e.g. Tesla beats earnings expectations",
        key="search_query"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        limit = st.slider("Results", 1, 20, 5, key="search_limit")
    with col2:
        sentiment = st.selectbox(
            "Sentiment Filter",
            ["All", "Bullish", "Bearish", "Neutral"],
            key="search_sentiment"
        )
    with col3:
        min_sim = st.slider(
            "Min Similarity", 0.0, 1.0, 0.3, 0.05,
            key="search_min_sim"
        )

    if st.button("🔎 Search", type="primary", key="search_btn"):
        if not query.strip():
            show_error("No query entered.",
                       "Type a financial topic above.")
            return
        sentiment_filter = None if sentiment == "All" else sentiment
        bar = st.progress(0, text="🔢 Encoding query...")
        time.sleep(0.2)
        bar.progress(50, text="🗄️ Searching Qdrant...")
        success, data = call_search(query, limit, sentiment_filter, min_sim)
        bar.progress(100, text="✅ Done!")
        time.sleep(0.2)
        bar.empty()

        if not success:
            show_error(data.get("detail", "Search failed."),
                       "Make sure FastAPI is running.")
            return
        st.session_state["search_result"] = {
            "data": data, "query": query, "sentiment": sentiment_filter
        }
        add_to_history(query, "🔎 Semantic Search", len(data))

    if "search_result" in st.session_state:
        saved    = st.session_state["search_result"]
        data     = saved["data"]
        q        = saved["query"]
        sf       = saved["sentiment"]
        keywords = re.findall(r"\b[a-zA-Z]{4,}\b", q.lower())

        st.markdown("---")
        if not data:
            st.warning("No results above the similarity threshold. "
                       "Try lowering Min Similarity.")
            return

        st.markdown(
            f'<p class="section-header">Found {len(data)} results for: <em>{q}</em></p>',
            unsafe_allow_html=True
        )
        if sf:
            st.markdown(f"Filtered by: {get_sentiment_badge(sf)}",
                        unsafe_allow_html=True)

        try:
            import plotly.graph_objects as go
            sentiments = [r["sentiment"] for r in data]
            counts = {
                "Bullish": sentiments.count("Bullish"),
                "Bearish": sentiments.count("Bearish"),
                "Neutral": sentiments.count("Neutral"),
            }
            fig = go.Figure(go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                marker_color=["#28a745", "#dc3545", "#ffc107"],
                text=list(counts.values()),
                textposition="auto"
            ))
            fig.update_layout(
                height=180, margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT),
                yaxis=dict(showgrid=False, showticklabels=False),
                xaxis=dict(showgrid=False),
                title_text="Sentiment in Results"
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        for result in data:
            render_headline_card(
                result["rank"], result["sentiment"],
                result["similarity"], result["headline"],
                result.get("ticker"),
                keywords_to_highlight=keywords
            )
        render_download_buttons(data, q, "semantic_search")


# ─── Hybrid Search ────────────────────────────────────────

def render_hybrid_search():
    st.markdown("## ⚡ Hybrid Search")
    st.write("Semantic similarity + keyword boosting + ticker-aware retrieval.")

    info1, info2 = st.columns(2)
    with info1:
        st.success("📌 **Ticker-Aware:** $AAPL, $TSLA etc prioritize matching headlines.")
    with info2:
        st.info("⚡ **Keyword Boost:** Query words give extra relevance signal.")

    query = st.text_input(
        "Search Query",
        placeholder="e.g. $AAPL iPhone revenue",
        key="hybrid_query"
    )
    if query:
        tickers = extract_tickers(query)
        if tickers:
            st.markdown(
                "**Detected tickers:** "
                + " ".join(f'<span class="badge-ticker">{t}</span>' for t in tickers)
                + " — these will be prioritized.",
                unsafe_allow_html=True
            )
        st.caption("💡 Use $TICKER format to trigger ticker-aware retrieval.")

    col1, col2, col3 = st.columns(3)
    with col1:
        limit = st.slider("Results", 1, 20, 5, key="hybrid_limit")
    with col2:
        sentiment = st.selectbox(
            "Sentiment Filter",
            ["All", "Bullish", "Bearish", "Neutral"],
            key="hybrid_sentiment"
        )
    with col3:
        min_sim = st.slider(
            "Min Similarity", 0.0, 1.0, 0.3, 0.05,
            key="hybrid_min_sim"
        )

    if st.button("⚡ Hybrid Search", type="primary", key="hybrid_btn"):
        if not query.strip():
            show_error("No query entered.", "Type a financial topic.")
            return
        sentiment_filter = None if sentiment == "All" else sentiment

        bar = st.progress(0, text="🎯 Extracting tickers...")
        time.sleep(0.15)
        bar.progress(25, text="🔢 Encoding query vector...")
        time.sleep(0.15)
        bar.progress(50, text="🗄️ Ticker-aware Qdrant search...")
        success, data = call_hybrid_search(query, limit, sentiment_filter, min_sim)
        bar.progress(85, text="⚡ Applying keyword boost...")
        time.sleep(0.15)
        bar.progress(100, text="✅ Done!")
        time.sleep(0.2)
        bar.empty()

        if not success:
            show_error(data.get("detail", "Search failed."),
                       "Make sure FastAPI is running.")
            return
        st.session_state["hybrid_result"] = {
            "data": data, "query": query, "sentiment": sentiment_filter
        }
        add_to_history(query, "⚡ Hybrid Search", len(data))

    if "hybrid_result" in st.session_state:
        saved    = st.session_state["hybrid_result"]
        data     = saved["data"]
        q        = saved["query"]
        sf       = saved["sentiment"]
        keywords = re.findall(r"\b[a-zA-Z]{4,}\b", q.lower())

        st.markdown("---")
        if not data:
            st.warning("No results above threshold. Try lowering Min Similarity.")
            return

        ticker_count   = sum(1 for r in data if r.get("retrieval_type") == "ticker_match")
        semantic_count = len(data) - ticker_count

        st.markdown(
            f'<p class="section-header">Found {len(data)} results for: <em>{q}</em></p>',
            unsafe_allow_html=True
        )
        if sf:
            st.markdown(f"Filtered by: {get_sentiment_badge(sf)}",
                        unsafe_allow_html=True)
        if ticker_count > 0:
            st.markdown(
                f"📌 **{ticker_count} ticker matches** + 🔎 **{semantic_count} semantic matches**"
            )

        try:
            import plotly.graph_objects as go
            labels   = [f"#{r['rank']}" for r in data]
            sim_vals = [r["similarity"] for r in data]
            boost    = [r.get("keyword_boost", 0) for r in data]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Similarity", x=labels, y=sim_vals,
                marker_color="#4a90e2",
                text=[f"{v:.3f}" for v in sim_vals],
                textposition="inside"
            ))
            fig.add_trace(go.Bar(
                name="Keyword Boost", x=labels, y=boost,
                marker_color="#9c27b0"
            ))
            fig.update_layout(
                barmode="stack", height=200,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT),
                legend=dict(orientation="h", y=1.1, font=dict(size=10)),
                yaxis=dict(showgrid=False),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        for result in data:
            render_headline_card(
                rank              = result["rank"],
                sentiment         = result["sentiment"],
                similarity        = result["similarity"],
                headline          = result["headline"],
                ticker            = result.get("ticker"),
                retrieval_type    = result.get("retrieval_type", "semantic"),
                hybrid_score      = result.get("hybrid_score"),
                keyword_boost     = result.get("keyword_boost"),
                keywords_to_highlight = keywords
            )

        with st.expander("📊 Scoring Details"):
            for r in data:
                st.markdown(
                    f"**#{r['rank']}** — "
                    f"Similarity: `{r['similarity']:.4f}` + "
                    f"Boost: `{r.get('keyword_boost', 0):.4f}` = "
                    f"Hybrid: `{r.get('hybrid_score', r['similarity']):.4f}` "
                    f"[{r.get('retrieval_type', 'semantic')}]"
                )

        render_download_buttons(data, q, "hybrid_search")


# ─── RAG Summary ──────────────────────────────────────────

def render_rag():
    st.markdown("## 🤖 RAG Market Intelligence Summary")
    st.write(
        "Retrieves the most relevant headlines from Qdrant, "
        "passes them as context to Llama3, "
        "and generates a 2-sentence market briefing."
    )
    st.info(
        "⏳ LLM generation takes 10-40 seconds on CPU. "
        "Please be patient after clicking Generate.",
        icon="ℹ️"
    )

    with st.expander("🔄 How RAG works"):
        p1, arr1, p2, arr2, p3 = st.columns([3, 1, 3, 1, 3])
        with p1:
            st.markdown(
                f'<div class="pipeline-step">'
                f'<div style="font-size:1.4rem">🗄️</div>'
                f'<b>Retrieve</b><br>'
                f'<span style="font-size:0.78rem;color:{MUTED}">'
                f'Top-k similar headlines from Qdrant via hybrid search'
                f'</span></div>',
                unsafe_allow_html=True
            )
        with arr1:
            st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
        with p2:
            st.markdown(
                f'<div class="pipeline-step">'
                f'<div style="font-size:1.4rem">🧠</div>'
                f'<b>Augment + Generate</b><br>'
                f'<span style="font-size:0.78rem;color:{MUTED}">'
                f'Headlines passed as context to Llama3'
                f'</span></div>',
                unsafe_allow_html=True
            )
        with arr2:
            st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
        with p3:
            st.markdown(
                f'<div class="pipeline-step">'
                f'<div style="font-size:1.4rem">📝</div>'
                f'<b>Summary</b><br>'
                f'<span style="font-size:0.78rem;color:{MUTED}">'
                f'2-sentence market intelligence briefing'
                f'</span></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    query = st.text_input(
        "Market Query",
        placeholder="e.g. Tesla earnings revenue performance",
        key="rag_query"
    )
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider(
            "Headlines as context",
            min_value=1, max_value=10, value=5,
            key="rag_top_k"
        )
    with col2:
        sentiment = st.selectbox(
            "Filter context by sentiment",
            ["None (All)", "Bullish", "Bearish", "Neutral"],
            key="rag_sentiment"
        )

    if st.button("🤖 Generate Summary", type="primary", key="rag_btn"):
        if not query.strip():
            show_error("No query entered.",
                       "Type a financial topic to summarize.")
            return
        sentiment_filter = None if sentiment == "None (All)" else sentiment

        bar = st.progress(0,  text="🗄️ Retrieving headlines from Qdrant...")
        time.sleep(0.3)
        bar.progress(30, text="📋 Building context for Llama3...")
        time.sleep(0.2)
        bar.progress(50, text="🤖 Llama3 generating summary — this takes 10-40s...")

        success, data = call_summarize(query, top_k, sentiment_filter)

        bar.progress(95, text="🧹 Cleaning output...")
        time.sleep(0.2)
        bar.progress(100, text="✅ Done!")
        time.sleep(0.3)
        bar.empty()

        if not success:
            detail = data.get("detail", "Unknown error.")
            suggestion = (
                "Make sure Ollama is running and llama3 is pulled: `ollama pull llama3`"
                if "LLM" in detail or "timeout" in detail.lower()
                else "Make sure FastAPI is running on port 8000."
            )
            show_error(detail, suggestion)
            return

        st.session_state["rag_result"] = data
        add_to_history(query, "🤖 RAG Summary",
                       len(data.get("sources", [])))

    # ── RAG Results — clearly separated summary + sources ──
    if "rag_result" in st.session_state:
        data = st.session_state["rag_result"]

        st.markdown("---")

        # ── SECTION 1: Generated Summary ──────────────────
        st.markdown(
            '<p class="section-header">📝 Generated Summary</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="summary-label">Market Intelligence Briefing · Generated by Llama3</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="summary-box">{data["summary"]}</div>',
            unsafe_allow_html=True
        )

        # download summary
        col_dl, _ = st.columns([1, 4])
        with col_dl:
            st.download_button(
                "⬇️ Download Summary",
                data=(
                    f"Query: {data['query']}\n"
                    f"Generated: {data['timestamp']}\n"
                    f"{'='*50}\n\n"
                    f"{data['summary']}"
                ),
                file_name = "market_summary.txt",
                mime      = "text/plain",
                key       = f"summary_dl_{int(time.time())}"
            )

        # ── SECTION 2: Performance Metrics ────────────────
        st.markdown(
            '<p class="section-header">⏱️ Performance</p>',
            unsafe_allow_html=True
        )
        t1, t2, t3 = st.columns(3)
        t1.metric("🗄️ Retrieval",   f"{data['retrieval_time_ms']:.0f}ms")
        t2.metric("🤖 LLM",         f"{data['llm_time_ms']/1000:.1f}s")
        t3.metric("⏱️ Total",        f"{data['processing_time_ms']/1000:.1f}s")

        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=["Retrieval", "LLM Generation"],
                y=[data["retrieval_time_ms"], data["llm_time_ms"]],
                marker_color=["#4a90e2", "#e74c3c"],
                text=[
                    f"{data['retrieval_time_ms']:.0f}ms",
                    f"{data['llm_time_ms']/1000:.1f}s"
                ],
                textposition="auto"
            ))
            fig.update_layout(
                height=160,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT),
                yaxis=dict(showgrid=False, showticklabels=False),
                xaxis=dict(showgrid=False),
                title_text="Time Breakdown (ms)"
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        # ── SECTION 3: Retrieved Sources ──────────────────
        if data.get("sources"):
            st.markdown(
                '<hr class="sources-divider">',
                unsafe_allow_html=True
            )
            st.markdown(
                '<p class="section-header">🗄️ Retrieved Context Headlines</p>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<p class="sources-label">'
                f'{len(data["sources"])} headlines retrieved from Qdrant '
                f'and used as context for Llama3</p>',
                unsafe_allow_html=True
            )

            keywords = re.findall(
                r"\b[a-zA-Z]{4,}\b", data["query"].lower()
            )

            # sentiment breakdown of sources
            if len(data["sources"]) > 1:
                try:
                    import plotly.graph_objects as go
                    src_sentiments = [h["sentiment"] for h in data["sources"]]
                    src_counts = {
                        "Bullish": src_sentiments.count("Bullish"),
                        "Bearish": src_sentiments.count("Bearish"),
                        "Neutral": src_sentiments.count("Neutral"),
                    }
                    fig2 = go.Figure(go.Bar(
                        x     = list(src_counts.keys()),
                        y     = list(src_counts.values()),
                        marker_color = ["#28a745", "#dc3545", "#ffc107"],
                        text  = list(src_counts.values()),
                        textposition = "auto"
                    ))
                    fig2.update_layout(
                        height=150,
                        margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=TEXT),
                        yaxis=dict(showgrid=False, showticklabels=False),
                        xaxis=dict(showgrid=False),
                        title_text="Sentiment of Retrieved Context"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                except ImportError:
                    pass

            for h in data["sources"]:
                render_headline_card(
                    h["rank"], h["sentiment"],
                    h["similarity"], h["headline"],
                    h.get("ticker"),
                    retrieval_type        = h.get("retrieval_type", "semantic"),
                    keywords_to_highlight = keywords
                )

            render_download_buttons(
                data["sources"], data["query"], "rag_sources"
            )

        with st.expander("📋 Response metadata"):
            st.json({
                "request_id":       data["request_id"],
                "timestamp":        data["timestamp"],
                "query":            data["query"],
                "sentiment_filter": data["sentiment_filter"],
                "sources_count":    len(data.get("sources", []))
            })


# ─── System Status ────────────────────────────────────────

def render_status():
    st.markdown("## 📊 System Status")
    st.write("Live monitoring of all pipeline components.")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh"):
            get_api_health(force_refresh=True)
            st.rerun()

    is_healthy, health_data, latency = get_api_health()

    if is_healthy:
        st.success(
            f"✅ FastAPI healthy — {latency}ms API response time "
            f"(benchmarked after TCP warmup)"
        )
    else:
        st.error(
            "❌ FastAPI not reachable. "
            "Run: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`"
        )
        return

    st.markdown("---")
    st.markdown('<p class="section-header">Component Health</p>', unsafe_allow_html=True)
    components = [
        ("🚀", "FastAPI",       True, "Port 8000",        f"{latency}ms"),
        ("🧠", "DistilBERT",    True, "distilbert-base",  "84.4% acc"),
        ("🔢", "all-MiniLM-L6", True, "384 dims",         "41.2s/9380"),
        ("🗄️","Qdrant",         health_data.get("qdrant") == "connected",
                                       "Port 6333",        "9,380 pts"),
        ("🤖", "Llama3/Ollama", True, "4.7GB model",      "10-40s/query"),
        ("⚡", "Hybrid Search", True, "Ticker + keyword", "~25ms"),
    ]
    comp_cols = st.columns(3)
    for i, (icon, name, ok, detail, perf) in enumerate(components):
        with comp_cols[i % 3]:
            status_color = SUCCESS if ok else DANGER
            status_text  = "● Online" if ok else "● Offline"
            st.markdown(
                f'<div class="card" style="margin-bottom:12px">'
                f'<div style="display:flex;justify-content:space-between">'
                f'<span style="font-size:1.4rem">{icon}</span>'
                f'<span style="color:{status_color};font-size:0.8rem;font-weight:600">'
                f'{status_text}</span></div>'
                f'<div style="font-weight:600;margin:6px 0 2px">{name}</div>'
                f'<div style="font-size:0.78rem;color:{MUTED}">{detail}</div>'
                f'<div style="font-size:0.78rem;color:{ACCENT};margin-top:4px">{perf}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown('<p class="section-header">Dataset Statistics</p>', unsafe_allow_html=True)
    success, stats = call_stats()
    if success:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Headlines", f"{stats.get('total_headlines', 0):,}")
        dist = stats.get("sentiment_distribution", {})
        m2.metric("Bearish", f"{dist.get('Bearish', 0):,}")
        m3.metric("Bullish", f"{dist.get('Bullish', 0):,}")
        m4.metric("Neutral", f"{dist.get('Neutral', 0):,}")
        m5.metric("Dimensions", "384")

        try:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Pie(
                labels=["Bearish", "Bullish", "Neutral"],
                values=[dist.get("Bearish", 0),
                        dist.get("Bullish", 0),
                        dist.get("Neutral", 0)],
                hole=0.6,
                marker=dict(colors=["#dc3545", "#28a745", "#ffc107"]),
                textinfo="label+percent"
            )])
            fig.update_layout(
                showlegend=False, height=250,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT)
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass
    else:
        st.warning("Could not load dataset statistics.")

    st.markdown("---")
    st.markdown('<p class="section-header">Performance Benchmarks</p>', unsafe_allow_html=True)
    st.caption("Measured using session-based connection reuse after TCP warmup")
    benchmarks = [
        ("GET /health",                  "~18ms",  "FAST"),
        ("POST /analyze (no similar)",   "~21ms",  "FAST"),
        ("POST /analyze (with similar)", "~72ms",  "FAST"),
        ("POST /batch (3 items)",        "~57ms",  "FAST"),
        ("POST /batch (10 items)",       "~186ms", "GOOD"),
        ("GET /search",                  "~25ms",  "FAST"),
        ("GET /hybrid-search",           "~35ms",  "FAST"),
        ("POST /summarize (LLM)",        "10-40s", "CPU"),
    ]
    b1, b2 = st.columns(2)
    for i, (endpoint, time_val, rating) in enumerate(benchmarks):
        col = b1 if i % 2 == 0 else b2
        with col:
            color = SUCCESS if rating == "FAST" \
                else (ACCENT if rating == "GOOD" else WARN)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:6px 0;border-bottom:1px solid {BORDER};'
                f'font-size:0.85rem">'
                f'<span style="color:{MUTED}">{endpoint}</span>'
                f'<span style="font-weight:600;color:{color}">{time_val}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown(
        f'<p style="font-size:0.8rem;color:{MUTED}">'
        f'Last checked: {datetime.now().strftime("%H:%M:%S")} · '
        f'<a href="{SWAGGER_URL}" target="_blank" style="color:{ACCENT}">'
        f'Open API Docs ↗</a></p>',
        unsafe_allow_html=True
    )


# ─── About ────────────────────────────────────────────────

def render_about():
    st.markdown("## ℹ️ About")

    left, right = st.columns([3, 2])

    with left:
        st.markdown('<p class="section-header">Project Overview</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card">The <b>Financial Intelligence & Alerting Engine</b> '
            f'is a production-grade NLP pipeline built to process financial news '
            f'headlines, classify market sentiment, enable semantic search across '
            f'historical data, and generate concise market intelligence briefings '
            f'using a local LLM.<br><br>'
            f'Built as an 8-week internship project covering the full ML engineering '
            f'stack — from raw data ingestion to containerized deployment.</div>',
            unsafe_allow_html=True
        )

        st.markdown('<p class="section-header">Pipeline Stages</p>', unsafe_allow_html=True)
        stages = [
            ("Week 1", "Data Pipeline & EDA",
             "HuggingFace dataset ingestion, text cleaning, class imbalance analysis"),
            ("Week 2", "Sentiment Classification",
             "DistilBERT fine-tuning, TF-IDF baseline, SMOTE oversampling"),
            ("Week 3", "Embedding Pipeline",
             "all-MiniLM-L6-v2 embeddings for 9,380 headlines"),
            ("Week 4", "Vector Database",
             "Qdrant Docker container, batch upload, cosine similarity search"),
            ("Week 5", "FastAPI Backend",
             "REST API with /analyze, /batch, /search, /hybrid-search, /summarize"),
            ("Week 6", "LLM + RAG",
             "Ollama Llama3 integration, LangChain RAG pipeline"),
            ("Week 7", "Streamlit Dashboard",
             "Interactive UI with charts, hybrid search, downloads, history"),
            ("Week 8", "Docker Containerization",
             "Multi-container deployment with docker-compose"),
        ]
        for week, title, desc in stages:
            st.markdown(
                f'<div style="display:flex;gap:12px;padding:8px 0;'
                f'border-bottom:1px solid {BORDER}">'
                f'<span style="color:{ACCENT};font-weight:600;'
                f'min-width:60px;font-size:0.82rem">{week}</span>'
                f'<div><b style="font-size:0.88rem">{title}</b>'
                f'<br><span style="font-size:0.78rem;color:{MUTED}">'
                f'{desc}</span></div></div>',
                unsafe_allow_html=True
            )

    with right:
        st.markdown('<p class="section-header">Tech Stack</p>', unsafe_allow_html=True)
        tech_groups = {
            "ML / NLP":  ["DistilBERT", "all-MiniLM-L6-v2", "HuggingFace", "Llama3", "LangChain"],
            "Data":      ["Pandas", "NumPy", "Scikit-learn", "SMOTE", "Qdrant"],
            "Backend":   ["FastAPI", "Uvicorn", "Pydantic", "Ollama", "Docker"],
            "Frontend":  ["Streamlit", "Plotly", "Python"]
        }
        for group, techs in tech_groups.items():
            st.markdown(f"**{group}**")
            st.markdown(
                " ".join(f'<span class="tech-badge">{t}</span>' for t in techs),
                unsafe_allow_html=True
            )
            st.markdown("")

        st.markdown('<p class="section-header">Model Performance</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card">'
            f'<b>DistilBERT (Test Set)</b><br>'
            f'<span style="font-size:0.85rem;color:{MUTED}">'
            f'Accuracy: 84.43% · Macro F1: 0.80<br>'
            f'Bearish Recall: 0.839 · Bullish F1: 0.77</span><br><br>'
            f'<b>Baseline TF-IDF</b><br>'
            f'<span style="font-size:0.85rem;color:{MUTED}">'
            f'Accuracy: 76.87% · Macro F1: 0.70</span></div>',
            unsafe_allow_html=True
        )

        st.markdown('<p class="section-header">Dataset</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card">'
            f'<b>zeroshot/twitter-financial-news-sentiment</b><br>'
            f'<span style="font-size:0.85rem;color:{MUTED}">'
            f'Raw: 9,543 rows → Clean: 9,380 rows<br>'
            f'Bearish: 15.2% · Bullish: 20.3% · Neutral: 64.5%</span></div>',
            unsafe_allow_html=True
        )

        st.markdown(f'🔗 **API Docs:** [Swagger UI]({SWAGGER_URL})')


# ─── Main ─────────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "🏠 Home":
        render_home()
    elif page == "🔍 Sentiment Analysis":
        render_sentiment()
    elif page == "🔎 Semantic Search":
        render_search()
    elif page == "⚡ Hybrid Search":
        render_hybrid_search()
    elif page == "🤖 RAG Summary":
        render_rag()
    elif page == "📊 System Status":
        render_status()
    elif page == "ℹ️ About":
        render_about()


if __name__ == "__main__":
    main()
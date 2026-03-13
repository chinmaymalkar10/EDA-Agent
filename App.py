"""
EDA Agent UI - Streamlit Interface
"""

import streamlit as st
import pandas as pd
import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from io import BytesIO
import io
import datetime

# ---------------------------------------------------------------------------
# Configuration — all tunable values in one place
# ---------------------------------------------------------------------------

FILE_SIZE_LIMIT_MB   = 100   # Max upload file size in megabytes
PREVIEW_ROWS         = 8     # Number of rows shown in the dataset preview table

# ---------------------------------------------------------------------------
# Page config
st.set_page_config(
    page_title="EDA Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def render_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #141720;
    --surface2: #1c2030;
    --accent: #7ee8a2;
    --accent2: #38b2ff;
    --accent3: #ff6b9d;
    --text: #e8eaf0;
    --muted: #6b7280;
    --border: #252a3a;
}

* { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}

.sub-header {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 300;
    margin-bottom: 2rem;
    letter-spacing: 0.01em;
}

.step-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    transition: border-color 0.2s;
}

.step-card:hover {
    border-color: var(--accent);
}

.step-number {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.step-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.75rem;
}

.step-status-success {
    display: inline-block;
    background: rgba(126, 232, 162, 0.1);
    color: var(--accent);
    border: 1px solid rgba(126, 232, 162, 0.3);
    border-radius: 20px;
    padding: 0.15rem 0.75rem;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
}

.step-status-error {
    display: inline-block;
    background: rgba(255, 107, 157, 0.1);
    color: var(--accent3);
    border: 1px solid rgba(255, 107, 157, 0.3);
    border-radius: 20px;
    padding: 0.15rem 0.75rem;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
}

.agent-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.75rem;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
    color: var(--muted);
    margin: 0.25rem;
}

.agent-badge.active {
    color: var(--accent2);
    border-color: var(--accent2);
    background: rgba(56, 178, 255, 0.08);
}

.plan-item {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
}

.plan-item:last-child { border-bottom: none; }

.plan-idx {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    background: rgba(126, 232, 162, 0.1);
    border-radius: 50%;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 0.1rem;
}

.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}

.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
}

.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

.code-block {
    background: #0a0c10;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    overflow-x: auto;
    color: #a8b4c8;
    line-height: 1.6;
    max-height: 300px;
    overflow-y: auto;
}

.output-text {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent2);
    white-space: pre-wrap;
    max-height: 250px;
    overflow-y: auto;
}

.error-box {
    background: rgba(255, 107, 157, 0.05);
    border: 1px solid rgba(255, 107, 157, 0.3);
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.8rem;
    color: var(--accent3);
    font-family: 'Space Mono', monospace;
}

.domain-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(56,178,255,0.15), rgba(126,232,162,0.15));
    border: 1px solid rgba(56,178,255,0.4);
    border-radius: 20px;
    padding: 0.35rem 1rem;
    font-size: 0.82rem;
    font-family: 'Space Mono', monospace;
    color: var(--accent2);
    margin-bottom: 1rem;
}

.insights-panel {
    background: linear-gradient(135deg, #141c28, #111820);
    border: 1px solid rgba(56, 178, 255, 0.25);
    border-left: 4px solid var(--accent2);
    border-radius: 12px;
    padding: 1.75rem 2rem;
    margin: 1rem 0 2rem 0;
    line-height: 1.8;
    font-size: 0.92rem;
    color: #c8d4e8;
}

.insights-panel h2 {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    color: var(--accent2);
    margin: 1.5rem 0 0.5rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(56,178,255,0.15);
}

.insights-panel h2:first-child { margin-top: 0; }

.insights-panel p { margin: 0.5rem 0; }

.insights-panel ul {
    margin: 0.4rem 0;
    padding-left: 1.5rem;
}

.insights-panel li { margin: 0.25rem 0; }

/* Chatbot styles */
.chatbot-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.chat-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: var(--accent2);
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}


.pipeline-flow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 1.5rem 0;
}

.pipeline-node {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
    color: var(--muted);
}

.pipeline-node.done {
    color: var(--accent);
    border-color: rgba(126, 232, 162, 0.4);
    background: rgba(126, 232, 162, 0.05);
}

.pipeline-arrow {
    color: var(--border);
    font-size: 1.2rem;
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: var(--accent);
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(126, 232, 162, 0.3);
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #38d68a) !important;
    color: #0d0f14 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(126, 232, 162, 0.3) !important;
}

/* Reset button — solid red, always visible */
[data-testid="stBaseButton-primary"] {
    background: #e05252 !important;
    background-image: none !important;
    color: #ffffff !important;
    border: none !important;
}
[data-testid="stBaseButton-primary"]:hover {
    background: #c43c3c !important;
    background-image: none !important;
    box-shadow: 0 4px 16px rgba(224, 82, 82, 0.4) !important;
}


.stFileUploader {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] > div > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileDropzoneInstructions"],
[data-testid="stFileUploader"] button {
    background: var(--bg) !important;
    border: 1px dashed var(--accent) !important;
    border-radius: 10px !important;
}

/* Text inside the browse section */
[data-testid="stFileDropzoneInstructions"] span,
[data-testid="stFileDropzoneInstructions"] p,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section p {
    color: var(--text) !important;
}

/* Uploaded filename text */
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploaderFileName"] {
    color: var(--accent) !important;
    font-weight: 600 !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px;
    padding: 0.25rem;
    gap: 0.25rem;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
}

.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #0d0f14 !important;
}

div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Expander header — always visible, not just on hover */
div[data-testid="stExpander"] summary {
    background: linear-gradient(135deg, rgba(126,232,162,0.06), rgba(56,178,255,0.04)) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}

div[data-testid="stExpander"] summary:hover {
    background: linear-gradient(135deg, rgba(126,232,162,0.12), rgba(56,178,255,0.08)) !important;
}

/* Expander title text color */
div[data-testid="stExpander"] summary span {
    color: var(--text) !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
}

/* Scroll to top/bottom button */
[data-testid="scrollToBottomButton"] button,
[data-testid="scrollToTopButton"] button {
    background: #ffffff !important;
    color: #0d0f14 !important;
    border: none !important;
    opacity: 1 !important;
}
[data-testid="scrollToBottomButton"] button svg,
[data-testid="scrollToTopButton"] button svg {
    fill: #0d0f14 !important;
    stroke: #0d0f14 !important;
}

/* Expander arrow icon */
div[data-testid="stExpander"] summary svg {
    fill: var(--accent) !important;
}

.stDataFrame {
    background: var(--surface) !important;
}

label { color: var(--muted) !important; font-size: 0.8rem !important; }

/* ── Chat toggle button ──────────────────────────────────────────────────── */
.chat-toggle-btn > button {
    background: linear-gradient(135deg, rgba(126,232,162,0.12), rgba(56,178,255,0.12)) !important;
    color: var(--accent2) !important;
    border: 1px solid rgba(56,178,255,0.35) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.chat-toggle-btn > button:hover {
    border-color: var(--accent2) !important;
    box-shadow: 0 0 16px rgba(56,178,255,0.2) !important;
    transform: none !important;
}

/* ── Chat panel container ────────────────────────────────────────────────── */
.chat-panel-wrap {
    background: #0d1117;
    border: 1px solid #252a3a;
    border-radius: 14px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.chat-panel-header {
    background: linear-gradient(135deg, rgba(126,232,162,0.06), rgba(56,178,255,0.06));
    border-bottom: 1px solid #252a3a;
    padding: 0.8rem 1.1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: #38b2ff;
}
.chat-msg-user {
    background: #1c2030;
    border: 1px solid #252a3a;
    border-radius: 12px 12px 4px 12px;
    padding: 0.7rem 0.9rem;
    margin: 0.35rem 0 0.35rem 18%;
    font-size: 0.86rem;
    color: #e8eaf0;
    line-height: 1.55;
}
.chat-msg-user .chat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: #6b7280;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
}
.chat-msg-assistant {
    background: #141c28;
    border: 1px solid rgba(56,178,255,0.15);
    border-left: 3px solid #38b2ff;
    border-radius: 4px 12px 12px 12px;
    padding: 0.7rem 0.9rem;
    margin: 0.35rem 18% 0.35rem 0;
    font-size: 0.86rem;
    color: #c8d4e8;
    line-height: 1.7;
}
.chat-msg-assistant .chat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: #38b2ff;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header & sidebar
# ---------------------------------------------------------------------------

def render_header():
    title_col, reset_col = st.columns([5, 1])
    with title_col:
        st.markdown('<div class="main-header">⬡ EDA AGENT</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Automated Exploratory Data Analysis powered by LangGraph multi-agent pipeline</div>', unsafe_allow_html=True)
    with reset_col:
        st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_btn", type="primary", use_container_width=True, help="Clear all results, chat history, and uploaded file"):
            _reset_session()
    st.markdown("""
<div class="pipeline-flow">
    <div class="pipeline-node">📂 CSV Input</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-node">🔎 Data Explorer</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-node">🧠 Planner</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-node">⚙️▶️ Gen + Execute</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-node">💡 Insights</div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-node">💬 Chatbot</div>
</div>
""", unsafe_allow_html=True)


def _reset_session():
    """Clear all session state and module-level caches, then rerun."""
    keys_to_clear = [
        'results', 'dataframe', 'eda_plan', 'explorer_log',
        'chat_history', 'chat_tool_log', 'chat_open', 'chat_prefill',
        'token_stats', 'pipeline_token_stats', 'chat_token_stats',
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    # Increment uploader version to force the file uploader widget to reset
    st.session_state['uploader_version'] = st.session_state.get('uploader_version', 0) + 1
    # Clear module-level caches in eda_agent
    try:
        import eda_agent as _ea
        _ea._clear_df_cache()
        _ea.reset_token_stats()
    except Exception:
        pass
    st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#1c2030;border:1px solid #252a3a;border-radius:8px;padding:0.75rem 1rem;font-size:0.8rem;color:#6b7280;line-height:1.6">
            ⚙️ Azure OpenAI credentials are configured in <code style="color:#7ee8a2">.env</code>.<br>
            Update <code>ENDPOINT</code>, <code>AZURE_OPENAI_API_KEY</code>, and <code>MODEL</code> there before running.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">About Agents</div>', unsafe_allow_html=True)

        agents_info = {
            "🔎 Data Explorer": "Receives only shape, column names, dtypes, and first 3 rows. Uses a query_dataframe tool to run arbitrary pandas expressions — deciding for itself what to explore — then writes a rich summary including statistical tests, time-series checks, and anomaly flags.",
            "🧠 Planner Agent": "Creates a column-specific EDA plan grounded in what the explorer found — covering statistical tests, outlier detection, time-series analysis, and feature engineering steps.",
            "⚙️▶️ Gen + Execute": "Generates and executes each step interleaved — findings from step N are passed as context to step N+1, so later steps build on earlier discoveries. Includes self-healing retry on failure: up to 3 attempts per step, regenerating code with error context on each retry.",
            "💡 Insights Agent": "Synthesizes all results into an executive report: dataset overview, statistical findings, anomalies detected, feature engineering recommendations, and next steps.",
            "💬 Data Chatbot": "After EDA completes, answers questions by combining the full EDA context (explorer findings, step outputs, insights) with live pandas queries executed on-demand via the query_dataframe tool."
        }

        for agent, desc in agents_info.items():
            with st.expander(agent, expanded=False):
                st.markdown(f'<div style="font-size:0.82rem;color:#6b7280;line-height:1.6">{desc}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Tech Stack</div>', unsafe_allow_html=True)
        stack = ["LangGraph", "LangChain", "Azure OpenAI", "Pandas", "Matplotlib", "Seaborn", "Streamlit"]
        for s in stack:
            st.markdown(f'<div class="agent-badge">◆ {s}</div>', unsafe_allow_html=True)




# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read CSV or any Excel/spreadsheet format into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xlsm", ".xlsb")):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    elif name.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="xlrd")
    elif name.endswith((".ods", ".odf", ".odt")):
        return pd.read_excel(uploaded_file, engine="odf")
    elif name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file, sep="\t")
    else:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            import logging as _log
            _log.getLogger("eda_agent").warning(f"[UPLOAD] CSV fallback failed ({e}), trying Excel")
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)


def save_uploaded_to_csv(uploaded_file, df: pd.DataFrame) -> str:
    """Save the dataframe to a temp CSV file for the agent pipeline."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False,
                                     newline='', encoding='utf-8') as tmp:
        df.to_csv(tmp, index=False, encoding='utf-8')
        return tmp.name


# ---------------------------------------------------------------------------
# Upload column (col1)
# ---------------------------------------------------------------------------

def render_upload_col():
    """Render the file upload section. Returns (uploaded_file, df_preview)."""
    st.markdown('<div class="section-title">Upload Dataset</div>', unsafe_allow_html=True)
    uploader_version = st.session_state.get("uploader_version", 0)
    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=["csv", "xlsx", "xls", "xlsm", "xlsb", "ods", "odf", "tsv", "txt"],
        label_visibility="collapsed",
        key=f"file_uploader_{uploader_version}"
    )

    df_preview = None
    if uploaded_file:
        if uploaded_file.size > FILE_SIZE_LIMIT_MB * 1024 * 1024:
            st.error(f"File too large ({uploaded_file.size / 1024 / 1024:.1f} MB). Please upload a file under {FILE_SIZE_LIMIT_MB} MB.")
            return None, None

        try:
            df_preview = read_uploaded_file(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
        uploaded_file.seek(0)

        if df_preview is not None:
            ext = uploaded_file.name.split(".")[-1].upper()
            st.markdown(f'<div class="agent-badge" style="margin-bottom:0.75rem">📄 {ext} file detected</div>', unsafe_allow_html=True)

            missing = df_preview.isnull().sum().sum()

            rows_str    = f"{df_preview.shape[0]:,}"
            cols_str    = f"{df_preview.shape[1]:,}"
            missing_str = f"{missing:,}"

            # One font size for all three based on the longest number
            max_len = max(len(rows_str), len(cols_str), len(missing_str))
            if max_len <= 5:  fs = "1.2rem"
            elif max_len <= 7: fs = "1rem"
            elif max_len <= 10: fs = "0.85rem"
            else: fs = "0.72rem"

            st.markdown(f"""
            <div style="display:flex;gap:0.5rem;margin:0.5rem 0">
                <div class="metric-box" style="flex:1;min-width:0;padding:0.75rem 0.5rem">
                    <div class="metric-val" style="font-size:{fs};white-space:nowrap">{rows_str}</div>
                    <div class="metric-label">Rows</div>
                </div>
                <div class="metric-box" style="flex:1;min-width:0;padding:0.75rem 0.5rem">
                    <div class="metric-val" style="font-size:{fs};white-space:nowrap">{cols_str}</div>
                    <div class="metric-label">Cols</div>
                </div>
                <div class="metric-box" style="flex:1;min-width:0;padding:0.75rem 0.5rem">
                    <div class="metric-val" style="font-size:{fs};white-space:nowrap">{missing_str}</div>
                    <div class="metric-label">Nulls</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
            st.dataframe(df_preview.head(PREVIEW_ROWS), width="stretch", height=230)

            with st.expander("Column Details"):
                col_info = pd.DataFrame({
                    'Type': df_preview.dtypes.astype(str),
                    'Nulls': df_preview.isnull().sum(),
                    'Unique': df_preview.nunique()
                })
                st.dataframe(col_info, width="stretch")

    return uploaded_file, df_preview


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(tmp_path: str, df_preview: pd.DataFrame):
    """Run all EDA agents sequentially and store results in session_state."""
    import eda_agent as ea
    from eda_agent import (EDAState, get_minimal_schema,
                           data_explorer_agent, planner_agent,
                           code_gen_exec_agent, insights_agent)

    progress_bar = st.progress(0, text="Initializing...")
    status_area  = st.empty()

    try:
        with st.spinner(""):
            _, schema = get_minimal_schema(tmp_path)

            state: EDAState = {
                "csv_path":           tmp_path,
                "df_schema":          schema,
                "df_info":            "",
                "explorer_log":       [],
                "eda_plan":           [],
                "generated_codes":    [],
                "execution_results":  [],
                "step_findings":      [],
                "insights":           "",
                "current_step_index": 0,
                "errors":             [],
                "status":             "starting"
            }

            status_area.markdown('<div class="agent-badge active">🔎 Data Explorer querying dataset dynamically...</div>', unsafe_allow_html=True)
            progress_bar.progress(10, text="Exploring dataset with tool calls...")
            state = data_explorer_agent(state)
            n_calls = len(state["explorer_log"])
            progress_bar.progress(25, text=f"Exploration done — {n_calls} tool calls made")
            st.session_state['explorer_log'] = state['explorer_log']

            status_area.markdown('<div class="agent-badge active">🧠 Planner building EDA plan...</div>', unsafe_allow_html=True)
            state = planner_agent(state)
            progress_bar.progress(42, text=f"Plan ready: {len(state['eda_plan'])} steps")
            st.session_state['eda_plan'] = state['eda_plan']

            status_area.markdown('<div class="agent-badge active">⚙️ Generating & executing steps with inter-step context...</div>', unsafe_allow_html=True)
            state = code_gen_exec_agent(state)
            progress_bar.progress(82, text="Generating insights...")
            status_area.markdown('<div class="agent-badge active">💡 Insights Agent synthesizing findings...</div>', unsafe_allow_html=True)
            state = insights_agent(state)
            progress_bar.progress(100, text="Complete! Chatbot is ready.")


        status_area.markdown('<div class="agent-badge active">✅ Pipeline complete!</div>', unsafe_allow_html=True)
        st.session_state['results'] = state

        from eda_agent import get_token_summary
        pipeline_ts = get_token_summary()
        st.session_state['pipeline_token_stats'] = pipeline_ts
        st.session_state['chat_token_stats']     = {"input": 0, "output": 0, "calls": 0}
        st.session_state['token_stats']          = _merge_token_stats(
            pipeline_ts, st.session_state['chat_token_stats']
        )

        try:
            os.unlink(tmp_path)
        except Exception as e:
            import logging as _log
            _log.getLogger("eda_agent").warning(f"[CLEANUP] Could not delete temp file {tmp_path}: {e}")

    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")


# ---------------------------------------------------------------------------
# Run column (col2)
# ---------------------------------------------------------------------------

def render_run_col(uploaded_file, df_preview):
    """Render the run button and trigger the pipeline."""
    if uploaded_file and df_preview is not None:
        st.markdown('<div class="section-title">Run EDA Pipeline</div>', unsafe_allow_html=True)
        run_col, _ = st.columns([1, 2])
        with run_col:
            run_button = st.button("▶  RUN EDA AGENT", width="stretch")

        if run_button:
            tmp_path = save_uploaded_to_csv(uploaded_file, df_preview)
            st.session_state['dataframe'] = df_preview.copy()
            run_pipeline(tmp_path, df_preview)
    else:
        st.markdown("""
        <div style="
            border: 1px dashed #252a3a;
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            color: #6b7280;
            margin-top: 2rem;
        ">
            <div style="font-size:3rem;margin-bottom:1rem">📂</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.85rem;letter-spacing:0.05em">
                Upload a CSV file to start
            </div>
            <div style="font-size:0.8rem;margin-top:0.5rem;opacity:0.7">
                The EDA agent will automatically analyze your dataset
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Results tabs
# ---------------------------------------------------------------------------

def render_summary_metrics(state):
    """Render the 4 metric boxes above the results tabs."""
    total      = len(state['execution_results'])
    successful = sum(1 for r in state['execution_results'] if not r.get('error'))
    with_plots = sum(1 for r in state['execution_results'] if r.get('images'))

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, val, label, color in zip(
        [mc1, mc2, mc3, mc4],
        [total, successful, total - successful, with_plots],
        ["Total Steps", "Successful", "Errors", "With Plots"],
        ["#7ee8a2", "#7ee8a2", "#ff6b9d", "#7ee8a2"]
    ):
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color}">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)


def render_eda_tab(state):
    """Render Tab 1: EDA Results."""
    if 'eda_plan' in st.session_state:
        st.markdown('<div class="section-title">EDA Plan — Generated by Planner Agent</div>', unsafe_allow_html=True)
        plan_html = '<div class="step-card">'
        for i, step in enumerate(st.session_state['eda_plan']):
            plan_html += f'<div class="plan-item"><div class="plan-idx">{i+1}</div><div>{step}</div></div>'
        plan_html += '</div>'
        st.markdown(plan_html, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Step-by-Step Results</div>', unsafe_allow_html=True)

    for result in state['execution_results']:
        retry_count = result.get('retry_count', 0)
        if not result.get('error'):
            if retry_count == 0:
                status_badge = '<span class="step-status-success">✓ SUCCESS</span>'
            else:
                status_badge = (f'<span class="step-status-success">✓ SUCCESS</span> '
                                f'<span style="font-size:0.7rem;color:#f6a623;font-family:\'Space Mono\',monospace;margin-left:0.4rem">'
                                f'↺ healed in {retry_count} retr{"y" if retry_count==1 else "ies"}</span>')
        else:
            status_badge = f'<span class="step-status-error">✗ FAILED ({1 + retry_count} attempts)</span>'

        with st.expander(f"Step {result['step_number']}: {result['step']}", expanded=True):
            st.markdown(status_badge, unsafe_allow_html=True)
            out_tab, code_tab, detail_tab = st.tabs(["📊 Output", "💻 Code", "🔍 Details"])

            with out_tab:
                if result.get('images'):
                    for img_b64 in result['images']:
                        st.image(base64.b64decode(img_b64), width="stretch")
                if result.get('text_output') and result['text_output'].strip():
                    st.markdown('<div class="section-title">Text Output</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="output-text">{result["text_output"]}</div>', unsafe_allow_html=True)
                if not result.get('images') and not (result.get('text_output') and result['text_output'].strip()):
                    if not result.get('error'):
                        st.info("Step executed successfully with no output to display.")
                if result.get('error'):
                    st.markdown(f'<div class="error-box">⚠ {result["error"][:500]}</div>', unsafe_allow_html=True)

            with code_tab:
                st.code(result.get('code', 'No code available'), language='python')

            with detail_tab:
                st.json({
                    "step_number": result['step_number'],
                    "step":        result['step'],
                    "has_plots":   bool(result.get('images')),
                    "plot_count":  len(result.get('images', [])),
                    "has_text":    bool(result.get('text_output', '').strip()),
                    "has_error":   bool(result.get('error')),
                    "retry_count": result.get('retry_count', 0),
                })


def render_insights_tab(state):
    """Render Tab 2: Insights."""
    if state.get('df_info'):
        with st.expander("📋 Dataset Summary (from Data Explorer)", expanded=False):
            st.markdown(
                f'<div style="font-size:0.85rem;color:#c8d4e8;line-height:1.7;white-space:pre-wrap">'
                f'{state["df_info"]}</div>',
                unsafe_allow_html=True
            )
    if state.get('insights'):
        st.markdown('<div class="insights-panel">', unsafe_allow_html=True)
        st.markdown(state['insights'])
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No insights generated yet.")


def generate_pdf(state) -> bytes:
    """Build and return PDF bytes from EDA results."""
    import re
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Image as RLImage, HRFlowable,
                                     PageBreak, Table, TableStyle)
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    buf      = io.BytesIO()
    doc      = SimpleDocTemplate(buf, pagesize=A4,
                                  leftMargin=2*cm, rightMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
    W, H     = A4
    usable_w = W - 4*cm

    title_style    = ParagraphStyle("title",    fontName="Helvetica-Bold", fontSize=22, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("subtitle", fontName="Helvetica",      fontSize=11, textColor=colors.HexColor("#6b7280"), spaceAfter=20, alignment=TA_CENTER)
    h1_style       = ParagraphStyle("h1",       fontName="Helvetica-Bold", fontSize=15, textColor=colors.HexColor("#0d1b2a"), spaceBefore=18, spaceAfter=8, borderPad=4)
    h2_style       = ParagraphStyle("h2",       fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#1a5276"), spaceBefore=14, spaceAfter=6)
    body_style     = ParagraphStyle("body",     fontName="Helvetica",      fontSize=9,  textColor=colors.HexColor("#2c3e50"), leading=14, spaceAfter=6)
    mono_style     = ParagraphStyle("mono",     fontName="Courier",        fontSize=8,  textColor=colors.HexColor("#34495e"), leading=12, backColor=colors.HexColor("#f4f6f8"), borderPad=6, spaceAfter=8)
    success_style  = ParagraphStyle("success",  fontName="Helvetica-Bold", fontSize=9,  textColor=colors.HexColor("#1e8449"))
    error_style    = ParagraphStyle("error",    fontName="Helvetica-Bold", fontSize=9,  textColor=colors.HexColor("#c0392b"))
    label_style    = ParagraphStyle("label",    fontName="Helvetica-Bold", fontSize=8,  textColor=colors.HexColor("#6b7280"), spaceAfter=3, spaceBefore=10)

    def clean(text):
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*',     r'\1', text)
        text = re.sub(r'#+\s*',         '',    text)
        text = re.sub(r'`(.*?)`',       r'\1', text)
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return text.strip()

    def para(text, style=None):
        return Paragraph(clean(text), style or body_style)

    def divider():
        return HRFlowable(width="100%", thickness=0.5,
                          color=colors.HexColor("#dde1e7"), spaceAfter=8, spaceBefore=4)

    story = []

    # Cover
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("EDA Agent Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", subtitle_style))
    story.append(divider())

    exec_results = state.get('execution_results', [])
    t_total = len(exec_results)
    t_ok    = sum(1 for r in exec_results if not r.get('error'))
    t_fail  = t_total - t_ok
    t_plots = sum(1 for r in exec_results if r.get('images'))

    summary_data = [
        ["Total Steps", "Successful", "Failed", "With Charts"],
        [str(t_total), str(t_ok), str(t_fail), str(t_plots)],
    ]
    tbl = Table(summary_data, colWidths=[usable_w/4]*4)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#0d1b2a")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#eaf4fb"), colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(Spacer(1, 0.5*cm))
    story.append(tbl)
    story.append(PageBreak())

    # Dataset summary
    if state.get('df_info'):
        story.append(Paragraph("Dataset Summary", h1_style))
        story.append(divider())
        for line in state['df_info'].split('\n'):
            if line.strip():
                story.append(para(line))
        story.append(PageBreak())

    # Insights
    if state.get('insights'):
        story.append(Paragraph("Executive Insights", h1_style))
        story.append(divider())
        for line in state['insights'].split('\n'):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 4))
            elif line.startswith('## '):
                story.append(Paragraph(line[3:], h2_style))
            elif line.startswith('# '):
                story.append(Paragraph(line[2:], h1_style))
            elif line.startswith('- ') or line.startswith('* '):
                story.append(para(f"• {line[2:]}"))
            else:
                story.append(para(line))
        story.append(PageBreak())

    # Step results
    story.append(Paragraph("EDA Step-by-Step Results", h1_style))
    story.append(divider())

    for result in exec_results:
        num   = result['step_number']
        title = result['step']
        ok    = not result.get('error')

        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(f"Step {num}: {title}", h2_style))
        story.append(Paragraph("✓  SUCCESS" if ok else "✗  FAILED", success_style if ok else error_style))
        story.append(Spacer(1, 0.2*cm))

        txt = (result.get('text_output') or '').strip()
        if txt:
            story.append(Paragraph("OUTPUT", label_style))
            for line in txt.split('\n')[:60]:
                safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(safe or " ", mono_style))

        if result.get('error'):
            story.append(Paragraph("ERROR", label_style))
            for line in result['error'].strip()[-600:].split('\n'):
                safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(safe or " ", mono_style))

        imgs = result.get('images', [])
        if imgs:
            story.append(Paragraph(f"CHART{'S' if len(imgs)>1 else ''} ({len(imgs)})", label_style))
            for img_b64 in imgs:
                try:
                    img_bytes = base64.b64decode(img_b64)
                    img_buf   = io.BytesIO(img_bytes)
                    rl_img    = RLImage(img_buf, width=usable_w, height=usable_w * 0.55)
                    story.append(rl_img)
                    story.append(Spacer(1, 0.3*cm))
                except Exception as e:
                    import logging as _log
                    _log.getLogger("eda_agent").warning(f"[PDF] Could not embed chart for step {result.get('step_number')}: {e}")

        story.append(divider())

    doc.build(story)
    return buf.getvalue()


def render_export_tab(state):
    """Render Tab 3: PDF Export."""
    st.markdown('<div class="section-title">Download EDA Report (PDF)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.82rem;color:#6b7280;margin-bottom:1rem">'
        'The PDF contains the dataset summary, insights, and every EDA step with its text output and charts.</div>',
        unsafe_allow_html=True
    )

    if st.button("📄 Generate & Download PDF Report"):
        with st.spinner("Building PDF..."):
            pdf_bytes = generate_pdf(state)
        fname = f"eda_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label="⬇  Download PDF Report",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf"
        )


def _merge_token_stats(a: dict, b: dict) -> dict:
    """Combine two token stat dicts into one cumulative total.
    Accepts keys 'input'/'output' or 'input_tokens'/'output_tokens'."""
    def _inp(d): return d.get("input", d.get("input_tokens", 0))
    def _out(d): return d.get("output", d.get("output_tokens", 0))
    def _calls(d): return d.get("calls", 0)
    return {
        "calls":         _calls(a) + _calls(b),
        "input_tokens":  _inp(a)   + _inp(b),
        "output_tokens": _out(a)   + _out(b),
        "total_tokens":  _inp(a)   + _inp(b) + _out(a) + _out(b),
    }




def render_results(state):
    """Render all three result tabs."""
    render_summary_metrics(state)

    tab_eda, tab_insights, tab_export = st.tabs([
        "📊 EDA Results",
        "💡 Insights",
        "⬇ Export"
    ])

    with tab_eda:
        render_eda_tab(state)
    with tab_insights:
        render_insights_tab(state)
    with tab_export:
        render_export_tab(state)


# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------

def render_chatbot(state):
    """Render the chat toggle button and chat panel."""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'chat_open' not in st.session_state:
        st.session_state['chat_open'] = False

    st.markdown("---")

    btn_label = "✕  Close Chat" if st.session_state['chat_open'] else "💬  Chat with Data"
    st.markdown('<div class="chat-toggle-btn">', unsafe_allow_html=True)
    if st.button(btn_label, key="chat_toggle"):
        st.session_state['chat_open'] = not st.session_state['chat_open']
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state['chat_open']:
        return

    st.markdown('<div class="chat-panel-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="chat-panel-header">💬 ASK YOUR DATA — powered by live queries + EDA context</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.markdown(
                f'<div class="chat-msg-user">'
                f'<div class="chat-label">YOU</div>'
                f'{msg["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-msg-assistant">'
                f'<div class="chat-label">DATA ASSISTANT</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(msg["content"])

    if not st.session_state['chat_history']:
        suggestions = [
            "What are the top 3 key findings?",
            "Which columns have the most missing data?",
            "What correlations exist between numeric columns?",
            "Are there any outliers I should know about?",
            "What analysis steps would you recommend next?",
            "Summarize all data quality issues found.",
        ]
        st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:0.65rem;letter-spacing:0.12em;color:#6b7280;margin:0.75rem 0 0.4rem">SUGGESTED QUESTIONS</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, sug in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(sug, key=f"sug_{i}", width="stretch"):
                    st.session_state['chat_prefill'] = sug
                    st.rerun()


    prefill      = st.session_state.pop('chat_prefill', None)
    user_input   = st.chat_input("Ask anything about your data...")
    active_input = user_input or prefill

    if active_input:
        from eda_agent import chat_with_data

        if 'chat_tool_log' not in st.session_state:
            st.session_state['chat_tool_log'] = []

        st.session_state['chat_history'].append({"role": "user", "content": active_input})

        with st.spinner("Thinking..."):
            try:
                reply, tool_calls = chat_with_data(
                    user_message=active_input,
                    chat_history=st.session_state['chat_history'][:-1],
                    eda_state=state,
                    df=st.session_state.get('dataframe')
                )
            except Exception as e:
                reply, tool_calls = f"Sorry, I encountered an error: {str(e)}", []

        # Accumulate chat tokens as a delta on top of pipeline stats
        from eda_agent import get_token_summary
        current_ts   = get_token_summary()
        pipeline_ts  = st.session_state.get('pipeline_token_stats', {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        chat_ts      = st.session_state.get('chat_token_stats',     {"input": 0, "output": 0, "calls": 0})
        p_in  = pipeline_ts.get("input_tokens", pipeline_ts.get("input", 0))
        p_out = pipeline_ts.get("output_tokens", pipeline_ts.get("output", 0))
        # Delta = what the module reports now minus what it reported after the pipeline
        delta = {
            "input":  max(0, current_ts["input_tokens"]  - p_in  - chat_ts["input"]),
            "output": max(0, current_ts["output_tokens"] - p_out - chat_ts["output"]),
            "calls":  max(0, current_ts["calls"]         - pipeline_ts.get("calls", 0) - chat_ts["calls"]),
        }
        chat_ts["input"]  += delta["input"]
        chat_ts["output"] += delta["output"]
        chat_ts["calls"]  += delta["calls"]
        st.session_state['chat_token_stats'] = chat_ts
        st.session_state['token_stats']      = _merge_token_stats(pipeline_ts, chat_ts)

        st.session_state['chat_history'].append({"role": "assistant", "content": reply})
        st.session_state['chat_tool_log'].append({
            "question":   active_input,
            "tool_calls": tool_calls
        })
        st.rerun()

    if st.session_state['chat_history']:
        if st.button("🗑  Clear conversation", key="clear_chat"):
            st.session_state['chat_history'] = []
            st.session_state['chat_tool_log'] = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_css()
    render_header()
    render_sidebar()

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        uploaded_file, df_preview = render_upload_col()
    with col2:
        render_run_col(uploaded_file, df_preview)

    if 'results' in st.session_state:
        state = st.session_state['results']
        st.markdown("---")
        st.markdown('<div class="main-header" style="font-size:1.5rem">Analysis Results</div>', unsafe_allow_html=True)
        render_results(state)
        render_chatbot(state)


main()

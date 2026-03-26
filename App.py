"""
EDA Agent UI - Streamlit Interface
"""

import streamlit as st
import pandas as pd
import base64
import json
import os
import tempfile
import io
import datetime
import traceback

from config import FILE_SIZE_LIMIT_MB, PREVIEW_ROWS, ERROR_LOG_FILE


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def friendly_error(exc: Exception) -> str:
    """Map exceptions to user-friendly messages."""
    name = type(exc).__name__
    msg  = str(exc).lower()

    if "authentication" in name.lower() or "api_key" in msg:
        return "Could not authenticate with Azure OpenAI. Please check your API key and endpoint in the .env file."
    if "timeout" in msg or "timed out" in msg:
        return "The AI service timed out. Please try again — the server may be under heavy load."
    if "rate" in msg and "limit" in msg:
        return "API rate limit reached. Please wait a moment and try again."
    if isinstance(exc, FileNotFoundError):
        return f"File not found: {exc}"
    if "emptydata" in name.lower() or "no columns" in msg:
        return "The uploaded file appears to be empty or has no readable columns."
    if "connection" in msg or "connect" in name.lower():
        return "Could not connect to the AI service. Please check your network and .env configuration."
    return f"An unexpected error occurred ({name}). See technical details below for more info."


def log_error(exc: Exception, context: str = ""):
    """Append a structured JSON error entry to errors.jsonl."""
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "context":   context,
        "type":      type(exc).__name__,
        "message":   str(exc),
        "traceback": traceback.format_exc(),
    }
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Sync .streamlit/config.toml with config.py
_toml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "config.toml")
os.makedirs(os.path.dirname(_toml_path), exist_ok=True)
with open(_toml_path, "w") as _f:
    _f.write(f"[server]\nmaxUploadSize = {FILE_SIZE_LIMIT_MB}\n")

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
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    with open(css_path) as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


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
        'pdf_bytes',
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
            log_error(e, context="file_upload_csv_fallback")
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
            log_error(e, context="file_upload")
            st.error(f"Could not read file. Please ensure it's a valid CSV, Excel, or TSV file.")
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

def run_pipeline(tmp_path: str):
    """Run all EDA agents sequentially and store results in session_state."""
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
            log_error(e, context="temp_file_cleanup")

    except Exception as e:
        log_error(e, context="run_pipeline")
        st.error(friendly_error(e))
        with st.expander("Technical details (for debugging)", expanded=False):
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
            run_pipeline(tmp_path)
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


def _pdf_styles():
    """Return all ReportLab paragraph styles used in the PDF report."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    return {
        "title":    ParagraphStyle("title",    fontName="Helvetica-Bold", fontSize=22, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica",      fontSize=11, textColor=colors.HexColor("#6b7280"), spaceAfter=20, alignment=TA_CENTER),
        "h1":       ParagraphStyle("h1",       fontName="Helvetica-Bold", fontSize=15, textColor=colors.HexColor("#0d1b2a"), spaceBefore=18, spaceAfter=8, borderPad=4),
        "h2":       ParagraphStyle("h2",       fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#1a5276"), spaceBefore=14, spaceAfter=6),
        "body":     ParagraphStyle("body",     fontName="Helvetica",      fontSize=9,  textColor=colors.HexColor("#2c3e50"), leading=14, spaceAfter=6),
        "mono":     ParagraphStyle("mono",     fontName="Courier",        fontSize=8,  textColor=colors.HexColor("#34495e"), leading=12, backColor=colors.HexColor("#f4f6f8"), borderPad=6, spaceAfter=8),
        "success":  ParagraphStyle("success",  fontName="Helvetica-Bold", fontSize=9,  textColor=colors.HexColor("#1e8449")),
        "error":    ParagraphStyle("error",    fontName="Helvetica-Bold", fontSize=9,  textColor=colors.HexColor("#c0392b")),
        "label":    ParagraphStyle("label",    fontName="Helvetica-Bold", fontSize=8,  textColor=colors.HexColor("#6b7280"), spaceAfter=3, spaceBefore=10),
    }


def _pdf_clean(text):
    """Strip markdown formatting and escape XML entities for ReportLab."""
    import re
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',     r'\1', text)
    text = re.sub(r'#+\s*',         '',    text)
    text = re.sub(r'`(.*?)`',       r'\1', text)
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return text.strip()


def _pdf_divider():
    """Return a horizontal rule flowable."""
    from reportlab.lib import colors
    from reportlab.platypus import HRFlowable
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#dde1e7"), spaceAfter=8, spaceBefore=4)


def _pdf_cover(story, styles, usable_w, exec_results):
    """Append the cover page (title + summary table) to story."""
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Spacer, PageBreak, Table, TableStyle

    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("EDA Agent Report", styles["title"]))
    story.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", styles["subtitle"]))
    story.append(_pdf_divider())

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


def _pdf_dataset_summary(story, styles, df_info):
    """Append the dataset summary section to story."""
    from reportlab.platypus import Paragraph, PageBreak

    story.append(Paragraph("Dataset Summary", styles["h1"]))
    story.append(_pdf_divider())
    for line in df_info.split('\n'):
        if line.strip():
            story.append(Paragraph(_pdf_clean(line), styles["body"]))
    story.append(PageBreak())


def _pdf_insights(story, styles, insights):
    """Append the executive insights section to story."""
    from reportlab.platypus import Paragraph, Spacer, PageBreak

    story.append(Paragraph("Executive Insights", styles["h1"]))
    story.append(_pdf_divider())
    for line in insights.split('\n'):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], styles["h2"]))
        elif line.startswith('# '):
            story.append(Paragraph(line[2:], styles["h1"]))
        elif line.startswith('- ') or line.startswith('* '):
            story.append(Paragraph(_pdf_clean(f"• {line[2:]}"), styles["body"]))
        else:
            story.append(Paragraph(_pdf_clean(line), styles["body"]))
    story.append(PageBreak())


def _pdf_step_results(story, styles, usable_w, exec_results):
    """Append the step-by-step EDA results to story."""
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Spacer, Image as RLImage
    story.append(Paragraph("EDA Step-by-Step Results", styles["h1"]))
    story.append(_pdf_divider())

    for result in exec_results:
        num   = result['step_number']
        title = result['step']
        ok    = not result.get('error')

        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(f"Step {num}: {title}", styles["h2"]))
        story.append(Paragraph(
            "✓  SUCCESS" if ok else "✗  FAILED",
            styles["success"] if ok else styles["error"]
        ))
        story.append(Spacer(1, 0.2*cm))

        txt = (result.get('text_output') or '').strip()
        if txt:
            story.append(Paragraph("OUTPUT", styles["label"]))
            for line in txt.split('\n')[:60]:
                safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(safe or " ", styles["mono"]))

        if result.get('error'):
            story.append(Paragraph("ERROR", styles["label"]))
            for line in result['error'].strip()[-600:].split('\n'):
                safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(safe or " ", styles["mono"]))

        imgs = result.get('images', [])
        if imgs:
            story.append(Paragraph(f"CHART{'S' if len(imgs)>1 else ''} ({len(imgs)})", styles["label"]))
            for img_b64 in imgs:
                try:
                    img_bytes = base64.b64decode(img_b64)
                    img_buf   = io.BytesIO(img_bytes)
                    rl_img    = RLImage(img_buf, width=usable_w, height=usable_w * 0.55)
                    story.append(rl_img)
                    story.append(Spacer(1, 0.3*cm))
                except Exception as e:
                    log_error(e, context=f"pdf_chart_step_{result.get('step_number')}")

        story.append(_pdf_divider())


def generate_pdf(state) -> bytes:
    """Build and return PDF bytes from EDA results."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate

    buf      = io.BytesIO()
    doc      = SimpleDocTemplate(buf, pagesize=A4,
                                  leftMargin=2*cm, rightMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
    usable_w = A4[0] - 4*cm
    styles   = _pdf_styles()
    story    = []

    exec_results = state.get('execution_results', [])

    _pdf_cover(story, styles, usable_w, exec_results)

    if state.get('df_info'):
        _pdf_dataset_summary(story, styles, state['df_info'])

    if state.get('insights'):
        _pdf_insights(story, styles, state['insights'])

    _pdf_step_results(story, styles, usable_w, exec_results)

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

    if 'pdf_bytes' not in st.session_state:
        with st.spinner("Building PDF..."):
            st.session_state['pdf_bytes'] = generate_pdf(state)

    fname = f"eda_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    st.download_button(
        label="📄 Download PDF Report",
        data=st.session_state['pdf_bytes'],
        file_name=fname,
        mime="application/pdf"
    )


def _normalize_token_stats(d: dict) -> dict:
    """Return a token stats dict with consistent keys."""
    inp   = d.get("input_tokens", d.get("input", 0))
    out   = d.get("output_tokens", d.get("output", 0))
    calls = d.get("calls", 0)
    return {"input_tokens": inp, "output_tokens": out, "calls": calls, "total_tokens": inp + out}


def _merge_token_stats(a: dict, b: dict) -> dict:
    """Combine two token stat dicts into one cumulative total.
    Accepts keys 'input'/'output' or 'input_tokens'/'output_tokens'."""
    na, nb = _normalize_token_stats(a), _normalize_token_stats(b)
    return {
        "calls":         na["calls"]        + nb["calls"],
        "input_tokens":  na["input_tokens"] + nb["input_tokens"],
        "output_tokens": na["output_tokens"]+ nb["output_tokens"],
        "total_tokens":  na["total_tokens"] + nb["total_tokens"],
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
                log_error(e, context="chatbot")
                reply, tool_calls = friendly_error(e), []

        # Accumulate chat tokens as a delta on top of pipeline stats
        from eda_agent import get_token_summary
        current_ts  = _normalize_token_stats(get_token_summary())
        pipeline_ts = _normalize_token_stats(st.session_state.get('pipeline_token_stats', {}))
        chat_ts     = _normalize_token_stats(st.session_state.get('chat_token_stats', {}))
        # Delta = what the module reports now minus pipeline baseline minus previous chat
        delta = {
            "input_tokens":  max(0, current_ts["input_tokens"]  - pipeline_ts["input_tokens"]  - chat_ts["input_tokens"]),
            "output_tokens": max(0, current_ts["output_tokens"] - pipeline_ts["output_tokens"] - chat_ts["output_tokens"]),
            "calls":         max(0, current_ts["calls"]         - pipeline_ts["calls"]         - chat_ts["calls"]),
        }
        chat_ts = _merge_token_stats(chat_ts, delta)
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

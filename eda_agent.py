"""
EDA Agent using LangGraph
Pipeline: Data Explorer (tool-calling) → Planner → Code Gen+Exec (interleaved) → Insights
"""
add
import os
import sys
import json
import traceback
import io
import base64
import datetime
from typing import TypedDict, List
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import pandas as pd
from openai import AzureOpenAI
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Configuration — imported from config.py
# ---------------------------------------------------------------------------
from config import (
    ENABLE_LOGGING, MAX_RETRIES,
    EXPLORER_MAX_ROUNDS_PER_COL,
    TOOL_OUTPUT_TRUNCATE, STEP_FINDING_TRUNCATE,
    CHATBOT_STEP_OUTPUT_CHARS, CHATBOT_ERROR_CHARS,
    INSIGHTS_STEP_OUTPUT_CHARS, INSIGHTS_ERROR_CHARS,
    CHATBOT_MAX_TOOL_ROUNDS,
    JSON_LOG_FILE,
)

# JSON structured log — single log file for all events
_json_log_handle = open(JSON_LOG_FILE, "w", encoding="utf-8")

_current_agent: str = "unknown"

def _log_json(event: str, **fields):
    """Append a single JSON line to logs.jsonl."""
    record = {
        "ts": datetime.datetime.now().isoformat(),
        "event": event,
        "agent": _current_agent,
        **fields,
    }
    _json_log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _json_log_handle.flush()

# ---------------------------------------------------------------------------
# Token / cost tracker
# ---------------------------------------------------------------------------

_token_stats: dict = {"input": 0, "output": 0, "calls": 0}

def reset_token_stats():
    _token_stats["input"]  = 0
    _token_stats["output"] = 0
    _token_stats["calls"]  = 0

def get_token_summary() -> dict:
    total = _token_stats["input"] + _token_stats["output"]
    return {
        "calls":         _token_stats["calls"],
        "input_tokens":  _token_stats["input"],
        "output_tokens": _token_stats["output"],
        "total_tokens":  total,
    }

def set_agent_context(name: str):
    global _current_agent
    _current_agent = name
    _log_json("agent_start", agent=name)

def _log_llm_call(agent: str, messages: list, tools: list, response_text: str, tool_calls_found: list):
    """Write full LLM call details to logs.jsonl when ENABLE_LOGGING is True."""
    if not ENABLE_LOGGING:
        return

    tool_names = []
    if tools:
        for t in tools:
            tool_names.append(t.get("name", str(t)) if isinstance(t, dict) else str(t))

    _log_json("llm_call_detail",
              agent=agent,
              tools=tool_names,
              message_count=len(messages),
              messages=[{"role": m.get("role", "?"), "content": str(m.get("content", ""))} for m in messages],
              response_text=(response_text or "").strip(),
              tool_calls=[{"name": tc.get("name", "?"), "arguments": tc.get("arguments", {})} for tc in (tool_calls_found or [])])

# ---------------------------------------------------------------------------
# DataFrame cache — avoids re-reading the same CSV multiple times per run
# ---------------------------------------------------------------------------

_df_cache: dict = {}

def _load_df(csv_path: str, nrows: int = None) -> pd.DataFrame:
    """Return a cached DataFrame for csv_path. Reads from disk only once per path+nrows combo."""
    key = (csv_path, nrows)
    if key not in _df_cache:
        _df_cache[key] = pd.read_csv(csv_path, encoding='utf-8', nrows=nrows)
    return _df_cache[key]

def _clear_df_cache():
    """Call at the start of each run so stale data doesn't carry over."""
    _df_cache.clear()

def _clear_output_dir():
    """Delete all plots from the previous run."""
    import shutil
    output_dir = Path("eda_outputs")
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            _log_json("warning", category="cleanup", error=str(e))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class EDAState(TypedDict):
    csv_path: str
    df_schema: str              # minimal schema sent upfront: shape + columns + dtypes only
    df_info: str                # rich context built dynamically by the Data Explorer agent
    explorer_log: List[dict]    # log of every tool call + result (for UI display)
    eda_plan: List[str]
    generated_codes: List[str]
    execution_results: List[dict]   # each has: step, code, text_output, images[], error, retry_count
    step_findings: List[str]        # brief findings from each completed step — fed to next step
    insights: str
    current_step_index: int
    errors: List[str]
    status: str


# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------

API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT    = os.getenv("ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
MODEL       = os.getenv("MODEL")

client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version=API_VERSION
)


def get_response(history: list, tools: list = None):
    """Call Azure OpenAI. Logs every call (agent, input, output) to eda_agent.log."""
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    kwargs   = {"model": MODEL, "input": messages}
    if tools:
        kwargs["tools"] = tools

    response = client.responses.create(**kwargs)

    # Track token usage
    in_tok = out_tok = 0
    if hasattr(response, "usage") and response.usage:
        in_tok  = getattr(response.usage, "input_tokens",  0)
        out_tok = getattr(response.usage, "output_tokens", 0)
        _token_stats["input"]  += in_tok
        _token_stats["output"] += out_tok
        _token_stats["calls"]  += 1

    # Extract text + tool calls for logging
    resp_text  = extract_text(response)
    resp_tools = extract_tool_calls(response)

    # JSON structured log — token stats + call metadata
    _log_json("llm_call",
              call_number=_token_stats["calls"],
              model=MODEL,
              input_tokens=in_tok,
              output_tokens=out_tok,
              cumulative_input=_token_stats["input"],
              cumulative_output=_token_stats["output"],
              message_count=len(messages),
              tool_calls=[tc.get("name", "?") for tc in (resp_tools or [])],
              response_chars=len(resp_text or ""))

    # Full LLM call detail (messages, response) — only when ENABLE_LOGGING is True
    _log_llm_call(_current_agent, messages, tools or [], resp_text, resp_tools)

    return response


def extract_text(response) -> str:
    """Pull plain text from a response that may also contain tool calls."""
    for block in response.output:
        if block.type == "message":
            for c in block.content:
                if hasattr(c, "text"):
                    return c.text
    return ""


def extract_tool_calls(response) -> list:
    """Return list of {id, name, arguments} dicts from the response output."""
    calls = []
    for block in response.output:
        if block.type == "function_call":
            calls.append({
                "id":        block.call_id,
                "name":      block.name,
                "arguments": json.loads(block.arguments)
            })
    return calls


# ---------------------------------------------------------------------------
# Tool definition exposed to the LLM
# ---------------------------------------------------------------------------

DATAFRAME_TOOL = {
    "type": "function",
    "name": "query_dataframe",
    "description": (
        "Execute a pandas expression against the dataframe `df` and return the string result. "
        "Use this to explore the dataset: check distributions, value counts, correlations, nulls, "
        "sample rows, data types, unique values, date ranges, etc. "
        "Call it as many times as needed until you fully understand the dataset."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "A single Python expression using `df`. Must return something printable. "
                    "Examples: `df.describe()`, `df['col'].value_counts().head(10)`, "
                    "`df.isnull().sum()`, `df.dtypes`, `df.head(3)`, "
                    "`df.select_dtypes(include='number').corr()`, "
                    "`df['date_col'].min(), df['date_col'].max()`"
                )
            },
            "reason": {
                "type": "string",
                "description": "Why you are running this query — what you want to learn."
            }
        },
        "required": ["expression", "reason"]
    }
}


def run_df_query(df: pd.DataFrame, expression: str) -> str:
    """Safely evaluate a pandas expression and return its string result."""
    try:
        result = eval(expression, {"df": df, "pd": pd})
        out = str(result)
        # Truncate very long outputs so the context window stays manageable
        if len(out) > 2000:
            out = out[:TOOL_OUTPUT_TRUNCATE] + f"\n... [truncated, {len(out)} chars total]"
        return out
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Minimal schema — only what fits on a napkin, sent upfront
# ---------------------------------------------------------------------------

def get_minimal_schema(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path, encoding='utf-8')
    schema = (
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        f"Columns & dtypes:\n{df.dtypes.to_string()}\n\n"
        f"First 3 rows:\n{df.head(3).to_string()}"
    )
    return df, schema


# ---------------------------------------------------------------------------
# Agent 0: Data Explorer  (tool-calling loop)
# ---------------------------------------------------------------------------

def data_explorer_agent(state: EDAState) -> EDAState:
    """
    The LLM sees only the minimal schema, then decides what pandas queries to run.
    It MUST explore every column before writing the summary.
    """
    set_agent_context("Data Explorer")
    print("\n[DATA EXPLORER] Starting dynamic dataset exploration...")
    state["status"] = "exploring"

    df = _load_df(state["csv_path"])
    col_list = list(df.columns)
    n_cols = len(col_list)
    explorer_log: List[dict] = []

    system_prompt = f"""You are an expert data scientist doing a thorough first-pass exploration of an unfamiliar dataset.
You have a tool `query_dataframe` that executes any pandas expression against `df` and returns the result as a string.

CRITICAL REQUIREMENT: You MUST explore EVERY column in this dataset. There are {n_cols} columns total:
{col_list}

Do NOT stop early. Do NOT skip columns. Every single column must be understood before you write your summary.

MANDATORY exploration checklist — work through ALL of these:

1. GLOBAL OVERVIEW (run first):
   - df.shape, df.dtypes (already known from schema)
   - df.isnull().sum() to see all missing values at once
   - df.duplicated().sum() to check for duplicate rows
   - df.describe(include='all') for a full statistical overview

2. NUMERIC COLUMNS — for EACH numeric column:
   - Distribution: df['col'].describe(), check skew with df['col'].skew()
   - Outliers: IQR check or df['col'].quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
   - Missing count: df['col'].isnull().sum()
   - Zero/negative values if relevant: (df['col'] <= 0).sum()

3. CATEGORICAL/OBJECT COLUMNS — for EACH categorical column:
   - df['col'].value_counts().head(15) — top values and frequency
   - df['col'].nunique() — cardinality
   - df['col'].isnull().sum() — missing values
   - Watch for: high-cardinality columns, columns with very few unique values (flags/binary)

4. DATE/DATETIME COLUMNS — for EACH date column:
   - df['col'].min(), df['col'].max() — date range
   - df['col'].dt.year.value_counts().sort_index() — distribution by year
   - Check for nulls

5. RELATIONSHIPS:
   - df.select_dtypes(include='number').corr() — full correlation matrix
   - For any strong correlations (>0.7 or <-0.7), investigate the pair
   - Group-by analysis: df.groupby('cat_col')['num_col'].describe() for important combinations

6. DATA QUALITY:
   - Columns with >20% nulls deserve special note
   - Constant or near-constant columns: df.nunique().sort_values()
   - Columns that look like IDs (nunique == nrows)

7. STATISTICAL TESTS:
   - Normality proxy: df['col'].skew() and df['col'].kurtosis() for every numeric column
     (skew > 1 or < -1 = significantly skewed; kurtosis > 3 = heavy tails)
   - Identify log-transform candidates: numeric columns with skew > 1 and all positive values
   - For categorical vs numeric: df.groupby('cat_col')['num_col'].agg(['mean','std','count','median'])
   - Note which columns are suitable for Pearson vs Spearman correlation based on normality

8. TIME-SERIES (if datetime columns exist):
   - For each datetime column: df['col'].dt.year.value_counts().sort_index()
   - Monthly/weekly trends: df.groupby(df['date_col'].dt.to_period('M')).size()
   - Check for gaps: df['date_col'].diff().describe() — are records evenly spaced?
   - Identify if any numeric columns show clear time-based trends

9. ANOMALY DETECTION:
   - IQR outlier count per numeric col:
     Q1=df['col'].quantile(0.25); Q3=df['col'].quantile(0.75); IQR=Q3-Q1
     outliers = df[(df['col'] < Q1-1.5*IQR) | (df['col'] > Q3+1.5*IQR)].shape[0]
   - Z-score outliers: ((df['col'] - df['col'].mean()) / df['col'].std()).abs() > 3
   - Flag rows with suspicious combinations: e.g. impossible values, negative where positive expected
   - Note which columns have the most outliers

You have up to {min(n_cols * 3, 30)} tool calls. Use them wisely to cover EVERY column.

After completing all checks, write a COMPREHENSIVE DATASET SUMMARY that covers:
- Every column: its type, range/values, missingness, notable characteristics
- Data quality issues found
- Key relationships between columns
- Anything unusual or interesting
The summary must be detailed enough that another agent can write a full multi-step EDA plan WITHOUT needing to query the data again.
Write plain text only — no code blocks in the summary."""

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": (
            f"Dataset schema:\n\n{state['df_schema']}\n\n"
            f"All {n_cols} columns you MUST cover: {col_list}\n\n"
            "Begin systematic exploration. Work through the mandatory checklist covering every column."
        )}
    ]

    max_rounds = max(n_cols * EXPLORER_MAX_ROUNDS_PER_COL, 30)  # scale ceiling with dataset width
    for round_idx in range(max_rounds):
        response   = get_response(history, tools=[DATAFRAME_TOOL])
        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            summary = extract_text(response)
            print(f"[DATA EXPLORER] Done after {round_idx} round(s). Summary: {len(summary)} chars")
            state["df_info"]      = summary
            state["explorer_log"] = explorer_log
            return state

        for call in tool_calls:
            expr   = call["arguments"]["expression"]
            reason = call["arguments"].get("reason", "")
            result = run_df_query(df, expr)

            short_expr   = expr[:70] + ("..." if len(expr) > 70 else "")
            short_result = result[:120] + ("..." if len(result) > 120 else "")
            print(f"  [TOOL {round_idx+1}] {short_expr}")
            print(f"           -> {short_result}")

            explorer_log.append({
                "round":      round_idx + 1,
                "expression": expr,
                "reason":     reason,
                "result":     result
            })

            history.append({"role": "assistant", "content": f"Ran: {expr}"})
            history.append({"role": "user",      "content": f"Result:\n{result}"})

    # Ceiling hit — force summary
    print("[DATA EXPLORER] Hit max rounds, requesting summary")
    history.append({
        "role":    "user",
        "content": (
            "You have reached the exploration limit. Write the comprehensive dataset summary now. "
            f"Make sure to cover all {n_cols} columns: {col_list}"
        )
    })
    response = get_response(history)
    state["df_info"]      = extract_text(response) or "Exploration completed."
    state["explorer_log"] = explorer_log
    return state


# ---------------------------------------------------------------------------
# Agent 1: Planner
# ---------------------------------------------------------------------------

def planner_agent(state: EDAState) -> EDAState:
    set_agent_context("Planner")
    print("\n[PLANNER] Building comprehensive EDA plan covering all columns...")
    state["status"] = "planning"

    system_prompt = f"""You are an expert data scientist creating a detailed EDA execution plan.

You have a comprehensive dataset summary from an automated exploration agent (see below).
Your job is to produce a step-by-step EDA plan that COVERS EVERY COLUMN IN THE DATASET.

CRITICAL RULES:
1. EVERY column listed in the schema MUST appear in at least one EDA step.
   Do NOT focus only on a few columns — distribute steps across ALL columns.
2. Each step must name the EXACT column names it will analyse.
3. Group related columns together logically within steps.
4. Steps must be concrete and actionable — a code generator will write Python for each one.
5. Produce 8-12 steps ensuring full column coverage.

MANDATORY STEP CATEGORIES (include all of these):
- Dataset overview: shape, dtypes, missing value summary for ALL columns
- Duplicate row analysis and row-level quality checks
- For ALL numeric columns: distribution histograms, boxplots, descriptive stats
- For ALL categorical/object columns: value counts bar charts, cardinality, top-N analysis
- For ALL date/datetime columns: time-series trend line, year/month/weekday distribution, gap analysis, rolling averages
- Full correlation heatmap across ALL numeric columns with significance
- Statistical tests: normality test (skew/kurtosis), Pearson/Spearman correlation with p-values for key numeric pairs, Kruskal-Wallis or Mann-Whitney for group comparisons
- Outlier & anomaly detection: IQR boxplots, Z-score flags, Isolation Forest on numeric columns
- Missing value visualisation: heatmap and bar chart of nulls per column
- Feature engineering recommendations: log-transform candidates (skewed cols), encoding strategies for categoricals, datetime feature extraction, binning suggestions
- Cross-column groupby analysis from exploration findings

Column schema:
{state['df_schema']}

Output format: Return ONLY a valid JSON array of step strings.
Each string must clearly state: what analysis, which specific columns, what output (chart/table).
Example:
[
  "Distribution analysis: histograms and boxplots for [col_a, col_b, col_c] to identify skew and outliers",
  "Categorical breakdown: value_counts bar charts for [col_x, col_y] showing top 15 categories each",
  ...
]"""

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": (
            f"Dataset exploration summary:\n\n{state['df_info']}\n\n"
            "Generate the comprehensive EDA plan covering every column."
        )}
    ]

    response = get_response(history)
    plan_raw = extract_text(response).strip()

    try:
        if plan_raw.startswith("```"):
            plan_raw = plan_raw.split("```")[1]
            if plan_raw.startswith("json"):
                plan_raw = plan_raw[4:]
        eda_plan = json.loads(plan_raw)
    except Exception as e:
        _log_json("warning", category="planner_fallback", error=str(e))
        eda_plan = [
            "Dataset overview: shape, dtypes, missing value summary for all columns",
            "Duplicate row detection and data quality assessment",
            "Numeric column distributions: histograms and boxplots for all numeric columns",
            "Categorical column analysis: value counts and cardinality for all categorical columns",
            "Missing value visualisation: heatmap and bar chart of nulls per column",
            "Full correlation heatmap for all numeric columns",
            "Outlier detection: IQR-based outlier count per numeric column",
            "Cross-column relationships: scatter plots for highly correlated column pairs"
        ]

    state["eda_plan"]           = eda_plan
    state["current_step_index"] = 0
    print(f"[PLANNER] {len(eda_plan)} steps generated covering all columns")
    return state


# ---------------------------------------------------------------------------
# Code Generation Helpers (used by Code Gen+Exec agent and self-healing retry)
# ---------------------------------------------------------------------------

def _build_codegen_system_prompt(all_cols, numeric_cols, cat_cols, date_cols) -> str:
    return f"""You are an expert Python data scientist writing standalone, executable EDA code snippets.

DATASET COLUMN REFERENCE (use these EXACT names — no guessing):
All columns   : {all_cols}
Numeric       : {numeric_cols}
Categorical   : {cat_cols}
Datetime      : {date_cols}

The dataframe is pre-loaded as `df` (pandas DataFrame).
Two variables are available:
  - `step_idx` (int): base index for this step — use for naming the first/only plot
  - `plot_counter` (list with one int): increment for multiple plots per step

PLOT SAVING RULES — CRITICAL:
- Each unique chart must be saved EXACTLY ONCE — never save the same data twice
- Name plots sequentially: plot_{{step_idx}}_0.png, plot_{{step_idx}}_1.png ...
- Use the plot_counter list to track the sub-index:
    plt.savefig(f'plot_{{step_idx}}_{{plot_counter[0]}}.png', bbox_inches='tight', dpi=100)
    plt.close()
    plot_counter[0] += 1
- CHOOSE ONE approach per group of related charts — either:
  (A) ONE combined subplot figure using plt.subplots() showing all columns together, OR
  (B) ONE individual figure per column saved separately
  NEVER DO BOTH — do not save a combined figure AND individual figures for the same data
- For a single chart: plt.savefig(f'plot_{{step_idx}}_0.png', ...)
- plt.close() MUST be called immediately after every savefig before creating the next figure

STRICT CODING RULES:
1. ALWAYS import at the top: pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
2. For statistical tests import: from scipy import stats
3. For anomaly detection import: from sklearn.ensemble import IsolationForest
4. Use ONLY column names from the reference above — never invent column names
5. Guard every column access: if 'col' in df.columns:
6. Use pd.to_numeric(df['col'], errors='coerce') before any numeric op on ambiguous columns
7. plt.figure(figsize=(12, 6)) minimum — use (16, 8) or larger for multi-column charts
8. Call plt.close() immediately after EVERY savefig — never leave figures open
9. NEVER use plt.show()
10. Print all text results with print()
11. Wrap blocks in try/except — print errors and continue rather than crashing
12. Loop over column lists for multi-column steps — do NOT hardcode individual column names
13. Return ONLY runnable Python code — no markdown fences, no prose

STATISTICAL TESTING PATTERNS:
- Normality: stat, p = stats.normaltest(df['col'].dropna()); print("p="+str(round(p,4))+", normal="+str(p>0.05))
- Pearson correlation: r, p = stats.pearsonr(df['a'].dropna(), df['b'].dropna())
- Spearman (non-normal): r, p = stats.spearmanr(df['a'].dropna(), df['b'].dropna())
- Group comparison (2 groups): stats.mannwhitneyu(group1, group2)
- Group comparison (3+ groups): stats.kruskal(*[g.values for _, g in df.groupby('cat')['num']])

ANOMALY DETECTION PATTERNS:
- IQR method: Q1=df['col'].quantile(0.25); Q3=df['col'].quantile(0.75); IQR=Q3-Q1; outliers=df[(df['col']<Q1-1.5*IQR)|(df['col']>Q3+1.5*IQR)]
- Z-score: z_scores = np.abs(stats.zscore(df[numeric_cols].dropna())); outlier_rows = (z_scores > 3).any(axis=1)
- Isolation Forest: clf = IsolationForest(contamination=0.05, random_state=42); preds = clf.fit_predict(df[numeric_cols].dropna())

TIME-SERIES PATTERNS:
- Rolling average: df.set_index('date_col')['val'].rolling('30D').mean()
- Monthly trend: df.groupby(df['date_col'].dt.to_period('M'))['val'].mean()
- Autocorrelation: pd.plotting.autocorrelation_plot(df['val'].dropna())"""


def _strip_code_fences(code: str) -> str:
    """Remove ```python ... ``` wrapping if present."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code  = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        if code.startswith("python"):
            code = code[6:].lstrip("\n")
    return code


def _generate_code_for_step(step: str, step_idx: int, total_steps: int,
                              df_info: str, system_prompt: str,
                              error_context: str = "",
                              step_findings: list = None) -> str:
    """Call the LLM to generate (or regenerate) code for one EDA step."""
    user_content = (
        f"Dataset summary:\n{df_info}\n\n"
    )

    # Inject findings from previous steps so this step can build on them
    if step_findings:
        findings_text = "\n".join(f"  Step {i+1}: {f}" for i, f in enumerate(step_findings))
        user_content += (
            f"FINDINGS FROM PREVIOUS STEPS (use these to inform this step's analysis):\n"
            f"{findings_text}\n\n"
        )

    user_content += (
        f"EDA Step {step_idx + 1} of {total_steps}: {step}\n"
        f"step_idx = {step_idx}\n"
        f"plot_counter = [0]  # increment this for each plot saved\n\n"
    )
    if error_context:
        user_content += (
            f"PREVIOUS ATTEMPT FAILED WITH THIS ERROR — fix it:\n"
            f"```\n{error_context}\n```\n\n"
        )
    user_content += (
        "Write the complete, executable Python code. "
        "Cover ALL columns mentioned in the step — do not skip any. "
        "Save every plot as plot_{{step_idx}}_{{plot_counter[0]}}.png and increment plot_counter[0]."
    )

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content}
    ]
    response = get_response(history)
    return _strip_code_fences(extract_text(response))




# ---------------------------------------------------------------------------
# Agent 2: Code Gen+Exec (interleaved generate-execute with self-healing)
# ---------------------------------------------------------------------------

def code_gen_exec_agent(state: EDAState) -> EDAState:
    """
    Interleaved generate→execute agent.
    For each EDA step: generate code → execute → extract findings → pass to next step.
    This allows later steps to build on findings from earlier ones (inter-step context).
    """
    set_agent_context("Code Gen+Exec")
    print("\n[CODE GEN+EXEC] Starting interleaved generate-execute pipeline...")
    state["status"] = "generating"

    df_ref       = _load_df(state["csv_path"], nrows=0)
    numeric_cols = df_ref.select_dtypes(include='number').columns.tolist()
    cat_cols     = df_ref.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols    = df_ref.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    all_cols     = df_ref.columns.tolist()

    system_prompt    = _build_codegen_system_prompt(all_cols, numeric_cols, cat_cols, date_cols)
    df               = _load_df(state["csv_path"])
    output_dir       = Path("eda_outputs")
    output_dir.mkdir(exist_ok=True)

    execution_results: list = []
    step_findings: list     = []    # grows after each successful step
    generated_codes: list   = []

    total_steps = len(state["eda_plan"])

    for idx, step in enumerate(state["eda_plan"]):
        print(f"\n  [Step {idx+1}/{total_steps}] {step[:60]}...")

        # --- Generate code with context from previous steps ---
        current_code = _generate_code_for_step(
            step, idx, total_steps,
            state["df_info"], system_prompt,
            step_findings=step_findings
        )
        generated_codes.append(current_code)

        # --- Execute with self-healing retries ---
        result = {
            "step_number": idx + 1,
            "step":        step,
            "code":        current_code,
            "text_output": "",
            "images":      [],
            "error":       None,
            "retry_count": 0,
        }

        last_error = None
        for attempt in range(1 + MAX_RETRIES):
            for old_plot in output_dir.glob(f"plot_{idx}_*.png"):
                old_plot.unlink(missing_ok=True)
            legacy = output_dir / f"plot_{idx}.png"
            if legacy.exists():
                legacy.unlink(missing_ok=True)

            text_out, error = _run_code(current_code, df, idx, output_dir)

            if error is None:
                result["text_output"] = text_out or ""
                result["error"]       = None
                result["retry_count"] = attempt
                result["images"]      = _collect_plots(output_dir, idx)
                n_plots = len(result["images"])
                result["code"]        = current_code
                print(f"    ✓ OK (attempt {attempt+1}) — {n_plots} plot(s)")

                # --- Extract findings to pass to next step ---
                finding = f"{step}: "
                if text_out and text_out.strip():
                    # Take first 300 chars of output as the key finding
                    finding += text_out.strip()[:STEP_FINDING_TRUNCATE].replace('\n', ' ')
                else:
                    finding += "completed successfully."
                step_findings.append(finding)
                break

            last_error = error
            print(f"    ✗ Attempt {attempt+1} failed — regenerating...")
            if attempt < MAX_RETRIES:
                current_code = _generate_code_for_step(
                    step, idx, total_steps,
                    state["df_info"], system_prompt,
                    error_context=last_error[-1500:],
                    step_findings=step_findings
                )
                result["code"] = current_code
        else:
            result["error"]       = last_error
            result["text_output"] = text_out or ""
            result["code"]        = current_code
            result["images"]      = _collect_plots(output_dir, idx)
            state["errors"].append(
                f"Step {idx+1} (all {1+MAX_RETRIES} attempts failed): "
                f"{last_error.strip().split(chr(10))[-1][:100]}"
            )
            print(f"    ✗ Step {idx+1} permanently failed after {1+MAX_RETRIES} attempts")

        execution_results.append(result)
        _log_json("step_executed",
                  step_number=idx + 1,
                  step=step,
                  success=not bool(result.get("error")),
                  retry_count=result.get("retry_count", 0),
                  charts=len(result.get("images", [])),
                  output_chars=len(result.get("text_output") or ""))

    state["generated_codes"]   = generated_codes
    state["execution_results"] = execution_results
    state["step_findings"]     = step_findings
    state["status"]            = "done"
    successful = sum(1 for r in execution_results if not r["error"])
    print(f"\n[CODE GEN+EXEC] Finished — {successful}/{len(execution_results)} steps succeeded")
    return state


# ---------------------------------------------------------------------------
# Code Execution Helpers (self-healing + multi-plot support)
# ---------------------------------------------------------------------------



def _collect_plots(output_dir: Path, step_idx: int) -> list:
    """
    Collect all plots saved for this step, deduplicating by file hash.
    Supports both naming patterns: plot_{idx}.png and plot_{idx}_{sub}.png.
    Returns list of base64-encoded PNG strings, sorted by sub-index.
    """
    import hashlib

    images = []
    seen_hashes: set = set()

    def _add_plot(path: Path):
        data = path.read_bytes()
        h = hashlib.sha256(data).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            images.append(base64.b64encode(data).decode())

    # Multi-plot pattern: plot_{step_idx}_{sub}.png
    sub_plots = sorted(
        output_dir.glob(f"plot_{step_idx}_*.png"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0
    )
    for plot_path in sub_plots:
        _add_plot(plot_path)

    # Legacy fallback: plot_{step_idx}.png
    if not images:
        legacy = output_dir / f"plot_{step_idx}.png"
        if legacy.exists():
            _add_plot(legacy)

    return images


def _run_code(code: str, df: pd.DataFrame, step_idx: int,
              output_dir: Path) -> tuple:
    """
    Execute code in an isolated context. Returns (text_output, error_traceback_or_None).
    Captures stdout. Saves working dir correctly.
    """
    old_stdout   = sys.stdout
    sys.stdout   = io.StringIO()
    original_dir = os.getcwd()

    try:
        exec_globals = {
            "df":           df.copy(),
            "pd":           pd,
            "step_idx":     step_idx,
            "plot_counter": [0],    # mutable list so code inside exec can mutate it
        }
        os.chdir(output_dir)
        exec(code, exec_globals)            # noqa: S102
        os.chdir(original_dir)
        text_out = sys.stdout.getvalue()
        return text_out, None
    except Exception:
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
        return sys.stdout.getvalue(), traceback.format_exc()
    finally:
        sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# Agent 3: Insights Agent
# ---------------------------------------------------------------------------

def insights_agent(state: EDAState) -> EDAState:
    set_agent_context("Insights Agent")
    print("\n[INSIGHTS] Writing executive summary...")
    state["status"] = "insights"

    results_context = ""
    for r in state["execution_results"]:
        results_context += f"\n--- Step {r['step_number']}: {r['step']} ---\n"
        if r.get("text_output") and r["text_output"].strip():
            results_context += f"Output:\n{r['text_output'][:INSIGHTS_STEP_OUTPUT_CHARS]}\n"
        if r.get("error"):
            results_context += f"Error: {r['error'][:INSIGHTS_ERROR_CHARS]}\n"
        if r.get("images"):
            results_context += "[Visualisation generated]\n"

    system_prompt = """You are a senior data analyst writing a comprehensive EDA report for a technical audience.

Structure your report with these exact markdown sections:
## 🔍 Dataset Overview
## 📋 Column-by-Column Summary
## 📊 Key Statistical Findings
## ⚠️ Data Quality Issues
## 🔗 Relationships & Correlations
## 🚨 Anomalies Detected
## 🔧 Feature Engineering Recommendations
## 💡 Actionable Recommendations
## 🔮 Suggested Next Steps

CRITICAL RULES:
- Reference EVERY column that was analysed — do not skip columns
- Use actual numbers from the EDA outputs (means, counts, percentages, correlations, p-values)
- Column-by-Column Summary: one bullet per column noting its range/values, missingness, skew, and anything notable
- Data Quality Issues: list ALL columns with nulls, outliers, or anomalies with exact counts
- Anomalies Detected: list rows/columns flagged by IQR, Z-score, or Isolation Forest with counts
- Feature Engineering Recommendations:
    * Log/sqrt transform candidates: list skewed numeric columns (|skew| > 1) and suggest transform
    * Encoding strategy per categorical column (one-hot if <10 unique, target/freq encoding if high-cardinality)
    * Datetime feature extraction suggestions (year, month, day_of_week, hour, is_weekend)
    * Binning suggestions for continuous columns where bins would be meaningful
    * Interaction features: mention any strongly correlated pairs that could be ratio/difference features
- Be precise and technical — the audience is data scientists and analysts
- Do NOT say "the analysis shows" or "as we can see" — be direct and assertive
- Each section should be substantive (5-10 sentences or bullets)"""

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": (
            f"Dataset exploration summary:\n{state['df_info']}\n\n"
            f"EDA plan executed:\n{json.dumps(state['eda_plan'], indent=2)}\n\n"
            f"Step results:\n{results_context}\n\n"
            "Write the executive report."
        )}
    ]

    try:
        response          = get_response(history)
        state["insights"] = extract_text(response)
        print("[INSIGHTS] Report ready")
    except Exception as e:
        state["insights"] = f"Could not generate insights: {e}"

    state["status"] = "done"
    return state


# ---------------------------------------------------------------------------
# Chatbot — answers user questions about the data after EDA is done
# ---------------------------------------------------------------------------

def build_chatbot_system_prompt(eda_state: dict, all_cols: list, numeric_cols: list, cat_cols: list) -> str:
    """Build a compact system prompt for the chatbot from EDA context.
    Caps per-step output at 300 chars and insights at 1500 chars to avoid token bloat.
    """
    results_context = ""
    for r in eda_state.get("execution_results", []):
        results_context += f"\n--- Step {r['step_number']}: {r['step']} ---\n"
        if r.get("text_output") and r["text_output"].strip():
            results_context += f"{r['text_output'][:CHATBOT_STEP_OUTPUT_CHARS]}\n"
        if r.get("error"):
            results_context += f"Error: {r['error'][:CHATBOT_ERROR_CHARS]}\n"
        elif not r.get("text_output", "").strip():
            results_context += "[Visualisation generated]\n"

    return f"""You are an expert data analyst chatbot. The user has uploaded a dataset and completed an automated EDA.
You answer questions about the data by COMBINING two sources:
  1. Pre-computed EDA context (below)
  2. Fresh live queries you run using the `query_dataframe` tool

DATASET COLUMN REFERENCE:
All columns   : {all_cols}
Numeric       : {numeric_cols}
Categorical   : {cat_cols}

=== DATASET SUMMARY (from exploration) ===
{eda_state.get('df_info', '')}

=== EDA STEP OUTPUTS ===
{results_context}

=== EXECUTIVE INSIGHTS ===
{eda_state.get('insights', '')}

BEHAVIOR RULES:
1. ALWAYS try to run at least one `query_dataframe` call to get fresh, precise data for the user's question.
   Even if the answer is partially in the context above, a live query gives exact numbers.
2. Combine the live query result WITH the context to give a complete answer.
3. Use ONLY column names from the column reference above — never invent column names.
4. If the user asks something that needs a filter, aggregation, groupby, or specific value lookup — run it.
5. For questions about relationships: run df[cols].corr() or groupby queries live.
6. For questions about specific rows or values: filter and show them.
7. Format your answer clearly: lead with the direct answer, then supporting details.
8. If a query fails, explain what went wrong and try an alternative expression.
9. Never say "I don't have access to run queries" — you DO have the tool, always use it."""


def chat_with_data(user_message: str, chat_history: List[dict], eda_state: dict,
                   df: pd.DataFrame = None, csv_path: str = None) -> tuple:
    """
    Chatbot that ALWAYS attempts live queries + combines with EDA context.
    Returns (reply_text, tool_call_log) where tool_call_log is a list of
    {expression, reason, result} dicts for every query executed this turn.
    """
    if df is not None:
        working_df = df.copy()
    elif csv_path and os.path.exists(csv_path):
        working_df = pd.read_csv(csv_path, encoding='utf-8')
    else:
        return ("I don't have access to the dataset for live queries. "
                "Please re-run the EDA pipeline to restore the session.", [])

    all_cols     = working_df.columns.tolist()
    numeric_cols = working_df.select_dtypes(include='number').columns.tolist()
    cat_cols     = working_df.select_dtypes(include=['object', 'category']).columns.tolist()

    system_prompt = build_chatbot_system_prompt(eda_state, all_cols, numeric_cols, cat_cols)

    history       = [{"role": "system", "content": system_prompt}]
    history.extend(chat_history)
    history.append({"role": "user", "content": user_message})

    set_agent_context("Chatbot")
    tool_call_log = []   # collect every query made this turn

    max_rounds = CHATBOT_MAX_TOOL_ROUNDS
    for _ in range(max_rounds):
        response   = get_response(history, tools=[DATAFRAME_TOOL])
        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            return extract_text(response), tool_call_log

        for call in tool_calls:
            expr   = call["arguments"]["expression"]
            reason = call["arguments"].get("reason", "")
            result = run_df_query(working_df, expr)
            print(f"  [CHAT TOOL] {expr[:70]} -> {result[:80]}...")

            tool_call_log.append({
                "expression": expr,
                "reason":     reason,
                "result":     result,
            })

            history.append({"role": "assistant", "content": f"Running query: {expr}"})
            history.append({"role": "user",      "content": f"Query result:\n{result}"})

    response = get_response(history)
    return extract_text(response) or "I was unable to generate a complete answer.", tool_call_log


# ---------------------------------------------------------------------------
# LangGraph
# ---------------------------------------------------------------------------

def build_eda_graph():
    workflow = StateGraph(EDAState)

    workflow.add_node("data_explorer",  data_explorer_agent)
    workflow.add_node("planner",        planner_agent)
    workflow.add_node("code_gen_exec",  code_gen_exec_agent)   # interleaved generate+execute
    workflow.add_node("insights",       insights_agent)

    workflow.set_entry_point("data_explorer")
    workflow.add_edge("data_explorer", "planner")
    workflow.add_edge("planner",       "code_gen_exec")
    workflow.add_edge("code_gen_exec", "insights")
    workflow.add_edge("insights",      END)

    return workflow.compile()


def run_eda(csv_path: str) -> dict:
    _clear_df_cache()
    _clear_output_dir()
    reset_token_stats()
    _, schema = get_minimal_schema(csv_path)

    initial_state: EDAState = {
        "csv_path":           csv_path,
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

    result = build_eda_graph().invoke(initial_state)

    _log_json("pipeline_complete", **get_token_summary())

    return result


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "sample.csv"
    results  = run_eda(csv_file)
    print(f"\nStatus: {results['status']}")
    print(f"Steps: {len(results['execution_results'])}")
    print(f"Explorer tool calls: {len(results['explorer_log'])}")

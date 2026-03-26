"""
Centralized configuration for EDA Agent.
All tunable values in one place — both App.py and eda_agent.py import from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
FILE_SIZE_LIMIT_MB          = 100   # Max upload file size in megabytes
PREVIEW_ROWS                = 8     # Number of rows shown in the dataset preview table

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ENABLE_LOGGING              = False  # Set to True to write full LLM prompt/response logs
LOG_TOKEN_STATS             = True   # Set to True to write per-call token usage to log file
LOG_FILE                    = Path("logs.log")
JSON_LOG_FILE               = Path("logs.jsonl")
ERROR_LOG_FILE              = Path("errors.jsonl")

# ---------------------------------------------------------------------------
# Self-healing code executor
# ---------------------------------------------------------------------------
MAX_RETRIES                 = 2      # Extra fix attempts per step after initial failure (total = 1 + MAX_RETRIES)

# ---------------------------------------------------------------------------
# Data Explorer
# ---------------------------------------------------------------------------
EXPLORER_MAX_ROUNDS_PER_COL = 3      # Tool-call rounds ceiling = n_cols * this value (min 30)

# ---------------------------------------------------------------------------
# Truncation limits (chars) — caps on text sent to LLM or stored in context
# ---------------------------------------------------------------------------
TOOL_OUTPUT_TRUNCATE        = 2000   # Max chars of a single query_dataframe result
STEP_FINDING_TRUNCATE       = 300    # Max chars of step output stored as a finding for next steps
CHATBOT_STEP_OUTPUT_CHARS   = 800    # Max chars of each step's text_output shown in chatbot context
CHATBOT_ERROR_CHARS         = 300    # Max chars of each step's error shown in chatbot context
INSIGHTS_STEP_OUTPUT_CHARS  = 1500   # Max chars of each step's text_output shown in insights context
INSIGHTS_ERROR_CHARS        = 300    # Max chars of error shown in insights context per step

# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------
CHATBOT_MAX_TOOL_ROUNDS     = 8      # Max query_dataframe rounds per chat turn

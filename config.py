"""Central configuration for the Finesse x Citadel Round 2 portfolio challenge.

Every constraint imposed by the competition guidelines lives here so that the
rest of the codebase reads parameters rather than hardcoding them. The
evaluators re-run this pipeline over a different date range, so nothing
downstream may assume a fixed window.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT / "outputs"

for _d in (DATA_DIR, CACHE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Competition constraints (Section 2 of the guidelines)
# --------------------------------------------------------------------------
STARTING_CAPITAL = 10_000_000.0      # Rs 1 crore
MAX_POSITIONS = 10                   # hard cap on portfolio size
TRANSACTION_COST = 0.001             # 0.1% of traded notional, each way

# In-sample backtest window mandated by the guidelines.
IS_START = "2021-01-01"
IS_END = "2025-12-31"

# Holdout window the organisers will run the submitted model over.
OOS_START = "2026-01-01"
OOS_END = "2026-06-30"

# Internal split of the permitted window. All parameter selection happens on
# the validation slice so that the 2026 holdout stays genuinely untouched:
# every configuration we test against 2026 spends some of its value as
# evidence, so we test against 2024-2025 instead and visit 2026 only to
# confirm a decision already made.
DEV_START = "2021-01-01"
DEV_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2025-12-31"

# Signals need history before the backtest begins (12-month momentum, 1-year
# volatility). We download from here but never evaluate performance on it.
WARMUP_START = "2019-01-01"

# --------------------------------------------------------------------------
# Universe (Section 2: Nifty 100 + Midcap 100 + Smallcap 100)
# --------------------------------------------------------------------------
UNIVERSE_SOURCES = {
    "nifty100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "midcap100": "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
    "smallcap100": "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
}

YF_SUFFIX = ".NS"  # NSE tickers on Yahoo Finance

# --------------------------------------------------------------------------
# Benchmarks (Section 8)
# --------------------------------------------------------------------------
BENCHMARKS = {
    "Nifty 100": "^CNX100",
    "Nifty 500": "^CRSLDX",
}
PRIMARY_BENCHMARK = "Nifty 500"  # spans all three permitted index tiers

# --------------------------------------------------------------------------
# Backtest conventions
# --------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0                 # guidelines specify a 0% risk-free rate

# Minimum fraction of the lookback window a stock must have priced data for
# before it is eligible for selection. Filters newly listed names.
MIN_DATA_COVERAGE = 0.95

# Ignore near-zero rebalance trades whose cost outweighs the benefit.
MIN_TRADE_WEIGHT = 0.001

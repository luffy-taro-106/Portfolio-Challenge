"""Shared evaluation helpers: window slicing, benchmarks, and summary tables.

Both `run_all.py` and `sweep.py` import from here so that a number quoted in
one script means exactly the same thing in the other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src import metrics
from src.backtest import BacktestResult


def slice_window(result: BacktestResult, start: str, end: str) -> metrics.Performance:
    """Metrics for one sub-period of a continuous backtest.

    Trades are attributed to the window in which they were *closed*, and costs
    and turnover to the window in which they were paid.
    """
    equity = result.equity.loc[start:end]
    if equity.empty:
        raise ValueError(f"{result.name}: no equity data in {start}..{end}")

    trades = result.trades
    if not trades.empty:
        exits = pd.to_datetime(trades["exit_date"])
        trades = trades[(exits >= pd.Timestamp(start)) & (exits <= pd.Timestamp(end))]

    notional = (
        float(result.daily_notional.loc[start:end].sum())
        if len(result.daily_notional) else 0.0
    )
    costs = (
        float(result.daily_costs.loc[start:end].sum())
        if len(result.daily_costs) else 0.0
    )
    return metrics.evaluate(equity, trades, notional, costs)


def benchmark_performance(series: pd.Series, start: str, end: str) -> metrics.Performance:
    """Buy-and-hold the index with the same starting capital."""
    prices = series.loc[start:end].dropna()
    equity = prices / prices.iloc[0] * config.STARTING_CAPITAL
    return metrics.evaluate(equity, pd.DataFrame())


def universe_hurdle(prices: pd.DataFrame, start: str, end: str) -> metrics.Performance:
    """Equal-weight the entire eligible universe: the bias-adjusted hurdle.

    Our constituent lists are today's index membership applied backwards, so
    the universe itself has already been filtered for survival. Comparing only
    against the published index would credit that filtering to the strategy.
    This row isolates it: any excess over *this* is attributable to selection
    and sizing rather than to knowing which stocks would still be in the index.
    """
    window = prices.loc[start:end]
    daily = window.pct_change().mean(axis=1).fillna(0.0)
    equity = (1 + daily).cumprod() * config.STARTING_CAPITAL
    return metrics.evaluate(equity, pd.DataFrame())


SUMMARY_FIELDS = {
    "Net PNL (Rs)": lambda p: p.net_pnl,
    "Total Return": lambda p: p.total_return,
    "Ann. Return": lambda p: p.annualised_return,
    "Ann. Vol": lambda p: p.annualised_volatility,
    "Sharpe": lambda p: p.sharpe,
    "Max DD": lambda p: p.max_drawdown,
    "Calmar": lambda p: p.calmar,
    "Trades": lambda p: p.total_trades,
    "Accuracy": lambda p: p.accuracy,
    "Gain/Loss": lambda p: p.gain_to_loss,
    "Turnover (ann.)": lambda p: p.annualised_turnover,
    "Costs (Rs)": lambda p: p.total_costs,
}

_MONEY = ("Net PNL (Rs)", "Costs (Rs)")
_PERCENT = ("Total Return", "Ann. Return", "Ann. Vol", "Max DD", "Accuracy")
_RATIO = ("Sharpe", "Calmar", "Gain/Loss", "Turnover (ann.)")


def summary_table(rows: dict[str, metrics.Performance]) -> pd.DataFrame:
    """Condense a set of Performance objects into the headline comparison."""
    return pd.DataFrame(
        {name: {k: f(p) for k, f in SUMMARY_FIELDS.items()} for name, p in rows.items()}
    ).T


def render(table: pd.DataFrame) -> str:
    """Format a summary table for console output."""
    out = table.copy()
    for column in out.columns:
        if column in _MONEY:
            out[column] = out[column].map(lambda v: f"{v:,.0f}")
        elif column in _PERCENT:
            out[column] = out[column].map(lambda v: f"{v:.2%}")
        elif column in _RATIO:
            out[column] = out[column].map(lambda v: f"{v:.2f}")
        elif column == "Trades":
            out[column] = out[column].map(lambda v: f"{int(v)}")
    return out.to_string()

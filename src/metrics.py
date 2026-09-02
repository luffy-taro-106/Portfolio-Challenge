"""Performance and risk metrics required by Section 7 of the guidelines.

Every metric the report must contain is computed here from two inputs: the
daily equity curve and the realised-trade ledger produced by the backtest
engine. Keeping them in one place means the report and the code cannot drift
apart.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


@dataclass
class Performance:
    """Full metric set for one strategy over one evaluation window."""

    start: pd.Timestamp
    end: pd.Timestamp
    starting_value: float
    final_value: float
    net_pnl: float                 # primary ranking metric (Section 5)
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe: float
    max_drawdown: float
    max_drawdown_date: pd.Timestamp | None
    calmar: float
    total_trades: int
    accuracy: float                # share of round trips closed at a profit
    gain_to_loss: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_holding_days: float
    total_turnover: float          # traded notional / mean equity
    annualised_turnover: float
    total_costs: float
    trades_per_stock: dict[str, int] = field(default_factory=dict)

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop("trades_per_stock")
        return row


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional decline from the running peak, at every point in time."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None]:
    """Largest peak-to-trough decline and the date it bottomed."""
    if equity.empty:
        return 0.0, None
    dd = drawdown_series(equity)
    return float(dd.min()), dd.idxmin()


def annualised_return(equity: pd.Series) -> float:
    """Geometric average return, annualised by trading-day count."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / config.TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def annualised_volatility(returns: pd.Series) -> float:
    """Standard deviation of daily returns, scaled to annual terms."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(config.TRADING_DAYS_PER_YEAR))


def sharpe_ratio(equity: pd.Series, returns: pd.Series) -> float:
    """Annualised return over annualised volatility, at a 0% risk-free rate.

    The guidelines phrase this as "annualised return divided by the standard
    deviation of daily returns". Taken literally that mixes an annual
    numerator with a daily denominator, so we use the standard convention of
    annualising both. This is noted in the report.
    """
    vol = annualised_volatility(returns)
    if vol == 0:
        return 0.0
    return float((annualised_return(equity) - config.RISK_FREE_RATE) / vol)


def evaluate(
    equity: pd.Series,
    trades: pd.DataFrame,
    traded_notional: float = 0.0,
    total_costs: float = 0.0,
) -> Performance:
    """Assemble the full metric set for one equity curve and trade ledger."""
    equity = equity.dropna()
    if equity.empty:
        raise ValueError("Cannot evaluate an empty equity curve")

    returns = equity.pct_change().dropna()
    mdd, mdd_date = max_drawdown(equity)
    ann_ret = annualised_return(equity)
    years = max(len(equity) / config.TRADING_DAYS_PER_YEAR, 1e-9)
    mean_equity = float(equity.mean())

    # Trade-level statistics (Section 7: gain-to-loss, accuracy, trade counts).
    if trades is not None and not trades.empty:
        pnl = trades["pnl"]
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        accuracy = len(wins) / len(pnl)
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        gain_to_loss = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        per_stock = trades["symbol"].value_counts().to_dict()
        stats = dict(
            total_trades=len(pnl),
            accuracy=accuracy,
            gain_to_loss=gain_to_loss,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=float(pnl.max()),
            worst_trade=float(pnl.min()),
            avg_holding_days=float(trades["holding_days"].mean()),
            trades_per_stock=per_stock,
        )
    else:
        stats = dict(
            total_trades=0, accuracy=0.0, gain_to_loss=0.0, avg_win=0.0,
            avg_loss=0.0, best_trade=0.0, worst_trade=0.0,
            avg_holding_days=0.0, trades_per_stock={},
        )

    turnover = traded_notional / mean_equity if mean_equity else 0.0

    return Performance(
        start=equity.index[0],
        end=equity.index[-1],
        starting_value=float(equity.iloc[0]),
        final_value=float(equity.iloc[-1]),
        net_pnl=float(equity.iloc[-1] - equity.iloc[0]),
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        annualised_return=ann_ret,
        annualised_volatility=annualised_volatility(returns),
        sharpe=sharpe_ratio(equity, returns),
        max_drawdown=mdd,
        max_drawdown_date=mdd_date,
        calmar=ann_ret / abs(mdd) if mdd else 0.0,
        total_turnover=turnover,
        annualised_turnover=turnover / years,
        total_costs=total_costs,
        **stats,
    )


def format_report(perf: Performance, title: str = "") -> str:
    """Human-readable metric block for console output."""
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    lines += [
        f"Period              {perf.start.date()} -> {perf.end.date()}",
        f"Starting value      Rs {perf.starting_value:,.0f}",
        f"Final value         Rs {perf.final_value:,.0f}",
        f"Net PNL             Rs {perf.net_pnl:,.0f}",
        f"Total return        {perf.total_return:>8.2%}",
        f"Annualised return   {perf.annualised_return:>8.2%}",
        f"Annualised vol      {perf.annualised_volatility:>8.2%}",
        f"Sharpe ratio        {perf.sharpe:>8.2f}",
        f"Max drawdown        {perf.max_drawdown:>8.2%}"
        + (f"  (trough {perf.max_drawdown_date.date()})" if perf.max_drawdown_date is not None else ""),
        f"Calmar ratio        {perf.calmar:>8.2f}",
        "",
        f"Total trades        {perf.total_trades}",
        f"Accuracy            {perf.accuracy:>8.2%}",
        f"Gain-to-loss        {perf.gain_to_loss:>8.2f}",
        f"Avg win / loss      Rs {perf.avg_win:,.0f} / Rs {perf.avg_loss:,.0f}",
        f"Avg holding days    {perf.avg_holding_days:>8.1f}",
        f"Turnover (ann.)     {perf.annualised_turnover:>8.2f}x",
        f"Total costs paid    Rs {perf.total_costs:,.0f}",
    ]
    return "\n".join(lines)

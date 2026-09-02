"""Event-driven backtest engine with explicit portfolio accounting.

Design decisions that matter for defensibility:

* **No lookahead.** A strategy sees prices only up to and including the signal
  date, and the resulting trades execute at the *next* session's close. This
  mirrors the realistic workflow of computing signals after the close and
  trading the following day.
* **Costs are unavoidable.** Every share that changes hands pays
  `TRANSACTION_COST` on its notional, on both the buy and the sell leg.
* **Trades are round trips.** A realised trade is booked whenever a position
  is reduced, using average-cost basis. This is what feeds the accuracy and
  gain-to-loss statistics required by Section 7.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


class Strategy(Protocol):
    """Anything that can turn a price history into target portfolio weights."""

    name: str

    def generate_weights(
        self,
        history: pd.DataFrame,
        equity: float | None = None,
        holdings: pd.Series | None = None,
    ) -> pd.Series:
        """Target weights keyed by ticker, summing to <= 1.0.

        `history` ends at the signal date, so the strategy is structurally
        prevented from seeing the future. `equity` is the current portfolio
        value, which a strategy needs if its rules depend on order size —
        a tradability screen, for instance, has to know how big the order is.
        `holdings` is the book as it stands, which a rule that treats an
        existing position differently from a new one has to see.
        """
        ...


@dataclass
class Position:
    shares: float = 0.0
    cost_basis: float = 0.0     # average per-share cost, including buy costs
    entry_date: pd.Timestamp | None = None
    peak_price: float = 0.0     # high-water mark since entry, for trailing stops


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    weights: pd.DataFrame           # target weights at each rebalance
    holdings: pd.DataFrame          # end-of-day market value per ticker
    trades: pd.DataFrame            # realised round trips
    traded_notional: float = 0.0
    total_costs: float = 0.0
    rebalance_dates: list = field(default_factory=list)
    daily_notional: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    daily_costs: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def rebalance_schedule(
    calendar: pd.DatetimeIndex, frequency: str = "ME"
) -> pd.DatetimeIndex:
    """Last trading session of each period in the calendar.

    Frequencies follow pandas offset aliases: "ME" monthly, "QE" quarterly,
    "W-FRI" weekly.
    """
    if len(calendar) == 0:
        return pd.DatetimeIndex([])
    grouped = pd.Series(calendar, index=calendar).groupby(
        pd.Grouper(freq=frequency)
    ).max()
    return pd.DatetimeIndex(grouped.dropna().values)


class Backtester:
    """Runs one strategy over one window and returns a full result bundle."""

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
        start: str,
        end: str,
        capital: float = config.STARTING_CAPITAL,
        cost: float = config.TRANSACTION_COST,
        frequency: str = "ME",
        trailing_stop: float | None = None,
    ) -> None:
        self.prices = prices.sort_index()
        # A stock that does not print on a given day still has value. Marking
        # at the last traded price stops a data gap from making a holding
        # vanish from equity and reappear the next session, which shows up as
        # a violent round trip in the curve and wrecks volatility and drawdown.
        # Only past prices are carried forward, so this leaks nothing.
        self.marks = self.prices.ffill()
        self.strategy = strategy
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.capital = capital
        self.cost = cost
        self.frequency = frequency
        # Checked daily between rebalances. Proceeds sit in cash until the next
        # signal date rather than being redeployed immediately, because a stop
        # is a statement that the position was wrong, not that a replacement
        # has been identified.
        self.trailing_stop = trailing_stop

    # -- internals ---------------------------------------------------------

    def _realise(
        self,
        ledger: list,
        symbol: str,
        position: Position,
        shares_sold: float,
        price: float,
        date: pd.Timestamp,
    ) -> None:
        """Book a round trip when a position is reduced."""
        proceeds_per_share = price * (1 - self.cost)
        pnl = (proceeds_per_share - position.cost_basis) * shares_sold
        held = (date - position.entry_date).days if position.entry_date else 0
        ledger.append(
            {
                "symbol": symbol,
                "entry_date": position.entry_date,
                "exit_date": date,
                "shares": shares_sold,
                "entry_price": position.cost_basis,
                "exit_price": proceeds_per_share,
                "pnl": pnl,
                "return": (
                    proceeds_per_share / position.cost_basis - 1.0
                    if position.cost_basis > 0
                    else 0.0
                ),
                "holding_days": held,
            }
        )

    # -- main loop ---------------------------------------------------------

    def run(self) -> BacktestResult:
        window = self.prices.loc[self.start : self.end]
        if window.empty:
            raise ValueError(f"No price data between {self.start} and {self.end}")

        calendar = window.index
        signal_dates = rebalance_schedule(calendar, self.frequency)

        cash = self.capital
        positions: dict[str, Position] = {}
        ledger: list[dict] = []
        equity_curve: dict[pd.Timestamp, float] = {}
        weight_log: dict[pd.Timestamp, pd.Series] = {}
        holdings_log: dict[pd.Timestamp, pd.Series] = {}
        traded_notional = 0.0
        total_costs = 0.0
        executed: list[pd.Timestamp] = []
        notional_log: dict[pd.Timestamp, float] = {}
        cost_log: dict[pd.Timestamp, float] = {}

        # Signals computed on date d execute on the following session.
        pending: pd.Series | None = None

        for i, date in enumerate(calendar):
            row = self.prices.loc[date]        # tradeable prices only
            mark = self.marks.loc[date]        # valuation, gaps carried forward

            # 1. Execute any order queued by yesterday's signal.
            if pending is not None:
                equity_now = cash + sum(
                    p.shares * mark.get(sym, np.nan)
                    for sym, p in positions.items()
                    if pd.notna(mark.get(sym, np.nan))
                )

                targets = pending.reindex(
                    sorted(set(pending.index) | set(positions))
                ).fillna(0.0)

                # Sells first so their proceeds fund the buys.
                for leg in ("sell", "buy"):
                    for symbol, target_weight in targets.items():
                        price = row.get(symbol, np.nan)
                        if pd.isna(price) or price <= 0:
                            continue

                        position = positions.get(symbol, Position())
                        target_shares = (target_weight * equity_now) / price
                        delta = target_shares - position.shares

                        if abs(delta * price) < config.MIN_TRADE_WEIGHT * equity_now:
                            continue
                        if (leg == "sell" and delta >= 0) or (leg == "buy" and delta <= 0):
                            continue

                        notional = abs(delta) * price
                        fee = notional * self.cost
                        traded_notional += notional
                        total_costs += fee
                        notional_log[date] = notional_log.get(date, 0.0) + notional
                        cost_log[date] = cost_log.get(date, 0.0) + fee

                        if delta < 0:  # reducing or closing
                            sold = min(-delta, position.shares)
                            if sold > 0:
                                self._realise(ledger, symbol, position, sold, price, date)
                            cash += sold * price * (1 - self.cost)
                            position.shares -= sold
                            if position.shares <= 1e-9:
                                positions.pop(symbol, None)
                            else:
                                positions[symbol] = position
                        else:  # adding or opening
                            spend = delta * price * (1 + self.cost)
                            if spend > cash:  # never lever up on rounding error
                                delta = cash / (price * (1 + self.cost))
                                spend = cash
                            if delta <= 0:
                                continue
                            new_total = position.shares + delta
                            buy_price = price * (1 + self.cost)
                            position.cost_basis = (
                                position.cost_basis * position.shares + buy_price * delta
                            ) / new_total
                            if position.shares <= 1e-9:
                                position.entry_date = date
                                position.peak_price = price
                            else:
                                position.peak_price = max(position.peak_price, price)
                            position.shares = new_total
                            cash -= spend
                            positions[symbol] = position

                executed.append(date)
                pending = None

            # 2. Trailing stops, checked before marking so a stopped-out name
            #    is valued as cash on the day it is sold.
            if self.trailing_stop and positions:
                for symbol in list(positions):
                    position = positions[symbol]
                    reference = mark.get(symbol, np.nan)
                    if pd.isna(reference):
                        continue
                    position.peak_price = max(position.peak_price, float(reference))
                    if position.peak_price <= 0:
                        continue
                    if reference > position.peak_price * (1 - self.trailing_stop):
                        continue

                    price = row.get(symbol, np.nan)
                    if pd.isna(price) or price <= 0:
                        continue        # cannot sell into a session with no print
                    notional = position.shares * price
                    fee = notional * self.cost
                    traded_notional += notional
                    total_costs += fee
                    notional_log[date] = notional_log.get(date, 0.0) + notional
                    cost_log[date] = cost_log.get(date, 0.0) + fee
                    self._realise(ledger, symbol, position, position.shares,
                                  price, date)
                    cash += notional * (1 - self.cost)
                    positions.pop(symbol, None)

            # 3. Mark to market.
            market_value = {
                sym: p.shares * mark[sym]
                for sym, p in positions.items()
                if sym in mark.index and pd.notna(mark[sym])
            }
            equity_curve[date] = cash + sum(market_value.values())
            holdings_log[date] = pd.Series(market_value, dtype=float)

            # 4. Generate tomorrow's order from data through today.
            if date in signal_dates and i + 1 < len(calendar):
                history = self.prices.loc[:date]
                book = pd.Series(market_value, dtype=float)
                if equity_curve[date] > 0:
                    book = book / equity_curve[date]
                weights = self.strategy.generate_weights(
                    history, equity=equity_curve[date], holdings=book)
                if weights is not None and not weights.empty:
                    weights = weights[weights > 0]
                    if weights.sum() > 1.0:
                        weights = weights / weights.sum()
                    pending = weights
                    weight_log[date] = weights

        # Book positions still open at the end as mark-to-market round trips.
        # Without this, a low-turnover strategy would report almost no trades
        # and its accuracy would describe only the names it happened to exit.
        # These bookings do not touch cash or the equity curve.
        final_date = calendar[-1]
        final_row = self.marks.loc[final_date]
        for symbol, position in positions.items():
            price = final_row.get(symbol, np.nan)
            if pd.isna(price) or position.shares <= 0:
                continue
            self._realise(ledger, symbol, position, position.shares, price, final_date)
            ledger[-1]["open_at_end"] = True

        equity = pd.Series(equity_curve).sort_index()
        trades = pd.DataFrame(ledger)
        if not trades.empty and "open_at_end" in trades.columns:
            trades["open_at_end"] = trades["open_at_end"].fillna(False)
        elif not trades.empty:
            trades["open_at_end"] = False
        weights_df = pd.DataFrame(weight_log).T.fillna(0.0)
        holdings_df = pd.DataFrame(holdings_log).T.fillna(0.0)

        return BacktestResult(
            name=self.strategy.name,
            equity=equity,
            weights=weights_df,
            holdings=holdings_df,
            trades=trades,
            traded_notional=traded_notional,
            total_costs=total_costs,
            rebalance_dates=executed,
            daily_notional=pd.Series(notional_log).sort_index(),
            daily_costs=pd.Series(cost_log).sort_index(),
        )
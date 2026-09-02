"""Every number and figure quoted in the report, generated from one backtest.

Run this before writing or checking the report. It restricts itself to the
2021-2025 modelling window so that nothing from the stress-test period can leak
into the submitted figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src import data, metrics, strategies
from src.backtest import Backtester
from src.evaluation import benchmark_performance, slice_window, universe_hurdle

FIGURES = config.OUTPUT_DIR / "figures"
YEARS = [2021, 2022, 2023, 2024, 2025]


def money(value: float) -> str:
    return f"Rs {value:,.0f}"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    prices = data.fetch_prices()
    benchmarks = data.fetch_benchmarks()

    result = Backtester(
        prices=prices,
        strategy=strategies.build("final"),
        start=config.IS_START,
        end=config.IS_END,
    ).run()
    perf = slice_window(result, config.IS_START, config.IS_END)

    print(metrics.format_report(perf, "FINAL STRATEGY  2021-01-01 to 2025-12-31"))
    print(f"\nFinal portfolio value   {money(config.STARTING_CAPITAL + perf.net_pnl)}")
    print(f"Total Net PNL           {money(perf.net_pnl)}")
    print(f"Rebalances executed     {len(result.rebalance_dates)}")

    # -- benchmark and hurdle -------------------------------------------------
    print("\n" + "=" * 70)
    print("BENCHMARKS")
    print("=" * 70)
    rows = {}
    for label, column in [("Nifty 500", "Nifty 500"), ("Nifty 100", "Nifty 100")]:
        if column in benchmarks.columns:
            rows[label] = benchmark_performance(
                benchmarks[column], config.IS_START, config.IS_END)
    rows["Equal-weight universe"] = universe_hurdle(
        prices, config.IS_START, config.IS_END)
    for label, bench in rows.items():
        print(f"{label:<24} PNL {money(bench.net_pnl):>18}  "
              f"ret {bench.total_return:>8.2%}  ann {bench.annualised_return:>7.2%}  "
              f"vol {bench.annualised_volatility:>6.2%}  "
              f"Sharpe {bench.sharpe:>5.2f}  MDD {bench.max_drawdown:>7.2%}")

    # -- composition ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("PORTFOLIO COMPOSITION")
    print("=" * 70)
    last = result.weights.iloc[-1]
    last = last[last > 0].sort_values(ascending=False)
    print(f"Closing portfolio ({result.weights.index[-1].date()}), "
          f"{len(last)} positions:")
    for symbol, weight in last.items():
        print(f"   {symbol:<18} {weight:>7.2%}")

    counts = result.weights.gt(0).sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    print(f"\nDistinct stocks held over the period: {len(counts)}")
    print(f"Median months held per stock:          {counts.median():.0f}")
    print("Most-held names: " + ", ".join(
        f"{s.replace('.NS','')} ({n})" for s, n in counts.head(8).items()))

    per_stock = pd.Series(perf.trades_per_stock).sort_values(ascending=False)
    print(f"\nTrades: {perf.total_trades} round trips across {len(per_stock)} stocks "
          f"({perf.total_trades / max(len(per_stock), 1):.1f} per stock)")
    print(f"Average holding period: {perf.avg_holding_days:.0f} days")
    print(f"Turnover: {perf.annualised_turnover:.2f}x per year, "
          f"costs {money(perf.total_costs)} "
          f"({perf.total_costs / perf.net_pnl:.2%} of net PNL)")

    # -- yearly ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CALENDAR YEARS")
    print("=" * 70)
    for year in YEARS:
        annual = slice_window(result, f"{year}-01-01", f"{year}-12-31")
        hurdle = universe_hurdle(prices, f"{year}-01-01", f"{year}-12-31")
        bench = benchmark_performance(
            benchmarks[config.PRIMARY_BENCHMARK], f"{year}-01-01", f"{year}-12-31")
        print(f"{year}   strategy {annual.total_return:>8.2%}   "
              f"Nifty 500 {bench.total_return:>8.2%}   "
              f"equal-weight universe {hurdle.total_return:>8.2%}   "
              f"MDD {annual.max_drawdown:>7.2%}")

    # -- figures --------------------------------------------------------------
    equity = result.equity
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    top.plot(equity.index, equity / 1e7, label="Strategy", linewidth=1.8,
             color="#1a3f6b")
    for label, column, colour in [("Nifty 500", "Nifty 500", "#b0562a"),
                                  ("Nifty 100", "Nifty 100", "#8a8a8a")]:
        if column not in benchmarks.columns:
            continue
        series = benchmarks[column].loc[config.IS_START:config.IS_END].dropna()
        top.plot(series.index, series / series.iloc[0], label=label,
                 linewidth=1.2, color=colour)
    hurdle_daily = prices.loc[config.IS_START:config.IS_END].pct_change().mean(axis=1)
    top.plot(equity.index, (1 + hurdle_daily.fillna(0)).cumprod(),
             label="Equal-weight universe", linewidth=1.1,
             color="#4c8055", linestyle="--")

    top.set_yscale("log")
    top.set_ylabel("Growth of Rs 1 crore (log scale)")
    top.set_title("Portfolio value against benchmarks, 2021-2025")
    top.legend(frameon=False, loc="upper left")
    top.grid(alpha=0.25, which="both")

    drawdown = metrics.drawdown_series(equity)
    bottom.fill_between(drawdown.index, drawdown * 100, 0,
                        color="#b0562a", alpha=0.35)
    bottom.set_ylabel("Drawdown (%)")
    bottom.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES / "equity_curve.png", dpi=170)

    # Annual returns against the benchmark.
    fig2, axis = plt.subplots(figsize=(9, 3.2))
    strat = [slice_window(result, f"{y}-01-01", f"{y}-12-31").total_return * 100
             for y in YEARS]
    bench = [benchmark_performance(benchmarks[config.PRIMARY_BENCHMARK],
                                   f"{y}-01-01", f"{y}-12-31").total_return * 100
             for y in YEARS]
    offset = pd.Series(range(len(YEARS)))
    axis.bar(offset - 0.2, strat, width=0.4, label="Strategy", color="#1a3f6b")
    axis.bar(offset + 0.2, bench, width=0.4, label="Nifty 500", color="#b0562a")
    axis.set_xticks(range(len(YEARS)))
    axis.set_xticklabels(YEARS)
    axis.set_ylabel("Return (%)")
    axis.set_title("Calendar-year returns")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25, axis="y")
    fig2.tight_layout()
    fig2.savefig(FIGURES / "annual_returns.png", dpi=170)

    print(f"\nFigures written to {FIGURES}")


if __name__ == "__main__":
    main()

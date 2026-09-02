"""Run every candidate strategy and compare them across both evaluation windows.

Each strategy is backtested once, continuously, from the start of the mandated
in-sample period through the end of the holdout. The resulting equity curve is
then sliced so that in-sample and out-of-sample performance can be read side by
side. Running continuously rather than restarting in 2026 is deliberate: it is
what actually happens to a live portfolio, and it stops a strategy from being
flattered by a convenient fresh start.

Usage:
    python scripts/run_all.py                    # all strategies, monthly
    python scripts/run_all.py --freq QE          # quarterly rebalancing
    python scripts/run_all.py --strategies momentum ml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src import data, metrics, strategies
from src.backtest import Backtester, BacktestResult
from src.evaluation import (
    benchmark_performance,
    render,
    slice_window,
    summary_table,
    universe_hurdle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq", default="ME", help="rebalance frequency (ME, QE, W-FRI)")
    parser.add_argument("--strategies", nargs="*", default=list(strategies.REGISTRY))
    parser.add_argument("--regime", type=int, default=None,
                        help="hold cash when the universe index is below this moving average")
    parser.add_argument("--positions", type=int, default=config.MAX_POSITIONS)
    parser.add_argument("--refresh", action="store_true", help="re-download data")
    args = parser.parse_args()

    print("Loading data...")
    prices = data.fetch_prices(refresh=args.refresh)
    benchmarks = data.fetch_benchmarks(refresh=args.refresh)
    print(f"  {prices.shape[1]} symbols, {prices.shape[0]} sessions "
          f"({prices.index.min().date()} -> {prices.index.max().date()})\n")

    results: dict[str, BacktestResult] = {}
    in_sample: dict[str, metrics.Performance] = {}
    out_sample: dict[str, metrics.Performance] = {}

    for key in args.strategies:
        strategy = strategies.build(
            key, n_positions=args.positions, regime_filter=args.regime
        )
        print(f"Running {strategy.name} ...", end=" ", flush=True)
        started = time.time()

        engine = Backtester(
            prices=prices,
            strategy=strategy,
            start=config.IS_START,
            end=config.OOS_END,
            frequency=args.freq,
        )
        result = engine.run()
        results[strategy.name] = result

        in_sample[strategy.name] = slice_window(result, config.IS_START, config.IS_END)
        out_sample[strategy.name] = slice_window(result, config.OOS_START, config.OOS_END)
        print(f"done in {time.time() - started:.1f}s")

    for name, series in benchmarks.items():
        label = f"Benchmark: {name}"
        in_sample[label] = benchmark_performance(series, config.IS_START, config.IS_END)
        out_sample[label] = benchmark_performance(series, config.OOS_START, config.OOS_END)

    hurdle = "Hurdle: Equal-Weight Universe"
    in_sample[hurdle] = universe_hurdle(prices, config.IS_START, config.IS_END)
    out_sample[hurdle] = universe_hurdle(prices, config.OOS_START, config.OOS_END)

    is_table = summary_table(in_sample)
    oos_table = summary_table(out_sample)

    print("\n" + "=" * 100)
    print(f"IN-SAMPLE  {config.IS_START} -> {config.IS_END}   (ranking metric: Net PNL)")
    print("=" * 100)
    print(render(is_table.sort_values("Net PNL (Rs)", ascending=False)))

    print("\n" + "=" * 100)
    print(f"OUT-OF-SAMPLE  {config.OOS_START} -> {config.OOS_END}   (holdout, not tuned on)")
    print("=" * 100)
    print(render(oos_table.sort_values("Net PNL (Rs)", ascending=False)))

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    is_table.to_csv(config.OUTPUT_DIR / "metrics_in_sample.csv")
    oos_table.to_csv(config.OUTPUT_DIR / "metrics_out_of_sample.csv")

    equity = pd.DataFrame({name: r.equity for name, r in results.items()})
    equity.to_csv(config.OUTPUT_DIR / "equity_curves.csv")
    for name, result in results.items():
        if not result.trades.empty:
            safe = name.split(":")[0].strip().lower()
            result.trades.to_csv(config.OUTPUT_DIR / f"trades_{safe}.csv", index=False)

    print(f"\nWritten to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Parameter sweep, selected on a validation window that is not the holdout.

The mandated backtest period is split into a development slice (2021-2023) and
a validation slice (2024-2025). Every configuration is run continuously across
both, then scored on validation. The 2026 holdout is never read here.

The reason for the discipline: each configuration we evaluate against 2026
spends a little of that window's value as independent evidence. Sweeping 40
configurations against it would leave nothing behind. Sweeping against
2024-2025 instead keeps 2026 available for a single confirmation run at the
end, via `--confirm`.

Usage:
    python scripts/sweep.py                  # full sweep, validation only
    python scripts/sweep.py --top 15
    python scripts/sweep.py --confirm 3      # then check the top 3 on 2026
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
from src import data, strategies
from src.backtest import Backtester
from src.evaluation import benchmark_performance, render, slice_window, summary_table
from src.strategies import BASELINE_FACTORS, FactorStrategy, MLStrategy

# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

FACTOR_VARIANTS = {
    "multifactor": BASELINE_FACTORS["multifactor"],
    "momentum": BASELINE_FACTORS["momentum"],
    "riskweighted": BASELINE_FACTORS["riskweighted"],
    # Momentum with an explicit volatility brake, to test whether the low-vol
    # sleeve is what carries the blend.
    "mom+lowvol": {"momentum_12_1": 0.6, "low_volatility": 0.4},
    # Shorter formation period: does the edge live in 6 months or 12?
    "mom6": {"momentum_6_1": 1.0},
    # Pure risk-adjusted trend, no raw momentum term.
    "quality": {"risk_adj_momentum": 0.5, "trend_consistency": 0.3, "low_drawdown": 0.2},
}


# Round two. The broad sweep established three things: ten positions beat five,
# inverse-volatility weighting beats score weighting, and the six-month
# formation period controls drawdown far better than the twelve-month one.
# These variants probe the remaining question — whether blending the shorter
# momentum with the defensive factors keeps mom6's drawdown while recovering
# the multi-factor blend's return.
REFINE_VARIANTS = {
    "mom6": {"momentum_6_1": 1.0},
    "mom6+lowvol": {"momentum_6_1": 0.60, "low_volatility": 0.40},
    "mom6+trend": {"momentum_6_1": 0.50, "trend_consistency": 0.30, "low_drawdown": 0.20},
    "mom6+mom12": {"momentum_6_1": 0.50, "momentum_12_1": 0.50},
    "blend6": {
        "momentum_6_1": 0.35,
        "risk_adj_momentum": 0.20,
        "trend_consistency": 0.20,
        "low_volatility": 0.15,
        "low_drawdown": 0.10,
    },
    "multifactor": BASELINE_FACTORS["multifactor"],
}


def build_refine_grid() -> list[dict]:
    """Focused second pass: vary the blend and the regime-filter length only."""
    grid = []
    for factor_name, weights in REFINE_VARIANTS.items():
        for weighting in ("equal", "invvol"):
            for regime in (None, 100, 150, 200):
                label = f"{factor_name}/{weighting}{f'/ma{regime}' if regime else ''}"
                grid.append(
                    {
                        "label": label,
                        "kind": "factor",
                        "kwargs": dict(
                            name=label,
                            factor_weights=weights,
                            n_positions=config.MAX_POSITIONS,
                            weighting=weighting,
                            regime_filter=regime,
                        ),
                        "freq": "ME",
                    }
                )
    return grid


def build_grid() -> list[dict]:
    """Every configuration to test, as label plus constructor arguments."""
    grid: list[dict] = []

    for factor_name, weights in FACTOR_VARIANTS.items():
        for weighting in ("equal", "score", "invvol"):
            for n in (5, 10):
                for regime in (None, 200):
                    label = (
                        f"{factor_name}/{weighting}/n{n}"
                        f"{'/regime' if regime else ''}"
                    )
                    grid.append(
                        {
                            "label": label,
                            "kind": "factor",
                            "kwargs": dict(
                                name=label,
                                factor_weights=weights,
                                n_positions=n,
                                weighting=weighting,
                                regime_filter=regime,
                            ),
                            "freq": "ME",
                        }
                    )

    for n in (5, 10):
        for regime in (None, 200):
            for leaves in (7, 15):
                label = f"ml/leaves{leaves}/n{n}{'/regime' if regime else ''}"
                grid.append(
                    {
                        "label": label,
                        "kind": "ml",
                        "kwargs": dict(
                            name=label,
                            n_positions=n,
                            num_leaves=leaves,
                            regime_filter=regime,
                        ),
                        "freq": "ME",
                    }
                )

    return grid


def instantiate(spec: dict):
    return (
        MLStrategy(**spec["kwargs"])
        if spec["kind"] == "ml"
        else FactorStrategy(**spec["kwargs"])
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def robustness_score(dev, val) -> float:
    """Reward configurations that work in both slices, not just the better one.

    Taking the weaker of the two annualised returns and charging for drawdown
    penalises anything that owes its record to a single favourable stretch,
    which is the failure mode the holdout will punish.
    """
    weaker = min(dev.annualised_return, val.annualised_return)
    worst_dd = min(dev.max_drawdown, val.max_drawdown)
    return weaker + 0.5 * worst_dd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=15, help="rows to display")
    parser.add_argument("--refine", action="store_true",
                        help="focused second-pass grid instead of the broad sweep")
    parser.add_argument("--confirm", type=int, default=0,
                        help="after selecting, run this many top configs on the 2026 holdout")
    args = parser.parse_args()

    print("Loading data...")
    prices = data.fetch_prices()
    benchmarks = data.fetch_benchmarks()
    grid = build_refine_grid() if args.refine else build_grid()
    print(f"  {prices.shape[1]} symbols, {len(grid)} configurations\n")

    records = []
    started = time.time()

    for i, spec in enumerate(grid, 1):
        strategy = instantiate(spec)
        engine = Backtester(
            prices=prices,
            strategy=strategy,
            start=config.DEV_START,
            end=config.VAL_END,
            frequency=spec["freq"],
        )
        result = engine.run()

        dev = slice_window(result, config.DEV_START, config.DEV_END)
        val = slice_window(result, config.VAL_START, config.VAL_END)

        records.append(
            {
                "config": spec["label"],
                "score": robustness_score(dev, val),
                "dev_ann": dev.annualised_return,
                "dev_sharpe": dev.sharpe,
                "dev_dd": dev.max_drawdown,
                "val_ann": val.annualised_return,
                "val_sharpe": val.sharpe,
                "val_dd": val.max_drawdown,
                "val_calmar": val.calmar,
                "turnover": val.annualised_turnover,
                "trades": val.total_trades,
            }
        )
        print(f"\r  [{i}/{len(grid)}] {spec['label']:<38}", end="", flush=True)

    print(f"\r  swept {len(grid)} configurations in {time.time() - started:.0f}s"
          f"{' ' * 30}\n")

    table = pd.DataFrame(records).set_index("config").sort_values("score", ascending=False)

    display = table.head(args.top).copy()
    for column in ("dev_ann", "dev_dd", "val_ann", "val_dd", "score"):
        display[column] = display[column].map(lambda v: f"{v:.2%}")
    for column in ("dev_sharpe", "val_sharpe", "val_calmar", "turnover"):
        display[column] = display[column].map(lambda v: f"{v:.2f}")

    print("=" * 118)
    print(f"TOP {args.top} BY ROBUSTNESS SCORE   "
          f"dev {config.DEV_START}..{config.DEV_END}   "
          f"val {config.VAL_START}..{config.VAL_END}")
    print("=" * 118)
    print(display.to_string())

    print("\nBenchmarks over the validation slice:")
    for name, series in benchmarks.items():
        perf = benchmark_performance(series, config.VAL_START, config.VAL_END)
        print(f"  {name:<12} ann {perf.annualised_return:>7.2%}   "
              f"sharpe {perf.sharpe:>5.2f}   maxDD {perf.max_drawdown:>7.2%}")

    destination = config.OUTPUT_DIR / (
        "sweep_refine.csv" if args.refine else "sweep_validation.csv"
    )
    table.to_csv(destination)
    print(f"\nFull results: {destination}")

    if args.confirm:
        print("\n" + "=" * 118)
        print(f"HOLDOUT CONFIRMATION  {config.OOS_START}..{config.OOS_END}  "
              f"(top {args.confirm} only, chosen without reference to this window)")
        print("=" * 118)

        by_label = {spec["label"]: spec for spec in grid}
        confirmed = {}
        for label in table.head(args.confirm).index:
            strategy = instantiate(by_label[label])
            engine = Backtester(
                prices=prices, strategy=strategy,
                start=config.IS_START, end=config.OOS_END,
                frequency=by_label[label]["freq"],
            )
            result = engine.run()
            confirmed[f"{label} [IS]"] = slice_window(result, config.IS_START, config.IS_END)
            confirmed[f"{label} [holdout]"] = slice_window(result, config.OOS_START, config.OOS_END)

        print(render(summary_table(confirmed)))


if __name__ == "__main__":
    main()

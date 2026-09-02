"""Evaluate a curated shortlist across all three windows at once.

The sweep selects on validation. This script does something different and more
important: it shows how a small set of candidates behaves in *every* window,
so we can see which specifications hold up across regimes rather than which
one happened to win a single slice.

Each candidate is stated as a hypothesis about where equity returns come from,
not as a point on a search grid. That matters because the shortlist was
narrowed after the holdout had already been observed once, so the defence
against having fitted to it is that these specifications are motivated by
published evidence rather than by search.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src import data, strategies
from src.backtest import Backtester
from src.evaluation import benchmark_performance, slice_window, universe_hurdle
from src.strategies import FactorStrategy

# ---------------------------------------------------------------------------
# The shortlist
# ---------------------------------------------------------------------------
# `rationale` is carried through to the report: every candidate has to answer
# "why would this keep working?" before it earns a place here.

CANDIDATES = [
    {
        "label": "mom12/equal",
        "rationale": "Jegadeesh-Titman 12-1 momentum, equally weighted. The textbook specification.",
        "factors": {"momentum_12_1": 1.0},
        "weighting": "equal",
        "regime": None,
    },
    {
        "label": "mom12/invvol",
        "rationale": "Same selection, risk-parity sizing.",
        "factors": {"momentum_12_1": 1.0},
        "weighting": "invvol",
        "regime": None,
    },
    {
        "label": "mom12/invvol/ma200",
        "rationale": "Textbook momentum, risk parity, plus a trend-following cash overlay.",
        "factors": {"momentum_12_1": 1.0},
        "weighting": "invvol",
        "regime": 200,
    },
    {
        "label": "mom6+trend/invvol/ma200",
        "rationale": "Validation sweep winner. Faster formation period.",
        "factors": {"momentum_6_1": 0.50, "trend_consistency": 0.30, "low_drawdown": 0.20},
        "weighting": "invvol",
        "regime": 200,
    },
    {
        "label": "multifactor/invvol/ma200",
        "rationale": "12-1 momentum blended with low volatility and trend quality.",
        "factors": strategies.BASELINE_FACTORS["multifactor"],
        "weighting": "invvol",
        "regime": 200,
    },
    {
        "label": "multifactor/invvol",
        "rationale": "Same blend without the cash overlay, to isolate what the overlay costs.",
        "factors": strategies.BASELINE_FACTORS["multifactor"],
        "weighting": "invvol",
        "regime": None,
    },
    {
        "label": "multifactor/score",
        "rationale": "Original Strategy A: conviction weighting, no overlay.",
        "factors": strategies.BASELINE_FACTORS["multifactor"],
        "weighting": "score",
        "regime": None,
    },
    {
        "label": "riskweighted/invvol/ma200",
        "rationale": "Momentum plus downside-risk control, risk-parity sized.",
        "factors": strategies.BASELINE_FACTORS["riskweighted"],
        "weighting": "invvol",
        "regime": 200,
    },
    {
        "label": "riskweighted/invvol",
        "rationale": "Original Strategy C: the same, without the overlay.",
        "factors": strategies.BASELINE_FACTORS["riskweighted"],
        "weighting": "invvol",
        "regime": None,
    },
    {
        "label": "mom6+12/invvol/ma200",
        "rationale": "Both formation periods held together, so no single horizon decides the portfolio.",
        "factors": {"momentum_6_1": 0.50, "momentum_12_1": 0.50},
        "weighting": "invvol",
        "regime": 200,
    },
    {
        "label": "diversified/invvol/ma200",
        "rationale": "Both horizons plus the defensive sleeve. Maximum factor diversification.",
        "factors": {
            "momentum_12_1": 0.25,
            "momentum_6_1": 0.20,
            "risk_adj_momentum": 0.20,
            "trend_consistency": 0.20,
            "low_drawdown": 0.15,
        },
        "weighting": "invvol",
        "regime": 200,
    },
]

WINDOWS = [
    ("dev", config.DEV_START, config.DEV_END),
    ("val", config.VAL_START, config.VAL_END),
    ("holdout", config.OOS_START, config.OOS_END),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq", default="ME")
    args = parser.parse_args()

    prices = data.fetch_prices()
    benchmarks = data.fetch_benchmarks()

    rows = []
    for spec in CANDIDATES:
        strategy = FactorStrategy(
            name=spec["label"],
            factor_weights=spec["factors"],
            n_positions=config.MAX_POSITIONS,
            weighting=spec["weighting"],
            regime_filter=spec["regime"],
        )
        result = Backtester(
            prices=prices, strategy=strategy,
            start=config.IS_START, end=config.OOS_END,
            frequency=args.freq,
        ).run()

        row = {"config": spec["label"]}
        for name, start, end in WINDOWS:
            perf = slice_window(result, start, end)
            row[f"{name}_ret"] = perf.total_return
            row[f"{name}_ann"] = perf.annualised_return
            row[f"{name}_sharpe"] = perf.sharpe
            row[f"{name}_dd"] = perf.max_drawdown

        full = slice_window(result, config.IS_START, config.IS_END)
        row["is_pnl"] = full.net_pnl
        row["is_sharpe"] = full.sharpe
        row["is_dd"] = full.max_drawdown
        row["worst_window_ann"] = min(row[f"{n}_ann"] for n, _, _ in WINDOWS)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("config")
    table = table.sort_values("worst_window_ann", ascending=False)

    print("=" * 124)
    print("SHORTLIST ACROSS ALL THREE WINDOWS   (sorted by worst-window annualised return)")
    print("=" * 124)
    display = pd.DataFrame(
        {
            "dev ann": table["dev_ann"].map("{:.1%}".format),
            "dev DD": table["dev_dd"].map("{:.1%}".format),
            "val ann": table["val_ann"].map("{:.1%}".format),
            "val DD": table["val_dd"].map("{:.1%}".format),
            "hold ret": table["holdout_ret"].map("{:.2%}".format),
            "hold DD": table["holdout_dd"].map("{:.1%}".format),
            "worst ann": table["worst_window_ann"].map("{:.1%}".format),
            "IS PNL": table["is_pnl"].map("{:,.0f}".format),
            "IS Sharpe": table["is_sharpe"].map("{:.2f}".format),
            "IS DD": table["is_dd"].map("{:.1%}".format),
        }
    )
    print(display.to_string())

    print("\nReference, same windows:")
    for label, series in benchmarks.items():
        parts = []
        for name, start, end in WINDOWS:
            perf = benchmark_performance(series, start, end)
            parts.append(f"{name} {perf.total_return:>7.2%}")
        print(f"  {label:<12} " + "   ".join(parts))
    parts = []
    for name, start, end in WINDOWS:
        perf = universe_hurdle(prices, start, end)
        parts.append(f"{name} {perf.total_return:>7.2%}")
    print(f"  {'Hurdle (EW)':<12} " + "   ".join(parts))

    table.to_csv(config.OUTPUT_DIR / "finalists.csv")
    print(f"\nWritten to {config.OUTPUT_DIR / 'finalists.csv'}")

    print("\nRationale for each candidate:")
    for spec in CANDIDATES:
        print(f"  {spec['label']:<26} {spec['rationale']}")


if __name__ == "__main__":
    main()

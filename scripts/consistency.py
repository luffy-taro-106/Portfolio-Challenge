"""Year-by-year robustness check for candidate strategies.

The single 2024-25 validation slice misled us badly once: it crowned a
six-month momentum blend that then lost 13.5% on the holdout. The lesson is
that one contiguous slice cannot separate a durable edge from a regime that
happened to last two years.

This script scores candidates on how they behave in *every* calendar year of
the mandated period. A strategy that clears the bias-adjusted hurdle in five
years out of five is making a very different claim from one that owes its
record to 2023. The headline ranking is by worst calendar year, because that
is the statistic a stress test actually probes.

Usage:
    python scripts/consistency.py
    python scripts/consistency.py --holdout   # also show Jan-Jun 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src import data, features, strategies
from src.backtest import Backtester
from src.evaluation import slice_window, universe_hurdle
from src.strategies import FactorStrategy

MF = strategies.BASELINE_FACTORS["multifactor"]

# Each candidate states a hypothesis about where returns come from. The three
# premia in play are momentum (price continuation), trend quality (risk-adjusted
# continuation) and price impact / size (compensation for trading friction).
# They are close to uncorrelated, which is the case for blending them.
CANDIDATES = {
    "FINAL: size-tilted, screened": dict(
        factors=strategies.FINAL_FACTORS, weighting="invvol",
        max_participation=strategies.MAX_PARTICIPATION),
    "final, no liquidity screen": dict(
        factors=strategies.FINAL_FACTORS, weighting="invvol"),
    "multifactor/invvol (no tilt)": dict(factors=MF, weighting="invvol"),
    "impact only": dict(factors={"price_impact": 1.0}, weighting="invvol"),
    "size only": dict(factors={"small_size": 1.0}, weighting="invvol"),
    "impact+mom12": dict(
        factors={"price_impact": 0.5, "momentum_12_1": 0.5}, weighting="invvol"),
    "impact+ram": dict(
        factors={"price_impact": 0.5, "risk_adj_momentum": 0.5}, weighting="invvol"),
    "multifactor+impact": dict(
        factors={**{k: v * 0.6 for k, v in MF.items()}, "price_impact": 0.4},
        weighting="invvol"),
    "multifactor+size": dict(
        factors={**{k: v * 0.6 for k, v in MF.items()}, "small_size": 0.4},
        weighting="invvol"),
    "three-premia": dict(
        factors={
            "price_impact": 0.30,
            "momentum_12_1": 0.25,
            "risk_adj_momentum": 0.25,
            "trend_consistency": 0.20,
        },
        weighting="invvol"),
    "three-premia+volume": dict(
        factors={
            "price_impact": 0.25,
            "momentum_12_1": 0.20,
            "risk_adj_momentum": 0.20,
            "trend_consistency": 0.20,
            "volume_surge": 0.15,
        },
        weighting="invvol"),
}

YEARS = [2021, 2022, 2023, 2024, 2025]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true",
                        help="also report the 2026 holdout column")
    parser.add_argument("--freq", default="ME")
    parser.add_argument("--tier", default=None,
                        choices=["nifty100", "midcap100", "smallcap100"],
                        help="restrict the tradeable universe to one tier")
    args = parser.parse_args()

    prices = data.fetch_prices()
    features.set_volume(data.fetch_volumes())

    if args.tier:
        # Large caps almost never leave the Nifty 100, so results here are far
        # less flattered by applying today's membership list backwards.
        keep = set(data.fetch_universe().query("tier == @args.tier")["ticker"])
        prices = prices[[c for c in prices.columns if c in keep]]
        features.set_volume(data.fetch_volumes()[prices.columns])
        print(f"Universe restricted to {args.tier}: {prices.shape[1]} symbols\n")

    end = config.OOS_END if args.holdout else config.IS_END
    rows = []

    for label, spec in CANDIDATES.items():
        strategy = FactorStrategy(
            name=label,
            factor_weights=spec["factors"],
            n_positions=config.MAX_POSITIONS,
            weighting=spec["weighting"],
            max_participation=spec.get("max_participation"),
        )
        result = Backtester(
            prices=prices, strategy=strategy,
            start=config.IS_START, end=end, frequency=args.freq,
        ).run()

        row = {"strategy": label}
        for year in YEARS:
            perf = slice_window(result, f"{year}-01-01", f"{year}-12-31")
            row[str(year)] = perf.total_return

        full = slice_window(result, config.IS_START, config.IS_END)
        row["IS PNL"] = full.net_pnl
        row["Sharpe"] = full.sharpe
        row["MaxDD"] = full.max_drawdown
        row["worst year"] = min(row[str(y)] for y in YEARS)
        if args.holdout:
            row["holdout"] = slice_window(
                result, config.OOS_START, config.OOS_END).total_return
        rows.append(row)

    # The hurdle is the bar that matters: beating the index is easy when the
    # universe itself is survivorship-selected.
    hurdle = {"strategy": "-- hurdle: equal-weight universe --"}
    for year in YEARS:
        hurdle[str(year)] = universe_hurdle(
            prices, f"{year}-01-01", f"{year}-12-31").total_return
    hurdle["worst year"] = min(hurdle[str(y)] for y in YEARS)
    hurdle["IS PNL"] = universe_hurdle(prices, config.IS_START, config.IS_END).net_pnl
    hurdle["Sharpe"] = universe_hurdle(prices, config.IS_START, config.IS_END).sharpe
    hurdle["MaxDD"] = universe_hurdle(prices, config.IS_START, config.IS_END).max_drawdown
    if args.holdout:
        hurdle["holdout"] = universe_hurdle(
            prices, config.OOS_START, config.OOS_END).total_return
    rows.append(hurdle)

    table = pd.DataFrame(rows).set_index("strategy").sort_values(
        "worst year", ascending=False)

    beats = {}
    for label in table.index:
        if label.startswith("--"):
            continue
        beats[label] = sum(
            table.loc[label, str(y)] > hurdle[str(y)] for y in YEARS)

    display = table.copy()
    for column in [str(y) for y in YEARS] + ["worst year", "MaxDD"]:
        display[column] = display[column].map("{:+.1%}".format)
    if args.holdout:
        display["holdout"] = display["holdout"].map("{:+.2%}".format)
    display["IS PNL"] = display["IS PNL"].map(lambda v: f"{v / 1e7:.2f}Cr")
    display["Sharpe"] = display["Sharpe"].map("{:.2f}".format)
    display["beats hurdle"] = pd.Series(beats).map(lambda v: f"{v}/5")

    print("=" * 132)
    print(f"CALENDAR-YEAR CONSISTENCY   sorted by worst year   "
          f"(rebalance {args.freq}, {config.MAX_POSITIONS} positions)")
    print("=" * 132)
    print(display.to_string())

    table.to_csv(config.OUTPUT_DIR / "consistency.csv")
    print(f"\nWritten to {config.OUTPUT_DIR / 'consistency.csv'}")


if __name__ == "__main__":
    main()

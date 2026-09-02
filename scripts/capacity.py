"""Could this book actually be traded at the size the backtest assumes?

The blend earns much of its edge from a size and illiquidity tilt, which means
it deliberately buys the names that are hardest to trade. Compounding makes this
worse over time: the same 10% weight is a far larger order in 2025 than it was
in 2021, because the capital behind it has grown many times over.

For every position at every rebalance this measures the order as a share of the
name's recent average daily traded value. Roughly speaking, under ~10% of ADV is
comfortable in a day, 10-25% needs working over several sessions, and above that
the fill assumed by the backtest is not credible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src import data, features, strategies
from src.backtest import Backtester
from src.strategies import FactorStrategy

MF = strategies.BASELINE_FACTORS["multifactor"]
ADV_WINDOW = 21
COMFORTABLE = 0.10      # share of ADV tradeable in a single session


def blended(tilt: float) -> dict[str, float]:
    weights = {k: v * (1 - tilt) for k, v in MF.items()}
    weights["small_size"] = tilt
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tilt", type=float, default=0.6)
    parser.add_argument("--capital", type=float, default=config.STARTING_CAPITAL)
    parser.add_argument("--max-participation", type=float, default=None,
                        help="apply the tradability screen at this share of ADV")
    args = parser.parse_args()

    prices = data.fetch_prices()
    volumes = data.fetch_volumes()
    features.set_volume(volumes)

    # DataFrame.asof would only return rows with no NaN in *any* column, which
    # silently discards every date before the last name listed, so look the
    # rebalance date up directly instead.
    traded_value = (
        (prices * volumes.reindex_like(prices)).rolling(ADV_WINDOW).median().ffill()
    )

    strategy = FactorStrategy(
        name=f"multifactor + size {args.tilt:.0%}",
        factor_weights=blended(args.tilt),
        n_positions=config.MAX_POSITIONS,
        weighting="invvol",
        max_participation=args.max_participation,
    )
    result = Backtester(
        prices=prices, strategy=strategy,
        start=config.IS_START, end=config.OOS_END,
        capital=args.capital,
    ).run()

    scale = args.capital / config.STARTING_CAPITAL
    rows = []
    for date, weights in result.weights.iterrows():
        equity = result.equity.asof(date)
        adv = traded_value.loc[date]
        for symbol, weight in weights[weights > 0].items():
            name_adv = adv.get(symbol, np.nan)
            if pd.isna(name_adv) or name_adv <= 0:
                continue
            rows.append({
                "date": date,
                "symbol": symbol,
                "notional": weight * equity,
                "adv": name_adv,
                "participation": weight * equity / name_adv,
            })

    book = pd.DataFrame(rows)
    book["year"] = book["date"].dt.year

    print("=" * 78)
    print(f"CAPACITY   {strategy.name}   starting capital "
          f"{args.capital / 1e7:.1f}Cr ({scale:.0f}x base)")
    print("=" * 78)

    yearly = book.groupby("year").agg(
        median_position=("notional", "median"),
        median_adv=("adv", "median"),
        median_participation=("participation", "median"),
        p90_participation=("participation", lambda s: s.quantile(0.90)),
        worst=("participation", "max"),
    )
    over = book.groupby("year")["participation"].apply(
        lambda s: (s > COMFORTABLE).mean())
    yearly["% over 10% ADV"] = over

    display = yearly.copy()
    display["median_position"] = display["median_position"].map(
        lambda v: f"{v / 1e5:.0f}L")
    display["median_adv"] = display["median_adv"].map(lambda v: f"{v / 1e7:.1f}Cr")
    for column in ("median_participation", "p90_participation", "worst",
                   "% over 10% ADV"):
        display[column] = display[column].map("{:.1%}".format)
    print(display.to_string())

    print(f"\nDays to exit the median position at {COMFORTABLE:.0%} of ADV: "
          f"{book['participation'].median() / COMFORTABLE:.1f}")
    print(f"Days to exit the 90th-percentile position:      "
          f"{book['participation'].quantile(0.90) / COMFORTABLE:.1f}")

    tight = book.nlargest(8, "participation")[
        ["date", "symbol", "notional", "adv", "participation"]]
    tight["date"] = tight["date"].dt.date
    tight["notional"] = tight["notional"].map(lambda v: f"{v / 1e5:.0f}L")
    tight["adv"] = tight["adv"].map(lambda v: f"{v / 1e7:.2f}Cr")
    tight["participation"] = tight["participation"].map("{:.1%}".format)
    print("\nTightest orders:")
    print(tight.to_string(index=False))

    book.to_csv(config.OUTPUT_DIR / "capacity.csv", index=False)


if __name__ == "__main__":
    main()

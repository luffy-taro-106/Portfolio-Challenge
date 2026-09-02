"""How much size/liquidity tilt should the blend carry?

Blending the size sleeve into the multi-factor score improved every metric at
once, which is unusual enough to deserve suspicion. The question this script
answers is whether the improvement sits on a broad plateau — in which case the
exact mix does not matter much and the effect is probably real — or on a narrow
spike, which would mean we had simply found a lucky coordinate.

Ranking is by worst calendar year and by how many years clear the bias-adjusted
hurdle. The holdout column is displayed but is not used for selection.
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
YEARS = [2021, 2022, 2023, 2024, 2025]


def blend(sleeve: dict[str, float], tilt: float) -> dict[str, float]:
    """Mix the multi-factor core with a tilt sleeve at the given weight."""
    combined = {k: v * (1 - tilt) for k, v in MF.items()}
    for name, share in sleeve.items():
        combined[name] = combined.get(name, 0.0) + share * tilt
    return combined


def build_candidates() -> dict[str, dict]:
    candidates: dict[str, dict] = {}

    for tilt in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        candidates[f"size tilt {tilt:.0%}"] = dict(
            factors=blend({"small_size": 1.0}, tilt), weighting="invvol")

    for tilt in (0.3, 0.4, 0.5):
        candidates[f"impact tilt {tilt:.0%}"] = dict(
            factors=blend({"price_impact": 1.0}, tilt), weighting="invvol")
        candidates[f"size+impact tilt {tilt:.0%}"] = dict(
            factors=blend({"small_size": 0.5, "price_impact": 0.5}, tilt),
            weighting="invvol")

    # Does the weighting scheme still matter once the tilt is in place?
    for weighting in ("equal", "score"):
        candidates[f"size tilt 40% / {weighting}"] = dict(
            factors=blend({"small_size": 1.0}, 0.4), weighting=weighting)

    # Volume participation as a third sleeve.
    candidates["size 40% + volume 15%"] = dict(
        factors=blend({"small_size": 0.73, "volume_surge": 0.27}, 0.55),
        weighting="invvol")

    # What does refusing to trade the thinnest names cost? The tilt hunts for
    # illiquidity, so this is the price of making the fills believable. The
    # limit is a share of daily traded value, so it tightens automatically as
    # the book compounds and the same weight becomes a larger order.
    for limit in (0.25, 0.10, 0.05, 0.02):
        candidates[f"size tilt 60% / max {limit:.0%} ADV"] = dict(
            factors=blend({"small_size": 1.0}, 0.6), weighting="invvol",
            max_participation=limit)

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--tier", default=None,
                        choices=["nifty100", "midcap100", "smallcap100"],
                        help="restrict the tradeable universe to one tier")
    args = parser.parse_args()

    prices = data.fetch_prices()
    features.set_volume(data.fetch_volumes())

    if args.tier:
        # A size tilt concentrates the book in exactly the names most likely to
        # have been added to the index because they already did well. Repeating
        # the sweep inside the Nifty 100, where membership barely changes,
        # separates the factor from that bias.
        keep = set(data.fetch_universe().query("tier == @args.tier")["ticker"])
        prices = prices[[c for c in prices.columns if c in keep]]
        features.set_volume(data.fetch_volumes()[prices.columns])
        print(f"Universe restricted to {args.tier}: {prices.shape[1]} symbols\n")

    end = config.OOS_END if args.holdout else config.IS_END

    hurdle_year = {
        y: universe_hurdle(prices, f"{y}-01-01", f"{y}-12-31").total_return
        for y in YEARS
    }

    rows = []
    for label, spec in build_candidates().items():
        strategy = FactorStrategy(
            name=label, factor_weights=spec["factors"],
            n_positions=config.MAX_POSITIONS, weighting=spec["weighting"],
            min_adv=spec.get("min_adv"),
            max_participation=spec.get("max_participation"),
        )
        result = Backtester(
            prices=prices, strategy=strategy,
            start=config.IS_START, end=end,
        ).run()

        yearly = {
            y: slice_window(result, f"{y}-01-01", f"{y}-12-31").total_return
            for y in YEARS
        }
        full = slice_window(result, config.IS_START, config.IS_END)

        row = {
            "config": label,
            "worst year": min(yearly.values()),
            "beats hurdle": sum(yearly[y] > hurdle_year[y] for y in YEARS),
            "IS PNL": full.net_pnl,
            "Sharpe": full.sharpe,
            "MaxDD": full.max_drawdown,
            "turnover": full.annualised_turnover,
        }
        if args.holdout:
            row["holdout"] = slice_window(
                result, config.OOS_START, config.OOS_END).total_return
        rows.append(row)

    table = pd.DataFrame(rows).set_index("config").sort_values(
        ["beats hurdle", "worst year"], ascending=False)

    display = table.copy()
    for column in ("worst year", "MaxDD"):
        display[column] = display[column].map("{:+.1%}".format)
    if args.holdout:
        display["holdout"] = display["holdout"].map("{:+.2%}".format)
    display["IS PNL"] = display["IS PNL"].map(lambda v: f"{v / 1e7:.2f}Cr")
    display["Sharpe"] = display["Sharpe"].map("{:.2f}".format)
    display["turnover"] = display["turnover"].map("{:.1f}x".format)
    display["beats hurdle"] = display["beats hurdle"].map(lambda v: f"{v}/5")

    print("=" * 108)
    print("BLEND SWEEP   sorted by years beating the hurdle, then worst year")
    print("=" * 108)
    print(display.to_string())
    print(f"\nHurdle by year: " +
          "  ".join(f"{y} {hurdle_year[y]:+.1%}" for y in YEARS))

    table.to_csv(config.OUTPUT_DIR / "blend_sweep.csv")


if __name__ == "__main__":
    main()

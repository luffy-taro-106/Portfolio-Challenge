"""Information-coefficient study: which factors actually predict returns?

Sweeping whole strategies is an expensive and noisy way to discover which
signals carry information, because every result confounds the signal with the
weighting scheme, the position count and the rebalance schedule. This script
isolates the signal.

At each month end we rank the cross-section on every factor and correlate that
ranking with the returns that followed (Spearman rank correlation, the
"information coefficient"). A factor with a reliably positive mean IC has
predictive content; one whose IC hovers around zero does not, however good the
backtest of a strategy built on it happens to look.

The reported t-statistic treats each month's IC as one observation, so it
answers the question that matters: is this factor's edge distinguishable from
luck across roughly sixty independent months?

Usage:
    python scripts/factor_ic.py                 # 21-day forward horizon
    python scripts/factor_ic.py --horizon 63    # one quarter ahead
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
from src import data, features
from src.backtest import rebalance_schedule


def compute_ic(
    prices: pd.DataFrame, horizon: int, start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-date IC for every factor, plus the summary across dates."""
    window = prices.loc[start:end]
    signal_dates = rebalance_schedule(window.index, "ME")
    all_dates = prices.index

    records = {}
    for date in signal_dates:
        position = all_dates.get_loc(date)
        if position + horizon >= len(all_dates):
            continue

        history = prices.iloc[: position + 1]
        panel = features.build_factor_panel(history)
        if panel.empty:
            continue

        future = prices.iloc[position + horizon]
        current = prices.iloc[position]
        forward = (future / current - 1.0).reindex(panel.index)

        row = {}
        for column in panel.columns:
            pair = pd.concat([panel[column], forward], axis=1).dropna()
            if len(pair) < 30:
                continue
            row[column] = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
        if row:
            records[date] = row

    per_date = pd.DataFrame(records).T.sort_index()

    n = per_date.notna().sum()
    mean = per_date.mean()
    std = per_date.std(ddof=1)
    summary = pd.DataFrame(
        {
            "mean_ic": mean,
            "std_ic": std,
            "t_stat": mean / (std / np.sqrt(n.replace(0, np.nan))),
            "hit_rate": (per_date > 0).sum() / n.replace(0, np.nan),
            "months": n,
        }
    ).sort_values("t_stat", ascending=False)
    return per_date, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=21,
                        help="forward return horizon in trading days")
    parser.add_argument("--tier", default=None,
                        choices=["nifty100", "midcap100", "smallcap100"],
                        help="restrict to one index tier, as a survivorship-bias check")
    args = parser.parse_args()

    prices = data.fetch_prices()
    features.set_volume(data.fetch_volumes())

    if args.tier:
        # Large caps rarely leave the Nifty 100, so a factor that survives here
        # is far less likely to be an artefact of backward-applied membership.
        universe = data.fetch_universe()
        keep = universe.loc[universe["tier"] == args.tier, "ticker"]
        prices = prices[[c for c in prices.columns if c in set(keep)]]
        print(f"Restricted to {args.tier}: {prices.shape[1]} symbols")

    print(f"Computing information coefficients "
          f"({args.horizon}-day forward returns, {config.IS_START}..{config.IS_END})\n")
    per_date, summary = compute_ic(prices, args.horizon, config.IS_START, config.IS_END)

    display = summary.copy()
    for column in ("mean_ic", "std_ic", "hit_rate"):
        display[column] = display[column].map("{:.3f}".format)
    display["t_stat"] = display["t_stat"].map("{:.2f}".format)
    display["months"] = display["months"].astype(int)

    print("=" * 78)
    print("FACTOR INFORMATION COEFFICIENTS   (higher t-stat = more reliable edge)")
    print("=" * 78)
    print(display.to_string())

    strong = summary[summary["t_stat"].abs() >= 2.0]
    print(f"\n{len(strong)} of {len(summary)} factors clear |t| >= 2.")

    # Stability across calendar years matters more than the full-period average:
    # the holdout failure came from a factor that worked in one regime only.
    print("\n" + "=" * 78)
    print("MEAN IC BY CALENDAR YEAR   (a factor that flips sign is regime-dependent)")
    print("=" * 78)
    yearly = per_date.groupby(per_date.index.year).mean().T
    ordered = yearly.loc[summary.index]
    consistent = (np.sign(ordered).nunique(axis=1) == 1)
    formatted = ordered.map("{:+.3f}".format)
    formatted["all_same_sign"] = consistent.map({True: "yes", False: ""})
    print(formatted.to_string())

    print("\nFactors positive in every single year:")
    for name in ordered.index[(ordered > 0).all(axis=1)]:
        print(f"  {name:<22} mean IC {summary.loc[name, 'mean_ic']:+.3f}   "
              f"t {summary.loc[name, 't_stat']:.2f}")

    summary.to_csv(config.OUTPUT_DIR / f"factor_ic_{args.horizon}d.csv")
    per_date.to_csv(config.OUTPUT_DIR / f"factor_ic_{args.horizon}d_bydate.csv")
    print(f"\nWritten to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

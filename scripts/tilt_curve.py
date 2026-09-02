"""How sensitive is the strategy to the size-tilt weight?

Sweeps the tilt on a fine grid and reports the response curve. The point is not
to read off a maximum — with one five-year history any maximum is partly noise —
but to see whether performance rises and falls smoothly over a wide region, and
whether the good region is the *same* region in the first half of the sample as
in the second. A parameter whose optimum jumps between sub-periods is fitting
noise; one whose optimum is stable is measuring something real.

Reports each window separately, so 2021-2023 and 2024-2025 can be compared
without either being contaminated by the other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def blended(tilt: float) -> dict[str, float]:
    weights = {k: v * (1 - tilt) for k, v in MF.items()}
    if tilt > 0:
        weights["small_size"] = tilt
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default=None,
                        choices=["nifty100", "midcap100", "smallcap100"])
    parser.add_argument("--no-screen", action="store_true",
                        help="drop the liquidity screen to isolate the tilt")
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    prices = data.fetch_prices()
    features.set_volume(data.fetch_volumes())
    if args.tier:
        keep = set(data.fetch_universe().query("tier == @args.tier")["ticker"])
        prices = prices[[c for c in prices.columns if c in keep]]
        features.set_volume(data.fetch_volumes()[prices.columns])
        print(f"Universe restricted to {args.tier}: {prices.shape[1]} symbols\n")

    screen = None if args.no_screen else strategies.MAX_PARTICIPATION
    hurdle_year = {
        y: universe_hurdle(prices, f"{y}-01-01", f"{y}-12-31").total_return
        for y in YEARS
    }

    grid = np.round(np.arange(0.0, 1.0 + 1e-9, args.step), 3)
    rows = []
    for tilt in grid:
        strategy = FactorStrategy(
            name=f"tilt {tilt:.2f}", factor_weights=blended(float(tilt)),
            n_positions=config.MAX_POSITIONS, weighting="invvol",
            max_participation=screen,
        )
        result = Backtester(
            prices=prices, strategy=strategy,
            start=config.IS_START, end=config.IS_END,
        ).run()

        full = slice_window(result, config.IS_START, config.IS_END)
        dev = slice_window(result, config.DEV_START, config.DEV_END)
        val = slice_window(result, config.VAL_START, config.VAL_END)
        yearly = {y: slice_window(result, f"{y}-01-01", f"{y}-12-31").total_return
                  for y in YEARS}

        rows.append({
            "tilt": float(tilt),
            "IS PNL": full.net_pnl,
            "Sharpe": full.sharpe,
            "MaxDD": full.max_drawdown,
            "worst year": min(yearly.values()),
            "beats hurdle": sum(yearly[y] > hurdle_year[y] for y in YEARS),
            "dev ret": dev.total_return,
            "dev Sharpe": dev.sharpe,
            "val ret": val.total_return,
            "val Sharpe": val.sharpe,
            "turnover": full.annualised_turnover,
        })

    table = pd.DataFrame(rows).set_index("tilt")

    label = args.tier or "full universe"
    screen_note = "no liquidity screen" if args.no_screen else "10% ADV screen"
    print("=" * 104)
    print(f"TILT RESPONSE CURVE   {label}, {screen_note}")
    print("=" * 104)

    display = table.copy()
    display["IS PNL"] = display["IS PNL"].map(lambda v: f"{v / 1e7:.2f}Cr")
    for column in ("MaxDD", "worst year", "dev ret", "val ret"):
        display[column] = display[column].map("{:+.1%}".format)
    for column in ("Sharpe", "dev Sharpe", "val Sharpe"):
        display[column] = display[column].map("{:.2f}".format)
    display["turnover"] = display["turnover"].map("{:.1f}x".format)
    display["beats hurdle"] = display["beats hurdle"].map(lambda v: f"{v}/5")
    display.index = [f"{t:.0%}" for t in table.index]
    print(display.to_string())

    # -- where does each criterion peak? --------------------------------------
    print("\n" + "-" * 104)
    print("PEAK LOCATION BY CRITERION")
    print("-" * 104)
    for column in ("IS PNL", "Sharpe", "worst year", "dev ret", "dev Sharpe",
                   "val ret", "val Sharpe"):
        best = table[column].idxmax()
        print(f"  {column:<12} peaks at tilt {best:.0%}")
    print(f"  {'MaxDD':<12} shallowest at tilt {table['MaxDD'].idxmax():.0%}")

    # A parameter worth trusting has a wide band of near-optimal values, not a
    # single spike. Report the span that stays within 10% of the best PNL.
    threshold = table["IS PNL"].max() * 0.90
    plateau = table.index[table["IS PNL"] >= threshold]
    print(f"\n  Within 10% of peak PNL: tilt {plateau.min():.0%} to "
          f"{plateau.max():.0%}  ({len(plateau)} of {len(table)} grid points)")

    dev_peak, val_peak = table["dev ret"].idxmax(), table["val ret"].idxmax()
    print(f"  Sub-period agreement: 2021-23 peaks at {dev_peak:.0%}, "
          f"2024-25 peaks at {val_peak:.0%}  (gap {abs(dev_peak - val_peak):.0%})")

    # -- figure ---------------------------------------------------------------
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))

    left.plot(table.index * 100, table["IS PNL"] / 1e7, marker="o",
              markersize=3, color="#1a3f6b")
    left.axvspan(plateau.min() * 100, plateau.max() * 100, color="#1a3f6b",
                 alpha=0.10, label="within 10% of peak")
    left.set_xlabel("Size tilt (% of score)")
    left.set_ylabel("Net PNL, 2021-2025 (Rs crore)")
    left.set_title(f"Net PNL vs tilt — {label}")
    left.legend(frameon=False, fontsize=8)
    left.grid(alpha=0.25)

    for column, colour, name in [("dev ret", "#b0562a", "2021-2023"),
                                 ("val ret", "#4c8055", "2024-2025")]:
        normalised = table[column] / table[column].max()
        right.plot(table.index * 100, normalised, marker="o", markersize=3,
                   color=colour, label=name)
    right.set_xlabel("Size tilt (% of score)")
    right.set_ylabel("Return, scaled to each period's best")
    right.set_title("Does the good region agree across sub-periods?")
    right.legend(frameon=False, fontsize=8)
    right.grid(alpha=0.25)

    fig.tight_layout()
    name = f"tilt_curve_{args.tier or 'full'}{'_noscreen' if args.no_screen else ''}"
    figures = config.OUTPUT_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{name}.png", dpi=170)
    table.to_csv(config.OUTPUT_DIR / f"{name}.csv")
    print(f"\nFigure: {figures / name}.png")


if __name__ == "__main__":
    main()

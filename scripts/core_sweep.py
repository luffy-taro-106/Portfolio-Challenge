"""Which factors belong in the non-size part of the score?

The size tilt is fixed at its chosen weight; this searches the remainder. Stage
one gives the whole remainder to a single factor, which ranks candidates by
what they add *alongside* the tilt rather than in isolation — a factor that
looks strong on its own is useless here if it selects the same names the tilt
already picks. Stage two blends the survivors.

Every configuration is scored on the full window, on 2021-2023 and 2024-2025
separately, and on how many calendar years clear the bias-adjusted hurdle, so a
configuration carried by one lucky regime is visible as such.
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

TILT = strategies.SIZE_TILT
CORE = 1.0 - TILT
YEARS = [2021, 2022, 2023, 2024, 2025]

# Every price- or volume-derived factor that could plausibly complement a size
# tilt. Deliberately wider than the incumbent core so the search can reject.
SINGLES = [
    "momentum_12_1", "momentum_6_1", "momentum_3_0", "momentum_accel",
    "residual_momentum", "risk_adj_momentum", "trend_consistency",
    "near_52w_high", "info_discreteness", "low_volatility", "low_downside_vol",
    "low_idio_vol", "low_beta", "low_drawdown", "low_max_return", "low_skew",
    "falling_vol", "reversal_1m", "volume_surge", "volume_agreement",
]


def with_tilt(core: dict[str, float]) -> dict[str, float]:
    """Scale a core to the leftover weight and add the size sleeve."""
    total = sum(core.values())
    weights = {k: v / total * CORE for k, v in core.items()}
    weights["small_size"] = weights.get("small_size", 0.0) + TILT
    return weights


def evaluate(label: str, core: dict[str, float], prices, hurdle) -> dict:
    strategy = FactorStrategy(
        name=label, factor_weights=with_tilt(core),
        n_positions=config.MAX_POSITIONS, weighting="invvol",
        max_participation=strategies.MAX_PARTICIPATION,
    )
    result = Backtester(
        prices=prices, strategy=strategy,
        start=config.IS_START, end=config.IS_END,
    ).run()

    full = slice_window(result, config.IS_START, config.IS_END)
    yearly = {y: slice_window(result, f"{y}-01-01", f"{y}-12-31").total_return
              for y in YEARS}
    return {
        "core": label,
        "IS PNL": full.net_pnl,
        "Sharpe": full.sharpe,
        "MaxDD": full.max_drawdown,
        "worst year": min(yearly.values()),
        "beats hurdle": sum(yearly[y] > hurdle[y] for y in YEARS),
        "dev ret": slice_window(result, config.DEV_START, config.DEV_END).total_return,
        "val ret": slice_window(result, config.VAL_START, config.VAL_END).total_return,
        "turnover": full.annualised_turnover,
    }


def show(rows: list[dict], title: str) -> pd.DataFrame:
    table = pd.DataFrame(rows).set_index("core").sort_values(
        "IS PNL", ascending=False)
    display = table.copy()
    display["IS PNL"] = display["IS PNL"].map(lambda v: f"{v / 1e7:.2f}Cr")
    for column in ("MaxDD", "worst year", "dev ret", "val ret"):
        display[column] = display[column].map("{:+.1%}".format)
    display["Sharpe"] = display["Sharpe"].map("{:.2f}".format)
    display["turnover"] = display["turnover"].map("{:.1f}x".format)
    display["beats hurdle"] = display["beats hurdle"].map(lambda v: f"{v}/5")
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(display.to_string())
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["singles", "blends", "refine", "both"],
                        default="both")
    parser.add_argument("--tier", default=None,
                        choices=["nifty100", "midcap100", "smallcap100"])
    args = parser.parse_args()

    prices = data.fetch_prices()
    features.set_volume(data.fetch_volumes())
    if args.tier:
        keep = set(data.fetch_universe().query("tier == @args.tier")["ticker"])
        prices = prices[[c for c in prices.columns if c in keep]]
        features.set_volume(data.fetch_volumes()[prices.columns])
        print(f"Universe restricted to {args.tier}: {prices.shape[1]} symbols")

    hurdle = {y: universe_hurdle(prices, f"{y}-01-01", f"{y}-12-31").total_return
              for y in YEARS}
    print(f"Size tilt fixed at {TILT:.0%}; core carries the remaining {CORE:.0%}.")

    if args.stage in ("singles", "both"):
        rows = [evaluate("(size only)", {"small_size": 1.0}, prices, hurdle)]
        rows += [evaluate(name, {name: 1.0}, prices, hurdle) for name in SINGLES]
        rows.append(evaluate("incumbent multifactor",
                             strategies.BASELINE_FACTORS["multifactor"],
                             prices, hurdle))
        table = show(rows, "STAGE 1 - single companion factor holding the core")
        table.to_csv(config.OUTPUT_DIR / "core_singles.csv")

    if args.stage in ("blends", "both"):
        blends = {
            "incumbent multifactor": strategies.BASELINE_FACTORS["multifactor"],
            "mom12 only": {"momentum_12_1": 1.0},
            "mom12 + trend": {"momentum_12_1": 0.5, "trend_consistency": 0.5},
            "mom12 + lowvol": {"momentum_12_1": 0.5, "low_volatility": 0.5},
            "mom12 + ram": {"momentum_12_1": 0.5, "risk_adj_momentum": 0.5},
            "mom12 + 52wk": {"momentum_12_1": 0.5, "near_52w_high": 0.5},
            "ram + trend": {"risk_adj_momentum": 0.5, "trend_consistency": 0.5},
            "mom12 + ram + trend": {
                "momentum_12_1": 0.34, "risk_adj_momentum": 0.33,
                "trend_consistency": 0.33},
            "mom12 + trend + lowvol": {
                "momentum_12_1": 0.4, "trend_consistency": 0.3,
                "low_volatility": 0.3},
            "mom12 + trend + lowdd": {
                "momentum_12_1": 0.4, "trend_consistency": 0.3,
                "low_drawdown": 0.3},
            "quality-heavy": {
                "momentum_12_1": 0.25, "risk_adj_momentum": 0.25,
                "trend_consistency": 0.25, "low_volatility": 0.25},
            "momentum-heavy": {
                "momentum_12_1": 0.60, "risk_adj_momentum": 0.20,
                "trend_consistency": 0.20},
        }
        rows = [evaluate(label, core, prices, hurdle)
                for label, core in blends.items()]
        table = show(rows, "STAGE 2 - blended cores")
        table.to_csv(config.OUTPUT_DIR / "core_blends.csv")

    if args.stage == "refine":
        # Perturb the incumbent along its two meaningful axes: how much of the
        # core goes to the momentum block rather than the defensive block, and
        # how concentrated the momentum block is in raw 12-month momentum.
        rows = [evaluate("incumbent multifactor",
                         strategies.BASELINE_FACTORS["multifactor"],
                         prices, hurdle)]
        for trend_share in (0.55, 0.65, 0.75, 0.85, 1.00):
            defensive = 1.0 - trend_share
            for mom_share in (0.33, 0.47, 0.60):
                rest = (1.0 - mom_share) / 2
                core = {
                    "momentum_12_1": trend_share * mom_share,
                    "risk_adj_momentum": trend_share * rest,
                    "trend_consistency": trend_share * rest,
                }
                if defensive > 1e-9:
                    core["low_volatility"] = defensive * 0.6
                    core["low_drawdown"] = defensive * 0.4
                label = f"mom-block {trend_share:.0%} / mom12 {mom_share:.0%}"
                rows.append(evaluate(label, core, prices, hurdle))
        table = show(rows, "STAGE 3 - refinement around the incumbent")
        table.to_csv(config.OUTPUT_DIR / "core_refine.csv")


if __name__ == "__main__":
    main()

"""Structural variants on the locked base: what else moves the result?

Screens turned out to be a dead end — excluding names on a hard threshold only
pushed the portfolio into worse replacements. These ideas change the shape of
the portfolio rather than its eligibility: what the tilt is actually built
from, how risk is spread, how often the book turns over, and whether exposure
should shrink when the book gets volatile.

The 2026 column is reported but never selected on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src import data, features, metrics, strategies
from src.backtest import Backtester
from src.evaluation import slice_window, universe_hurdle
from src.strategies import FactorStrategy

YEARS = [2021, 2022, 2023, 2024, 2025]
CORE = strategies.BASELINE_FACTORS["multifactor"]
TILT = strategies.SIZE_TILT


def weights_with(sleeve: dict[str, float]) -> dict[str, float]:
    """Core scaled to its share, plus a size sleeve of the given composition."""
    total = sum(sleeve.values())
    blended = {k: v * (1 - TILT) for k, v in CORE.items()}
    for name, share in sleeve.items():
        blended[name] = blended.get(name, 0.0) + TILT * share / total
    return blended


def evaluate(label: str, strategy, prices, hurdle: dict, frequency: str = "ME") -> dict:
    result = Backtester(prices=prices, strategy=strategy, start=config.IS_START,
                        end=config.IS_END, frequency=frequency).run()
    full = slice_window(result, config.IS_START, config.IS_END)
    yearly = {y: slice_window(result, f"{y}-01-01", f"{y}-12-31").total_return
              for y in YEARS}
    holdout = Backtester(prices=prices, strategy=strategy, start=config.OOS_START,
                         end=config.OOS_END, frequency=frequency).run()
    oos = metrics.evaluate(holdout.equity, holdout.trades,
                           holdout.traded_notional, holdout.total_costs)

    trades = result.trades
    return {
        "variant": label,
        "win %": float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0,
        "IS PNL": full.net_pnl,
        "Sharpe": full.sharpe,
        "MaxDD": full.max_drawdown,
        "gain/loss": full.gain_to_loss,
        "turnover": full.annualised_turnover,
        "worst yr": min(yearly.values()),
        "hurdle": sum(yearly[y] > hurdle[y] for y in YEARS),
        "2026 ret": oos.total_return,
        "2026 Sharpe": oos.sharpe,
    }


def show(rows: list[dict], title: str) -> pd.DataFrame:
    table = pd.DataFrame(rows).set_index("variant").sort_values(
        "IS PNL", ascending=False)
    display = table.copy()
    display["win %"] = display["win %"].map("{:.1%}".format)
    display["IS PNL"] = display["IS PNL"].map(lambda v: f"{v / 1e7:.2f}Cr")
    for column in ("MaxDD", "worst yr", "2026 ret"):
        display[column] = display[column].map("{:+.1%}".format)
    for column in ("Sharpe", "gain/loss", "2026 Sharpe"):
        display[column] = display[column].map("{:.2f}".format)
    display["turnover"] = display["turnover"].map("{:.1f}x".format)
    display["hurdle"] = display["hurdle"].map(lambda v: f"{v}/5")
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    print(display.to_string())
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["sleeve", "risk", "cadence", "all"])
    args = parser.parse_args()

    prices = data.fetch_prices()
    features.set_volume(data.fetch_volumes())
    universe = data.fetch_universe()
    features.set_industry(
        universe.assign(ticker=universe["symbol"] + ".NS")
        .set_index("ticker")["industry"])
    hurdle = {y: universe_hurdle(prices, f"{y}-01-01", f"{y}-12-31").total_return
              for y in YEARS}

    def build(**kwargs) -> FactorStrategy:
        kwargs.setdefault("factor_weights", strategies.FINAL_FACTORS)
        return FactorStrategy(
            name=kwargs.pop("label", "variant"),
            n_positions=config.MAX_POSITIONS, weighting="invvol",
            max_participation=strategies.MAX_PARTICIPATION, **kwargs)

    base = evaluate("BASE", build(), prices, hurdle)

    if args.stage in ("sleeve", "all"):
        # The tilt is the edge, so what it is built from matters more than
        # anything bolted on afterwards. small_size is pure scale;
        # price_impact is how far the price moves per rupee traded.
        sleeves = {
            "sleeve: small_size (base)": {"small_size": 1.0},
            "sleeve: price_impact": {"price_impact": 1.0},
            "sleeve: liquidity (Amihud)": {"liquidity": 1.0},
            "sleeve: size 75 / impact 25": {"small_size": 0.75, "price_impact": 0.25},
            "sleeve: size 50 / impact 50": {"small_size": 0.50, "price_impact": 0.50},
            "sleeve: size 25 / impact 75": {"small_size": 0.25, "price_impact": 0.75},
        }
        rows = [evaluate(label, build(factor_weights=weights_with(s)), prices, hurdle)
                for label, s in sleeves.items()]
        show(rows, "WHAT THE SIZE SLEEVE IS BUILT FROM").to_csv(
            config.OUTPUT_DIR / "ideas_sleeve.csv")

    if args.stage in ("risk", "all"):
        rows = [base]
        for cap in (0.15, 0.20, 0.30, 1.00):
            rows.append(evaluate(f"position cap {cap:.0%}",
                                 build(cap=cap), prices, hurdle))
        for window in (63, 126):
            rows.append(evaluate(f"risk-parity window {window}d",
                                 build(vol_window=window), prices, hurdle))
        for limit in (2, 3, 4):
            rows.append(evaluate(f"max {limit} per industry",
                                 build(max_per_industry=limit), prices, hurdle))
        for target in (0.15, 0.20, 0.25):
            rows.append(evaluate(f"target vol {target:.0%}",
                                 build(target_vol=target), prices, hurdle))
        show(rows, "RISK SHAPING").to_csv(config.OUTPUT_DIR / "ideas_risk.csv")

    if args.stage in ("cadence", "all"):
        rows = [base]
        for frequency, label in [("W-FRI", "weekly"), ("SME", "twice monthly"),
                                 ("QE", "quarterly")]:
            rows.append(evaluate(f"rebalance {label}", build(), prices, hurdle,
                                 frequency=frequency))
        show(rows, "REBALANCE CADENCE").to_csv(
            config.OUTPUT_DIR / "ideas_cadence.csv")


if __name__ == "__main__":
    main()

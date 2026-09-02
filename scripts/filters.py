"""Do hard screens or rank buffering improve the base strategy?

Each idea is added to the locked base one at a time, so any change in the
result is attributable. The table reports win rate first, since that is what
these ideas are meant to move, but a screen that lifts win rate while cutting
PNL has not helped: dropping every position that might lose money also drops
the ones that would have won.

The 2026 column is shown throughout at the user's request. It is *not* used to
choose anything — it covers six months and one regime, and selecting on it
would convert the only untouched window into another fitted one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src import data, features, metrics, screens, strategies
from src.backtest import Backtester
from src.evaluation import slice_window, universe_hurdle
from src.strategies import FactorStrategy

YEARS = [2021, 2022, 2023, 2024, 2025]


def base(**kwargs) -> FactorStrategy:
    """The locked configuration, before any of the ideas under test."""
    return FactorStrategy(
        name=kwargs.pop("name", "base"),
        factor_weights=strategies.FINAL_FACTORS,
        n_positions=config.MAX_POSITIONS,
        weighting="invvol",
        max_participation=strategies.MAX_PARTICIPATION,
        **kwargs,
    )


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float((trades["pnl"] > 0).mean())


def evaluate(label: str, strategy, prices, hurdle: dict) -> dict:
    result = Backtester(prices=prices, strategy=strategy,
                        start=config.IS_START, end=config.IS_END).run()
    full = slice_window(result, config.IS_START, config.IS_END)
    yearly = {y: slice_window(result, f"{y}-01-01", f"{y}-12-31").total_return
              for y in YEARS}

    # The holdout is run as a separate backtest starting from fresh capital,
    # so it measures the rules rather than the compounded 2021-2025 book.
    holdout = Backtester(prices=prices, strategy=strategy,
                         start=config.OOS_START, end=config.OOS_END).run()
    oos = metrics.evaluate(holdout.equity, holdout.trades,
                           holdout.traded_notional, holdout.total_costs)

    return {
        "variant": label,
        "win %": win_rate(result.trades),
        "IS PNL": full.net_pnl,
        "Sharpe": full.sharpe,
        "MaxDD": full.max_drawdown,
        "gain/loss": full.gain_to_loss,
        "trades": len(result.trades),
        "turnover": full.annualised_turnover,
        "worst yr": min(yearly.values()),
        "hurdle": sum(yearly[y] > hurdle[y] for y in YEARS),
        "2026 ret": oos.total_return,
        "2026 Sharpe": oos.sharpe,
    }


def show(rows: list[dict], title: str, sort: str = "IS PNL") -> pd.DataFrame:
    table = pd.DataFrame(rows).set_index("variant")
    ordered = table.sort_values(sort, ascending=False)
    display = ordered.copy()
    display["win %"] = display["win %"].map("{:.1%}".format)
    display["IS PNL"] = display["IS PNL"].map(lambda v: f"{v / 1e7:.2f}Cr")
    for column in ("MaxDD", "worst yr", "2026 ret"):
        display[column] = display[column].map("{:+.1%}".format)
    for column in ("Sharpe", "gain/loss", "2026 Sharpe"):
        display[column] = display[column].map("{:.2f}".format)
    display["turnover"] = display["turnover"].map("{:.1f}x".format)
    display["hurdle"] = display["hurdle"].map(lambda v: f"{v}/5")
    print("\n" + "=" * 122)
    print(title)
    print("=" * 122)
    print(display.to_string())
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["screens", "buffer", "combo", "all"])
    args = parser.parse_args()

    prices = data.fetch_prices()
    features.set_volume(data.fetch_volumes())
    hurdle = {y: universe_hurdle(prices, f"{y}-01-01", f"{y}-12-31").total_return
              for y in YEARS}

    baseline = evaluate("BASE (no filter)", base(), prices, hurdle)

    if args.stage in ("screens", "all"):
        rows = [baseline]
        for key, screen in screens.CATALOGUE.items():
            rows.append(evaluate(f"+ {screen.label}",
                                 base(screens=[screen]), prices, hurdle))
        table = show(rows, "ONE SCREEN AT A TIME")
        table.to_csv(config.OUTPUT_DIR / "filters_screens.csv")

    if args.stage in ("buffer", "all"):
        rows = [baseline]
        for slots in (2, 3, 5, 8, 12, 20):
            rows.append(evaluate(f"+ buffer {slots} slots",
                                 base(buffer=slots), prices, hurdle))
        table = show(rows, "RANK BUFFERING - incumbents held until rank 10 + N")
        table.to_csv(config.OUTPUT_DIR / "filters_buffer.csv")


if __name__ == "__main__":
    main()

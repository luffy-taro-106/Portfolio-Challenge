"""Hard eligibility screens applied before selection.

A screen differs from a factor. A factor says "prefer this"; a screen says
"never hold this". Blending a condition into the score lets a strong reading
elsewhere override it, which is exactly what you do *not* want for a condition
that describes a way to lose money — a stock in freefall should be excluded
however cheap or thinly traded it happens to be.

Each screen is a comparison against one raw, un-z-scored quantity, so its
threshold has a plain reading ("price at least 70% of the 52-week high")
rather than an opaque one ("z-score above 0.4").
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features

_SCREEN_CACHE: dict[pd.Timestamp, pd.DataFrame] = {}


def screen_panel(history: pd.DataFrame) -> pd.DataFrame:
    """Raw quantities the screens compare against, for every eligible name."""
    tickers = features.eligible_universe(history)
    if len(tickers) == 0:
        return pd.DataFrame()

    prices = history[tickers]
    last = prices.iloc[-1]
    moving_avg = prices.rolling(200, min_periods=100).mean().iloc[-1]
    quarter = prices.tail(64).pct_change().iloc[1:]
    month = prices.tail(22).pct_change().iloc[1:]

    short_vol = features.volatility(prices, 42)
    long_vol = features.volatility(prices, 252).replace(0, np.nan)

    frame = pd.DataFrame(
        {
            # Is the stock in an uptrend on its own terms?
            "price_to_ma200": (last / moving_avg.replace(0, np.nan)),
            # Anchor effect: distance below the 52-week high.
            "near_high": last / prices.tail(252).max().replace(0, np.nan),
            # Absolute, not relative, momentum.
            "momentum_12_1": features.momentum(prices, 252, 21),
            # Crash risk: the worst single session of the last quarter.
            "worst_day_3m": quarter.min(),
            # Lottery demand: the best single session of the last month.
            "best_day_1m": month.max(),
            # Is volatility breaking out of its own long-run range?
            "vol_ratio": (short_vol / long_vol),
            # Does volume arrive on up days or down days?
            "volume_agreement": features.price_volume_agreement(prices, 63),
            # Recent drawdown from the one-year peak.
            "drawdown_1y": last / prices.tail(252).cummax().iloc[-1].replace(0, np.nan)
            - 1.0,
        }
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def cached_screen_panel(history: pd.DataFrame) -> pd.DataFrame:
    date = history.index[-1]
    if date not in _SCREEN_CACHE:
        _SCREEN_CACHE[date] = screen_panel(history)
    return _SCREEN_CACHE[date]


@dataclass(frozen=True)
class Screen:
    """One threshold test against a column of the screen panel.

    `minimum` keeps names at or above the threshold, `maximum` keeps names at
    or below it. Names with a missing reading are kept: a screen exists to
    exclude on evidence, and absent data is not evidence.
    """

    label: str
    column: str
    threshold: float
    direction: str = "minimum"

    def mask(self, panel: pd.DataFrame) -> pd.Series:
        if self.column not in panel.columns:
            return pd.Series(True, index=panel.index)
        values = panel[self.column]
        if self.direction == "minimum":
            passes = values >= self.threshold
        else:
            passes = values <= self.threshold
        return passes | values.isna()


def apply(scores: pd.Series, panel: pd.DataFrame, screens) -> pd.Series:
    """Drop every name failing any screen."""
    if not screens or panel.empty or scores.empty:
        return scores
    keep = pd.Series(True, index=panel.index)
    for screen in screens:
        keep &= screen.mask(panel)
    allowed = panel.index[keep]
    return scores[scores.index.isin(allowed)]


# Candidate screens, each with a defensible motivation rather than a swept
# threshold. Widths are round numbers on purpose.
CATALOGUE = {
    "above_ma200": Screen("price above 200-day average", "price_to_ma200", 1.0),
    "near_high_70": Screen("within 30% of 52-week high", "near_high", 0.70),
    "near_high_80": Screen("within 20% of 52-week high", "near_high", 0.80),
    "near_high_85": Screen("within 15% of 52-week high", "near_high", 0.85),
    "positive_momentum": Screen("12-month return positive", "momentum_12_1", 0.0),
    "no_crash_15": Screen("no single day worse than -15%", "worst_day_3m", -0.15),
    "no_crash_10": Screen("no single day worse than -10%", "worst_day_3m", -0.10),
    "no_lottery": Screen("no single day above +20%", "best_day_1m", 0.20, "maximum"),
    "vol_stable_150": Screen("42d vol under 1.5x its 1y level", "vol_ratio", 1.5, "maximum"),
    "vol_stable_125": Screen("42d vol under 1.25x its 1y level", "vol_ratio", 1.25, "maximum"),
    "not_extended": Screen("price under 1.5x its 200-day average", "price_to_ma200", 1.5, "maximum"),
    "accumulation": Screen("volume arrives on up days", "volume_agreement", 0.0),
    "shallow_dd": Screen("within 25% of the 1-year peak", "drawdown_1y", -0.25),
}

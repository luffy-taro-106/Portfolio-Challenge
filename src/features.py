"""Cross-sectional factor construction.

Every factor here is computed purely from the price and volume history passed
in, which ends at the signal date. Nothing reads fundamentals: reported
financials are only available with a lag and Yahoo Finance serves current
restated figures rather than point-in-time ones, so using them would leak
future information into the backtest.

The library deliberately reaches past the standard momentum/volatility pair.
Several factors here are documented anomalies that most entrants will not
think to include — residual momentum, information discreteness, the 52-week
high effect, the lottery-demand (MAX) effect and Amihud illiquidity — and
several more are derived from volume, which price-only pipelines cannot see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Volume is supplied once by the entry-point script rather than threaded
# through the strategy interface, which only carries prices. `build_factor_panel`
# slices it to match whatever history it is given, so no future volume can leak.
_VOLUME: pd.DataFrame | None = None
_INDUSTRY: pd.Series | None = None


def set_volume(volume: pd.DataFrame) -> None:
    global _VOLUME
    _VOLUME = volume


def set_industry(industry: pd.Series) -> None:
    """Ticker -> industry map, used for sector-neutral selection."""
    global _INDUSTRY
    _INDUSTRY = industry


def get_industry() -> pd.Series | None:
    return _INDUSTRY


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def eligible_universe(history: pd.DataFrame, lookback: int = 252) -> pd.Index:
    """Tickers with enough recent price history to be scored fairly.

    Also drops names that stopped trading before the signal date, which is the
    partial mitigation available to us for using a present-day constituent
    list: a stock cannot be selected on a date where it had no price.
    """
    window = history.tail(lookback)
    if window.empty:
        return pd.Index([])

    coverage = window.notna().mean()
    latest = history.iloc[-1]
    return history.columns[(coverage >= config.MIN_DATA_COVERAGE) & latest.notna() & (latest > 0)]


# ---------------------------------------------------------------------------
# Return-based factors
# ---------------------------------------------------------------------------

def momentum(history: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Total return over `lookback` days, excluding the most recent `skip`.

    Skipping the latest month is the standard Jegadeesh-Titman construction: it
    sidesteps the short-term reversal effect that contaminates raw momentum.
    """
    if len(history) < lookback + 1:
        lookback = len(history) - 1
    if lookback <= skip:
        return pd.Series(dtype=float)
    end = history.iloc[-(skip + 1)]
    start = history.iloc[-(lookback + 1)]
    return (end / start - 1.0).replace([np.inf, -np.inf], np.nan)


def volatility(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Annualised standard deviation of daily returns."""
    returns = history.tail(window + 1).pct_change()
    return returns.std(ddof=1) * np.sqrt(config.TRADING_DAYS_PER_YEAR)


def downside_volatility(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Annualised deviation of negative returns only."""
    returns = history.tail(window + 1).pct_change()
    return returns.where(returns < 0).std(ddof=1) * np.sqrt(config.TRADING_DAYS_PER_YEAR)


def risk_adjusted_momentum(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Momentum per unit of volatility: a price-based 'quality of trend'."""
    vol = volatility(history, window=window)
    return (momentum(history, lookback=window) / vol.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def trend_consistency(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Fraction of the last `window` days spent above the 200-day average."""
    window_prices = history.tail(window)
    if len(window_prices) < 20:
        return pd.Series(dtype=float)
    moving_avg = history.rolling(200, min_periods=50).mean().tail(window)
    return (window_prices > moving_avg).mean()


def max_drawdown_factor(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Worst peak-to-trough decline over the window, as a positive number."""
    window_prices = history.tail(window)
    if window_prices.empty:
        return pd.Series(dtype=float)
    return -(window_prices / window_prices.cummax() - 1.0).min()


def reversal(history: pd.DataFrame, window: int = 21) -> pd.Series:
    """Negative of the last month's return: buys recent underperformers."""
    if len(history) < window + 1:
        return pd.Series(dtype=float)
    return -(history.iloc[-1] / history.iloc[-(window + 1)] - 1.0)


def momentum_acceleration(history: pd.DataFrame) -> pd.Series:
    """Is the trend speeding up? Three-month momentum minus six-month.

    A second derivative of price. Picks up stocks whose advance is gathering
    pace rather than decaying, which raw momentum cannot distinguish.
    """
    fast = momentum(history, 63, 0)
    slow = momentum(history, 126, 0)
    return fast - slow / 2.0


# ---------------------------------------------------------------------------
# Market-model factors (beta, residual momentum, idiosyncratic volatility)
# ---------------------------------------------------------------------------

def _market_model(history: pd.DataFrame, window: int = 252) -> tuple[pd.Series, pd.DataFrame]:
    """Regress every stock on the equal-weight universe return.

    Returns per-stock beta and the matrix of residual daily returns. Vectorised
    via covariance ratios rather than looping stock by stock.
    """
    returns = history.tail(window + 1).pct_change().iloc[1:]
    if len(returns) < 60:
        return pd.Series(dtype=float), pd.DataFrame()

    market = returns.mean(axis=1)
    market_var = market.var(ddof=1)
    if market_var == 0 or np.isnan(market_var):
        return pd.Series(dtype=float), pd.DataFrame()

    centred = returns.sub(returns.mean())
    beta = centred.mul(market - market.mean(), axis=0).mean() * (
        len(returns) / (len(returns) - 1)
    ) / market_var

    alpha = returns.mean() - beta * market.mean()
    predicted = pd.DataFrame(
        np.outer(market.to_numpy(), beta.to_numpy()),
        index=returns.index,
        columns=returns.columns,
    ).add(alpha, axis=1)
    return beta, returns - predicted


def residual_momentum(history: pd.DataFrame, window: int = 252, skip: int = 21) -> pd.Series:
    """Momentum measured on market-adjusted returns, scaled by residual risk.

    Blitz, Huij and Martens (2011) show this is materially more robust than raw
    momentum because it strips out the beta and sector bets that raw momentum
    unintentionally accumulates. That is precisely the fragility that sank the
    six-month variant on the holdout, so it is the most promising single idea
    in this library.
    """
    beta, residuals = _market_model(history, window)
    if residuals.empty:
        return pd.Series(dtype=float)

    scored = residuals.iloc[:-skip] if skip and len(residuals) > skip else residuals
    if scored.empty:
        return pd.Series(dtype=float)

    sigma = scored.std(ddof=1).replace(0, np.nan)
    return (scored.mean() / sigma).replace([np.inf, -np.inf], np.nan)


def idiosyncratic_volatility(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Residual volatility after removing market exposure.

    Ang, Hodrick, Xing and Zhang (2006): high idiosyncratic volatility predicts
    *low* returns, the opposite of what a risk story would suggest.
    """
    _, residuals = _market_model(history, window)
    if residuals.empty:
        return pd.Series(dtype=float)
    return residuals.std(ddof=1) * np.sqrt(config.TRADING_DAYS_PER_YEAR)


def market_beta(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Sensitivity to the equal-weight universe return."""
    beta, _ = _market_model(history, window)
    return beta


# ---------------------------------------------------------------------------
# Path-shape factors
# ---------------------------------------------------------------------------

def information_discreteness(history: pd.DataFrame, window: int = 252, skip: int = 21) -> pd.Series:
    """The 'frog in the pan' signal of Da, Gurun and Warachka (2014).

    Two stocks can post the same twelve-month return by very different routes:
    one drifting up on many small days, the other jumping on a handful of large
    ones. Gradual information is under-reacted to and keeps drifting; discrete
    information is noticed immediately and does not. ID is negative for gradual
    movers, so we flip the sign — higher is better, as everywhere else.
    """
    prices = history.tail(window + skip + 1)
    if len(prices) < 60:
        return pd.Series(dtype=float)

    scored = prices.iloc[:-skip] if skip else prices
    returns = scored.pct_change().iloc[1:]
    if returns.empty:
        return pd.Series(dtype=float)

    total = scored.iloc[-1] / scored.iloc[0] - 1.0
    positive = (returns > 0).mean()
    negative = (returns < 0).mean()
    return -(np.sign(total) * (negative - positive))


def proximity_to_high(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Current price as a fraction of the 52-week high.

    George and Hwang (2004): nearness to the 52-week high predicts returns
    better than raw momentum does, because the high acts as an anchor that
    investors are slow to revise past.
    """
    window_prices = history.tail(window)
    if window_prices.empty:
        return pd.Series(dtype=float)
    return (window_prices.iloc[-1] / window_prices.max()).replace([np.inf, -np.inf], np.nan)


def max_daily_return(history: pd.DataFrame, window: int = 21) -> pd.Series:
    """Largest single-day gain in the past month, sign-flipped.

    Bali, Cakici and Whitelaw (2011): stocks with lottery-like payoffs attract
    demand from investors who overpay for the small chance of a large gain, and
    subsequently underperform. Avoiding them is a cheap defensive edge.
    """
    returns = history.tail(window + 1).pct_change().iloc[1:]
    return -returns.max() if not returns.empty else pd.Series(dtype=float)


def return_skewness(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Negative of past daily-return skewness, for the same lottery reason."""
    returns = history.tail(window + 1).pct_change().iloc[1:]
    return -returns.skew() if len(returns) > 30 else pd.Series(dtype=float)


def volatility_trend(history: pd.DataFrame, short: int = 42, long: int = 252) -> pd.Series:
    """Falling volatility, expressed as a positive score.

    Rising short-horizon volatility relative to its own long-run level is an
    early warning that a trend is becoming unstable.
    """
    recent = volatility(history, short)
    baseline = volatility(history, long).replace(0, np.nan)
    return -(recent / baseline).replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Volume-based factors
# ---------------------------------------------------------------------------

def _aligned_volume(history: pd.DataFrame) -> pd.DataFrame | None:
    """Volume matching the given price history, loading it if nobody set it.

    A caller that forgets `set_volume` would otherwise get an all-NaN size
    factor, which `composite_score` drops row by row until nothing is left and
    the strategy silently holds cash for the entire backtest. Falling back to
    the cached file makes that failure impossible.
    """
    global _VOLUME
    if _VOLUME is None:
        from src import data

        _VOLUME = data.fetch_volumes()
    volume = _VOLUME.reindex(index=history.index, columns=history.columns)
    return volume if not volume.dropna(how="all").empty else None


def average_daily_value(history: pd.DataFrame, window: int = 21) -> pd.Series | None:
    """Median rupee value traded per session over the trailing window.

    The median rather than the mean, because a single block deal can multiply a
    thin name's average and make it look tradeable when it is not.
    """
    volume = _aligned_volume(history)
    if volume is None:
        return None
    return (history * volume).tail(window).median()


def amihud_illiquidity(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Average absolute return per rupee of turnover, sign-flipped.

    Amihud (2002). Illiquid stocks earn a premium for the trading friction they
    impose, but that friction is exactly what a real portfolio has to pay. We
    flip the sign to *prefer liquid* names: our fills should be defensible.
    """
    volume = _aligned_volume(history)
    if volume is None:
        return pd.Series(dtype=float)

    prices = history.tail(window + 1)
    volume = volume.tail(window + 1)
    returns = prices.pct_change().abs().iloc[1:]
    traded_value = (prices * volume).iloc[1:].replace(0, np.nan)
    return -(returns / traded_value).mean() * 1e9


def small_traded_value(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Negative log of average daily rupee turnover: a size proxy.

    Amihud illiquidity mixes two things — how much a stock trades, and how much
    its price moves per unit traded. This isolates the first so we can tell
    whether the illiquidity result is really a size effect wearing a costume.
    """
    volume = _aligned_volume(history)
    if volume is None:
        return pd.Series(dtype=float)

    traded_value = (history.tail(window) * volume.tail(window)).mean()
    return -np.log(traded_value.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def small_share_volume(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Negative log of average daily *share* volume.

    The counterpart to `small_traded_value`, which multiplies volume by price.
    Separating the two tells us whether the premium attaches to how little a
    stock trades in money terms or to how few hands its shares change, since a
    high-priced stock can post a large rupee turnover on very few shares.
    """
    volume = _aligned_volume(history)
    if volume is None:
        return pd.Series(dtype=float)
    shares = volume.tail(window).mean()
    return -np.log(shares.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def return_per_turnover(history: pd.DataFrame, window: int = 252) -> pd.Series:
    """The other half of Amihud: price impact, holding turnover scale aside.

    Average absolute daily return divided by the *rank* of turnover rather than
    its level, so the size dimension is largely differenced out.
    """
    volume = _aligned_volume(history)
    if volume is None:
        return pd.Series(dtype=float)

    prices = history.tail(window + 1)
    absolute_return = prices.pct_change().abs().iloc[1:].mean()
    turnover_rank = (prices.iloc[1:] * volume.tail(window)).mean().rank(pct=True)
    return (absolute_return / turnover_rank.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def volume_surge(history: pd.DataFrame, short: int = 21, long: int = 252) -> pd.Series:
    """Recent turnover against its own baseline: is participation rising?

    A trend that the rest of the market is joining is better supported than one
    running on thinning volume.
    """
    volume = _aligned_volume(history)
    if volume is None:
        return pd.Series(dtype=float)

    traded_value = history.tail(long) * volume.tail(long)
    recent = traded_value.tail(short).mean()
    baseline = traded_value.mean().replace(0, np.nan)
    return (recent / baseline).replace([np.inf, -np.inf], np.nan)


def price_volume_agreement(history: pd.DataFrame, window: int = 63) -> pd.Series:
    """Does volume arrive on up days or down days?

    Positive means the heavy days are the advancing ones — accumulation.
    Negative means the crowd shows up to sell, which often precedes a break in
    the trend even while price momentum still looks healthy.
    """
    volume = _aligned_volume(history)
    if volume is None:
        return pd.Series(dtype=float)

    returns = history.tail(window + 1).pct_change().iloc[1:]
    traded = volume.tail(window)
    if returns.empty or traded.empty:
        return pd.Series(dtype=float)

    weights = traded.div(traded.mean().replace(0, np.nan))
    signed = np.sign(returns) * weights.reindex(returns.index)
    return signed.mean()


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

FACTOR_BUILDERS = {
    "momentum_12_1": lambda h: momentum(h, 252, 21),
    "momentum_6_1": lambda h: momentum(h, 126, 21),
    "momentum_3_0": lambda h: momentum(h, 63, 0),
    "momentum_accel": momentum_acceleration,
    "low_volatility": lambda h: -volatility(h, 252),
    "low_downside_vol": lambda h: -downside_volatility(h, 252),
    "risk_adj_momentum": lambda h: risk_adjusted_momentum(h, 252),
    "trend_consistency": lambda h: trend_consistency(h, 252),
    "low_drawdown": lambda h: -max_drawdown_factor(h, 252),
    "reversal_1m": lambda h: reversal(h, 21),
    "residual_momentum": lambda h: residual_momentum(h, 252, 21),
    "low_idio_vol": lambda h: -idiosyncratic_volatility(h, 252),
    "low_beta": lambda h: -market_beta(h, 252),
    "info_discreteness": lambda h: information_discreteness(h, 252, 21),
    "near_52w_high": lambda h: proximity_to_high(h, 252),
    "low_max_return": lambda h: max_daily_return(h, 21),
    "low_skew": lambda h: return_skewness(h, 252),
    "falling_vol": lambda h: volatility_trend(h, 42, 252),
    "liquidity": lambda h: amihud_illiquidity(h, 252),
    "small_size": lambda h: small_traded_value(h, 252),
    "small_size_63": lambda h: small_traded_value(h, 63),
    "small_size_126": lambda h: small_traded_value(h, 126),
    "small_share_volume": small_share_volume,
    "price_impact": lambda h: return_per_turnover(h, 252),
    "volume_surge": volume_surge,
    "volume_agreement": lambda h: price_volume_agreement(h, 63),
}


def build_factor_panel(history: pd.DataFrame) -> pd.DataFrame:
    """All factors for every eligible ticker at the current signal date.

    Returns a ticker-indexed frame in which higher is better for every column,
    so volatility, drawdown, beta and illiquidity are all sign-flipped at
    construction.
    """
    tickers = eligible_universe(history)
    if len(tickers) == 0:
        return pd.DataFrame()

    prices = history[tickers]
    panel = pd.DataFrame(index=tickers)
    for name, builder in FACTOR_BUILDERS.items():
        try:
            values = builder(prices)
        except Exception:
            values = pd.Series(dtype=float)
        panel[name] = values.reindex(tickers) if len(values) else np.nan

    return panel.replace([np.inf, -np.inf], np.nan)


def zscore(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional z-score, winsorised to limit single-name influence."""
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty or series.std(ddof=0) == 0:
        return pd.Series(0.0, index=series.index)
    return ((series - series.mean()) / series.std(ddof=0)).clip(-clip, clip)


def composite_score(
    panel: pd.DataFrame,
    weights: dict[str, float],
    mode: str = "zscore",
    clip: float = 3.0,
) -> pd.Series:
    """Weighted blend of standardised factors, over names with complete data.

    Each factor is standardised across the cross-section before weighting so
    that a factor measured in return units and one measured as a ratio
    contribute on comparable scales.

    `mode` chooses the standardisation. "zscore" preserves how far apart names
    are, so a genuine outlier keeps its influence up to the winsorisation
    limit. "rank" keeps only the ordering, which discards that magnitude but
    is immune to a skewed factor distribution — and several factors here are
    badly skewed, traded value most of all, since it spans four orders of
    magnitude and its z-score is dominated by a handful of giants.
    """
    if panel.empty:
        return pd.Series(dtype=float)

    columns = [c for c in weights if c in panel.columns]
    usable = panel[columns].dropna()
    if usable.empty:
        return pd.Series(dtype=float)

    score = pd.Series(0.0, index=usable.index)
    for column in columns:
        if mode == "rank":
            # Centred percentile rank, so the scale matches a z-score closely
            # enough that weights mean the same thing in both modes.
            standardised = (usable[column].rank(pct=True) - 0.5) * 2.0
        else:
            standardised = zscore(usable[column], clip=clip)
        score += weights[column] * standardised.reindex(usable.index).fillna(0.0)
    return score.sort_values(ascending=False)

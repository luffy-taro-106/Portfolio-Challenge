"""Data acquisition: index constituents and adjusted price history.

Both fetchers cache to disk so that repeated backtests and the strategy sweep
do not re-hit the network. Delete `data/cache` to force a refresh.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_universe(refresh: bool = False) -> pd.DataFrame:
    """Return the combined eligible universe as a symbol/industry/tier table.

    NSE only publishes *current* index membership, so this is a present-day
    snapshot applied backwards through the backtest. That introduces
    survivorship bias, which is disclosed in the report rather than silently
    ignored; see README for the mitigation applied at selection time.
    """
    cache = config.CACHE_DIR / "universe.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    frames = []
    for tier, url in config.UNIVERSE_SOURCES.items():
        response = requests.get(url, headers=_NSE_HEADERS, timeout=30)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        frame.columns = [c.strip() for c in frame.columns]
        frame = frame.rename(
            columns={
                "Company Name": "company",
                "Industry": "industry",
                "Symbol": "symbol",
                "ISIN Code": "isin",
            }
        )
        frame["tier"] = tier
        frames.append(frame[["symbol", "company", "industry", "tier", "isin"]])

    universe = pd.concat(frames, ignore_index=True)

    # A handful of names sit in more than one index; keep the largest-cap tier.
    tier_rank = {"nifty100": 0, "midcap100": 1, "smallcap100": 2}
    universe["_rank"] = universe["tier"].map(tier_rank)
    universe = (
        universe.sort_values("_rank")
        .drop_duplicates(subset="symbol", keep="first")
        .drop(columns="_rank")
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    universe["ticker"] = universe["symbol"].str.strip() + config.YF_SUFFIX
    universe.to_csv(cache, index=False)
    return universe


def _download(
    tickers: list[str], start: str, end: str, field: str = "Close"
) -> pd.DataFrame:
    """Download one auto-adjusted field, returning a date x ticker frame."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        frame = raw[field]
    else:  # single ticker collapses the column index
        frame = raw[[field]]
        frame.columns = tickers

    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.sort_index()


def fetch_prices(refresh: bool = False) -> pd.DataFrame:
    """Adjusted daily closes for the full universe, warmup through OOS end."""
    cache = config.CACHE_DIR / "prices.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    universe = fetch_universe(refresh=refresh)
    tickers = universe["ticker"].tolist()

    # Chunked so one bad symbol cannot poison the whole request.
    chunks = []
    for i in range(0, len(tickers), 50):
        batch = tickers[i : i + 50]
        chunks.append(_download(batch, config.WARMUP_START, config.OOS_END))

    prices = pd.concat(chunks, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    # Drop symbols with no usable history at all.
    prices = prices.dropna(axis=1, how="all")

    prices.to_parquet(cache)
    return prices


def fetch_volumes(refresh: bool = False) -> pd.DataFrame:
    """Daily traded share volume for the full universe.

    Opens up the liquidity and participation signals — Amihud illiquidity,
    volume surges, price-volume divergence — none of which can be computed
    from closes alone.
    """
    cache = config.CACHE_DIR / "volumes.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    tickers = fetch_universe(refresh=refresh)["ticker"].tolist()
    chunks = []
    for i in range(0, len(tickers), 50):
        chunks.append(
            _download(tickers[i : i + 50], config.WARMUP_START, config.OOS_END, "Volume")
        )

    volumes = pd.concat(chunks, axis=1)
    volumes = volumes.loc[:, ~volumes.columns.duplicated()].dropna(axis=1, how="all")
    volumes.to_parquet(cache)
    return volumes


def fetch_benchmarks(refresh: bool = False) -> pd.DataFrame:
    """Adjusted daily closes for the benchmark indices, named not tickered."""
    cache = config.CACHE_DIR / "benchmarks.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    tickers = list(config.BENCHMARKS.values())
    data = _download(tickers, config.WARMUP_START, config.OOS_END)
    data = data.rename(columns={v: k for k, v in config.BENCHMARKS.items()})
    data = data.ffill().dropna(how="all")

    data.to_parquet(cache)
    return data


def trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Session dates on which a meaningful share of the universe traded."""
    coverage = prices.notna().sum(axis=1)
    return prices.index[coverage >= max(10, 0.5 * coverage.max())]


if __name__ == "__main__":
    universe = fetch_universe(refresh=True)
    print(f"Universe: {len(universe)} symbols")
    print(universe["tier"].value_counts().to_string())

    prices = fetch_prices(refresh=True)
    print(f"\nPrices: {prices.shape[0]} sessions x {prices.shape[1]} symbols")
    print(f"Range: {prices.index.min().date()} -> {prices.index.max().date()}")

    missing = sorted(set(universe["ticker"]) - set(prices.columns))
    if missing:
        print(f"\nNo data for {len(missing)}: {', '.join(missing)}")

    benchmarks = fetch_benchmarks(refresh=True)
    print(f"\nBenchmarks: {benchmarks.shape}")
    print(benchmarks.tail(2).to_string())

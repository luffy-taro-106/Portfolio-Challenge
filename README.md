# Finesse x Citadel — Round 2 Portfolio Challenge

A reproducible systematic long-only equity portfolio for the Nifty 100, Nifty Midcap 100, and Nifty Smallcap 100 universe.

The submitted strategy holds at most 10 stocks, starts with ₹1 crore, and applies the mandated 0.1% transaction cost on every buy and sell. Its backtest window is 1 January 2021 to 31 December 2025.

## Submission result

| Metric | Result |
| --- | ---: |
| Final portfolio value | ₹17,83,05,098 |
| Total Net PNL | ₹16,83,05,098 |
| Total return | 1,683.05% |
| Annualised return | 79.93% |
| Sharpe ratio, 0% risk-free rate | 3.86 |
| Maximum drawdown | −18.72% |
| Profitable trades | 82.15% |

See [`REPORT.pdf`](REPORT.pdf) for the full report and methodology.

## Strategy in one paragraph

At each month-end, the model scores the permitted stock universe using a 65% traded-value/size tilt and a 35% trend-quality sleeve. The trend sleeve rewards 12–1 month momentum, risk-adjusted momentum, stable trends, low volatility, and low drawdown. Stocks failing the liquidity test are excluded, the ten highest-scoring eligible names are selected, and capital is allocated inversely to trailing volatility. This combines momentum with a liquidity premium while keeping position sizes executable.

### Final parameters

| Rule | Value |
| --- | --- |
| Portfolio size | Top 10 eligible stocks |
| Rebalance | Monthly, signals at close and execution at next close |
| Selection score | 65.0% small traded value; 12.3% 12–1 momentum; 7.0% risk-adjusted momentum; 7.0% trend consistency; 5.3% low volatility; 3.5% low drawdown |
| Score construction | Cross-sectional z-scores, winsorised at 3 standard deviations |
| Position sizing | Inverse 252-day volatility, 25% maximum position weight |
| Liquidity safeguard | Intended position cannot exceed 10% of the stock’s trailing 21-session median traded value |
| Trading cost | 0.1% of notional on each buy and sell |

## Quick start

Requires Python 3.13 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download and cache the official universe, prices, and benchmarks.
python src/data.py

# Run the submitted strategy over the competition backtest window.
python scripts/run_all.py --strategies final
```

The first data run downloads approximately 300 NSE symbols and caches data under `data/cache/`. Volume data is fetched automatically on the first run of the submitted strategy. Delete that directory or pass `--refresh` to fetch data again.

## Reproduce the report

```bash
# Recreate in-sample metrics and figures.
python scripts/report_assets.py

# Rebuild REPORT.pdf from REPORT.md.
python scripts/build_report.py
```

`REPORT.pdf` is already included in the repository. Its figures are generated under `outputs/figures/`, which is intentionally ignored because it is reproducible output.

## Validation commands

```bash
# Smoke test of data retrieval and cache generation.
python src/data.py

# Final strategy only.
python scripts/run_all.py --strategies final

# Factor-level predictive-content check.
python scripts/factor_ic.py

# Re-run the tilt sensitivity curve.
python scripts/tilt_curve.py
```

There is no separate unit-test suite; these scripts are the end-to-end validation suite. They check data ingestion, signal construction, cost accounting, portfolio construction, performance metrics, and report reproduction.

## Repository layout

| Path | Purpose |
| --- | --- |
| `config.py` | Competition constraints, date ranges, costs, benchmarks, and paths |
| `src/data.py` | NSE constituents and Yahoo Finance market-data retrieval with local caching |
| `src/features.py` | Price/volume factors, liquidity and tradability calculations |
| `src/strategies.py` | Submitted strategy plus comparison strategies |
| `src/backtest.py` | Lookahead-safe event-driven backtester, transaction costs, trade ledger |
| `src/metrics.py` | Return, drawdown, Sharpe, gain-to-loss, accuracy, and trade metrics |
| `scripts/` | Reproducible analysis, report generation, and strategy-validation scripts |
| `REPORT.md` / `REPORT.pdf` | Submission report |

## Data sources and safeguards

- **Constituents:** official NSE index CSVs for Nifty 100, Midcap 100, and Smallcap 100.
- **Prices and volume:** Yahoo Finance through `yfinance`; adjusted daily close is used for price returns.
- **No lookahead:** a signal sees market data only through its signal date and trades at the next session’s close.
- **Missing-price handling:** existing holdings are marked at their latest observed price; unavailable prices are never used for new trades.
- **Liquidity:** volume is used both as a factor and as a hard 10%-of-ADV participation safeguard.

## Limitations

NSE publishes current index membership, so using current constituent lists in a historical test introduces survivorship bias. The report quantifies this using an equal-weight-universe hurdle and documents the limitation. The strategy also depends on Yahoo Finance data and uses the prescribed 0.1% cost model rather than a full market-impact model.

## License

Submitted for the Finesse x Citadel Round 2 competition.

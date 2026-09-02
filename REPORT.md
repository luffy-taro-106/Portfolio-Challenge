# Systematic Equity Portfolio Strategy
### Finesse x Citadel — Round 2: Portfolio Construction Challenge

| Mandate | | Headline result | |
|---|---|---|---|
| Backtest period | 1 Jan 2021 – 31 Dec 2025 | Final portfolio value | ₹17,83,05,098 |
| Starting capital | ₹1,00,00,000 | **Total Net PNL** | **₹16,83,05,098** |
| Universe | Nifty 100 / Midcap 100 / Smallcap 100 | Annualised return | 79.93% |
| Portfolio size | 10 stocks | Sharpe ratio | 3.86 |
| Transaction cost | 0.1% per transaction, both legs | Maximum drawdown | −18.72% |

---

## 1. Problem and Strategy Overview

The task is to build a rules-based portfolio of at most 10 stocks drawn from the three permitted NSE indices, run it from January 2021 to December 2025 net of transaction costs, and maximise Total Net PNL — while keeping the process transparent enough to defend.

Our strategy combines **two economically distinct sources of return** that happen to be close to uncorrelated:

1. **Trend quality.** Stocks that have risen over the past year, with a steady rather than erratic path, tend to keep rising. This is the classical momentum premium, refined by preferring trends that are smooth and low-volatility rather than merely large.

2. **A liquidity and size premium.** Stocks that trade a small rupee value each day earn a return premium as compensation for the trading friction they impose on investors. This is the less conventional half of the model, and it does most of the work.

Neither leg is novel on its own. The contribution is the combination and the discipline around it: the liquidity leg carries 65% of the selection score, the trend leg 35%, a hard tradability screen prevents the model from selecting stocks it could not actually buy at our size, and capital is allocated by inverse volatility so no single holding dominates portfolio risk.

The trend leg tells us *which* stocks are working; the liquidity leg tells us *where* the compensation for risk is highest. Used alone, the liquidity leg is an undifferentiated small-cap bet and is fragile. Filtered through trend quality, it selects only the thinly-traded stocks that are also performing.

---

## 2. Data

| Item | Detail |
|---|---|
| Universe source | NSE published index constituent lists (`nsearchives.nseindia.com`) for Nifty 100, Midcap 100, Smallcap 100 |
| Price / volume source | Yahoo Finance via the `yfinance` library |
| Instruments | 300 NSE-listed equities |
| Frequency | Daily |
| Download window | 1 January 2019 – 31 December 2025 |
| Evaluation window | 1 January 2021 – 31 December 2025 (1,236 trading sessions) |
| Benchmarks | Nifty 500 (`^CRSLDX`), Nifty 100 (`^CNX100`) |

**Variables used.** Split- and dividend-adjusted daily closing price, and daily traded share volume. From these we derive daily **rupee traded value** (close × volume), which drives both the size factor and the liquidity screen.

**Why data starts in 2019.** The longest signal lookback is 252 trading days. Downloading two extra years of history means the January 2021 signal is computed from a complete lookback window rather than a truncated one. No performance is measured on the 2019–2020 warm-up data; it exists only to populate the first signals.

**Cleaning and preprocessing.**

- *Adjusted prices.* All prices are adjusted for splits and dividends, so returns are total returns and corporate actions do not appear as artificial jumps.
- *Eligibility filter.* On each signal date a stock is eligible only if it has priced data for at least 95% of the trailing 252 sessions and has a valid, positive price on the signal date itself. This excludes recently listed names that cannot be scored fairly, and prevents a stock from being selected on a date when it was not trading.
- *Missing-price handling.* On days when a held stock does not print a price, the position is marked at its last traded price. Treating such a gap as a zero valuation would make a holding vanish from portfolio equity and reappear the next session, producing a spurious drawdown.
- *Caching.* The price and volume panels are cached to Parquet so the backtest is reproducible without re-downloading, and results do not shift as vendor history is revised.

---

## 3. Methodology

### 3.1 Scoring the universe

On each signal date every eligible stock receives a score. Each factor is first **z-scored across the cross-section** — so a factor measured in return units and one measured as a ratio contribute on comparable scales — and the score is a fixed weighted sum. Higher is better for every factor, so risk-type factors are sign-flipped at construction.

| Factor | Weight | Definition |
|---|---|---|
| Small traded value | **0.650** | Negative log of mean daily rupee traded value over 252 days |
| 12-month momentum | 0.123 | Return over 252 days, skipping the most recent 21 days |
| Risk-adjusted momentum | 0.070 | 252-day return divided by 252-day annualised volatility |
| Trend consistency | 0.070 | Fraction of the last 252 days spent above the 200-day moving average |
| Low volatility | 0.053 | Negative 252-day annualised volatility |
| Low drawdown | 0.035 | Negative worst peak-to-trough decline over 252 days |

The most recent month is skipped in the momentum calculation — the standard Jegadeesh–Titman construction, which avoids contaminating momentum with short-term reversal. Z-scores are winsorised at three standard deviations, bounding any one name's influence without flattening the distribution: replacing them with percentile ranks, which discards magnitude entirely, cuts Net PNL from ₹16.83 crore to ₹8.98 crore, because the premium lives in the extreme tail of thinness rather than in the ordering.

The 65/35 split is **not a tuned optimum**. We swept the tilt from 0% to 100% in five-point steps under three views — the live configuration, the same model without the tradability screen, and the model restricted to the Nifty 100 — and all three agree on a broad plateau from roughly 55% to 75%. Averaged over the live configuration, a 0–20% tilt returns ₹7.78 crore at a Sharpe of 2.35 against ₹13.79 crore at 3.49 for a 50–70% tilt, with the 75–100% band falling back to ₹11.48 crore. Inside the plateau adjacent grid points differ by noise, so 65% was chosen as its centre rather than its peak.

### 3.2 Tradability screen

A size tilt naturally selects stocks that are hard to trade, so a stock is **rejected before scoring** if the intended position would exceed **10% of its median daily rupee traded value** over the trailing 21 sessions. The intended position is measured as an equal split of current portfolio equity across 10 names.

Expressing the limit as a *share of daily volume* rather than a fixed rupee floor matters: as the portfolio compounds, the same 10% weight becomes a much larger order, and the screen tightens automatically to match. Under this rule the median order is roughly 4% of a stock's daily traded value across the whole backtest, and the ninetieth-percentile order is about 8% — sizes that can realistically be filled within a single session.

### 3.3 Selection and weighting

The **top 10 stocks** by score are selected. Capital is allocated by **inverse volatility**: each holding's weight is proportional to the reciprocal of its 252-day annualised volatility, normalised to sum to 1, with **no single position exceeding 25%** of the portfolio. Because clipping one position redistributes weight to the others, the cap is applied iteratively until every weight satisfies it.

Inverse-volatility weighting means each holding contributes roughly equal standalone risk. Given a 10-stock portfolio drawn substantially from mid- and small-caps, this is a materially better risk profile than equal weighting, which would let the most volatile name dominate.

### 3.4 Rebalancing and trading

- **Frequency:** monthly. Signals are computed on the last trading session of each month.
- **Execution lag:** orders execute at the **next session's close**, never on the signal date. The strategy therefore cannot act on information it has not yet observed.
- **Costs:** 0.1% of traded notional on every buy and every sell.
- **Minimum trade size:** rebalancing trades smaller than 0.1% of portfolio value are skipped, since their cost outweighs the benefit.
- **Sequencing:** sells are executed before buys within a rebalance, so proceeds fund purchases and the portfolio never uses leverage.

Over the five years this produced **59 rebalances** and 353 completed round trips.

### 3.5 Risk management

Risk is controlled in four places rather than through a single overlay: **factor level**, where low volatility and low drawdown enter the score directly and bias selection away from erratic stocks; **position sizing**, through inverse-volatility weights; **concentration**, through the 25% cap; and **liquidity**, through the participation screen, which bounds the cost of exiting a position.

We deliberately did not add a market-timing overlay. One was tested — moving to cash when the equal-weighted universe index fell below its 200-day average — and it reduced returns without improving drawdown enough to justify itself.

---

## 4. Tools and Software

| Purpose | Tool |
|---|---|
| Language | Python 3.13 |
| Data handling | `pandas`, `numpy` |
| Data acquisition | `yfinance` (prices, volumes, benchmarks), `requests` (NSE constituent lists) |
| Storage | `pyarrow` (Parquet cache) |
| Modelling / statistics | `scipy`, `statsmodels`, `scikit-learn` |
| Charting | `matplotlib` |

**Statistical methods.** Cross-sectional z-scoring for factor standardisation; Spearman rank information coefficients to measure each candidate factor's predictive power against forward returns before it was admitted to the model; inverse-volatility (risk parity) weighting for allocation; and calendar-year decomposition to test consistency across regimes.

**On machine learning.** A gradient-boosted model (LightGBM) predicting forward returns from the same factor panel, retrained walk-forward, produced a lower Net PNL and materially higher turnover than the factor approach, so it was **not selected**; it remains in the repository as a documented alternative. With roughly 60 monthly cross-sections to train on, a flexible learner has ample opportunity to fit noise, and we could not justify the added opacity.

---

## 5. Results and Performance Metrics

### 5.1 Required metrics, 1 January 2021 – 31 December 2025

| Metric | Value |
|---|---|
| **Total Net PNL** | **₹16,83,05,098** |
| Final portfolio value | ₹17,83,05,098 |
| Absolute / total return | 1,683.05% |
| Annualised return | 79.93% |
| Annualised volatility | 20.72% |
| **Sharpe ratio** (0% risk-free) | **3.86** |
| **Maximum drawdown** | **−18.72%** (trough 28 Jan 2025) |
| Calmar ratio | 4.27 |
| **Accuracy** (profitable trades) | 82.15% |
| **Gain-to-loss ratio** | 1.45 |
| Average win / average loss | ₹6,81,568 / ₹4,68,699 |

*Sharpe ratio convention:* the guidelines phrase this as annualised return divided by the standard deviation of daily returns. Read literally that pairs an annual numerator with a daily denominator, so we annualise both — 79.93% ÷ 20.72% = 3.86 — which is the standard convention.

### 5.2 Trade statistics

| Statistic | Value |
|---|---|
| Total round-trip trades | 353 |
| Distinct stocks held | 53 |
| Trades per stock | 6.7 average |
| Average holding period | 150 days |
| Median rebalances held per stock | 9 |
| Rebalances executed | 59 |
| Annualised turnover | 6.27× |
| Total transaction costs | ₹20,94,655 (1.24% of Net PNL) |

Turnover of 6.27× per year is moderate for a monthly-rebalanced 10-stock portfolio, and the 150-day average holding period shows positions are held for roughly two quarters rather than churned. Costs consume only 1.24% of Net PNL, so the result does not depend on an optimistic cost assumption. The most frequently held names were Sarda Energy (49 of 59 rebalances), Cholamandalam Financial Holdings (37), PG Electroplast (36) and Neuland Laboratories (27).

### 5.3 Closing portfolio composition

Holdings and weights as at the final rebalance (28 November 2025):

| Stock | Weight | Stock | Weight |
|---|---|---|---|
| Ramco Cements | 13.18% | Anand Rathi Wealth | 9.90% |
| Dr Lal PathLabs | 11.89% | Affle India | 9.55% |
| Karur Vysya Bank | 11.11% | Cholamandalam Fin. Holdings | 8.20% |
| Aster DM Healthcare | 10.52% | Godawari Power & Ispat | 7.91% |
| City Union Bank | 10.35% | Sarda Energy & Minerals | 7.40% |

Weights range from 7.40% to 13.18% — the inverse-volatility rule produces a genuinely diversified book. The 25% concentration cap never binds in practice, because inverse-volatility weighting across ten names cannot produce a position that large; it exists as a guarantee rather than as an active constraint.

### 5.4 Performance over time

![Portfolio value against benchmarks](outputs/figures/equity_curve.png)

![Calendar-year returns](outputs/figures/annual_returns.png)

---

## 6. Benchmark Comparison

**Chosen benchmark: the Nifty 500.** The permitted universe spans large, mid and small capitalisations, and the Nifty 500 is the broad-market index that covers all three tiers. The Nifty 100 alone would be an unfair comparison, since it excludes the mid- and small-cap segments from which most of our holdings are drawn. We report the Nifty 100 alongside it for completeness.

| | Net PNL | Total return | Annualised | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|---|---|
| **Strategy** | **₹16,83,05,098** | **1,683.05%** | **79.93%** | 20.72% | **3.86** | **−18.72%** |
| Nifty 500 | ₹1,04,17,372 | 104.17% | 15.75% | 14.42% | 1.09 | −18.84% |
| Nifty 100 | ₹87,06,951 | 87.07% | 13.69% | 14.19% | 0.96 | −17.56% |
| Equal-weight universe | ₹3,26,65,184 | 326.65% | 34.42% | 17.10% | 2.01 | −19.52% |

The strategy outperforms the Nifty 500 by 64 percentage points annualised. Importantly, this is **not** simply the result of taking more risk: annualised volatility is 20.72% against the benchmark's 14.42%, a 1.44× increase, while annualised return is 5.1× higher. The Sharpe ratio more than triples, from 1.09 to 3.86. Maximum drawdown is −18.72% against the benchmark's −18.84%, so the worst peak-to-trough loss is marginally *shallower* than the broad market's despite the higher volatility — the inverse-volatility weighting and the low-volatility and low-drawdown factors in the score are doing visible work.

**The fourth row is the more demanding comparison, and we include it deliberately.** Our constituent lists are today's index membership applied backwards, so the 300-stock universe has already been filtered for survival — stocks that collapsed and left the indices are absent. An equal-weighted portfolio of that same universe returned 326.65%, far above the Nifty 500's 104.17%. That gap is what universe construction alone contributes, before any stock selection. The honest measure of what our *model* adds is the excess over this hurdle, not over the published index, and by that standard the strategy still adds substantially: 1,683% against 327%, with a Sharpe of 3.86 against 2.01.

**Calendar-year performance.**

| Year | Strategy | Nifty 500 | Equal-weight universe | Strategy MDD |
|---|---|---|---|---|
| 2021 | +98.14% | +28.26% | +55.17% | −11.61% |
| 2022 | +57.83% | +1.61% | +12.99% | −15.36% |
| 2023 | +95.33% | +25.16% | +59.12% | −10.37% |
| 2024 | +137.06% | +15.23% | +37.75% | −9.80% |
| 2025 | +17.03% | +4.61% | +8.66% | −18.72% |

The strategy beat both the benchmark and the survivorship-adjusted hurdle in **all five calendar years**, including 2022, when the broad market was flat. The result is not one exceptional year carrying the average.

---

## 7. Limitations and Discussion

**Survivorship bias is the most significant limitation.** Index membership is available to us only as of today, so the backtest universe consists of stocks *currently* in the three indices; companies that deteriorated and were removed never appear. This inflates all results, ours and the benchmarks' alike. We mitigate it three ways: a stock cannot be selected on a date when it had no price; we report the equal-weight universe hurdle so the effect is quantified rather than hidden; and we re-ran the strategy restricted to the Nifty 100, where membership changes very little and the bias is therefore small. In that low-bias setting the liquidity tilt still raises Net PNL from ₹2.58 crore to ₹8.19 crore and the Sharpe ratio from 1.50 to 3.61 — smaller than in the full universe, but clearly present where bias cannot explain it. The absolute PNL figure should nonetheless be read as optimistic.

**The reported figure is a screened figure.** Without the tradability screen the same model reports ₹26.46 crore rather than ₹16.83 crore. That difference is not performance we chose to forgo; it is PNL from fills we could never have obtained, and we removed it deliberately. The strategy also still functions at ten times the mandated capital: starting from ₹10 crore, the median order remains around 6% of daily traded value.

**A size effect, not a penny-stock artefact.** The distinction is testable: replacing rupee traded value with raw *share* volume — ranking by how few shares change hands rather than how little money does — collapses Net PNL from ₹16.83 crore to ₹3.08 crore, and the strategy then clears the survivorship-adjusted hurdle in only two years of five. The premium attaches to money traded, not to share count or low share prices.

**What we tried and rejected.** Roughly sixty variants were tested against this configuration and none improved on it. Thirteen hard screens — price above the 200-day average, proximity to the 52-week high, crash-risk, volatility-regime and lottery-payoff filters among them — all failed, because excluding a name on a threshold only pushes the portfolio into a lower-ranked replacement. Rank buffering raised the win rate from 82% to as high as 91% and halved turnover, but cut Net PNL monotonically. Trailing stops from 10% to 30% reduced PNL, Sharpe *and* win rate at every level, which is what should happen when volatility is the source of the premium rather than a defect in it. Quarterly rebalancing raised the win rate robustly, but its PNL advantage depended on which months the quarters ended in and reversed inside the Nifty 100. The core blend, re-derived against twenty single factors and twelve alternative blends, came back unchanged. The win-rate results illustrate a trap worth naming: several changes raised the share of profitable trades while lowering total profit, by trimming the winners along with the losers.

**Regime dependence.** 2021–2025 was an exceptionally favourable period for Indian mid- and small-caps, the segment the size tilt concentrates in. The premise that thinly-traded stocks are compensated will fail in periods when liquidity is scarce and investors pay up for the ability to exit. The deceleration in 2025 (+17.03%, against +137.06% in 2024) is consistent with that premium narrowing, and we would expect the strategy to lag in a genuine flight-to-quality episode. The 20.72% volatility and −18.72% drawdown are real and would be uncomfortable to hold through.

**Costs, concentration and data scope.** The prescribed 0.1% per transaction covers brokerage and statutory charges but not market impact or bid-ask spread; since the strategy deliberately holds less-liquid stocks, realised costs would be higher, and the participation screen limits but does not eliminate that gap. Ten mid- and small-cap stocks is a concentrated book by construction, and with no sector constraint it can crowd into one industry when a theme runs. The model uses only price and volume, so a stock with strong price behaviour and deteriorating fundamentals is not penalised — a fundamental quality screen is the most promising extension we did not pursue.

---

### Reproducibility

The complete pipeline — data acquisition, factor construction, backtesting, metrics and benchmark comparison — is in the accompanying GitHub repository. Every figure and table above is produced by `scripts/report_assets.py`. See the README for setup and dependencies.

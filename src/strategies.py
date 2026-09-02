"""Candidate strategies, parameterised so they can be swept.

Strategies A, B and C are all instances of one `FactorStrategy` class: they
differ only in which factors drive selection and how capital is spread across
the names selected. Expressing them as configurations rather than as separate
classes makes the comparison honest, because every variant runs through
identical machinery.

Strategy D is separate because a learned model needs its own training loop.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src import features, screens

warnings.filterwarnings("ignore", category=UserWarning)

# Factor panels and the synthetic index depend only on the signal date, so the
# sweep computes each one once and every configuration reuses it.
_PANEL_CACHE: dict[pd.Timestamp, pd.DataFrame] = {}
_INDEX_CACHE: dict[pd.Timestamp, pd.Series] = {}
_ADV_CACHE: dict[pd.Timestamp, pd.Series | None] = {}


def cached_panel(history: pd.DataFrame) -> pd.DataFrame:
    date = history.index[-1]
    if date not in _PANEL_CACHE:
        _PANEL_CACHE[date] = features.build_factor_panel(history)
    return _PANEL_CACHE[date]


def cached_adv(history: pd.DataFrame) -> pd.Series | None:
    date = history.index[-1]
    if date not in _ADV_CACHE:
        _ADV_CACHE[date] = features.average_daily_value(history)
    return _ADV_CACHE[date]


def universe_index(history: pd.DataFrame) -> pd.Series:
    """Equal-weight index of the universe, rebuilt from the history itself.

    Used by the regime filter. Deriving it from the same price panel keeps the
    strategy self-contained: it needs no data the backtest has not already
    given it, so the evaluators can re-run it on any window.
    """
    date = history.index[-1]
    if date not in _INDEX_CACHE:
        daily = history.pct_change().mean(axis=1).fillna(0.0)
        _INDEX_CACHE[date] = (1 + daily).cumprod()
    return _INDEX_CACHE[date]


# ---------------------------------------------------------------------------
# Weighting schemes
# ---------------------------------------------------------------------------

def _cap_and_normalise(weights: pd.Series, cap: float) -> pd.Series:
    """Scale to sum 1 while holding every position under `cap`.

    Iterative because clipping one name redistributes weight to the others,
    which can push a second name over the cap.
    """
    weights = weights[weights > 0]
    if weights.empty:
        return weights
    weights = weights / weights.sum()
    if cap >= 1.0:
        return weights

    for _ in range(10):
        if (weights <= cap + 1e-9).all():
            break
        spare = (weights[weights > cap] - cap).sum()
        weights[weights > cap] = cap
        room = weights[weights < cap]
        if room.empty:
            break
        weights[room.index] += spare * room / room.sum()

    return weights / weights.sum()


def _equal(selected, scores, history, cap):
    return pd.Series(1.0 / len(selected), index=selected)


def _by_score(selected, scores, history, cap):
    """Allocate in proportion to conviction, shifted to keep weights positive."""
    subset = scores.reindex(selected)
    shifted = subset - subset.min() + subset.std(ddof=0) * 0.5 + 1e-6
    return _cap_and_normalise(shifted, cap)


def _inverse_volatility(selected, scores, history, cap, window: int = 252):
    """Risk parity: each name contributes equal standalone volatility."""
    vol = features.volatility(history[selected], window=window)
    vol = vol.replace(0, np.nan)
    if vol.isna().all():
        return _equal(selected, scores, history, cap)
    vol = vol.fillna(vol.median())
    return _cap_and_normalise(1.0 / vol, cap)


WEIGHTING = {
    "equal": _equal,
    "score": _by_score,
    "invvol": _inverse_volatility,
}


# ---------------------------------------------------------------------------
# Factor strategies (A, B, C and every sweep variant)
# ---------------------------------------------------------------------------

class FactorStrategy:
    """Rank the universe on a weighted factor blend, hold the top N.

    Parameters
    ----------
    factor_weights
        Which columns of the factor panel to score on, and how heavily. Keys
        must exist in `features.build_factor_panel`.
    weighting
        How capital is spread across the selected names: "equal", "score"
        (proportional to conviction) or "invvol" (risk parity).
    regime_filter
        When set, hold cash on any signal date where the equal-weight universe
        index is below its own moving average of this many days. A crude but
        transparent drawdown control.
    max_participation
        Reject any name whose intended position would exceed this share of its
        median daily traded value. A size tilt walks straight into names too
        thin to trade, so without this the backtest books fills it could never
        have got. Expressed as participation rather than as a rupee floor
        because the same floor is punitive on a small book and far too lax
        once the book has compounded.
    min_adv
        Optional absolute rupee floor on traded value, applied alongside the
        participation test. Useful for excluding names that are thin in
        absolute terms even when the book is small.
    screens
        Hard exclusions applied before selection. Unlike factors, these cannot
        be outvoted by a strong reading elsewhere.
    buffer
        Rank buffer, in slots. A name already held is retained while its rank
        stays inside `n_positions + buffer`, rather than being sold the moment
        it slips out of the top N. Two names either side of a hard cutoff
        otherwise swap places on noise alone, and each swap is a round trip
        that is paid for and earns nothing.
    """

    def __init__(
        self,
        name: str,
        factor_weights: dict[str, float],
        n_positions: int = config.MAX_POSITIONS,
        weighting: str = "score",
        cap: float = 0.25,
        regime_filter: int | None = None,
        min_adv: float | None = None,
        max_participation: float | None = None,
        screens: list | None = None,
        buffer: int = 0,
        max_per_industry: int | None = None,
        target_vol: float | None = None,
        vol_window: int = 252,
        score_mode: str = "zscore",
        size_pool: float | None = None,
        score_clip: float = 3.0,
    ) -> None:
        if weighting not in WEIGHTING:
            raise ValueError(f"Unknown weighting '{weighting}'. Options: {list(WEIGHTING)}")
        self.name = name
        self.factor_weights = factor_weights
        self.n_positions = n_positions
        self.weighting = weighting
        self.cap = cap
        self.regime_filter = regime_filter
        self.min_adv = min_adv
        self.max_participation = max_participation
        self.screens = screens or []
        self.buffer = buffer
        self.max_per_industry = max_per_industry
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.score_mode = score_mode
        self.size_pool = size_pool
        self.score_clip = score_clip

    def _risk_off(self, history: pd.DataFrame) -> bool:
        if not self.regime_filter:
            return False
        index = universe_index(history)
        if len(index) < self.regime_filter:
            return False
        return bool(index.iloc[-1] < index.tail(self.regime_filter).mean())

    def _drop_untradeable(
        self, scores: pd.Series, history: pd.DataFrame, equity: float | None
    ) -> pd.Series:
        """Remove names we could not build a position in without moving them."""
        if scores.empty or not (self.min_adv or self.max_participation):
            return scores
        adv = cached_adv(history)
        if adv is None:
            return scores

        traded = adv.reindex(scores.index)
        keep = traded.notna()
        if self.min_adv:
            keep &= traded >= self.min_adv
        if self.max_participation:
            # Size the test on an equal split of the book. The actual weights
            # are not known until after selection, and an equal split is the
            # right order of magnitude for any of the weighting schemes.
            order = (equity or config.STARTING_CAPITAL) / self.n_positions
            keep &= traded * self.max_participation >= order
        return scores[keep]

    def _size_pool(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Keep only the thinnest-traded slice of the universe, then score.

        The alternative to blending size into the score additively. Adding it
        lets a strong momentum reading buy a large, heavily traded name by
        outvoting the size term; gating on it cannot. That makes the size
        exposure a property of the universe rather than an average, so it
        holds regardless of how the other factors happen to read.
        """
        if not self.size_pool or panel.empty or "small_size" not in panel.columns:
            return panel
        size = panel["small_size"].dropna()
        if size.empty:
            return panel
        keep = size.nlargest(max(self.n_positions, int(len(size) * self.size_pool)))
        return panel.loc[panel.index.isin(keep.index)]

    def _industry_limited(self, ranked: pd.Index) -> pd.Index:
        """Drop names once their industry has filled its quota.

        Ten names chosen purely on score can land in three sectors, so the book
        carries a thematic bet nobody decided to take. Capping per industry
        makes that bet explicit and bounded.
        """
        if not self.max_per_industry:
            return ranked
        industry = features.get_industry()
        if industry is None:
            return ranked

        counts: dict[str, int] = {}
        allowed = []
        for symbol in ranked:
            sector = industry.get(symbol)
            if sector is None or pd.isna(sector):
                allowed.append(symbol)
                continue
            if counts.get(sector, 0) < self.max_per_industry:
                counts[sector] = counts.get(sector, 0) + 1
                allowed.append(symbol)
        return pd.Index(allowed)

    def _scale_to_target_vol(
        self, weights: pd.Series, history: pd.DataFrame
    ) -> pd.Series:
        """Shrink gross exposure when the book's own recent volatility is high.

        Barroso and Santa-Clara's risk-managed momentum, with the constraint
        that we cannot lever: the scale is capped at 1, so this can only take
        risk off, never add it.
        """
        if not self.target_vol or weights.empty:
            return weights
        returns = history[weights.index].tail(64).pct_change().iloc[1:]
        if returns.empty:
            return weights
        realised = (returns @ weights).std(ddof=1) * np.sqrt(
            config.TRADING_DAYS_PER_YEAR)
        if not np.isfinite(realised) or realised <= 0:
            return weights
        return weights * min(1.0, self.target_vol / realised)

    def _select(self, scores: pd.Series, holdings: pd.Series | None) -> pd.Index:
        """Top N by score, with incumbents given a stay of execution.

        Without a buffer the book is rebuilt around a hard cutoff, so a name
        oscillating between ranks 10 and 11 is bought and sold repeatedly for
        no reason other than noise. With one, an incumbent has to fall clear
        of the buffer before it is replaced.
        """
        ranked = self._industry_limited(scores.index)
        if not self.buffer or holdings is None or holdings.empty:
            return ranked[: self.n_positions]

        held = set(holdings[holdings > 0].index)
        reprieved = [s for s in ranked[: self.n_positions + self.buffer] if s in held]
        selected = reprieved[: self.n_positions]

        chosen = set(selected)
        for symbol in ranked:
            if len(selected) >= self.n_positions:
                break
            if symbol not in chosen:
                selected.append(symbol)
                chosen.add(symbol)
        return pd.Index(selected)

    def generate_weights(
        self,
        history: pd.DataFrame,
        equity: float | None = None,
        holdings: pd.Series | None = None,
    ) -> pd.Series:
        if self._risk_off(history):
            return pd.Series(dtype=float)

        panel = cached_panel(history)
        panel = self._size_pool(panel)
        scores = features.composite_score(panel, self.factor_weights,
                                          mode=self.score_mode,
                                          clip=self.score_clip)
        scores = self._drop_untradeable(scores, history, equity)
        if self.screens:
            scores = screens.apply(scores, screens.cached_screen_panel(history),
                                   self.screens)
        if scores.empty:
            return pd.Series(dtype=float)

        selected = self._select(scores, holdings)
        if self.weighting == "invvol":
            weights = _inverse_volatility(selected, scores, history, self.cap,
                                          window=self.vol_window)
        else:
            weights = WEIGHTING[self.weighting](selected, scores, history, self.cap)
        return self._scale_to_target_vol(weights, history)


# ---------------------------------------------------------------------------
# Strategy D: gradient-boosted forward-return model
# ---------------------------------------------------------------------------

class MLStrategy:
    """LightGBM ranking stocks by predicted forward return.

    Trained walk-forward: at every retraining date the model sees only factor
    snapshots whose forward return had already been observed by that date. The
    model is deliberately shallow and heavily regularised, because the binding
    constraint here is generalisation, not in-sample fit.
    """

    FEATURES = [
        "momentum_12_1", "momentum_6_1", "momentum_3_0",
        "low_volatility", "low_downside_vol", "risk_adj_momentum",
        "trend_consistency", "low_drawdown", "reversal_1m",
    ]

    def __init__(
        self,
        name: str = "D: Machine Learning",
        n_positions: int = config.MAX_POSITIONS,
        horizon: int = 21,
        retrain_every: int = 6,
        weighting: str = "score",
        cap: float = 0.25,
        n_estimators: int = 200,
        num_leaves: int = 7,
        learning_rate: float = 0.03,
        regime_filter: int | None = None,
        min_train_samples: int = 500,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.n_positions = n_positions
        self.horizon = horizon
        self.retrain_every = retrain_every
        self.weighting = weighting
        self.cap = cap
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.regime_filter = regime_filter
        self.min_train_samples = min_train_samples
        self.seed = seed
        self._model = None
        self._calls = 0

    def _training_set(self, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Factor snapshots paired with their realised forward returns.

        The last usable snapshot is `horizon` days before the signal date;
        anything later would need a return we could not yet have seen.
        """
        cutoff = len(history) - self.horizon
        if cutoff <= 300:
            return pd.DataFrame(), pd.Series(dtype=float)

        frames, targets = [], []
        for i in range(300, cutoff, 21):
            snapshot_date = history.index[i]
            panel = _PANEL_CACHE.get(snapshot_date)
            if panel is None:
                panel = features.build_factor_panel(history.iloc[: i + 1])
                _PANEL_CACHE[snapshot_date] = panel
            if panel.empty:
                continue

            forward = history.iloc[i + self.horizon] / history.iloc[i] - 1.0
            usable = panel[self.FEATURES].dropna()
            forward = forward.reindex(usable.index).dropna()
            if forward.empty:
                continue

            # Rank within the cross-section: the model learns relative
            # attractiveness, which is what a top-N selection actually needs.
            frames.append(usable.loc[forward.index])
            targets.append(forward.rank(pct=True))

        if not frames:
            return pd.DataFrame(), pd.Series(dtype=float)
        return pd.concat(frames), pd.concat(targets)

    def _fit(self, history: pd.DataFrame) -> None:
        import lightgbm as lgb

        X, y = self._training_set(history)
        if len(X) < self.min_train_samples:
            self._model = None
            return

        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=3,
            min_child_samples=50,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=self.seed,
            verbose=-1,
        )
        self._model.fit(X, y)

    def _risk_off(self, history: pd.DataFrame) -> bool:
        if not self.regime_filter:
            return False
        index = universe_index(history)
        if len(index) < self.regime_filter:
            return False
        return bool(index.iloc[-1] < index.tail(self.regime_filter).mean())

    def generate_weights(
        self,
        history: pd.DataFrame,
        equity: float | None = None,
        holdings: pd.Series | None = None,
    ) -> pd.Series:
        if self._calls % self.retrain_every == 0:
            self._fit(history)
        self._calls += 1

        if self._risk_off(history):
            return pd.Series(dtype=float)

        panel = cached_panel(history)
        if panel.empty or self._model is None:
            return pd.Series(dtype=float)

        usable = panel[self.FEATURES].dropna()
        if usable.empty:
            return pd.Series(dtype=float)

        predicted = pd.Series(self._model.predict(usable), index=usable.index)
        predicted = predicted.sort_values(ascending=False)
        selected = predicted.head(self.n_positions).index
        return WEIGHTING[self.weighting](selected, predicted, history, self.cap)


# ---------------------------------------------------------------------------
# The four headline configurations
# ---------------------------------------------------------------------------

BASELINE_FACTORS = {
    "multifactor": {
        "momentum_12_1": 0.35,
        "risk_adj_momentum": 0.20,
        "trend_consistency": 0.20,
        "low_volatility": 0.15,
        "low_drawdown": 0.10,
    },
    "momentum": {"momentum_12_1": 1.0},
    "riskweighted": {
        "momentum_12_1": 0.40,
        "risk_adj_momentum": 0.25,
        "trend_consistency": 0.20,
        "low_downside_vol": 0.15,
    },
}

# Share of the score coming from the size and traded-value tilt, the rest from
# the multi-factor core. Swept from 0% to 100% in 5-point steps across three
# views — the live configuration, without the liquidity screen, and restricted
# to the Nifty 100. All three agree on a broad plateau between roughly 55% and
# 75%, with individual grid points inside it separated by noise rather than by
# signal. 65% is the centre of that plateau, so it is the value furthest from
# the edges where performance degrades, rather than a peak found by searching.
SIZE_TILT = 0.65

FINAL_FACTORS = {
    **{k: v * (1 - SIZE_TILT) for k, v in BASELINE_FACTORS["multifactor"].items()},
    "small_size": SIZE_TILT,
}

# Standard participation ceiling. Costs roughly 40% of the unscreened paper PNL,
# which is the honest measure of how much of that PNL was never obtainable.
MAX_PARTICIPATION = 0.10


def build(key: str, **kwargs):
    """Construct one of the headline strategies by short name."""
    if key == "final":
        # The submission candidate: the multi-factor core (momentum, trend
        # quality, low volatility) tilted towards small, thinly traded names,
        # sized by inverse volatility, and screened so no position exceeds a
        # tenth of its stock's daily traded value.
        #
        # The tilt is the part that does the work, and it survives the two
        # tests that matter. Restricted to the Nifty 100, where index
        # membership barely changes and survivorship bias is therefore small,
        # it still lifts net PNL from Rs 2.58Cr to Rs 8.19Cr. And it beats the
        # bias-adjusted universe hurdle in every one of the five calendar
        # years, so it is not one good regime carrying the average.
        kwargs.pop("regime_filter", None)
        kwargs.setdefault("max_participation", MAX_PARTICIPATION)
        return FactorStrategy("Final: Size-Tilted Multi-Factor / Risk Parity",
                              FINAL_FACTORS, weighting="invvol", **kwargs)
    if key == "multifactor":
        return FactorStrategy("A: Multi-Factor", BASELINE_FACTORS["multifactor"],
                              weighting="score", **kwargs)
    if key == "momentum":
        return FactorStrategy("B: Momentum", BASELINE_FACTORS["momentum"],
                              weighting="equal", **kwargs)
    if key == "riskweighted":
        return FactorStrategy("C: Risk-Weighted", BASELINE_FACTORS["riskweighted"],
                              weighting="invvol", **kwargs)
    if key == "ml":
        return MLStrategy(**kwargs)
    raise KeyError(f"Unknown strategy '{key}'. Options: {list(REGISTRY)}")


REGISTRY = ["final", "multifactor", "momentum", "riskweighted", "ml"]

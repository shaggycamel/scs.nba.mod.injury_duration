"""Duration models, from flat baselines up to gradient-boosted hazards.

All of them answer the same question — given what is known the moment a
player is ruled out, how many games will they miss — so they can be scored
against each other on the same spells with the same metrics.

Three families:

1. Baselines that ignore the features. A Kaplan-Meier mean for everybody,
   then one per injury type, then the player's own history. These matter
   because the duration distribution is so skewed (median 1 game, p99 30)
   that "predict the median" is a genuinely hard baseline to beat on MAE.
2. Discrete-time hazard models, which is where the real work happens: a
   regularised logistic regression, a random forest, sklearn's histogram
   gradient booster, and LightGBM, all fitted to the per-game "do they play
   the next game" target.
3. A direct regression on observed duration, fitted only to spells whose
   return was seen. Not a serious candidate — it is here to measure what
   discarding censored spells costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import hazard as hz
from .design import Design

MAX_K = 100

# Typical days between a team's games, used only to date the grid's final row.
MEAN_GAP_DAYS = 2.3


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


@dataclass
class KMBaseline:
    """Kaplan-Meier point estimate, optionally stratified, shrunk to the pool.

    Stratum estimates are shrunk toward the global value by
    `n / (n + prior_weight)`, so a body-part/ailment cell seen twice does not
    get to assert its own mean.

    `statistic` matters more than it looks. Duration here has a median of 1
    game and a 99th percentile of 30, so the mean sits far above most of the
    distribution: predicting the mean is right for squared error and badly
    wrong for absolute error. Both are reported rather than picking one.
    """

    by: tuple[str, ...] = ()
    prior_weight: float = 10.0
    statistic: str = "mean"
    name: str = "km"
    _global: float = 0.0
    _table: dict = field(default_factory=dict)

    def _point(self, d: np.ndarray, e: np.ndarray) -> float:
        return hz.km_mean(d, e) if self.statistic == "mean" else hz.km_median(d, e)

    def fit(self, spells: pl.DataFrame) -> KMBaseline:
        d = spells["games_missed"].to_numpy()
        e = spells["event"].to_numpy()
        self._global = self._point(d, e)
        self._table = {}
        if self.by:
            for key, grp in spells.group_by(list(self.by)):
                dd = grp["games_missed"].to_numpy()
                ee = grp["event"].to_numpy()
                n = len(dd)
                w = n / (n + self.prior_weight)
                self._table[tuple(key)] = w * self._point(dd, ee) + (1 - w) * self._global
        return self

    def predict(self, spells: pl.DataFrame) -> np.ndarray:
        if not self.by:
            return np.full(spells.height, self._global)
        keys = list(zip(*[spells[c].to_list() for c in self.by]))
        return np.array([self._table.get(k, self._global) for k in keys])


@dataclass
class PlayerHistoryBaseline:
    """The player's own mean prior duration, shrunk toward the global mean."""

    prior_weight: float = 3.0
    name: str = "player_history"
    _global: float = 0.0

    def fit(self, spells: pl.DataFrame) -> PlayerHistoryBaseline:
        self._global = hz.km_mean(spells["games_missed"].to_numpy(), spells["event"].to_numpy())
        return self

    def predict(self, spells: pl.DataFrame) -> np.ndarray:
        n = spells["prior_spells_career"].fill_null(0).to_numpy().astype(float)
        total = spells["prior_games_missed_career"].fill_null(0).to_numpy().astype(float)
        mean_prior = np.where(n > 0, total / np.maximum(n, 1), self._global)
        w = n / (n + self.prior_weight)
        return w * mean_prior + (1 - w) * self._global


# --------------------------------------------------------------------------
# Hazard models
# --------------------------------------------------------------------------


@dataclass
class HazardModel:
    """A binary classifier over per-missed-game rows, read as a hazard.

    `kind` picks both the estimator and the encoding it can digest: the
    boosters take integer category codes and raw NaNs, the linear model and
    the forest take one-hot columns and imputed values.
    """

    name: str
    kind: str
    numeric: list[str]
    categorical: list[str]
    params: dict = field(default_factory=dict)
    _model: object | None = None
    _design: Design | None = None
    _cat_idx: list[int] = field(default_factory=list)
    _names: list[str] = field(default_factory=list)

    NATIVE_CATEGORICAL = ("lightgbm", "histgb")

    @property
    def uses_codes(self) -> bool:
        return self.kind in self.NATIVE_CATEGORICAL

    def _build(self) -> Any:
        if self.kind == "logistic":
            # Scaling is not optional here: the one-hot block sits next to
            # raw minute totals in the thousands, and an L2 penalty applied
            # to both on the same scale is not the model we asked for.
            return Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("est", LogisticRegression(max_iter=3000, **self.params)),
                ]
            )
        if self.kind == "forest":
            return RandomForestClassifier(random_state=0, n_jobs=-1, **self.params)
        if self.kind == "histgb":
            return HistGradientBoostingClassifier(
                random_state=0, categorical_features=self._cat_idx, **self.params
            )
        if self.kind == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                random_state=0,
                verbose=-1,
                n_jobs=-1,
                # Without these the multithreaded histogram build is not
                # reproducible, and held-out log loss moves by enough
                # between runs to flip the smallest ablation deltas.
                deterministic=True,
                force_row_wise=True,
                **self.params,
            )
        raise ValueError(f"unknown kind {self.kind!r}")

    def _matrix(self, rows: pl.DataFrame) -> np.ndarray:
        if self.uses_codes:
            X, self._cat_idx, self._names = self._design.codes(rows)
            return X
        X, self._names = self._design.onehot(rows)
        return X

    def fit(self, rows: pl.DataFrame, target: str = "returns_next") -> HazardModel:
        self._design = Design(self.numeric, self.categorical).fit(rows)
        X = self._matrix(rows)
        y = rows[target].to_numpy()
        self._model = self._build()
        if self.kind == "lightgbm":
            self._model.fit(X, y, categorical_feature=self._cat_idx)
        else:
            self._model.fit(X, y)
        return self

    def hazard(self, rows: pl.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit first")
        return self._model.predict_proba(self._matrix(rows))[:, 1]


# Horizons, in days, for the "will he be available by then" answers.
DAY_HORIZONS = (7, 14, 21, 30, 60)


def predict_from_grid(model: HazardModel, grid: pl.DataFrame, max_k: int = MAX_K) -> pl.DataFrame:
    """Turn per-(spell, k) hazards into per-spell duration summaries.

    Reports duration in both games and days. Days are converted from the
    per-game hazard using the team's real calendar rather than modelled
    directly, because a player returns *for a game*, not on an arbitrary
    Tuesday: gaps between games run from 1 to 7 days, so a daily hazard would
    be mostly structural zeros dictated by the schedule. The conversion is
    exact, not an average-gap approximation.

    The arithmetic: a spell that misses exactly k games ends when the player
    plays game k+1, so

        P(misses exactly k) = S(k-1) - S(k)
        days out            = date(game k+1) - date(first missed game)

    and the grid already carries that date difference as
    `days_missed_so_far`, shifted one row forward.
    """
    h = model.hazard(grid)
    g = (
        grid.select("spell_id", "k", "days_missed_so_far")
        .with_columns(pl.Series("h", h))
        .sort("spell_id", "k")
        .with_columns(
            (1.0 - pl.col("h").clip(1e-9, 1 - 1e-9))
            .log()
            .cum_sum()
            .over("spell_id")
            .exp()
            .alias("surv")
        )
        .with_columns(
            pl.col("surv").shift(1, fill_value=1.0).over("spell_id").alias("surv_prev"),
            # The date the player would be back if this is the last game missed.
            pl.col("days_missed_so_far")
            .shift(-1)
            .over("spell_id")
            .cast(pl.Float64)
            .alias("return_days"),
        )
        .with_columns((pl.col("surv_prev") - pl.col("surv")).alias("p_return_here"))
    )
    # Past the last grid game the return date is unknown; carry the final gap
    # so the last row still has a date to attribute its mass to.
    g = g.with_columns(
        pl.col("return_days").fill_null(
            pl.col("days_missed_so_far").cast(pl.Float64) + MEAN_GAP_DAYS
        )
    )

    aggs = [
        (1.0 + pl.col("surv").sum()).alias("pred_mean_games"),
        pl.col("k").filter(pl.col("surv") <= 0.5).min().alias("pred_median_games"),
        (1.0 - pl.col("surv").filter(pl.col("k") == 1).first()).alias("p_back_next_game"),
        (1.0 - pl.col("surv").filter(pl.col("k") == 3).first()).alias("p_back_within_3"),
        (1.0 - pl.col("surv").filter(pl.col("k") == 10).first()).alias("p_back_within_10"),
        pl.col("surv").filter(pl.col("k") == 20).first().alias("p_out_past_20"),
        # Days. The mean adds the surviving tail at the last grid date, which
        # makes it a lower bound for the few spells with mass beyond max_k.
        (
            (pl.col("p_return_here") * pl.col("return_days")).sum()
            + pl.col("surv").last() * pl.col("return_days").last()
        ).alias("pred_mean_days"),
        pl.col("return_days").filter(pl.col("surv") <= 0.5).min().alias("pred_median_days"),
    ]
    aggs += [
        pl.col("p_return_here")
        .filter(pl.col("return_days") <= d)
        .sum()
        .alias(f"p_back_within_{d}d")
        for d in DAY_HORIZONS
    ]

    out = g.group_by("spell_id").agg(aggs)
    last_day = g.group_by("spell_id").agg(pl.col("return_days").max().alias("_last_day"))
    return (
        out.join(last_day, on="spell_id", how="left")
        .with_columns(
            pl.col("pred_median_games").fill_null(max_k + 1).cast(pl.Float64),
            # No median inside the grid means more than half the mass is past
            # its end; report the horizon rather than a silent null.
            pl.col("pred_median_days").fill_null(pl.col("_last_day")),
        )
        .drop("_last_day")
    )


# --------------------------------------------------------------------------
# Direct regression on observed duration (censoring-blind, for contrast)
# --------------------------------------------------------------------------


@dataclass
class ObservedOnlyRegressor:
    """LightGBM regression fitted to uncensored spells only.

    Included to quantify the cost of the obvious shortcut: dropping the 22%
    of spells whose return was never seen removes most of the long ones, and
    the fitted model inherits that. `objective` chooses what it targets, so
    it can be compared against a hazard model's mean and median on equal
    footing instead of only where its training population happens to sit.
    """

    numeric: list[str]
    categorical: list[str]
    objective: str = "poisson"
    name: str = "lgbm (observed spells only)"
    params: dict = field(default_factory=dict)
    _model: object | None = None
    _design: Design | None = None
    _cat_idx: list[int] = field(default_factory=list)

    def fit(self, spells: pl.DataFrame) -> ObservedOnlyRegressor:
        import lightgbm as lgb

        obs = spells.filter(pl.col("event") == 1)
        self._design = Design(self.numeric, self.categorical).fit(obs)
        X, self._cat_idx, _ = self._design.codes(obs)
        extra = {"alpha": 0.5} if self.objective == "quantile" else {}
        self._model = lgb.LGBMRegressor(
            objective=self.objective,
            random_state=0,
            verbose=-1,
            n_jobs=-1,
            deterministic=True,
            force_row_wise=True,
            **extra,
            **self.params,
        )
        self._model.fit(X, obs["games_missed"].to_numpy(), categorical_feature=self._cat_idx)
        return self

    def predict(self, spells: pl.DataFrame) -> np.ndarray:
        X, _, _ = self._design.codes(spells)
        return self._model.predict(X)

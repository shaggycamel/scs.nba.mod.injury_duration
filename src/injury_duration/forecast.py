"""Produce usable forecasts from a fitted hazard model.

Two modes, matching the two moments someone actually asks the question:

* `forecast_at_onset` — the player has just been ruled out. Everything comes
  from the injury as filed, the player's recent workload, and the schedule.
* `forecast_in_progress` — the player has been out for a while and the report
  has moved. Adds what the report has said since, which is worth about a
  point of AUC on the next-game call.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import experiment as ex
from . import models

BEST_LGBM = {
    "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15,
    "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8,
}


def fit(design: pl.DataFrame, cols: list[str], params: dict | None = None):
    num, cat = ex.split_numeric_categorical(cols)
    return models.HazardModel(
        "lightgbm", "lightgbm", num, cat, params or BEST_LGBM
    ).fit(design)


def forecast_at_onset(
    model: models.HazardModel,
    spell_features: pl.DataFrame,
    tctx: pl.DataFrame,
    max_k: int = models.MAX_K,
) -> pl.DataFrame:
    """Expected games missed and return probabilities for each spell's day one."""
    grid = ex.grid_design(spell_features, tctx, max_k=max_k)
    pred = models.predict_from_grid(model, grid, max_k=max_k)
    return spell_features.select(
        "spell_id", "player_name", "season", "start_date", "team_slug_start",
        "body_region", "ailment_class", "index_status", "index_reason",
        "games_missed", "days_out", "event", "censor_reason",
    ).join(pred, on="spell_id", how="left")


def forecast_in_progress(
    model: models.HazardModel, design_rows: pl.DataFrame
) -> pl.DataFrame:
    """P(plays the next game) at each game of each spell, as it unfolded."""
    p = model.hazard(design_rows)
    return design_rows.select(
        "spell_id", "games_missed_so_far", "game_date", "status_clean",
        "returns_next",
    ).with_columns(pl.Series("p_plays_next", np.round(p, 4)))


def survival_curve(
    model: models.HazardModel,
    spell_features: pl.DataFrame,
    tctx: pl.DataFrame,
    max_k: int = models.MAX_K,
) -> pl.DataFrame:
    """P(still out after k games), per spell, for k = 1..max_k."""
    grid = ex.grid_design(spell_features, tctx, max_k=max_k)
    h = model.hazard(grid)
    return (
        grid.select("spell_id", "k")
        .with_columns(pl.Series("hazard", h))
        .sort("spell_id", "k")
        .with_columns(
            (1 - pl.col("hazard").clip(1e-9, 1 - 1e-9)).log().cum_sum()
            .over("spell_id").exp().alias("p_still_out")
        )
    )


def summarise(forecast: pl.DataFrame) -> pl.DataFrame:
    """Round a forecast table down to the columns a person would read."""
    return forecast.select(
        "player_name",
        "start_date",
        pl.col("body_region").alias("region"),
        pl.col("ailment_class").alias("ailment"),
        pl.col("index_status").alias("status"),
        pl.col("pred_median_games").round(0).alias("pred_games_median"),
        pl.col("pred_mean_games").round(1).alias("pred_games_mean"),
        pl.col("p_back_next_game").round(2).alias("p_next"),
        pl.col("p_back_within_3").round(2).alias("p_3"),
        pl.col("p_back_within_10").round(2).alias("p_10"),
        pl.col("games_missed").alias("actual_games"),
        pl.col("event").alias("return_seen"),
    )

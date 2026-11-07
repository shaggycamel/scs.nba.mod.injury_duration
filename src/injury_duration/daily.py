"""Daily as-of states and return-date forecasts.

The stored model is a per-missed-game hazard: at each missed game k it asks
whether the player returns for game k+1. A daily forecast starts from the
last missed game on or before the prediction date, freezes everything the
report said by then, and rolls the hazard forward over the team's remaining
schedule.

The as-of rule is conservative: a report row is used only once its game date
has arrived (`game_date <= prediction_date`), so the next game's report cannot
leak into today's prediction.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from . import experiment as ex
from . import hazard as hz
from . import models

EPOCH = date(1970, 1, 1)


def _date_from_days(expr: pl.Expr) -> pl.Expr:
    return pl.lit(EPOCH) + pl.duration(days=expr.round().cast(pl.Int64))


def build_daily_states(
    spells: pl.DataFrame,
    features: pl.DataFrame,
    rows: pl.DataFrame,
    cutoff: date,
    through: date | None = None,
) -> pl.DataFrame:
    """One row per injured player per calendar day, as known that day.

    `through` forces predictions to the data cutoff even when a spell has a
    non-null next_date. The daily job uses it for active spells; the historical
    backfill leaves it unset and stops the day before the return/censor event.
    """
    design = ex.hazard_design(rows, features).filter(
        pl.col("spell_id").is_in(spells["spell_id"].to_list())
    )
    meta = spells.select(
        "spell_id", "player_name", "start_date", "last_missed_date", "next_date",
        "event", "games_missed", "censor_reason",
    )

    # Generate the calendar days a prediction is wanted for: from the first
    # missed game through the day before the return (observed spells), or
    # through the data cutoff (censored spells).
    days = []
    for r in meta.iter_rows(named=True):
        # A non-null next_date is the first game after the spell: a return,
        # a G League assignment, a trade or a dropped report. Predictions stop
        # the day before it. Only a season-end censor has no next date, and
        # those predictions continue to the data cutoff.
        if through is not None:
            end = through
        else:
            end = r["next_date"] - timedelta(days=1) if r["next_date"] else cutoff
        end = max(end, r["last_missed_date"])
        if end < r["start_date"]:
            continue
        d = r["start_date"]
        while d <= end:
            days.append(
                {
                    "spell_id": r["spell_id"],
                    "player_name": r["player_name"],
                    "prediction_date": d,
                    "start_date": r["start_date"],
                    "next_date": r["next_date"],
                    "censor_reason": r["censor_reason"],
                }
            )
            d += timedelta(days=1)
    days = pl.DataFrame(days)
    if days.is_empty():
        return days

    design = design.sort("spell_id", "game_date")
    states = days.sort("spell_id", "prediction_date").join_asof(
        design,
        left_on="prediction_date",
        right_on="game_date",
        by="spell_id",
        strategy="backward",
    )
    states = states.filter(pl.col("game_date").is_not_null()).with_columns(
        (pl.col("prediction_date") - pl.col("start_date")).dt.total_days()
        .alias("days_missed_so_far"),
        (pl.col("games_missed_so_far").cast(pl.Float64) + 1).log()
        .alias("log_games_missed_so_far"),
        pl.int_range(0, pl.len()).alias("state_id"),
    )
    return states


def _impute_missing_history(
    states: pl.DataFrame, model: models.HazardModel
) -> pl.DataFrame:
    """Fill player-history features for players with no NBA games on record.

    The model was trained with nulls, and for ordinary players those nulls are
    informative. A two-way player with no box-score history is different: every
    attribute and workload column is null, which sends the booster down a leaf
    learned for fringe players and produces absurd 90-game medians for a
    one-week injury. For those rows only, impute the training medians and mark
    the row so the caller can see it.
    """
    no_history = pl.col("career_games").is_null() | (pl.col("career_games") == 0)
    med = {c: v for c, v in model._design.medians.items() if c in states.columns}
    cats = [c for c in model._design.categorical if c in states.columns]
    numeric = [
        pl.when(no_history)
        .then(pl.col(c).cast(pl.Float64, strict=False).fill_null(float(med[c])))
        .otherwise(pl.col(c).cast(pl.Float64, strict=False))
        .alias(c)
        for c in med
    ]
    categorical = [
        pl.when(no_history)
        .then(pl.col(c).fill_null("unknown"))
        .otherwise(pl.col(c))
        .alias(c)
        for c in cats
    ]
    return states.with_columns(numeric + categorical + [no_history.alias("history_imputed")])


def forecast_daily(
    model: models.HazardModel,
    states: pl.DataFrame,
    tctx: pl.DataFrame,
    max_k: int = 100,
) -> pl.DataFrame:
    """Roll the hazard forward from each daily state and summarise the return."""
    if states.is_empty():
        return states

    states = _impute_missing_history(states, model)

    steps = pl.DataFrame({"step": list(range(1, max_k + 1))}, schema={"step": pl.Int64})
    grid = states.join(steps, how="cross").with_columns(
        (pl.col("team_game_idx") + pl.col("step")).alias("future_idx")
    )

    regular = tctx.filter(pl.col("season_type") == "Regular Season")
    fut = regular.select(
        "season", "team_slug", "team_game_idx", "game_date", "season_type",
        "days_rest", "team_games_next_14d",
    )
    grid = grid.join(
        fut.rename(
            {
                "game_date": "future_date",
                "season_type": "future_season_type",
                "days_rest": "future_days_rest",
                "team_games_next_14d": "future_next_14d",
            }
        ),
        left_on=["season", "team_slug", "future_idx"],
        right_on=["season", "team_slug", "team_game_idx"],
        how="left",
    )
    # Gap from the future missed game to the one after it, matching how the
    # training rows define `days_to_next_game`.
    nxt = fut.select(
        "season", "team_slug",
        (pl.col("team_game_idx") - 1).alias("future_idx"),
        pl.col("days_rest").cast(pl.Float64).alias("_future_gap"),
    )
    grid = grid.join(nxt, on=["season", "team_slug", "future_idx"], how="left")

    last = regular.group_by("season", "team_slug").agg(
        pl.col("team_game_idx").max().alias("last_idx"),
        pl.col("game_date").max().alias("last_date"),
    )
    grid = grid.join(last, on=["season", "team_slug"], how="left")
    beyond = pl.col("future_date").is_null()
    grid = grid.with_columns(
        pl.when(beyond)
        .then(
            pl.col("last_date")
            + pl.duration(
                days=(
                    (pl.col("future_idx") - pl.col("last_idx")).cast(pl.Float64)
                    * hz.MEAN_DAYS_BETWEEN_GAMES
                ).round()
            )
        )
        .otherwise(pl.col("future_date"))
        .alias("future_date"),
        pl.col("_future_gap").fill_null(hz.MEAN_DAYS_BETWEEN_GAMES)
        .alias("days_to_next_game"),
        pl.col("future_next_14d").fill_null(round(14 / hz.MEAN_DAYS_BETWEEN_GAMES)),
        pl.col("future_season_type").eq("Playoffs").fill_null(False)
        .alias("in_playoffs_now"),
    )

    grid = grid.with_columns(
        (pl.col("games_missed_so_far") + pl.col("step")).alias("games_missed_so_far"),
        (pl.col("future_date") - pl.col("start_date")).dt.total_days()
        .alias("days_missed_so_far"),
        (pl.col("days_to_next_game") == 1).alias("next_is_b2b"),
        pl.col("future_next_14d").alias("team_games_next_14d_now"),
        (pl.col("games_missed_so_far") + pl.col("step")).cast(pl.Float64).add(1).log()
        .alias("log_games_missed_so_far"),
    )

    h = model.hazard(grid)
    grid = grid.with_columns(pl.Series("hazard", h)).sort("state_id", "step")
    grid = grid.with_columns(
        (1 - pl.col("hazard")).cum_prod().over("state_id").alias("surv")
    ).with_columns(
        (
            pl.col("hazard")
            * pl.col("surv").shift(1).fill_null(1).over("state_id")
        ).alias("p_return"),
    ).with_columns(
        pl.col("p_return").cum_sum().over("state_id").alias("cum_return"),
        (
            pl.col("p_return")
            * pl.col("future_date").dt.epoch("d")
        ).sum().over("state_id").alias("_mean_ord"),
    ).with_columns(
        (1 - pl.col("cum_return").last().over("state_id")).alias("_tail"),
        pl.col("future_date").dt.epoch("d").last().over("state_id").alias("_last_ord"),
    )

    summary = grid.group_by("state_id").agg(
        pl.col("step").filter(pl.col("cum_return") >= 0.5).min().alias("median_step"),
        (1 + pl.col("surv").sum()).alias("mean_games"),
        pl.col("p_return").first().alias("p_return_next_game"),
        pl.col("p_return")
        .filter((pl.col("future_date") - pl.col("prediction_date")).dt.total_days() <= 7)
        .sum().alias("p_return_within_7d"),
        pl.col("p_return")
        .filter((pl.col("future_date") - pl.col("prediction_date")).dt.total_days() <= 14)
        .sum().alias("p_return_within_14d"),
        pl.col("p_return")
        .filter((pl.col("future_date") - pl.col("prediction_date")).dt.total_days() <= 30)
        .sum().alias("p_return_within_30d"),
        pl.coalesce(
            pl.col("future_date").filter(pl.col("cum_return") >= 0.5).min(),
            pl.col("future_date").max(),
        ).alias("pred_return_date_median"),
        pl.col("_mean_ord").first().alias("_mean_ord"),
        pl.col("_tail").first().alias("_tail"),
        pl.col("_last_ord").first().alias("_last_ord"),
    ).with_columns(
        _date_from_days(
            pl.col("_mean_ord") + pl.col("_tail") * (pl.col("_last_ord") + hz.MEAN_DAYS_BETWEEN_GAMES)
        ).alias("pred_return_date_mean"),
        pl.col("median_step").fill_null(max_k).alias("pred_games_remaining_median"),
        pl.col("mean_games").alias("pred_games_remaining_mean"),
    )

    keep = [
        "state_id", "spell_id", "player_id", "player_name", "season", "team_slug",
        "start_date", "prediction_date", "games_missed_so_far", "days_missed_so_far",
        "status_clean", "body_region", "ailment_class", "games_missed", "event",
        "next_date", "censor_reason", "history_imputed",
    ]
    return states.select(keep).join(summary, on="state_id", how="left")

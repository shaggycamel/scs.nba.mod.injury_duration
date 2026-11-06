"""Discrete-time hazard framing, survival utilities, and censoring-aware metrics.

Why a hazard model rather than regression on games missed
---------------------------------------------------------
22% of spells never show a return — the season ends, or the player is traded
or sent to the G League while still out. Regressing on observed duration
throws those away (and they are the long ones, so the model learns a world
where nobody misses 40 games) or keeps them at their truncated length (and
learns one where season-ending injuries last two weeks). A discrete-time
hazard model uses every game of every spell: each game contributes "still
out" and only the last game of an observed spell contributes "came back".

The framing: for a spell that misses games 1..m, row k asks "the player has
now missed k games — do they play game k+1?". The answer is known for every
k < m (no) and for k = m only when the return was observed. So a censored
spell contributes its m "no" rows and no "yes" row, which is exactly the
information it carries.

Everything downstream — expected games missed, the probability of being back
within three games, survival curves by injury type — is then a product over
these per-game hazards.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# Time-varying columns the dynamic model may use. Each is a function of the
# report up to and including game k, or of the schedule, which is fixed in
# advance. Nothing here peeks at game k+1.
MEAN_DAYS_BETWEEN_GAMES = 2.3

TIME_VARYING = [
    "games_missed_so_far",
    "days_missed_so_far",
    "log_games_missed_so_far",
    "status_now_out",
    "status_now_doubtful",
    "status_now_questionable",
    "status_now_probable",
    "status_ever_softened",
    "games_since_softened",
    "region_changed",
    "ailment_changed",
    "is_management_now",
    "is_recovery_now",
    "is_surgical_now",
    "days_to_next_game",
    "next_is_b2b",
    "team_games_next_14d_now",
    "in_playoffs_now",
    "hope_now",
    "eliminated_now",
]


def build_hazard_rows(spells: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Expand spells into one row per missed game, with the per-game target.

    Returns the panel rows belonging to a spell, carrying `returns_next`
    (the hazard target), the elapsed-time columns, and the report state as of
    that game.
    """
    rows = panel.filter(pl.col("in_spell")).select(
        "spell_id", "spell_game_no", "player_id", "season", "team_slug",
        "game_date", "game_id", "team_game_idx", "player_game_idx", "season_type",
        "days_rest", "status_clean", "reason_category", "body_region",
        "ailment_class", "is_management", "is_recovery_stage", "is_surgical",
        "is_bridged",
    )

    meta = spells.select(
        "spell_id", "games_missed", "event", "start_date", "body_region", "ailment_class"
    ).rename({"body_region": "index_region", "ailment_class": "index_ailment"})
    rows = rows.join(meta, on="spell_id", how="inner")

    # The target: did the player come back for the next game? Known for every
    # game but the last of a censored spell, where it is simply unobserved —
    # and a censored spell has no row with the answer.
    rows = rows.with_columns(
        (
            (pl.col("spell_game_no") == pl.col("games_missed")) & (pl.col("event") == 1)
        ).cast(pl.Int8).alias("returns_next")
    )

    rows = rows.sort("spell_id", "spell_game_no").with_columns(
        pl.col("spell_game_no").alias("games_missed_so_far"),
        (pl.col("game_date") - pl.col("start_date")).dt.total_days()
        .alias("days_missed_so_far"),
        (pl.col("season_type") == "Playoffs").alias("in_playoffs_now"),
        (pl.col("status_clean") == "Out").alias("status_now_out"),
        (pl.col("status_clean") == "Doubtful").alias("status_now_doubtful"),
        (pl.col("status_clean") == "Questionable").alias("status_now_questionable"),
        (pl.col("status_clean") == "Probable").alias("status_now_probable"),
        (pl.col("body_region") != pl.col("index_region")).alias("region_changed"),
        (pl.col("ailment_class") != pl.col("index_ailment")).alias("ailment_changed"),
        pl.col("is_management").fill_null(False).alias("is_management_now"),
        pl.col("is_recovery_stage").fill_null(False).alias("is_recovery_now"),
        pl.col("is_surgical").fill_null(False).alias("is_surgical_now"),
    )

    # "Softened" = the report moved off Out to any of the maybe-plays labels.
    # Once that happens the player is usually days away, and how long ago it
    # happened matters too.
    softened = pl.col("status_clean").is_in(["Doubtful", "Questionable", "Probable", "Available"])
    rows = rows.with_columns(softened.alias("_soft")).with_columns(
        pl.col("_soft").cum_sum().over("spell_id").gt(0).alias("status_ever_softened"),
        pl.when(pl.col("_soft"))
        .then(pl.col("spell_game_no"))
        .otherwise(None)
        .forward_fill()
        .over("spell_id")
        .alias("_first_soft_at"),
    )
    rows = rows.with_columns(
        (pl.col("spell_game_no") - pl.col("_first_soft_at")).fill_null(-1)
        .alias("games_since_softened"),
        (pl.col("spell_game_no").cast(pl.Float64) + 1).log().alias("log_games_missed_so_far"),
    ).drop("_soft", "_first_soft_at")

    return rows


def attach_schedule_density(rows: pl.DataFrame, tctx: pl.DataFrame) -> pl.DataFrame:
    """Add the forward schedule as of each missed game.

    Both columns come from the team's own schedule rather than from the
    spell's rows. Shifting within the spell looks equivalent and is not: the
    last row of a spell has no next row, and the last row of an *observed*
    spell is precisely the one where the player came back, so whatever value
    fills that gap becomes a near-perfect marker for the target. That leak
    pushed per-game AUC to 0.98 and made the rolled-forward duration
    predictions useless, because a real schedule never contains the sentinel.
    """
    incentive_now = tctx.select(
        "season", "game_id", "team_slug",
        pl.col("playin_probability").alias("hope_now"),
        pl.col("eliminated").alias("eliminated_now"),
    )
    next_gap = tctx.select(
        "season", "team_slug",
        (pl.col("team_game_idx") - 1).alias("team_game_idx"),
        pl.col("days_rest").cast(pl.Float64).alias("days_to_next_game"),
    )
    out = (
        rows.join(
            tctx.select(
                "season", "game_id", "team_slug",
                pl.col("team_games_next_14d").alias("team_games_next_14d_now"),
            ),
            on=["season", "game_id", "team_slug"],
            how="left",
        )
        .join(next_gap, on=["season", "team_slug", "team_game_idx"], how="left")
        .join(incentive_now, on=["season", "game_id", "team_slug"], how="left")
        .with_columns(
            pl.col("team_games_next_14d_now").fill_null(0),
            # Only the team's final game of a season has no successor. Filling
            # with the typical gap keeps that value indistinguishable from an
            # ordinary two-day rest rather than making it a flag.
            pl.col("days_to_next_game").fill_null(MEAN_DAYS_BETWEEN_GAMES),
        )
    )
    return out.with_columns((pl.col("days_to_next_game") == 1).alias("next_is_b2b"))


# --------------------------------------------------------------------------
# Survival arithmetic
# --------------------------------------------------------------------------

def kaplan_meier(duration: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival curve for discrete durations.

    Returns `(times, survival)` where `survival[i]` is P(duration > times[i]).
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=float)
    times = np.unique(duration[event == 1])
    surv = np.ones(len(times))
    s = 1.0
    for i, t in enumerate(times):
        at_risk = (duration >= t).sum()
        died = ((duration == t) & (event == 1)).sum()
        if at_risk > 0:
            s *= 1.0 - died / at_risk
        surv[i] = s
    return times, surv


def km_mean(duration: np.ndarray, event: np.ndarray, horizon: int | None = None) -> float:
    """Restricted mean duration from the KM curve.

    The mean of a right-censored sample is only identified up to the largest
    observed event time, so it is restricted to `horizon` (default: the
    largest observed duration).
    """
    times, surv = kaplan_meier(duration, event)
    if len(times) == 0:
        return float(np.mean(duration))
    hi = horizon if horizon is not None else int(np.max(duration))
    grid = np.arange(1, hi + 1)
    s_grid = np.ones(len(grid))
    for i, g in enumerate(grid):
        idx = np.searchsorted(times, g, side="right") - 1
        s_grid[i] = surv[idx] if idx >= 0 else 1.0
    # E[T] = sum_{j>=0} P(T > j), and P(T > 0) = 1 for every spell.
    return float(1.0 + s_grid[:-1].sum()) if len(s_grid) > 1 else 1.0


def km_median(duration: np.ndarray, event: np.ndarray) -> float:
    times, surv = kaplan_meier(duration, event)
    below = np.where(surv <= 0.5)[0]
    if len(below) == 0:
        return float(np.max(duration))
    return float(times[below[0]])


def survival_from_hazards(h: np.ndarray) -> np.ndarray:
    """P(still out after k games) for k = 1..len(h), given per-game hazards."""
    return np.cumprod(1.0 - np.clip(h, 1e-9, 1 - 1e-9))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def concordance_index(risk: np.ndarray, duration: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C for right-censored data.

    A pair is comparable when the shorter duration is an observed event.
    `risk` should be higher for spells expected to end sooner.
    """
    risk = np.asarray(risk, dtype=float)
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=bool)

    order = np.argsort(duration)
    risk, duration, event = risk[order], duration[order], event[order]

    conc = disc = tied = 0.0
    for i in range(len(duration)):
        if not event[i]:
            continue
        later = duration > duration[i]
        if not later.any():
            continue
        r_j = risk[later]
        conc += float((risk[i] > r_j).sum())
        disc += float((risk[i] < r_j).sum())
        tied += float((risk[i] == r_j).sum())
    total = conc + disc + tied
    return float("nan") if total == 0 else (conc + 0.5 * tied) / total


def horizon_table(duration: np.ndarray, event: np.ndarray, horizon: int) -> np.ndarray:
    """Mask of spells whose status at `horizon` games is known.

    A spell censored at or before the horizon is unusable: we do not know
    whether it would still have been running. Everything else is usable with
    a hard label, which keeps the horizon metrics free of inverse-probability
    weighting.
    """
    duration = np.asarray(duration)
    event = np.asarray(event, dtype=bool)
    returned_by = event & (duration <= horizon)
    still_out = duration > horizon
    return returned_by | still_out


def horizon_label(duration: np.ndarray, event: np.ndarray, horizon: int) -> np.ndarray:
    """1 if the spell was still running after `horizon` games."""
    duration = np.asarray(duration)
    return (duration > horizon).astype(int)


# --------------------------------------------------------------------------
# Prediction grid
# --------------------------------------------------------------------------
# Columns that vary with elapsed time but are still knowable the moment a
# player is ruled out, because the schedule is fixed months in advance.
SCHEDULE_TIME_COLS = [
    "games_missed_so_far",
    "days_missed_so_far",
    "log_games_missed_so_far",
    "days_to_next_game",
    "next_is_b2b",
    "team_games_next_14d_now",
    "in_playoffs_now",
]

# Columns that only become known as the spell unfolds. The index model must
# not use these; the dynamic model exists to use them.
REPORT_TIME_COLS = [c for c in TIME_VARYING if c not in SCHEDULE_TIME_COLS]

def build_prediction_grid(
    spells: pl.DataFrame, tctx: pl.DataFrame, max_k: int = 100
) -> pl.DataFrame:
    """One row per (spell, k) for k = 1..max_k, with the schedule-time columns.

    This is what turns a hazard model into a duration prediction: evaluate
    the hazard at every k, then take the product.

    Two deliberate choices about the calendar:

    * The grid runs past the end of the season on the average gap between
      games, so the resulting expectation measures how long the injury keeps
      the player out rather than how many games happened to be left. The two
      differ a lot for a March injury and only the first is a property of the
      injury.
    * Only the regular-season schedule is read from the calendar. How deep a
      team goes in the playoffs is not known in December, so letting the grid
      see those dates would quietly feed the model the team's eventual
      postseason run. Everything past game 82 is extrapolated.
    """
    starts = spells.select(
        "spell_id", "season", pl.col("team_slug_start").alias("team_slug"),
        "start_team_game_idx", "start_date",
        (pl.col("start_season_type") == "Playoffs").alias("started_in_playoffs"),
        "hope_at_onset", "eliminated_at_onset",
    )
    regular = tctx.filter(pl.col("season_type") == "Regular Season").select(
        "season", "team_slug", "team_game_idx", "game_date", "days_rest",
        "team_games_next_14d",
    )

    grid = starts.join(
        pl.DataFrame({"k": list(range(1, max_k + 1))}, schema={"k": pl.Int64}),
        how="cross",
    ).with_columns((pl.col("start_team_game_idx") + pl.col("k") - 1).alias("team_game_idx"))

    grid = grid.join(regular, on=["season", "team_slug", "team_game_idx"], how="left")

    last = regular.group_by("season", "team_slug").agg(
        pl.col("team_game_idx").max().alias("last_idx"),
        pl.col("game_date").max().alias("last_date"),
    )
    grid = grid.join(last, on=["season", "team_slug"], how="left").with_columns(
        (pl.col("team_game_idx") > pl.col("last_idx")).alias("beyond_season")
    )
    # Anchor the extrapolation on the spell's own start when the spell begins
    # after the regular season, so a playoff injury is not dated from April.
    anchor = (
        pl.when(pl.col("started_in_playoffs"))
        .then(pl.col("start_date"))
        .otherwise(pl.col("last_date"))
    )
    steps = (
        pl.when(pl.col("started_in_playoffs"))
        .then(pl.col("k").cast(pl.Float64) - 1)
        .otherwise((pl.col("team_game_idx") - pl.col("last_idx")).cast(pl.Float64))
    )
    grid = grid.with_columns(
        pl.when(pl.col("beyond_season"))
        .then(anchor + pl.duration(days=(steps * MEAN_DAYS_BETWEEN_GAMES).round()))
        .otherwise(pl.col("game_date"))
        .alias("game_date"),
        pl.col("days_rest").cast(pl.Float64).fill_null(MEAN_DAYS_BETWEEN_GAMES),
        pl.col("team_games_next_14d").fill_null(round(14 / MEAN_DAYS_BETWEEN_GAMES)),
    )

    grid = grid.sort("spell_id", "k").with_columns(
        pl.col("k").alias("games_missed_so_far"),
        (pl.col("game_date") - pl.col("start_date")).dt.total_days()
        .alias("days_missed_so_far"),
        (pl.col("k").cast(pl.Float64) + 1).log().alias("log_games_missed_so_far"),
        pl.col("days_rest").shift(-1).over("spell_id").alias("days_to_next_game"),
        pl.col("started_in_playoffs").alias("in_playoffs_now"),
        pl.col("team_games_next_14d").alias("team_games_next_14d_now"),
    ).with_columns(
        pl.col("days_to_next_game").fill_null(MEAN_DAYS_BETWEEN_GAMES),
    ).with_columns(
        (pl.col("days_to_next_game") == 1).alias("next_is_b2b"),
    )

    # The report-state columns are frozen at their value on the first missed
    # game: ruled Out, nothing re-filed, nothing softened yet.
    return grid.with_columns(
        # Frozen at the first missed game: at onset nobody knows how the
        # team's season will turn. Overriding these two is how the de-biased
        # ("what would this injury cost at neutral urgency") prediction is
        # produced -- see scripts/07_debias.py.
        pl.col("hope_at_onset").alias("hope_now"),
        pl.col("eliminated_at_onset").alias("eliminated_now"),
        pl.lit(True).alias("status_now_out"),
        pl.lit(False).alias("status_now_doubtful"),
        pl.lit(False).alias("status_now_questionable"),
        pl.lit(False).alias("status_now_probable"),
        pl.lit(False).alias("status_ever_softened"),
        pl.lit(-1, dtype=pl.Int64).alias("games_since_softened"),
        pl.lit(False).alias("region_changed"),
        pl.lit(False).alias("ailment_changed"),
    )

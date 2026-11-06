"""Canonical team-game calendar, and the report-to-schedule date alignment.

`league_game_schedule` is the authority on which games exist and in what
order, but its `game_date` is not always the date the injury report filed
the game under: for 2021-22 the whole schedule sits one day early (its
Finals end on 2022-06-15, the real Game 6 was 2022-06-16). Where the report
carries a `game_id` the two agree exactly, which is how the offset below is
measured rather than assumed.

A constant per-season offset is harmless for everything except joining the
report to a game, because game ordering and day-gaps within a season are
differences and the offset cancels. So the fix is applied only to a separate
`report_date` column and the original date is kept.
"""

from __future__ import annotations

import polars as pl

from . import paths

CANDIDATE_OFFSETS = (0, -1, 1)


def nba_team_slugs() -> list[str]:
    """The 30 current NBA franchises, by slug."""
    teams = pl.read_parquet(paths.TEAMS)
    return sorted(teams.filter(pl.col("active"))["team_slug"].unique().to_list())


def team_id_lookup() -> pl.DataFrame:
    return pl.read_parquet(paths.TEAMS).select("team_id", "team_slug")


def _raw_schedule() -> pl.DataFrame:
    slugs = nba_team_slugs()
    return (
        pl.read_parquet(paths.SCHEDULE)
        .filter(
            pl.col("team").is_in(slugs)
            & pl.col("opponent").is_in(slugs)
            & pl.col("season_type").is_in(["Regular Season", "Playoffs"])
            & (pl.col("season") >= paths.FIRST_SEASON)
        )
        .select(
            "season",
            "season_type",
            "game_id",
            "game_date",
            pl.col("team").alias("team_slug"),
            pl.col("opponent").alias("opp_slug"),
            "home",
        )
        .unique(subset=["season", "game_id", "team_slug"])
    )


def report_date_offsets(verbose: bool = False) -> dict[str, int]:
    """Days to add to a schedule date to get the date the report filed it.

    Measured per season as the offset that aligns the most (team, date)
    pairs between the report and the schedule.
    """
    sched = _raw_schedule().select("season", "team_slug", "game_date").unique()
    report = (
        pl.read_parquet(paths.INJURIES)
        .select(pl.col("team_slug"), "game_date")
        .unique()
    )

    offsets: dict[str, int] = {}
    for season in sched["season"].unique().sort():
        s = sched.filter(pl.col("season") == season)
        scores = {}
        for off in CANDIDATE_OFFSETS:
            shifted = s.with_columns(pl.col("game_date") + pl.duration(days=off))
            scores[off] = shifted.join(
                report, on=["team_slug", "game_date"], how="semi"
            ).height
        best = max(scores, key=lambda k: scores[k])
        offsets[season] = best
        if verbose:
            total = s.height
            print(f"{season}: offset {best:+d} matches {scores[best]}/{total} "
                  f"(alternatives { {k: v for k, v in scores.items() if k != best} })")
    return offsets


def game_calendar(verbose: bool = False) -> pl.DataFrame:
    """One row per (season, team, game), ordered, with the report-side date.

    Columns
    -------
    season, season_type, game_id, team_slug, opp_slug, home
    game_date       as published in the schedule
    report_date     the date the injury report files the game under
    team_game_idx   1-based index of the game within (team, season)
    days_rest       days since the team's previous game that season
    """
    sched = _raw_schedule()
    offsets = report_date_offsets(verbose=verbose)
    off_df = pl.DataFrame(
        {"season": list(offsets), "date_offset": [offsets[s] for s in offsets]},
        schema={"season": pl.String, "date_offset": pl.Int32},
    )

    cal = (
        sched.join(off_df, on="season", how="left")
        .with_columns(
            (pl.col("game_date") + pl.duration(days=pl.col("date_offset"))).alias("report_date")
        )
        .sort("team_slug", "season", "game_date", "game_id")
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("team_slug", "season").alias("team_game_idx"),
            (pl.col("game_date") - pl.col("game_date").shift(1).over("team_slug", "season"))
            .dt.total_days()
            .alias("days_rest"),
        )
    )
    return cal

"""Features for injury-duration models, all evaluated strictly before tip-off.

Every feature here is a function of games that finished before the first game
of the spell. The box score reaches back to 2009-10, so career load is real
rather than truncated at the start of the injury report in 2021-22.

The leakage rule that matters most for this problem: nothing about the spell
itself may enter the index feature set. That rules out the obvious
temptations — the reason filed on *later* games of the spell, the number of
games the team ended up playing while the player was out, and the player's
minutes in the game they came back for.
"""

from __future__ import annotations

import polars as pl

from . import calendar as cal_mod
from . import paths, standings

POSITION_GROUPS = {
    "Guard": "guard",
    "Guard-Forward": "wing",
    "Forward-Guard": "wing",
    "Forward": "forward",
    "Forward-Center": "big",
    "Center-Forward": "big",
    "Center": "big",
}


def player_attributes() -> pl.DataFrame:
    """Time-invariant player attributes.

    `player_info` only carries 2024-25 and 2025-26, but birthdate, listed
    height and weight, and draft position do not change, so the most recent
    row per player is used for all seasons. Age is computed per spell from
    the birthdate rather than taken from a season row.
    """
    pi = pl.read_parquet(paths.PLAYER_INFO)
    return (
        pi.sort("season")
        .group_by("player_id")
        .agg(pl.all().last())
        .select(
            "player_id",
            pl.col("birthdate").str.slice(0, 10).str.to_date().alias("birthdate"),
            pl.col("height_cm").cast(pl.Float64),
            pl.col("weight_kg").cast(pl.Float64),
            pl.col("position").replace_strict(POSITION_GROUPS, default="unknown")
            .alias("position_group"),
            pl.col("draft_number").cast(pl.Int64, strict=False).alias("draft_number"),
            pl.col("draft_round").cast(pl.Int64, strict=False).alias("draft_round"),
            pl.col("draft_year").cast(pl.Int64, strict=False).alias("draft_year"),
        )
        .with_columns(
            (pl.col("weight_kg") / (pl.col("height_cm") / 100).pow(2)).alias("bmi"),
            pl.col("draft_number").fill_null(61),
            pl.col("draft_round").fill_null(0),
        )
    )


def played_game_history() -> pl.DataFrame:
    """Every game a player actually played, with trailing load aggregates.

    Each row carries the state of the player's workload *including* that
    game, so an as-of join from a spell's first game picks up the last game
    played before the injury and nothing after it.
    """
    sched = (
        pl.read_parquet(paths.SCHEDULE)
        .filter(pl.col("season_type").is_in(["Regular Season", "Playoffs"]))
        .select("game_id", "game_date", "season", "season_type")
        .unique(subset="game_id")
    )
    box = (
        pl.read_parquet(
            paths.BOX,
            columns=["game_id", "player_id", "min", "usg_pct", "pts", "poss", "pace"],
        )
        .filter(pl.col("min") > 0)
        .join(sched, on="game_id", how="inner")
        .unique(subset=["game_id", "player_id"])
        .sort("player_id", "game_date", "game_id")
    )

    over = ["player_id"]
    # A length-matched column of ones, for counting inside date windows.
    one = pl.col("game_id").is_not_null().cast(pl.Int32)
    return box.with_columns(
        # Career wear. Truncated at 2009-10 for the handful of players whose
        # careers started earlier; `draft_year` carries the untruncated signal.
        pl.col("game_id").cum_count().over(over).alias("career_games"),
        pl.col("min").cum_sum().over(over).alias("career_minutes"),
        # Season-to-date.
        pl.col("game_id").cum_count().over(["player_id", "season"]).alias("season_games"),
        pl.col("min").cum_sum().over(["player_id", "season"]).alias("season_minutes"),
        # Recent form and recent load.
        pl.col("min").alias("min_last"),
        pl.col("min").rolling_mean(3, min_samples=1).over(over).alias("min_roll3"),
        pl.col("min").rolling_mean(5, min_samples=1).over(over).alias("min_roll5"),
        pl.col("min").rolling_mean(10, min_samples=1).over(over).alias("min_roll10"),
        pl.col("min").rolling_std(10, min_samples=2).over(over).alias("min_sd10"),
        pl.col("usg_pct").rolling_mean(5, min_samples=1).over(over).alias("usg_roll5"),
        pl.col("usg_pct").rolling_mean(10, min_samples=1).over(over).alias("usg_roll10"),
        pl.col("pace").rolling_mean(5, min_samples=1).over(over).alias("pace_roll5"),
        # Date-window load: how much basketball in the days right before.
        pl.col("min").rolling_sum_by("game_date", "7d").over(over).alias("min_7d"),
        pl.col("min").rolling_sum_by("game_date", "14d").over(over).alias("min_14d"),
        pl.col("min").rolling_sum_by("game_date", "30d").over(over).alias("min_30d"),
        one.rolling_sum_by("game_date", "7d").over(over).alias("games_7d"),
        one.rolling_sum_by("game_date", "14d").over(over).alias("games_14d"),
        (pl.col("game_date") - pl.col("game_date").shift(1).over(over))
        .dt.total_days()
        .alias("rest_before_last_game"),
    ).select(
        "player_id", "game_date", "season",
        "career_games", "career_minutes", "season_games", "season_minutes",
        "min_last", "min_roll3", "min_roll5", "min_roll10", "min_sd10",
        "usg_roll5", "usg_roll10", "pace_roll5",
        "min_7d", "min_14d", "min_30d", "games_7d", "games_14d",
        "rest_before_last_game",
    )


def team_context() -> pl.DataFrame:
    """Per team game: record to date and the shape of the schedule ahead.

    Schedule density ahead matters because a borderline player sitting out is
    partly a scheduling decision, and a team's record matters because a team
    out of the race stops rushing players back.
    """
    cal = cal_mod.game_calendar()
    results = (
        pl.read_parquet(paths.SCHEDULE)
        .select("game_id", "team", "team_winner")
        .unique(subset=["game_id", "team"])
    )
    cal = cal.join(
        results, left_on=["game_id", "team_slug"], right_on=["game_id", "team"], how="left"
    ).with_columns((pl.col("team_winner") == pl.col("team_slug")).alias("won"))

    cal = (
        cal.sort("team_slug", "season", "team_game_idx")
        .with_columns(
            # Strictly prior games: this game's result is not known pre-tip.
            pl.col("won").cum_sum().shift(1).over("team_slug", "season")
            .alias("team_wins_before"),
            (pl.col("team_game_idx") - 1).alias("team_games_before"),
            pl.col("team_game_idx").max().over("team_slug", "season")
            .alias("team_games_total"),
            pl.col("team_game_idx")
            .filter(pl.col("season_type") == "Regular Season")
            .max()
            .over("team_slug", "season")
            .alias("team_regular_total"),
        )
        .with_columns(
            (pl.col("team_wins_before") / pl.col("team_games_before").cast(pl.Float64))
            .alias("team_win_pct_before"),
            (pl.col("team_regular_total") - pl.col("team_game_idx"))
            .alias("team_regular_remaining"),
        )
    )

    # Games the team plays in the fortnight *after* this one. Fixed by the
    # schedule months in advance, so it is known pre-tip.
    dates = cal.select("season", "team_slug", "team_game_idx", "game_date")
    ahead = (
        dates.join(
            dates.select("season", "team_slug", pl.col("game_date").alias("other")),
            on=["season", "team_slug"],
            how="left",
        )
        .filter(
            (pl.col("other") > pl.col("game_date"))
            & (pl.col("other") <= pl.col("game_date") + pl.duration(days=14))
        )
        .group_by("season", "team_slug", "team_game_idx")
        .agg(pl.len().alias("team_games_next_14d"))
    )
    cal = cal.join(ahead, on=["season", "team_slug", "team_game_idx"], how="left")
    cal = cal.with_columns(pl.col("team_games_next_14d").fill_null(0))

    # Team incentive, from the standings on the day. Win percentage is a
    # smooth proxy that confounds quality with incentive; playoff hope is the
    # quantity a front office actually responds to.
    inc = standings.playin_probability().select(
        "season", "game_date", "team_slug", "playin_probability", "playin_stakes",
        "eliminated", "clinched_playin", "games_since_elimination",
        "wins_vs_playin_line",
    )
    cal = cal.join(inc, on=["season", "game_date", "team_slug"], how="left")

    return cal.select(
        "season", "game_id", "team_slug", "team_game_idx", "game_date", "season_type",
        "days_rest", "home", "opp_slug",
        "team_games_before", "team_win_pct_before", "team_regular_remaining",
        "team_regular_total", "team_games_next_14d",
        "playin_probability", "playin_stakes", "eliminated", "clinched_playin",
        "games_since_elimination", "wins_vs_playin_line",
    )


def injury_history(spells: pl.DataFrame) -> pl.DataFrame:
    """Prior-spell features for each spell, using only spells that started first.

    Self-joining the spell table on player and keeping the pairs where the
    other spell began earlier gives every "how beaten up is this player
    already" feature without any look-ahead. Spells cannot overlap by
    construction, so an earlier start also means an earlier finish.
    """
    s = spells.select(
        "spell_id", "player_id", "season", "start_date", "last_missed_date",
        "games_missed", "body_region",
    )
    prior = s.select(
        "player_id",
        pl.col("start_date").alias("p_start"),
        pl.col("last_missed_date").alias("p_end"),
        pl.col("games_missed").alias("p_games"),
        pl.col("body_region").alias("p_region"),
        pl.col("season").alias("p_season"),
    )
    pairs = s.join(prior, on="player_id", how="inner").filter(
        pl.col("p_start") < pl.col("start_date")
    )

    overall = pairs.group_by("spell_id").agg(
        pl.len().alias("prior_spells_career"),
        pl.col("p_games").sum().alias("prior_games_missed_career"),
        (pl.col("p_season") == pl.col("season")).sum().alias("prior_spells_season"),
        pl.col("p_games").filter(pl.col("p_season") == pl.col("season")).sum()
        .alias("prior_games_missed_season"),
        (pl.col("p_start") >= pl.col("start_date") - pl.duration(days=365)).sum()
        .alias("prior_spells_365d"),
        pl.col("p_games")
        .filter(pl.col("p_start") >= pl.col("start_date") - pl.duration(days=365))
        .sum()
        .alias("prior_games_missed_365d"),
        (pl.col("start_date").first() - pl.col("p_end").max()).dt.total_days()
        .alias("days_since_prior_spell"),
        pl.col("p_games").sort_by("p_start").last().alias("prior_spell_games"),
    )

    same_region = (
        pairs.filter(pl.col("p_region") == pl.col("body_region"))
        .group_by("spell_id")
        .agg(
            pl.len().alias("prior_spells_same_region"),
            pl.col("p_games").sum().alias("prior_games_missed_same_region"),
            pl.col("p_games").sort_by("p_start").last().alias("prior_same_region_games"),
            (pl.col("start_date").first() - pl.col("p_end").max()).dt.total_days()
            .alias("days_since_same_region"),
        )
    )

    counts = [
        "prior_spells_career", "prior_games_missed_career", "prior_spells_season",
        "prior_games_missed_season", "prior_spells_365d", "prior_games_missed_365d",
        "prior_spells_same_region", "prior_games_missed_same_region",
    ]
    out = (
        s.select("spell_id")
        .join(overall, on="spell_id", how="left")
        .join(same_region, on="spell_id", how="left")
        .with_columns([pl.col(c).fill_null(0) for c in counts])
    )
    return out.with_columns(
        # A fresh absence to the same body part within a month is a different
        # animal from a first-time one.
        (pl.col("days_since_same_region") <= 30).fill_null(False).alias("is_quick_recurrence"),
        (pl.col("days_since_prior_spell") <= 7).fill_null(False)
        .alias("is_immediate_reaggravation"),
    )


def build_index_features(spells: pl.DataFrame) -> pl.DataFrame:
    """Assemble the spell-level design matrix, as known at the spell's first game."""
    attrs = player_attributes()
    hist = played_game_history()
    tctx = team_context()

    df = spells.join(attrs, on="player_id", how="left")

    # Last game played before the injury: as-of join, strictly backwards.
    df = (
        df.sort("start_date")
        .join_asof(
            hist.sort("game_date").drop("season"),
            left_on="start_date",
            right_on="game_date",
            by="player_id",
            strategy="backward",
            allow_exact_matches=False,
        )
        .rename({"game_date": "last_played_date"})
        .with_columns(
            (pl.col("start_date") - pl.col("last_played_date")).dt.total_days()
            .alias("days_since_last_played")
        )
    )

    df = df.join(
        tctx.drop("game_date", "season_type").rename({
            "playin_probability": "hope_at_onset",
            "playin_stakes": "stakes_at_onset",
            "eliminated": "eliminated_at_onset",
            "clinched_playin": "clinched_at_onset",
            "games_since_elimination": "games_since_elim_at_onset",
            "wins_vs_playin_line": "wins_vs_playin_at_onset",
        }),
        left_on=["season", "start_game_id", "team_slug_start"],
        right_on=["season", "game_id", "team_slug"],
        how="left",
    )

    df = df.join(injury_history(spells), on="spell_id", how="left")

    df = df.with_columns(
        ((pl.col("start_date") - pl.col("birthdate")).dt.total_days() / 365.25).alias("age"),
        # Divided by the 82-game regular season, not by the games the team
        # ended up playing: how deep they went in the playoffs is not
        # knowable when the player is ruled out in December.
        (pl.col("start_team_game_idx") / pl.col("team_regular_total"))
        .alias("season_progress"),
        pl.col("start_date").dt.month().alias("start_month"),
        (pl.col("start_season_type") == "Playoffs").alias("starts_in_playoffs"),
        # A spell that is already running when the season opens began in the
        # off-season, so none of the in-season load history applies to it.
        (pl.col("start_team_game_idx") == 1).alias("starts_at_season_open"),
        (pl.col("index_category") == "illness").alias("is_illness"),
        # A named major structure only matters alongside a pathology that
        # actually takes one out: a torn, operated or fractured ACL or
        # Achilles, not a sore one.
        (
            pl.col("mentions_major_structure")
            & pl.col("ailment_class").is_in(["rupture_tear", "surgery", "fracture"])
        ).alias("is_catastrophic"),
        (pl.col("days_rest") == 1).fill_null(False).alias("start_on_b2b"),
        pl.col("season").str.slice(0, 4).cast(pl.Int32).alias("season_start_year"),
    ).with_columns(
        # A spell starting on the team's opening night has no standings row to
        # join to -- a team with no games played has no record. On that night
        # every team's hope is simply the league base rate: twenty of thirty
        # teams reach the play-in.
        pl.col("hope_at_onset").fill_null(standings.PLAYIN_SEEDS * 2 / 30).alias("hope_at_onset"),
        pl.col("stakes_at_onset").fill_null(
            4 * (standings.PLAYIN_SEEDS * 2 / 30) * (1 - standings.PLAYIN_SEEDS * 2 / 30)
        ).alias("stakes_at_onset"),
        pl.col("eliminated_at_onset").fill_null(False).alias("eliminated_at_onset"),
        pl.col("clinched_at_onset").fill_null(False).alias("clinched_at_onset"),
        pl.col("wins_vs_playin_at_onset").fill_null(0).alias("wins_vs_playin_at_onset"),
        (pl.col("season_start_year") - pl.col("draft_year")).alias("years_since_draft"),
        (pl.col("career_minutes") / pl.col("career_games")).alias("career_min_per_game"),
        (pl.col("min_roll5") - pl.col("min_roll10")).alias("min_trend"),
    )
    return df


# Feature blocks, used for the ablation study. Keeping them named rather than
# inline is what makes "which variables actually drive duration" answerable.
BLOCKS: dict[str, list[str]] = {
    "injury": [
        "body_region", "body_side", "ailment_class", "is_surgical",
        "is_recovery_stage", "is_management", "is_bone_stress", "is_catastrophic",
        "n_reason_separators", "is_illness", "index_status",
    ],
    "player": [
        "age", "height_cm", "weight_kg", "bmi", "position_group",
        "draft_number", "draft_round", "years_since_draft",
    ],
    "load": [
        "min_last", "min_roll3", "min_roll5", "min_roll10", "min_sd10", "min_trend",
        "usg_roll5", "usg_roll10", "pace_roll5", "min_7d", "min_14d", "min_30d",
        "games_7d", "games_14d", "rest_before_last_game", "career_games",
        "career_minutes", "career_min_per_game", "season_games", "season_minutes",
        "days_since_last_played",
    ],
    "history": [
        "prior_spells_career", "prior_games_missed_career", "prior_spells_season",
        "prior_games_missed_season", "prior_spells_same_region",
        "prior_games_missed_same_region", "prior_games_missed_365d",
        "prior_spells_365d", "days_since_prior_spell", "prior_spell_games",
        "prior_same_region_games", "days_since_same_region",
        "is_quick_recurrence", "is_immediate_reaggravation",
    ],
    # Kept separate from `context` on purpose: the question "how much of a
    # published duration is medicine and how much is the team's situation"
    # only has an answer if the two are separable in the ablation.
    "incentive": [
        "hope_at_onset", "stakes_at_onset", "eliminated_at_onset",
        "clinched_at_onset", "games_since_elim_at_onset",
        "wins_vs_playin_at_onset",
    ],
    "context": [
        "season_progress", "start_month", "starts_in_playoffs",
        "starts_at_season_open", "start_on_b2b", "days_rest", "home",
        "team_games_before", "team_win_pct_before", "team_regular_remaining",
        "team_games_next_14d",
    ],
}

CATEGORICAL = [
    "body_region", "body_side", "ailment_class", "position_group", "index_status",
]


def all_features() -> list[str]:
    return [c for block in BLOCKS.values() for c in block]

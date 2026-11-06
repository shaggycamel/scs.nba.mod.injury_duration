"""Turn the per-game injury report into player injury spells.

Why a panel first
-----------------
The box score is not a record of availability: a player on a long-term
absence is simply missing from it. Klay Thompson has no 2021-22 regular
season box-score row until the night he came back, so "games missed" cannot
be counted from the box score alone. The injury report is the only source
that says a player was unavailable, and it says it one game at a time.

So the build goes report + box score -> a per-(player, game) availability
panel -> runs of consecutive unavailable games -> spells.

Censoring
---------
A spell is only *observed* if we see the player play again. Three things stop
us seeing that, and all of them are right-censoring rather than a short
spell:

* the season runs out while the player is still out;
* the player is sent to the G League, traded, suspended, or leaves the team
  mid-absence, after which their return is no longer observable on this
  roster (a competing risk);
* the data ends.

Treating those as "recovered on the last game we saw" is the single biggest
way to bias a duration model low, so they are flagged, not dropped.
"""

from __future__ import annotations

import polars as pl

from . import calendar as cal_mod
from . import ids, paths, taxonomy

# Report categories that may *open* an injury spell.
START_CATEGORIES = ("injury", "illness")

# Categories that keep an open spell alive without being able to start one.
# "Return to Competition Reconditioning" is the league's ramp-up filing after
# a long absence and `unknown` is a blank/"-" reason, which in practice
# carries on whatever came before.
CONTINUE_CATEGORIES = ("injury", "illness", "reconditioning", "unknown", "rest")

# Categories that end our ability to observe the return. The player is still
# absent, but for a reason that makes the recovery date unobservable.
CENSOR_CATEGORIES = ("gleague", "roster", "suspension", "personal", "protocol")

# Longest run of games with no report row at all that we are willing to treat
# as "still out" when it is sandwiched between two reported-out games.
MAX_BRIDGE_GAMES = 3


def health_missed(categories: tuple[str, ...] = CONTINUE_CATEGORIES) -> pl.Expr:
    """Game missed for a health reason the report names.

    Status "Available" is excluded: an available player with an injury note
    who does not appear is a coach's decision, not an absence.
    """
    return (
        ~pl.col("played")
        & pl.col("reason_category").is_in(list(categories))
        & (pl.col("status_clean") != "Available")
    )


def load_report() -> tuple[pl.DataFrame, dict]:
    """Report rows, deduplicated, re-keyed to a player id, with reason parsed."""
    raw = pl.read_parquet(paths.INJURIES).unique()
    resolved, diag = ids.resolve_player_ids(raw)
    out = (
        resolved.with_columns(taxonomy.reason_features())
        .with_columns(
            # The status field is occasionally two statuses concatenated by the
            # scraper ("Out Out"); take the first token.
            pl.col("status").str.extract(r"^(\w+)", 1).alias("status_clean")
        )
        # The report's own game_id is null for most of 2021-22 and would
        # shadow the calendar's on join, so it is renamed and kept only as a
        # cross-check. `game_date` is renamed for the same reason: the report
        # files a game under a date that the schedule does not always share.
        .rename({"game_id": "report_game_id", "game_date": "report_date"})
        .drop("team", "matchup")
    )
    return out, diag


def build_panel() -> tuple[pl.DataFrame, dict]:
    """One row per (player, season, team game) the player was attached for.

    State is one of
      played           : box-score minutes > 0
      out              : listed Out on the report
      listed_dnp       : on the report but not ruled out, and did not play
      unlisted_absent  : neither in the box score nor on the report
    """
    cal = cal_mod.game_calendar()
    report, diag = load_report()

    # Map each report row onto a scheduled team game. (team_slug, report_date)
    # is unique because no team plays twice in a day.
    rep = report.join(
        cal.select("season", "game_id", "team_slug", "report_date", "team_game_idx",
                   "game_date", "season_type", "days_rest", "opp_slug", "home"),
        on=["team_slug", "report_date"],
        how="inner",
    )
    mismatched = rep.filter(
        pl.col("report_game_id").is_not_null()
        & (pl.col("report_game_id") != pl.col("game_id"))
    ).height
    diag["report_game_id_mismatch"] = mismatched
    diag["report_rows_mapped"] = rep.height
    diag["report_rows_unmapped"] = report.height - rep.height

    rep = rep.select(
        "season", "game_id", "team_slug", "team_game_idx", "game_date",
        "season_type", "days_rest", "opp_slug", "home",
        "player_id", "player_name", "status_clean", "reason", "reason_category",
        "body_region", "body_side", "body_part_raw", "ailment_class", "ailment_raw",
        "is_surgical", "is_recovery_stage", "is_management", "is_bone_stress",
        "mentions_major_structure", "n_reason_separators", "reason_len",
    ).unique(subset=["season", "game_id", "player_id"], keep="first")

    # Box score -> who actually played.
    slugs = cal_mod.team_id_lookup()
    box = (
        pl.read_parquet(paths.BOX, columns=["game_id", "team_id", "player_id", "min", "comment"])
        .join(slugs, on="team_id", how="inner")
        .select("game_id", "team_slug", "player_id", "min", "comment")
        .unique(subset=["game_id", "player_id"], keep="first")
        .with_columns(pl.col("min").fill_null(0.0))
    )
    box = box.join(
        cal.select("season", "game_id", "team_slug", "team_game_idx", "game_date",
                   "season_type", "days_rest", "opp_slug", "home"),
        on=["game_id", "team_slug"],
        how="inner",
    )

    # The player-game universe: every game at which the player was either in
    # the box score or on their team's report, plus the games in between (a
    # player can be neither for a stretch, e.g. a report gap).
    events = pl.concat(
        [
            rep.select("season", "team_slug", "player_id", "team_game_idx"),
            box.select("season", "team_slug", "player_id", "team_game_idx"),
        ]
    ).unique()
    windows = events.group_by("season", "team_slug", "player_id").agg(
        pl.col("team_game_idx").min().alias("lo"),
        pl.col("team_game_idx").max().alias("hi"),
    )
    panel = (
        windows.join(
            cal.select("season", "team_slug", "team_game_idx", "game_id", "game_date",
                       "season_type", "days_rest", "opp_slug", "home"),
            on=["season", "team_slug"],
            how="inner",
        )
        .filter(pl.col("team_game_idx").is_between("lo", "hi"))
        .drop("lo", "hi")
    )

    panel = (
        panel.join(
            box.select("season", "game_id", "player_id", "min", "comment"),
            on=["season", "game_id", "player_id"],
            how="left",
        )
        .join(
            rep.drop("team_slug", "team_game_idx", "game_date", "season_type",
                     "days_rest", "opp_slug", "home", "player_name"),
            on=["season", "game_id", "player_id"],
            how="left",
        )
        .with_columns(
            pl.col("min").fill_null(0.0),
            pl.col("status_clean").fill_null(""),
        )
    )

    # A handful of regular-season dates have no report at all (scraper gaps);
    # knowing that lets the bridge below distinguish "nobody was reported" from
    # "this player specifically was not reported".
    reported_dates = rep.select("season", "game_date").unique().with_columns(
        pl.lit(True).alias("league_reported")
    )
    panel = panel.join(reported_dates, on=["season", "game_date"], how="left").with_columns(
        pl.col("league_reported").fill_null(False)
    )

    # Whether a game was missed is a question about minutes, not about the
    # label on the report: a player listed Doubtful or even Questionable who
    # does not appear has missed that game just as surely as one listed Out,
    # and reading the label as the state would score those as returns. The
    # label is kept as a feature instead.
    panel = panel.with_columns(
        (pl.col("min") > 0).alias("played"),
        pl.when(pl.col("min") > 0).then(pl.lit("played"))
        .when(pl.col("status_clean") == "Out").then(pl.lit("out"))
        .when(pl.col("status_clean") != "").then(pl.format("missed_{}", pl.col("status_clean")))
        .otherwise(pl.lit("unlisted_absent"))
        .alias("state"),
    )

    # Trades: a player can appear for two teams in one season. Order their
    # games by date so "games missed" follows the player, not a franchise.
    # team_slug breaks the last tie, and it is not hypothetical: a player
    # traded between two teams that then play each other is attached to both
    # sides of one game, so (player, season, date, game_id) is not unique.
    # polars' sort is not stable on ties, so without this the whole index
    # shifts between runs and nothing downstream reproduces.
    panel = panel.sort(
        "player_id", "season", "game_date", "game_id", "team_slug"
    ).with_columns(
        pl.int_range(1, pl.len() + 1).over("player_id", "season").alias("player_game_idx")
    )

    names = (
        pl.concat([
            rep.select("player_id", "player_name"),
            pl.read_parquet(paths.BOX, columns=["player_id", "player_name"]),
        ])
        .drop_nulls()
        .group_by("player_id")
        .agg(pl.col("player_name").mode().first())
    )
    panel = panel.join(names, on="player_id", how="left")
    diag["panel_rows"] = panel.height
    diag["panel_states"] = (
        panel.group_by("state").len().sort("len", descending=True).to_dicts()
    )
    return panel, diag


def _bridge_report_gaps(panel: pl.DataFrame) -> pl.DataFrame:
    """Relabel short unreported gaps that sit between two reported-out games.

    A player missing from both the box score and the report is usually just
    absent from the report, not available: either the whole date is missing
    from the scrape, or their row was dropped. Only gaps of at most
    MAX_BRIDGE_GAMES games, bounded on both sides by a health-driven Out, are
    bridged; anything longer is left alone so a quiet stretch at the back of
    a roster cannot be mistaken for a four-month injury.
    """
    p = panel.with_columns(health_missed().alias("_health_out"))

    # Nearest decided state on each side, ignoring the unreported rows.
    decided = pl.when(pl.col("state") != "unlisted_absent").then(pl.col("_health_out"))
    decided_cat = pl.when(pl.col("state") != "unlisted_absent").then(pl.col("reason_category"))
    decided_reason = pl.when(pl.col("state") != "unlisted_absent").then(pl.col("reason"))

    p = p.sort("player_id", "season", "player_game_idx").with_columns(
        decided.forward_fill().over("player_id", "season").alias("_prev_out"),
        decided.backward_fill().over("player_id", "season").alias("_next_out"),
        decided_cat.forward_fill().over("player_id", "season").alias("_prev_cat"),
        decided_reason.forward_fill().over("player_id", "season").alias("_prev_reason"),
    )

    # Length of the unreported run this row belongs to.
    p = p.with_columns(
        (pl.col("state") != "unlisted_absent")
        .cum_sum()
        .over("player_id", "season")
        .alias("_run_id")
    )
    p = p.with_columns(
        pl.when(pl.col("state") == "unlisted_absent")
        .then(pl.len().over("player_id", "season", "_run_id"))
        .otherwise(0)
        .alias("_gap_len")
    )

    bridge = (
        (pl.col("state") == "unlisted_absent")
        & pl.col("_prev_out").fill_null(False)
        & pl.col("_next_out").fill_null(False)
        & (pl.col("_gap_len") <= MAX_BRIDGE_GAMES)
    )
    p = p.with_columns(
        pl.when(bridge).then(pl.lit("out")).otherwise(pl.col("state")).alias("state"),
        pl.when(bridge).then(pl.lit("Out")).otherwise(pl.col("status_clean")).alias("status_clean"),
        pl.when(bridge).then(pl.col("_prev_cat")).otherwise(pl.col("reason_category"))
        .alias("reason_category"),
        pl.when(bridge).then(pl.col("_prev_reason")).otherwise(pl.col("reason")).alias("reason"),
        bridge.alias("is_bridged"),
    )
    return p.drop(
        "_health_out", "_prev_out", "_next_out", "_prev_cat", "_prev_reason",
        "_run_id", "_gap_len",
    )


def build_spells() -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Segment the panel into injury spells.

    Returns `(spells, panel, diagnostics)`. `panel` carries a `spell_id` so
    the per-game rows of a spell can be recovered for the hazard model.
    """
    panel, diag = build_panel()
    panel = _bridge_report_gaps(panel)

    blocked = health_missed(CONTINUE_CATEGORIES)
    starts_ok = health_missed(START_CATEGORIES)
    censors = (
        ~pl.col("played")
        & pl.col("reason_category").is_in(list(CENSOR_CATEGORIES))
    )

    p = panel.sort("player_id", "season", "player_game_idx").with_columns(
        blocked.alias("blocked"), starts_ok.alias("start_ok"), censors.alias("censor_state")
    )

    # Consecutive blocked games form a block. A block becomes a spell from its
    # first start-eligible game onward; blocks with no such game (a pure
    # reconditioning or blank-reason run) are not spells.
    p = p.with_columns(
        ((~pl.col("blocked")) | (pl.col("blocked") & ~pl.col("blocked").shift(1).fill_null(False)))
        .cum_sum()
        .over("player_id", "season")
        .alias("_blk")
    )
    p = p.with_columns(
        pl.when(pl.col("blocked"))
        .then(pl.col("player_game_idx").filter(pl.col("start_ok")).min()
              .over("player_id", "season", "_blk"))
        .alias("_spell_from")
    )
    in_spell = pl.col("blocked") & pl.col("_spell_from").is_not_null() & (
        pl.col("player_game_idx") >= pl.col("_spell_from")
    )
    p = p.with_columns(in_spell.alias("in_spell"))
    p = p.with_columns(
        pl.when(pl.col("in_spell"))
        .then(
            pl.col("player_id").cast(pl.String)
            + "_" + pl.col("season")
            + "_" + pl.col("_spell_from").cast(pl.String)
        )
        .alias("spell_id")
    )
    p = p.with_columns(
        pl.when(pl.col("in_spell"))
        .then(pl.col("player_game_idx") - pl.col("_spell_from") + 1)
        .alias("spell_game_no")
    )

    # What happens at the first game after the spell decides the outcome.
    nxt = pl.struct(
        state=pl.col("state").shift(-1).over("player_id", "season"),
        cat=pl.col("reason_category").shift(-1).over("player_id", "season"),
        date=pl.col("game_date").shift(-1).over("player_id", "season"),
        idx=pl.col("player_game_idx").shift(-1).over("player_id", "season"),
        mins=pl.col("min").shift(-1).over("player_id", "season"),
    )
    p = p.with_columns(nxt.alias("_next"))

    spell_rows = p.filter(pl.col("in_spell"))
    last = (
        spell_rows.sort("player_id", "season", "player_game_idx")
        .group_by("spell_id")
        .agg(pl.all().last())
    )

    # `.sort_by("player_game_idx").first()` throughout: these columns mean
    # "as filed on the first game of the spell", and group_by().agg() makes no
    # promise about within-group row order, so a bare .first() silently picks
    # an arbitrary game of the spell and stops the build being reproducible.
    spells = (
        spell_rows.group_by("spell_id")
        .agg(
            pl.col("player_id").sort_by("player_game_idx").first(),
            pl.col("player_name").sort_by("player_game_idx").first(),
            pl.col("season").sort_by("player_game_idx").first(),
            pl.col("team_slug").sort_by("player_game_idx").first().alias("team_slug_start"),
            pl.col("game_date").min().alias("start_date"),
            pl.col("game_date").max().alias("last_missed_date"),
            pl.col("player_game_idx").min().alias("start_player_game_idx"),
            pl.col("team_game_idx").sort_by("player_game_idx").first().alias("start_team_game_idx"),
            pl.col("game_id").sort_by("player_game_idx").first().alias("start_game_id"),
            pl.col("season_type").sort_by("player_game_idx").first().alias("start_season_type"),
            pl.len().alias("games_missed"),
            pl.col("is_bridged").sum().alias("n_bridged_games"),
            # Index (first-game) description of the injury.
            pl.col("reason").sort_by("player_game_idx").first().alias("index_reason"),
            pl.col("status_clean").sort_by("player_game_idx").first().alias("index_status"),
            pl.col("reason_category").sort_by("player_game_idx").first().alias("index_category"),
            pl.col("body_region").sort_by("player_game_idx").first().alias("body_region"),
            pl.col("body_side").sort_by("player_game_idx").first().alias("body_side"),
            pl.col("body_part_raw").sort_by("player_game_idx").first().alias("body_part_raw"),
            pl.col("ailment_class").sort_by("player_game_idx").first().alias("ailment_class"),
            pl.col("is_surgical").sort_by("player_game_idx").first().alias("is_surgical"),
            pl.col("is_recovery_stage").sort_by("player_game_idx").first().alias("is_recovery_stage"),
            pl.col("is_management").sort_by("player_game_idx").first().alias("is_management"),
            pl.col("is_bone_stress").sort_by("player_game_idx").first().alias("is_bone_stress"),
            pl.col("mentions_major_structure").sort_by("player_game_idx").first().alias("mentions_major_structure"),
            pl.col("n_reason_separators").sort_by("player_game_idx").first().alias("n_reason_separators"),
            # Did the filing change during the spell?
            pl.col("body_region").n_unique().alias("n_regions_in_spell"),
            pl.col("ailment_class").n_unique().alias("n_ailments_in_spell"),
            pl.col("team_slug").n_unique().alias("n_teams_in_spell"),
        )
        .join(
            last.select(
                "spell_id",
                pl.col("_next").struct.field("state").alias("next_state"),
                pl.col("_next").struct.field("cat").alias("next_cat"),
                pl.col("_next").struct.field("date").alias("next_date"),
                pl.col("_next").struct.field("mins").alias("next_min"),
            ),
            on="spell_id",
            how="left",
        )
    )

    # Event vs censoring.
    returned = (pl.col("next_state") == "played").fill_null(False)
    censor_cat = pl.col("next_cat").is_in(list(CENSOR_CATEGORIES)).fill_null(False)
    spells = spells.with_columns(
        returned.cast(pl.Int8).alias("event"),
        pl.when(returned).then(pl.lit("returned"))
        .when(pl.col("next_state").is_null()).then(pl.lit("season_end"))
        .when(censor_cat).then(pl.format("left_{}", pl.col("next_cat")))
        .otherwise(pl.lit("dropped_from_report"))
        .alias("censor_reason"),
    ).with_columns(
        pl.when(pl.col("event") == 1)
        .then((pl.col("next_date") - pl.col("start_date")).dt.total_days())
        .otherwise((pl.col("last_missed_date") - pl.col("start_date")).dt.total_days() + 1)
        .alias("days_out")
    )

    diag["spells"] = spells.height
    diag["spell_event_rate"] = float(spells["event"].mean())
    diag["censor_reasons"] = (
        spells.group_by("censor_reason").len().sort("len", descending=True).to_dicts()
    )
    return spells.sort("start_date", "player_id"), p, diag

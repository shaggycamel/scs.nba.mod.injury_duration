"""Find the spells that are still active at the current data cutoff.

Stage 1 of the lean daily job. It pulls only the current season (all players),
runs the existing spell builder on that small snapshot, and writes the ongoing
spells. The target player ids then drive the full-history lean fetch.

Usage:
    INJURY_DURATION_DATA=data/active uv run --group data \
        python scripts/18_active_spells.py --season 2026-27
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl
import scs_hub

PLAYER_INFO_COLUMNS = [
    "season",
    "player_id",
    "first_name",
    "last_name",
    "player_slug",
    "birthdate",
    "school",
    "country",
    "last_affiliation",
    "height_cm",
    "weight_kg",
    "season_exp",
    "jersey",
    "position",
    "rosterstatus",
    "games_played_current_season_flag",
    "team_id",
    "team_name",
    "team_abbreviation",
    "team_code",
    "team_city",
    "playercode",
    "from_year",
    "to_year",
    "dleague_flag",
    "nba_flag",
    "games_played_flag",
    "draft_year",
    "draft_round",
    "draft_number",
    "greatest_75_flag",
]
TEAMS_COLUMNS = [
    "team_id",
    "team_slug",
    "team_long",
    "team_name",
    "conference",
    "division",
    "first_season",
    "last_season",
    "stadium",
    "stadium_capacity",
    "active",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--db-con", default="cockroach-read")
    parser.add_argument("--out", type=Path, default=Path("artifacts/active_spells.parquet"))
    parser.add_argument("--data", type=Path, default=Path("data/active"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["INJURY_DURATION_DATA"] = str(args.data.resolve())

    hub = scs_hub.SportsHub(db_con=args.db_con)
    season = args.season
    bounds = hub.db.read(
        f"SELECT min(game_date) AS first, max(game_date) AS last "
        f"FROM nba.league_game_schedule_vw WHERE season = '{season}'"
    ).row(0, named=True)
    first, last = bounds["first"], bounds["last"]
    if first is None:
        raise SystemExit(f"no schedule for {season!r}")

    queries = {
        "nba/league_game_schedule": f"""
            SELECT season_type, season, game_id, game_date, matchup, team,
                   opponent, home, team_winner
            FROM nba.league_game_schedule_vw
            WHERE season = '{season}'
        """,
        "nba/injuries": f"""
            SELECT game_date, game_id, matchup, team, team_slug, player_name,
                   nba_id, status, reason
            FROM nba.injuries_vw
            WHERE game_date BETWEEN '{first}' AND '{last}'
        """,
        "nba/player_box_score": f"""
            SELECT DISTINCT game_id, team_id, player_id::BIGINT AS player_id,
                   player_name, min, inj_reason AS comment,
                   usg_pct, pts, poss, pace
            FROM nba.player_box_score_vw
            WHERE min > 0
              AND game_id IN (
                  SELECT game_id FROM nba.league_game_schedule_vw
                  WHERE season = '{season}'
              )
        """,
        "nba/player_info": f"""
            SELECT {", ".join(PLAYER_INFO_COLUMNS)}
            FROM nba.player_info_vw WHERE season <= '{season}'
        """,
        "nba/teams": f"SELECT {', '.join(TEAMS_COLUMNS)} FROM nba.teams_vw",
        "util/player_id_map_vw": "SELECT * FROM util.player_id_map_vw",
    }
    for rel, sql in queries.items():
        frame = hub.db.read(sql)
        out = args.data / f"{rel}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(out)

    from injury_duration import spells

    all_spells, _, _ = spells.build_spells()
    active = all_spells.filter(
        (pl.col("season") == season)
        & (pl.col("event") == 0)
        & pl.col("censor_reason").is_in(["season_end", "dropped_from_report"])
    ).sort("last_missed_date", descending=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    active.write_parquet(args.out)
    print(f"cutoff {last}, {all_spells.height:,} current-season spells, {active.height:,} active")
    print(active.group_by("censor_reason").len().sort("len", descending=True))
    print(
        active.select(
            [
                "spell_id",
                "player_id",
                "player_name",
                "start_date",
                "last_missed_date",
                "games_missed",
                "censor_reason",
            ]
        ).head(20)
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

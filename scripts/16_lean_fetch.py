"""Build a player-filtered snapshot for lean inference experiments.

Instead of pulling every box score back to 2009-10, take a set of players
(from a spells table) and pull only their injuries and box-score history, plus
the small reference tables. The existing feature builder then runs on that
much smaller snapshot; the parity check compares the result with the full
build.

Usage:
    uv run --group data python scripts/16_lean_fetch.py \
        --season 2025-26 --out data/lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import scs_hub

PLAYER_INFO_COLUMNS = [
    "season", "player_id", "first_name", "last_name", "player_slug", "birthdate",
    "school", "country", "last_affiliation", "height_cm", "weight_kg",
    "season_exp", "jersey", "position", "rosterstatus",
    "games_played_current_season_flag", "team_id", "team_name",
    "team_abbreviation", "team_code", "team_city", "playercode",
    "from_year", "to_year", "dleague_flag", "nba_flag", "games_played_flag",
    "draft_year", "draft_round", "draft_number", "greatest_75_flag",
]
TEAMS_COLUMNS = [
    "team_id", "team_slug", "team_long", "team_name", "conference", "division",
    "first_season", "last_season", "stadium", "stadium_capacity", "active",
]


def sql_list(values: list[str] | list[int]) -> str:
    if not values:
        return "NULL"
    if isinstance(values[0], str):
        return ", ".join("'" + v.replace("'", "''") + "'" for v in values)
    return ", ".join(str(v) for v in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spells", type=Path, default=Path("data/build/spells.parquet"))
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--out", type=Path, default=Path("data/lean"))
    parser.add_argument("--db-con", default="cockroach-read")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spells = pl.read_parquet(args.spells).filter(pl.col("season") == args.season)
    if spells.is_empty():
        raise SystemExit(f"no spells for {args.season!r} in {args.spells}")
    players = sorted(spells["player_id"].unique().to_list())
    through = spells["season"].max()
    print(f"target: {len(players)} players, {spells.height:,} spells, through {through}")

    hub = scs_hub.SportsHub(db_con=args.db_con)
    ids = sql_list(players)
    queries = {
        "nba/injuries": f"""
            SELECT game_date, game_id, matchup, team, team_slug, player_name,
                   nba_id, status, reason
            FROM nba.injuries
            WHERE game_date <= (
                SELECT max(game_date) FROM nba.league_game_schedule
                WHERE season = '{through}'
            )
        """,
        "nba/league_game_schedule": f"""
            SELECT season_type, season, game_id, game_date, matchup, team,
                   opponent, home, team_winner
            FROM nba.league_game_schedule
            WHERE season <= '{through}'
        """,
        "nba/player_box_score": f"""
            SELECT game_id, team_id, player_id, player_name, min, comment,
                   usg_pct, pts, poss, pace
            FROM nba.player_box_score
            WHERE player_id IN ({ids})
        """,
        "nba/player_info": f"""
            SELECT {", ".join(PLAYER_INFO_COLUMNS)}
            FROM nba.player_info
            WHERE season <= '{through}'
        """,
        "nba/teams": f"SELECT {', '.join(TEAMS_COLUMNS)} FROM nba.teams",
        "util/player_id_map_vw": "SELECT * FROM util.player_id_map_vw",
    }

    for rel, sql in queries.items():
        frame = hub.db.read(sql)
        out = args.out / f"{rel}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(out)
        print(f"{rel:35} {frame.height:>8,} rows  {frame.width:>2} cols")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

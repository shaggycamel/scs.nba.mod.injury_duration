"""Recreate the model's Parquet snapshot from the live database.

The analysis window is frozen at 2025-26: schedule and box-score history runs
from 2009-10 through that season, and injury rows stop at its last game. The
snapshot is written to `data/` (or `$INJURY_DURATION_DATA`); every later script
reads it, so a fetcher run is the only time the model touches the database.

Usage:
    uv run --group data python scripts/00_fetch.py
    uv run --group data python scripts/00_fetch.py --db-con postgres
    uv run --group data python scripts/00_fetch.py --through 2024-25 --out /tmp/injury-data

Only the columns the model uses are selected. The Cockroach tables carry an
implicit `rowid`; explicit column lists keep it out of the snapshot, where it
would break exact-row deduplication.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import scs_hub

from injury_duration import paths

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
    parser.add_argument(
        "--db-con",
        default="cockroach-read",
        help="credentials.ini section to read through (default: cockroach-read)",
    )
    parser.add_argument(
        "--through",
        default="2025-26",
        help="last season to include (default: 2025-26)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.DATA,
        help="output directory (default: data/ or $INJURY_DURATION_DATA)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hub = scs_hub.SportsHub(db_con=args.db_con)

    cutoff = hub.db.read(
        f"SELECT max(game_date) AS d FROM nba.league_game_schedule WHERE season = '{args.through}'"
    )["d"][0]
    if cutoff is None:
        raise SystemExit(f"no games found for season {args.through!r}")

    queries = {
        "nba/injuries": f"""
            SELECT game_date, game_id, matchup, team, team_slug, player_name,
                   nba_id, status, reason
            FROM nba.injuries
            WHERE game_date <= DATE '{cutoff}'
        """,
        "nba/league_game_schedule": f"""
            SELECT season_type, season, game_id, game_date, matchup, team,
                   opponent, home, team_winner
            FROM nba.league_game_schedule
            WHERE season <= '{args.through}'
        """,
        "nba/player_box_score": f"""
            SELECT game_id, team_id, player_id, player_name, min, comment,
                   usg_pct, pts, poss, pace
            FROM nba.player_box_score
            WHERE game_id IN (
                SELECT game_id FROM nba.league_game_schedule
                WHERE season <= '{args.through}'
            )
        """,
        "nba/player_info": f"""
            SELECT {", ".join(PLAYER_INFO_COLUMNS)}
            FROM nba.player_info
            WHERE season <= '{args.through}'
        """,
        "nba/teams": f"SELECT {', '.join(TEAMS_COLUMNS)} FROM nba.teams",
        "util/player_id_map_vw": "SELECT * FROM util.player_id_map_vw",
        "util/unmatched_player_source_vw": "SELECT * FROM util.unmatched_player_source_vw",
    }

    counts: dict[str, int] = {}
    for rel, sql in queries.items():
        frame = hub.db.read(sql)
        out = args.out / f"{rel}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(out)
        counts[rel] = frame.height
        print(f"{rel:35} {frame.height:>8,} rows  {frame.width:>2} cols")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "db_con": args.db_con,
        "through": args.through,
        "cutoff": str(cutoff),
        "rows": counts,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out} (cutoff {cutoff})")


if __name__ == "__main__":
    main()

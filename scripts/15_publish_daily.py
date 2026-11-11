"""Publish the daily backfill to Cockroach `mod.injury_daily_forecast`.

This is the one-off/repeatable writer for the results already on disk. It
replaces the season's rows in a single transaction, so rerunning after a
correction is safe. The public view is a live join: it fills the actual
return date and games missed from the box score and roster once the outcome
is known, because a daily forecast row is written before that.

Usage:
    uv run --group data python scripts/15_publish_daily.py
    uv run --group data python scripts/15_publish_daily.py --season 2026-27
    uv run --group data python scripts/15_publish_daily.py --view-only
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import scs_hub
from sqlalchemy import text

PUBLIC_TABLE = "mod.injury_daily_forecast"
TABLE = "mod.injury_daily_forecast_base"

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  prediction_date date NOT NULL,
  spell_id text NOT NULL,
  season text NOT NULL,
  player_id bigint,
  player_name text,
  team_slug text,
  injury_start_date date,
  games_missed_so_far bigint,
  days_missed_so_far bigint,
  status text,
  body_region text,
  ailment_class text,
  pred_return_date date,
  pred_return_date_mean date,
  pred_games_median bigint,
  pred_games_mean double precision,
  p_next_game double precision,
  p_return_7d double precision,
  p_return_14d double precision,
  p_return_30d double precision,
  actual_return_date date,
  actual_games_missed bigint,
  return_seen boolean,
  censored boolean,
  history_imputed boolean,
  censor_reason text,
  model_trained_through text,
  model_created_at timestamptz,
  generated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (prediction_date, spell_id)
)
"""

# The public view fills actuals for rows written before the outcome was known.
# A daily forecast is a snapshot; without this join its `actual_*` columns stay
# null forever. The live return is the first game the player played after the
# spell began in that season, and the missed-game count follows the player's
# roster stints so a mid-absence trade is counted correctly. The box lookup
# uses the raw table rather than `player_box_score_vw`: `min > 0` already
# selects played games, and the raw table makes the live join much faster.
VIEW_SQL = f"""
CREATE OR REPLACE VIEW {PUBLIC_TABLE} AS
WITH spells AS (
    SELECT spell_id, player_id, season, min(injury_start_date) AS injury_start_date
    FROM {TABLE}
    GROUP BY spell_id, player_id, season
),
returns AS (
    SELECT sp.spell_id, min(s.game_date) AS live_return_date
    FROM spells sp
    JOIN nba.player_box_score bx
      ON bx.player_id = sp.player_id
     AND bx.min > 0
    JOIN nba.league_game_schedule_vw s
      ON s.game_id = bx.game_id
     AND s.season = sp.season
    WHERE s.game_date > sp.injury_start_date
    GROUP BY sp.spell_id
),
missed AS (
    SELECT sp.spell_id, count(*) AS live_games_missed
    FROM spells sp
    JOIN returns r ON r.spell_id = sp.spell_id
    JOIN nba.team_roster_vw ro
      ON ro.player_id::BIGINT = sp.player_id
     AND ro.season = sp.season
    JOIN nba.league_game_schedule_vw s
      ON s.team = ro.team_slug
     AND s.season = sp.season
     AND s.game_date >= ro.entry_date
     AND s.game_date < ro.exit_date
    WHERE s.game_date >= sp.injury_start_date
      AND s.game_date < r.live_return_date
    GROUP BY sp.spell_id
)
SELECT
    b.prediction_date,
    b.spell_id,
    b.season,
    b.player_id,
    b.player_name,
    b.team_slug,
    b.injury_start_date,
    b.games_missed_so_far,
    b.days_missed_so_far,
    b.status,
    b.body_region,
    b.ailment_class,
    b.pred_return_date,
    b.pred_return_date_mean,
    b.pred_games_median,
    b.pred_games_mean,
    b.p_next_game,
    b.p_return_7d,
    b.p_return_14d,
    b.p_return_30d,
    COALESCE(b.actual_return_date, r.live_return_date) AS actual_return_date,
    COALESCE(b.actual_games_missed, m.live_games_missed) AS actual_games_missed,
    (COALESCE(b.actual_return_date, r.live_return_date) IS NOT NULL) AS return_seen,
    (COALESCE(b.actual_return_date, r.live_return_date) IS NULL) AS censored,
    b.history_imputed,
    b.censor_reason,
    b.model_trained_through,
    b.model_created_at,
    b.generated_at
FROM {TABLE} b
LEFT JOIN returns r ON r.spell_id = b.spell_id
LEFT JOIN missed m ON m.spell_id = b.spell_id
"""

COLUMNS = [
    "prediction_date",
    "spell_id",
    "season",
    "player_id",
    "player_name",
    "team_slug",
    "injury_start_date",
    "games_missed_so_far",
    "days_missed_so_far",
    "status",
    "body_region",
    "ailment_class",
    "pred_return_date",
    "pred_return_date_mean",
    "pred_games_median",
    "pred_games_mean",
    "p_next_game",
    "p_return_7d",
    "p_return_14d",
    "p_return_30d",
    "actual_return_date",
    "actual_games_missed",
    "return_seen",
    "censored",
    "history_imputed",
    "censor_reason",
    "model_trained_through",
    "model_created_at",
    "generated_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/backfill_daily_2026_27.parquet"),
    )
    parser.add_argument(
        "--model-meta",
        type=Path,
        default=Path("artifacts/injury_duration/model.json"),
    )
    parser.add_argument("--db-con", default="cockroach-write")
    parser.add_argument("--season", default="2026-27")
    parser.add_argument(
        "--append",
        action="store_true",
        help="replace only one prediction_date instead of the whole season",
    )
    parser.add_argument(
        "--prediction-date",
        type=date.fromisoformat,
        default=None,
        help="date to replace with --append (default: latest in the input)",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="recreate the public view without writing prediction rows",
    )
    return parser.parse_args()


def _refresh_view(conn) -> None:
    conn.execute(text(VIEW_SQL))
    conn.execute(text(f"GRANT SELECT ON {PUBLIC_TABLE} TO kobe_public"))
    conn.execute(text(f"REVOKE SELECT ON {TABLE} FROM kobe_public"))


def main() -> None:
    args = parse_args()
    if args.view_only:
        hub = scs_hub.SportsHub(db_con=args.db_con)
        with hub.db.engine.begin() as conn:
            _refresh_view(conn)
        print(f"{args.db_con}: recreated {PUBLIC_TABLE}")
        return

    raw = pl.read_parquet(args.input)
    meta = json.loads(args.model_meta.read_text())
    trained = meta["seasons"][-1]
    created = datetime.fromisoformat(meta["created_at"])
    generated = datetime.now(UTC)

    out = raw.select(
        pl.col("prediction_date"),
        pl.col("spell_id"),
        pl.col("season"),
        pl.col("player_id").cast(pl.Int64),
        pl.col("player_name"),
        pl.col("team_slug"),
        pl.col("start_date").alias("injury_start_date"),
        pl.col("games_missed_so_far").cast(pl.Int64),
        pl.col("days_missed_so_far").cast(pl.Int64),
        pl.col("status_clean").alias("status"),
        pl.col("body_region"),
        pl.col("ailment_class"),
        pl.col("pred_return_date_median").alias("pred_return_date"),
        pl.col("pred_return_date_mean"),
        pl.col("pred_games_remaining_median").alias("pred_games_median"),
        pl.col("pred_games_remaining_mean").alias("pred_games_mean"),
        pl.col("p_return_next_game").alias("p_next_game"),
        pl.col("p_return_within_7d").alias("p_return_7d"),
        pl.col("p_return_within_14d").alias("p_return_14d"),
        pl.col("p_return_within_30d").alias("p_return_30d"),
        pl.col("actual_return_date"),
        pl.col("actual_games_missed").cast(pl.Int64),
        pl.col("return_seen"),
        pl.col("censored"),
        pl.col("history_imputed"),
        pl.col("censor_reason"),
        pl.lit(trained).alias("model_trained_through"),
        pl.lit(created).alias("model_created_at"),
        pl.lit(generated).alias("generated_at"),
    ).select(COLUMNS)

    hub = scs_hub.SportsHub(db_con=args.db_con)
    latest_report = hub.db.read(
        f"SELECT max(game_date) AS d FROM nba.injuries_vw WHERE season = '{args.season}'"
    ).item()
    latest_prediction = out["prediction_date"].max()
    if (
        latest_report is not None
        and latest_prediction is not None
        and latest_prediction > latest_report
    ):
        raise SystemExit(
            f"refusing to publish prediction_date {latest_prediction}, later than the "
            f"latest injury report {latest_report}"
        )
    if args.append:
        replace_date = args.prediction_date or out["prediction_date"].max()
        out = out.filter(pl.col("prediction_date") == replace_date)
        where = f"season = '{args.season}' AND prediction_date = DATE '{replace_date}'"
    else:
        replace_date = None
        where = f"season = '{args.season}'"

    with hub.db.engine.begin() as conn:
        conn.execute(text(DDL))
        conn.execute(text(f"DELETE FROM {TABLE} WHERE {where}"))
        out.write_database(TABLE, conn, if_table_exists="append")
        _refresh_view(conn)

    count = hub.db.read(f"SELECT count(*) AS n FROM {PUBLIC_TABLE} WHERE {where}").item()
    suffix = f" on {replace_date}" if replace_date else ""
    print(
        f"{args.db_con}: wrote {count:,} rows for {args.season}{suffix} to {PUBLIC_TABLE} "
        f"(model trained through {trained})"
    )


if __name__ == "__main__":
    main()

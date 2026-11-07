"""Publish the daily backfill to Cockroach `mod.injury_daily_forecast`.

This is the one-off/repeatable writer for the results already on disk. It
replaces the season's rows in a single transaction, so rerunning after a
correction is safe.

Usage:
    uv run --group data python scripts/15_publish_daily.py
    uv run --group data python scripts/15_publish_daily.py --season 2026-27
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import scs_hub
from sqlalchemy import text

TABLE = "mod.injury_daily_forecast"

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

COLUMNS = [
    "prediction_date", "spell_id", "season", "player_id", "player_name",
    "team_slug", "injury_start_date", "games_missed_so_far",
    "days_missed_so_far", "status", "body_region", "ailment_class",
    "pred_return_date", "pred_return_date_mean", "pred_games_median",
    "pred_games_mean", "p_next_game", "p_return_7d", "p_return_14d",
    "p_return_30d", "actual_return_date", "actual_games_missed",
    "return_seen", "censored", "history_imputed", "censor_reason",
    "model_trained_through", "model_created_at", "generated_at",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    with hub.db.engine.begin() as conn:
        conn.execute(text(DDL))
        conn.execute(
            text(f"DELETE FROM {TABLE} WHERE season = '{args.season}'")
        )
        out.write_database(TABLE, conn, if_table_exists="append")
        conn.execute(text(f"GRANT SELECT ON {TABLE} TO kobe_public"))

    count = hub.db.read(
        f"SELECT count(*) AS n FROM {TABLE} WHERE season = '{args.season}'"
    ).item()
    print(
        f"{args.db_con}: wrote {count:,} rows for {args.season} to {TABLE} "
        f"(model trained through {trained})"
    )


if __name__ == "__main__":
    main()

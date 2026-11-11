"""Backfill daily return-date forecasts for one season.

For every injured player-day, the model is asked when the player will return
using only what was known that day. Output is one row per (spell,
prediction_date); censored spells stop at the data cutoff and are marked by
`return_seen = 0`.

Usage:
    uv run python scripts/14_backfill_daily.py --season 2026-27
    uv run python scripts/14_backfill_daily.py --season 2026-27 --max-k 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from injury_duration import daily, persist
from injury_duration import experiment as ex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/injury_duration/model.pkl"),
        help="persisted ModelBundle",
    )
    parser.add_argument("--season", default="2026-27", help="season to backfill")
    parser.add_argument("--max-k", type=int, default=100)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="spells per forecast chunk (keeps the 100-step grid in memory)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="readable CSV (default: reports/backfill_daily_<season>.csv)",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help="full Parquet (default: artifacts/backfill_daily_<season>.parquet)",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="load even when library versions differ from fit time",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = persist.load(args.model, strict=not args.allow_version_mismatch)
    d = ex.load()

    spells = d["spells"].filter(pl.col("season") == args.season)
    if spells.is_empty():
        raise SystemExit(f"no spells for {args.season!r} in the current build")

    # The build can contain future games through the roster stints; the
    # observation cutoff is the last report/box event for this season.
    cutoff = spells["last_missed_date"].max()
    if cutoff is None:
        raise SystemExit(f"no observed dates for {args.season!r}")
    out_chunks: list[pl.DataFrame] = []
    n_states = 0
    for offset in range(0, spells.height, args.chunk_size):
        chunk = spells.slice(offset, args.chunk_size)
        states = daily.build_daily_states(chunk, d["features"], d["rows"], cutoff)
        if states.is_empty():
            continue
        n_states += states.height
        chunk_out = daily.forecast_daily(bundle.live, states, d["team_context"], max_k=args.max_k)
        if "state_id" in chunk_out.columns:
            chunk_out = chunk_out.drop("state_id")
        out_chunks.append(chunk_out)
    if not out_chunks:
        raise SystemExit(f"no daily states for {args.season!r}")
    out = pl.concat(out_chunks, how="vertical_relaxed")
    out = out.with_columns(
        pl.when(pl.col("event") == 1)
        .then(pl.col("next_date"))
        .otherwise(None)
        .alias("actual_return_date"),
        pl.col("games_missed").alias("actual_games_missed"),
        (pl.col("event") == 1).alias("return_seen"),
        (pl.col("event") == 0).alias("censored"),
    )

    slug = args.season.replace("-", "_")
    csv_path = args.out_csv or Path(f"reports/backfill_daily_{slug}.csv")
    parquet_path = args.out_parquet or Path(f"artifacts/backfill_daily_{slug}.parquet")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    readable = out.select(
        "prediction_date",
        "spell_id",
        "player_id",
        "player_name",
        "team_slug",
        "start_date",
        "games_missed_so_far",
        "days_missed_so_far",
        "status_clean",
        "body_region",
        "ailment_class",
        pl.col("pred_return_date_median").alias("pred_return_date"),
        pl.col("pred_games_remaining_median").alias("pred_games_median"),
        pl.col("pred_games_remaining_mean").alias("pred_games_mean"),
        pl.col("p_return_next_game").round(3).alias("p_next_game"),
        pl.col("p_return_within_7d").round(3).alias("p_7d"),
        pl.col("p_return_within_14d").round(3).alias("p_14d"),
        pl.col("p_return_within_30d").round(3).alias("p_30d"),
        "actual_return_date",
        "actual_games_missed",
        "return_seen",
        "censored",
        "history_imputed",
        "censor_reason",
    ).sort("prediction_date", "player_name")
    readable.write_csv(csv_path)
    out.write_parquet(parquet_path)

    n_spells = spells.height
    n_censored = spells.filter(pl.col("event") == 0).height
    print(
        f"loaded {args.model}  (trained {bundle.metadata['seasons'][0]}.."
        f"{bundle.metadata['seasons'][-1]})"
    )
    print(
        f"{args.season}: {n_spells:,} spells "
        f"({n_censored:,} censored at {cutoff}); "
        f"{n_states:,} player-days"
    )
    print(f"  {csv_path}")
    print(f"  {parquet_path}")
    print()
    top = (
        readable.group_by("spell_id", "player_name", "start_date", "ailment_class")
        .agg(
            pl.col("pred_games_median").last().alias("pred_games_median"),
            pl.col("pred_return_date").last().alias("pred_return_date"),
            pl.col("actual_games_missed").last().alias("actual_games_so_far"),
            pl.col("return_seen").last().alias("returned"),
        )
        .sort("pred_games_median", descending=True)
        .head(10)
    )
    print(top)

    resolved = out.filter(
        pl.col("return_seen")
        & pl.col("actual_return_date").is_not_null()
        & pl.col("pred_return_date_median").is_not_null()
    ).with_columns(
        (pl.col("pred_return_date_median") - pl.col("actual_return_date"))
        .dt.total_days()
        .abs()
        .alias("date_error_days"),
        (
            pl.col("pred_games_remaining_median")
            - (pl.col("games_missed") - pl.col("games_missed_so_far"))
        )
        .abs()
        .alias("games_error"),
        pl.when(pl.col("days_missed_so_far") == 0)
        .then(pl.lit("onset"))
        .when(pl.col("days_missed_so_far") <= 3)
        .then(pl.lit("1-3 days"))
        .when(pl.col("days_missed_so_far") <= 7)
        .then(pl.lit("4-7 days"))
        .otherwise(pl.lit("8+ days"))
        .alias("bucket"),
    )
    if resolved.height:
        print()
        print("resolved-spell error by prediction day (median absolute):")
        print(
            resolved.group_by("bucket")
            .agg(
                pl.len().alias("player_days"),
                pl.col("date_error_days").median().alias("date_error_days"),
                pl.col("games_error").median().alias("games_error"),
            )
            .sort("bucket")
        )


if __name__ == "__main__":
    main()

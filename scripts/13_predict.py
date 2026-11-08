"""Load the persisted model and forecast a season.

Usage:
    uv run python scripts/13_predict.py
    uv run python scripts/13_predict.py --season 2025-26 --max-k 100
    uv run python scripts/13_predict.py --model /tmp/injury-model/model.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from injury_duration import experiment as ex
from injury_duration import forecast, persist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/injury_duration/model.pkl"),
        help="path to the persisted bundle",
    )
    parser.add_argument("--season", default="2025-26", help="season to score")
    parser.add_argument("--max-k", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="forecast Parquet path (default: artifacts/forecasts_<season>.parquet)",
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

    feats = d["features"].filter(pl.col("season") == args.season)
    if feats.is_empty():
        raise SystemExit(f"no spells for {args.season!r} in the build")

    onset = forecast.forecast_at_onset(bundle.onset, feats, d["team_context"], max_k=args.max_k)
    out = args.out or Path(f"artifacts/forecasts_{args.season}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    onset.write_parquet(out)

    rows = ex.hazard_design(d["rows"], d["features"]).filter(pl.col("season") == args.season)
    live = forecast.forecast_in_progress(bundle.live, rows)
    live_out = out.with_name(out.stem + "_next_game.parquet")
    live.write_parquet(live_out)

    print(
        f"loaded {args.model}  (trained {bundle.metadata['seasons'][0]}.."
        f"{bundle.metadata['seasons'][-1]})"
    )
    if args.season <= bundle.metadata["seasons"][-1]:
        print(
            f"note: {args.season} is inside the training window; this is an in-sample demonstration"
        )
    print(f"{args.season}: {onset.height:,} spells -> {out}")
    print(f"{args.season}: {live.height:,} per-game rows -> {live_out}")
    print()
    print(forecast.summarise(onset.sort("pred_mean_games", descending=True)).head(10))


if __name__ == "__main__":
    main()

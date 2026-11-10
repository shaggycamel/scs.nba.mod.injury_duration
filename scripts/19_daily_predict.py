"""Score today's active spells and write the rows for the daily append.

Run against a lean current-season build:

    INJURY_DURATION_DATA=data/lean_current uv run python \
        scripts/19_daily_predict.py --season 2026-27

Output is the same shape as the historical backfill, restricted to the latest
prediction date. `15_publish_daily.py --append` writes it to Cockroach.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from injury_duration import daily, paths, persist
from injury_duration import experiment as ex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/injury_duration/model.pkl"))
    parser.add_argument("--active", type=Path, default=Path("artifacts/active_spells.parquet"))
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--max-k", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = persist.load(args.model)
    active = pl.read_parquet(args.active)
    spell_ids = active.filter(pl.col("season") == args.season)["spell_id"].to_list()
    if not spell_ids:
        raise SystemExit(f"no active spells for {args.season!r}")

    d = ex.load()
    spells = d["spells"].filter(pl.col("spell_id").is_in(spell_ids))
    if spells.is_empty():
        raise SystemExit("active spells are not present in the build")
    # The panel now extends to the end of the season through the roster
    # stints, but the report is only available through the current data date.
    # Score as of the latest report, not the last scheduled game.
    cutoff = pl.read_parquet(paths.INJURIES, columns=["game_date"])["game_date"].max()
    panel_max = pl.read_parquet(paths.BUILD / "panel.parquet")["game_date"].max()
    if cutoff is None or (panel_max is not None and cutoff > panel_max):
        cutoff = panel_max

    states = daily.build_daily_states(spells, d["features"], d["rows"], cutoff, through=cutoff)
    states = states.filter(pl.col("prediction_date") == cutoff)
    out = daily.forecast_daily(bundle.live, states, d["team_context"], max_k=args.max_k)
    out = out.with_columns(
        pl.when(pl.col("event") == 1)
        .then(pl.col("next_date"))
        .otherwise(None)
        .alias("actual_return_date"),
        pl.col("games_missed").alias("actual_games_missed"),
        (pl.col("event") == 1).alias("return_seen"),
        (pl.col("event") == 0).alias("censored"),
    )
    path = args.out or Path(f"artifacts/daily_predictions_{args.season}.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    print(f"{args.season}: {out.height:,} active spells at {cutoff} -> {path}")
    print(
        out.select(
            [
                "player_name",
                "days_missed_so_far",
                "pred_games_remaining_median",
                "pred_return_date_median",
                "p_return_next_game",
            ]
        ).head(10)
    )


if __name__ == "__main__":
    main()

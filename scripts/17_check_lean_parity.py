"""Compare lean-snapshot features with the full build.

The lean snapshot filters box scores to the target players and injuries to
those players; if no feature for a target spell depends on another player's
rows, every feature row should be identical. This script checks that.

Usage:
    uv run python scripts/17_check_lean_parity.py --season 2025-26
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from injury_duration import features
from injury_duration import hazard as hz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, default=Path("data/build"))
    parser.add_argument("--lean", type=Path, default=Path("data/lean/build"))
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def compare_frame(
    full: pl.DataFrame,
    lean: pl.DataFrame,
    key: list[str],
    columns: list[str],
    tolerance: float,
) -> list[str]:
    joined = full.join(lean, on=key, how="inner", suffix="_lean")
    problems = []
    for col in columns:
        if col not in full.columns or col not in lean.columns:
            problems.append(f"{col}: missing from one side")
            continue
        if col in key:
            continue
        right = pl.col(col + "_lean")
        if full.schema[col].is_numeric():
            left = pl.col(col).cast(pl.Float64)
            r = right.cast(pl.Float64)
            bad = joined.filter(
                (pl.col(col).is_null() != right.is_null())
                | ((left - r).abs() > tolerance)
            )
            diff = (left - r).abs().max()
            if bad.height:
                problems.append(
                    f"{col}: {bad.height}/{joined.height} differ "
                    f"(max abs diff {diff})"
                )
        else:
            bad = joined.filter(pl.col(col).ne_missing(right))
            if bad.height:
                problems.append(f"{col}: {bad.height}/{joined.height} differ")
    return problems


def main() -> None:
    args = parse_args()
    full_sp = pl.read_parquet(args.full / "spell_features.parquet")
    lean_sp = pl.read_parquet(args.lean / "spell_features.parquet")
    target = pl.read_parquet(args.full / "spells.parquet").filter(
        pl.col("season") == args.season
    )["spell_id"].to_list()
    full_sp = full_sp.filter(pl.col("spell_id").is_in(target)).sort("spell_id")
    lean_sp = lean_sp.filter(pl.col("spell_id").is_in(target)).sort("spell_id")

    print(f"spell_features: full {full_sp.height:,}, lean {lean_sp.height:,}")
    if full_sp.height != lean_sp.height:
        raise SystemExit("row counts differ; compare spell ids first")

    cols = features.all_features() + ["next_date", "event", "games_missed"]
    problems = compare_frame(full_sp, lean_sp, ["spell_id"], cols, args.tolerance)

    full_rows = pl.read_parquet(args.full / "hazard_rows.parquet").filter(
        pl.col("spell_id").is_in(target)
    )
    lean_rows = pl.read_parquet(args.lean / "hazard_rows.parquet").filter(
        pl.col("spell_id").is_in(target)
    )
    print(f"hazard_rows:   full {full_rows.height:,}, lean {lean_rows.height:,}")
    row_cols = [c for c in hz.TIME_VARYING + ["returns_next", "game_date", "game_id"]
                if c in full_rows.columns and c in lean_rows.columns]
    problems += compare_frame(
        full_rows, lean_rows, ["spell_id", "spell_game_no"], row_cols, args.tolerance
    )

    if problems:
        print("\nPARITY FAILED")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("\nPARITY OK — every checked feature matches the full build")


if __name__ == "__main__":
    main()

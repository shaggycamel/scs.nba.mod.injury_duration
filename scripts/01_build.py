"""Build the injury spell dataset and the hazard rows, and report diagnostics.

Run: uv run python scripts/01_build.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import polars as pl

from injury_duration import features, paths, spells
from injury_duration import hazard as hz

warnings.filterwarnings("ignore", message="Sortedness of columns")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=None,
        help="where to write build_diagnostics.json "
             "(default: reports/build_diagnostics.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build = paths.ensure_build()

    sp, panel, diag = spells.build_spells()
    tctx = features.team_context()

    idx = features.build_index_features(sp)
    rows = hz.attach_schedule_density(hz.build_hazard_rows(sp, panel), tctx)

    sp.write_parquet(build / "spells.parquet")
    panel.write_parquet(build / "panel.parquet")
    idx.write_parquet(build / "spell_features.parquet")
    rows.write_parquet(build / "hazard_rows.parquet")
    tctx.write_parquet(build / "team_context.parquet")

    diag["hazard_rows"] = rows.height
    diag["hazard_return_rate"] = float(rows["returns_next"].mean())
    diag["feature_cols"] = len(features.all_features())
    diagnostics = args.diagnostics_out or (paths.ensure_reports() / "build_diagnostics.json")
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.write_text(json.dumps(diag, indent=2, default=str))

    print(f"spells            {sp.height:>8,}")
    print(f"  observed return {int(sp['event'].sum()):>8,}  "
          f"({sp['event'].mean():.1%})")
    print(f"panel rows        {diag['panel_rows']:>8,}")
    print(f"hazard rows       {rows.height:>8,}  "
          f"return rate {diag['hazard_return_rate']:.3f}")
    print(f"features          {len(features.all_features()):>8,}")
    print()
    print("censoring:")
    for r in diag["censor_reasons"]:
        print(f"  {r['censor_reason']:<22} {r['len']:>6,}")
    print()
    print("spells per season:")
    per = sp.group_by("season").agg(
        pl.len().alias("spells"),
        pl.col("event").mean().round(3).alias("observed"),
        pl.col("games_missed").median().alias("median_games"),
    ).sort("season")
    print(per)
    print(f"\nwrote {build}")


if __name__ == "__main__":
    main()

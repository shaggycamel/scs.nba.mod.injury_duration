"""Fit and persist the onset and live hazard models.

The final fit uses every season in the build (through 2025-26), not the
train/validation split used for the published metrics. `model.json` records the
training window, feature lists, hyperparameters, package versions and the
held-out metrics from `reports/`.

Usage:
    uv run python scripts/12_fit_model.py
    uv run python scripts/12_fit_model.py --out /tmp/injury-model
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import polars as pl

from injury_duration import experiment as ex
from injury_duration import forecast, paths, persist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/injury_duration"),
        help="directory for model.pkl and model.json",
    )
    parser.add_argument(
        "--through",
        default="2025-26",
        help="last season used for the final fit (default: 2025-26)",
    )
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=paths.ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def held_out_metrics() -> dict:
    dm = pl.read_csv(paths.REPORTS / "duration_metrics.csv")
    pg = pl.read_csv(paths.REPORTS / "per_game_return_metrics.csv")
    hist = dm.filter((pl.col("model") == "hazard lightgbm") & (pl.col("point") == "median")).row(
        0, named=True
    )
    live = pg.filter(pl.col("model") == "lightgbm / with_report_state").row(0, named=True)
    return {
        "duration": {
            "model": hist["model"],
            "c_index": round(float(hist["c_index"]), 4),
            "mae_observed": round(float(hist["mae_observed"]), 4),
            "mae_long": round(float(hist["mae_observed_long"]), 4),
            "n_spells": int(hist["n_spells"]),
        },
        "next_game": {
            "model": live["model"],
            "auc": round(float(live["auc"]), 4),
            "log_loss": round(float(live["log_loss"]), 4),
            "n_rows": int(live["n_rows"]),
        },
    }


def main() -> None:
    args = parse_args()
    d = ex.load()
    design = ex.hazard_design(d["rows"], d["features"])
    train = design.filter(pl.col("season") <= args.through)
    if train.is_empty():
        raise SystemExit(f"no rows through {args.through!r}")

    onset_cols = ex.index_feature_cols()
    live_cols = ex.dynamic_feature_cols()
    onset = forecast.fit(train, onset_cols)
    live = forecast.fit(train, live_cols)

    metadata = persist.build_metadata(
        seasons=sorted(train["season"].unique().to_list()),
        rows=train.height,
        spells=int(d["features"].filter(pl.col("season") <= args.through).height),
        features={"onset": onset_cols, "live": live_cols},
        params={"onset": forecast.BEST_LGBM, "live": forecast.BEST_LGBM},
        metrics=held_out_metrics(),
        git_commit=git_commit(),
    )
    path = persist.save(persist.ModelBundle(onset, live, metadata), args.out / "model.pkl")
    print(
        f"wrote {path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)\n"
        f"  seasons {metadata['seasons'][0]}..{metadata['seasons'][-1]}, "
        f"{metadata['spells']:,} spells, {metadata['rows']:,} hazard rows"
    )
    print(f"  metadata {path.with_name('model.json')}")


if __name__ == "__main__":
    main()

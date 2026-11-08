"""Forecast injury duration for the held-out season, and show the working.

Trains on 2021-22..2024-25 and forecasts every 2025-26 spell as it would have
been forecast the night the player was ruled out, then prints the cases a
person would want to check: the absences the model called longest, the ones
it got worst, and a game-by-game trace of one long spell.

Run: uv run python scripts/05_forecast.py
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from injury_duration import experiment as ex
from injury_duration import forecast, models, paths

warnings.filterwarnings("ignore")

FIT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
FORECAST_SEASON = "2025-26"


def main() -> None:
    rep = paths.ensure_reports()
    d = ex.load()
    design = ex.hazard_design(d["rows"], d["features"])
    fit_rows = design.filter(pl.col("season").is_in(FIT_SEASONS))
    target_rows = design.filter(pl.col("season") == FORECAST_SEASON)
    sp = d["features"].filter(pl.col("season") == FORECAST_SEASON)

    onset = forecast.fit(fit_rows, ex.index_feature_cols())
    live = forecast.fit(fit_rows, ex.dynamic_feature_cols())

    fc = forecast.forecast_at_onset(onset, sp, d["team_context"])
    # The readable summary goes in reports/; the wide per-row and per-(spell, k)
    # tables are regenerable model output and belong with the other build
    # artefacts, not in version control.
    forecast.summarise(fc).write_csv(rep / "forecast_2025_26.csv")
    fc.write_csv(paths.ensure_build() / "forecast_2025_26_full.csv")

    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(190)
    pl.Config.set_fmt_str_lengths(30)

    print("=" * 78)
    print("LONGEST ABSENCES THE MODEL CALLED AT ONSET (2025-26)")
    print("=" * 78)
    print(forecast.summarise(fc.sort("pred_mean_games", descending=True).head(15)))

    print()
    print("=" * 78)
    print("WHERE IT WAS MOST WRONG (observed returns only)")
    print("=" * 78)
    obs = fc.filter(pl.col("event") == 1).with_columns(
        (pl.col("pred_median_games") - pl.col("games_missed")).alias("err")
    )
    print("under-called (missed far more than predicted):")
    print(forecast.summarise(obs.sort("err").head(8)))
    print("\nover-called (came back far sooner than predicted):")
    print(forecast.summarise(obs.sort("err", descending=True).head(8)))

    print()
    print("=" * 78)
    print("GAME-BY-GAME: P(plays the next game), with the live report")
    print("=" * 78)
    long_spells = (
        fc.filter((pl.col("games_missed") >= 12) & (pl.col("event") == 1))
        .sort("games_missed", descending=True)
        .head(2)
    )
    trace = forecast.forecast_in_progress(live, target_rows)
    for row in long_spells.iter_rows(named=True):
        print(f"\n{row['player_name']}  {row['index_reason']}")
        print(
            f"  ruled out {row['start_date']}, forecast at onset: "
            f"median {row['pred_median_games']:.0f} games, "
            f"mean {row['pred_mean_games']:.1f}; actual {row['games_missed']}"
        )
        t = trace.filter(pl.col("spell_id") == row["spell_id"]).sort("games_missed_so_far")
        for r in t.iter_rows(named=True):
            bar = "#" * int(r["p_plays_next"] * 60)
            flag = "  <- played" if r["returns_next"] == 1 else ""
            print(
                f"    after {r['games_missed_so_far']:>3} missed "
                f"({r['game_date']}, {r['status_clean'] or 'not listed':<12}) "
                f"p={r['p_plays_next']:.3f} {bar}{flag}"
            )

    # Day-horizon calibration, written out so the HTML report quotes measured
    # numbers rather than a copy of them.
    dur = fc["days_out"].to_numpy().astype(float)
    ev = fc["event"].to_numpy().astype(int)
    cal_rows = []
    for dh in models.DAY_HORIZONS:
        mask = ((ev == 1) & (dur <= dh)) | (dur > dh)
        y = (dur[mask] <= dh).astype(int)
        pr = fc[f"p_back_within_{dh}d"].to_numpy()[mask]
        base = np.full_like(pr, y.mean())
        cal_rows.append(
            {
                "horizon_days": dh,
                "n_evaluable": int(mask.sum()),
                "actual_rate": round(float(y.mean()), 4),
                "mean_pred": round(float(pr.mean()), 4),
                "brier": round(float(np.mean((pr - y) ** 2)), 5),
                "brier_skill": round(
                    float(1 - np.mean((pr - y) ** 2) / np.mean((base - y) ** 2)), 4
                ),
            }
        )
    pl.DataFrame(cal_rows).write_csv(rep / "day_horizon_calibration.csv")
    print("\nP(back within D days), held-out season:")
    print(pl.DataFrame(cal_rows))

    curves = forecast.survival_curve(
        onset, sp.filter(pl.col("games_missed") >= 20), d["team_context"], max_k=60
    )
    curves.write_csv(paths.ensure_build() / "survival_curves_long_spells.csv")
    print(f"\nwrote {rep / 'forecast_2025_26.csv'}")


if __name__ == "__main__":
    main()

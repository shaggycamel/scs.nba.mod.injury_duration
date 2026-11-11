"""Compare the LightGBM + HistGB hazard ensemble with each single model.

Time split: train 2021-24, tune/validate 2024-25, test 2025-26. The ensemble
averages the per-game hazards; each member keeps its own encoder.

The per-game score uses the live report-state features; the duration grid uses
the onset features because the grid has no report-state history.

Usage:
    uv run python scripts/22_ensemble.py
"""

from __future__ import annotations

import polars as pl

from injury_duration import evaluate as ev
from injury_duration import experiment as ex
from injury_duration import forecast, models


def per_game(model, design: pl.DataFrame, label: str) -> dict:
    rows = design.filter(pl.col("season") == "2025-26")
    p = model.hazard(rows)
    return {"kind": "per_game", **ev.hazard_row_metrics(rows["returns_next"].to_numpy(), p, label)}


def duration(model, feats: pl.DataFrame, tctx: pl.DataFrame, label: str) -> list[dict]:
    grid = ex.grid_design(feats, tctx, max_k=models.MAX_K)
    pred = models.predict_from_grid(model, grid)
    spells = feats.select("spell_id", "season", "games_missed", "event").filter(
        pl.col("season") == "2025-26"
    )
    joined = spells.join(pred, on="spell_id", how="left")
    return [
        {
            "kind": "duration_mean",
            **ev.duration_metrics(joined, joined["pred_mean_games"].to_numpy(), label),
        },
        {
            "kind": "duration_median",
            **ev.duration_metrics(joined, joined["pred_median_games"].to_numpy(), label),
        },
    ]


def main() -> None:
    d = ex.load()
    design = ex.hazard_design(d["rows"], d["features"])
    train = design.filter(pl.col("season") <= "2023-24")
    onset = ex.index_feature_cols()
    live = ex.dynamic_feature_cols()

    onset_num, onset_cat = ex.split_numeric_categorical(onset)
    live_num, live_cat = ex.split_numeric_categorical(live)

    onset_models = {
        "lightgbm": models.HazardModel(
            "lightgbm", "lightgbm", onset_num, onset_cat, forecast.BEST_LGBM
        ).fit(train),
        "histgb": models.HazardModel("histgb", "histgb", onset_num, onset_cat, forecast.HISTGB).fit(
            train
        ),
        "ensemble": forecast.fit_ensemble(train, onset),
    }
    live_models = {
        "lightgbm": models.HazardModel(
            "lightgbm", "lightgbm", live_num, live_cat, forecast.BEST_LGBM
        ).fit(train),
        "histgb": models.HazardModel("histgb", "histgb", live_num, live_cat, forecast.HISTGB).fit(
            train
        ),
        "ensemble": forecast.fit_ensemble(train, live),
    }

    rows = []
    for label in ("lightgbm", "histgb", "ensemble"):
        rows.append(per_game(live_models[label], design, label))
        rows.extend(duration(onset_models[label], d["features"], d["team_context"], label))

    out = pl.DataFrame(rows)
    out.write_csv("reports/ensemble_metrics.csv")
    print(
        out.select(
            "kind",
            "model",
            "n_rows",
            "n_spells",
            "c_index",
            "mae_observed",
            "mae_observed_long",
            "auc",
            "brier",
            "brier_skill",
        ).sort("kind", "model")
    )
    print("wrote reports/ensemble_metrics.csv")


if __name__ == "__main__":
    main()

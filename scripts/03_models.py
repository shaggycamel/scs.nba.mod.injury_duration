"""Fit and compare injury-duration models on time-based splits.

Two questions, scored separately:

A. Per-game: a player has missed k games — do they play the next one? Scored
   on the hazard rows. Comparing a model that only sees the injury as first
   filed against one that also sees how the report has moved since says how
   much of the signal is in the live report versus the injury itself.

B. Per-spell: at the moment the player is ruled out, how many games will
   they miss? Scored against baselines that ignore the features.

Train 2021-22..2023-24, tune on 2024-25, refit on all four, score 2025-26.

Run: uv run python scripts/03_models.py
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import polars as pl

from injury_duration import evaluate as ev
from injury_duration import experiment as ex
from injury_duration import models, paths

warnings.filterwarnings("ignore")

TRAIN = ["2021-22", "2022-23", "2023-24"]
VALID = ["2024-25"]
TEST = ["2025-26"]

GRIDS = {
    "logistic": [{"C": c} for c in (0.03, 0.1, 0.3, 1.0)],
    "forest": [{"n_estimators": 400, "min_samples_leaf": leaf} for leaf in (5, 20, 50)],
    "histgb": [
        {
            "max_iter": 400,
            "learning_rate": lr,
            "max_leaf_nodes": leaves,
            "early_stopping": True,
            "validation_fraction": 0.15,
        }
        for lr in (0.05, 0.1)
        for leaves in (15, 31)
    ],
    "lightgbm": [
        {
            "n_estimators": n,
            "learning_rate": lr,
            "num_leaves": leaves,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
        }
        for n in (300, 700)
        for lr in (0.03, 0.06)
        for leaves in (15, 31)
    ],
}


def tune(kind: str, cols: list[str], train: pl.DataFrame, valid: pl.DataFrame):
    """Pick hyperparameters by log loss on the held-out season."""
    num, cat = ex.split_numeric_categorical(cols)
    best, best_ll = None, np.inf
    trace = []
    for params in GRIDS[kind]:
        m = models.HazardModel(kind, kind, num, cat, params).fit(train)
        p = m.hazard(valid)
        ll = ev.hazard_row_metrics(valid["returns_next"].to_numpy(), p)["log_loss"]
        trace.append({"kind": kind, **params, "valid_log_loss": round(ll, 5)})
        if ll < best_ll:
            best, best_ll = params, ll
    return best, best_ll, trace


def main() -> None:
    rep = paths.ensure_reports()
    d = ex.load()
    design = ex.hazard_design(d["rows"], d["features"])
    feats = d["features"]

    sets = {
        "index_only": ex.index_feature_cols(),
        "with_report_state": ex.dynamic_feature_cols(),
    }

    tr = design.filter(pl.col("season").is_in(TRAIN))
    va = design.filter(pl.col("season").is_in(VALID))
    te = design.filter(pl.col("season").is_in(TEST))
    trva = design.filter(pl.col("season").is_in(TRAIN + VALID))
    print(f"hazard rows  train {tr.height:,}  valid {va.height:,}  test {te.height:,}")

    # ---------------------------------------------------------------- A
    print("\n" + "=" * 78)
    print("A. PER-GAME RETURN PREDICTION (test season 2025-26)")
    print("=" * 78)
    chosen, per_game, traces = {}, [], []
    y_te = te["returns_next"].to_numpy()
    base = np.full(te.height, tr["returns_next"].mean())
    per_game.append(ev.hazard_row_metrics(y_te, base, "base rate (train mean)"))

    for set_name, cols in sets.items():
        for kind in GRIDS:
            params, ll, trace = tune(kind, cols, tr, va)
            traces.extend({"feature_set": set_name, **t} for t in trace)
            num, cat = ex.split_numeric_categorical(cols)
            m = models.HazardModel(kind, kind, num, cat, params).fit(trva)
            p = m.hazard(te)
            per_game.append(ev.hazard_row_metrics(y_te, p, f"{kind} / {set_name}"))
            chosen[(set_name, kind)] = params
            print(f"  {kind:<10} {set_name:<18} valid_ll={ll:.4f}  {params}")

    pg = pl.DataFrame(per_game).with_columns(
        pl.col(c).round(4) for c in ("auc", "log_loss", "log_loss_skill", "brier", "brier_skill")
    )
    print()
    print(pg.select("model", "n_rows", "auc", "log_loss", "brier", "brier_skill"))
    pg.write_csv(rep / "per_game_return_metrics.csv")
    pl.DataFrame(traces).write_csv(rep / "hyperparameter_trace.csv")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 78)
    print("B. DURATION AT THE MOMENT THE PLAYER IS RULED OUT (test 2025-26)")
    print("=" * 78)
    sp_trva = feats.filter(pl.col("season").is_in(TRAIN + VALID))
    sp_te = feats.filter(pl.col("season").is_in(TEST))

    results, surv_by_model = [], {}

    strata = [
        ((), "km global"),
        (("ailment_class",), "km by ailment"),
        (("body_region",), "km by region"),
        (("body_region", "ailment_class"), "km by region x ailment"),
        (("body_region", "ailment_class", "index_status"), "km by region x ailment x status"),
    ]
    for by, name in strata:
        for stat in ("mean", "median"):
            b = models.KMBaseline(by=by, statistic=stat, name=name).fit(sp_trva)
            results.append(ev.duration_metrics(sp_te, b.predict(sp_te), name) | {"point": stat})
    ph = models.PlayerHistoryBaseline().fit(sp_trva)
    results.append(ev.duration_metrics(sp_te, ph.predict(sp_te), ph.name) | {"point": "mean"})

    # Censoring-blind regression, for contrast. Poisson targets the mean and
    # quantile(0.5) the median, so it meets the hazard models on both.
    num, cat = ex.split_numeric_categorical(ex.index_feature_cols())
    num_sp = [c for c in num if c in feats.columns]
    for obj, point in (("poisson", "mean"), ("quantile", "median")):
        reg = models.ObservedOnlyRegressor(num_sp, cat, objective=obj).fit(sp_trva)
        results.append(ev.duration_metrics(sp_te, reg.predict(sp_te), reg.name) | {"point": point})

    # Hazard models, rolled forward over the prediction grid.
    grid_te = ex.grid_design(sp_te, d["team_context"], max_k=models.MAX_K)
    cols = ex.index_feature_cols()
    num, cat = ex.split_numeric_categorical(cols)
    for kind in GRIDS:
        params = chosen[("index_only", kind)]
        m = models.HazardModel(kind, kind, num, cat, params).fit(trva)
        pred = models.predict_from_grid(m, grid_te)
        joined = sp_te.select("spell_id", "games_missed", "event").join(
            pred, on="spell_id", how="left"
        )
        for col, point in (("pred_mean_games", "mean"), ("pred_median_games", "median")):
            results.append(
                ev.duration_metrics(joined, joined[col].to_numpy(), f"hazard {kind}")
                | {"point": point}
            )
        # Survival at each horizon, for a calibration-aware score.
        h = m.hazard(grid_te)
        g = grid_te.select("spell_id", "k").with_columns(pl.Series("h", h))
        g = g.sort("spell_id", "k").with_columns(
            (1 - pl.col("h").clip(1e-9, 1 - 1e-9))
            .log()
            .cum_sum()
            .over("spell_id")
            .exp()
            .alias("surv")
        )
        surv = {}
        for horizon in ev.HORIZONS:
            s = g.filter(pl.col("k") == horizon).select("spell_id", "surv")
            s = joined.select("spell_id").join(s, on="spell_id", how="left")
            surv[horizon] = s["surv"].fill_null(0.0).to_numpy()
        surv_by_model[f"hazard {kind}"] = surv

    res = pl.DataFrame(results)
    show = [
        "model",
        "point",
        "mae_observed",
        "mae_observed_long",
        "medae_observed",
        "rmse_observed",
        "bias_observed",
        "spearman",
        "c_index",
    ]
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(200)
    rounded = res.select(show).with_columns(
        pl.col(c).round(3) for c in show if c not in ("model", "point")
    )
    print("--- point estimate: MEAN (minimises squared error) ---")
    print(rounded.filter(pl.col("point") == "mean").drop("point"))
    print()
    print("--- point estimate: MEDIAN (minimises absolute error) ---")
    print(rounded.filter(pl.col("point") == "median").drop("point"))
    print()
    print("Ranking (c_index) uses the censored spells too and does not depend")
    print("on the point estimate, so it is the fairest single column here.")
    res.write_csv(rep / "duration_metrics.csv")

    print("\nHORIZON CALIBRATION: P(still out after k games)")
    brier_rows = []
    for name, surv in surv_by_model.items():
        joined = sp_te.select("spell_id", "games_missed", "event")
        brier_rows.extend(ev.horizon_brier(joined, surv, name))
    br = pl.DataFrame(brier_rows)
    print(
        br.with_columns(
            pl.col(c).round(4) for c in ("base_rate", "brier", "brier_skill", "log_loss")
        )
    )
    br.write_csv(rep / "horizon_brier.csv")

    (rep / "chosen_hyperparameters.json").write_text(
        json.dumps({f"{k[0]}|{k[1]}": v for k, v in chosen.items()}, indent=2)
    )
    print(f"\nwrote metrics to {rep}")


if __name__ == "__main__":
    main()

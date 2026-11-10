"""Which variables drive injury duration, and how much does each block add.

Three views, because any one of them on its own is easy to over-read:

* **Block ablations.** Drop a whole family of features and refit. This is the
  only one that answers "do we need this data at all", and it is the answer
  to trust when a feature is correlated with another block.
* **Permutation importance** on the held-out season, scored by the drop in
  log loss of the per-game return prediction. Measures what the fitted model
  actually leans on, out of sample.
* **Partial dependence** of the per-game return hazard on the handful of
  variables that matter, so the direction and shape are visible rather than
  just the magnitude.

Run: uv run python scripts/04_explain.py
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from injury_duration import evaluate as ev
from injury_duration import experiment as ex
from injury_duration import features, models, paths
from injury_duration import hazard as hz

warnings.filterwarnings("ignore")

TRAIN = ["2021-22", "2022-23", "2023-24"]
VALID = ["2024-25"]
TEST = ["2025-26"]
RNG = np.random.default_rng(0)
N_REPEATS = 5

BEST = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}


def fit_lgbm(cols: list[str], train: pl.DataFrame) -> models.HazardModel:
    num, cat = ex.split_numeric_categorical(cols)
    return models.HazardModel("lightgbm", "lightgbm", num, cat, BEST).fit(train)


def main() -> None:
    rep = paths.ensure_reports()
    d = ex.load()
    design = ex.hazard_design(d["rows"], d["features"])
    trva = design.filter(pl.col("season").is_in(TRAIN + VALID))
    te = design.filter(pl.col("season").is_in(TEST))
    y_te = te["returns_next"].to_numpy()

    sp_te = d["features"].filter(pl.col("season").is_in(TEST))
    grid_te = ex.grid_design(sp_te, d["team_context"], max_k=models.MAX_K)

    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(190)

    # ------------------------------------------------------------ ablations
    print("=" * 78)
    print("BLOCK ABLATIONS (LightGBM hazard, test season 2025-26)")
    print("=" * 78)
    print("Each row drops one feature block and refits. Negative delta = the")
    print("block was carrying signal.\n")

    schedule_block = {"schedule_time": hz.SCHEDULE_TIME_COLS}
    all_blocks = {**features.BLOCKS, **schedule_block}
    full_cols = ex.index_feature_cols()

    rows = []
    base_model = fit_lgbm(full_cols, trva)
    base = ev.hazard_row_metrics(y_te, base_model.hazard(te), "full")
    base_dur = ev.duration_metrics(
        sp_te.select("spell_id", "games_missed", "event").join(
            models.predict_from_grid(base_model, grid_te), on="spell_id", how="left"
        ),
        sp_te.select("spell_id")
        .join(models.predict_from_grid(base_model, grid_te), on="spell_id", how="left")[
            "pred_median_games"
        ]
        .to_numpy(),
        "full",
    )
    rows.append(
        {
            "variant": "full model",
            "n_features": len(full_cols),
            "log_loss": round(base["log_loss"], 4),
            "d_log_loss": 0.0,
            "auc": round(base["auc"], 4),
            "c_index": round(base_dur["c_index"], 4),
            "mae_median_pred": round(base_dur["mae_observed"], 3),
        }
    )

    for name, block in all_blocks.items():
        cols = [c for c in full_cols if c not in block]
        m = fit_lgbm(cols, trva)
        met = ev.hazard_row_metrics(y_te, m.hazard(te), f"-{name}")
        pred = models.predict_from_grid(m, grid_te)
        joined = sp_te.select("spell_id", "games_missed", "event").join(
            pred, on="spell_id", how="left"
        )
        dur = ev.duration_metrics(joined, joined["pred_median_games"].to_numpy(), f"-{name}")
        rows.append(
            {
                "variant": f"drop {name}",
                "n_features": len(cols),
                "log_loss": round(met["log_loss"], 4),
                "d_log_loss": round(met["log_loss"] - base["log_loss"], 4),
                "auc": round(met["auc"], 4),
                "c_index": round(dur["c_index"], 4),
                "mae_median_pred": round(dur["mae_observed"], 3),
            }
        )

    # The other direction: one block at a time, on its own.
    for name, block in all_blocks.items():
        cols = [c for c in full_cols if c in block]
        if not cols:
            continue
        m = fit_lgbm(cols, trva)
        met = ev.hazard_row_metrics(y_te, m.hazard(te), name)
        pred = models.predict_from_grid(m, grid_te)
        joined = sp_te.select("spell_id", "games_missed", "event").join(
            pred, on="spell_id", how="left"
        )
        dur = ev.duration_metrics(joined, joined["pred_median_games"].to_numpy(), name)
        rows.append(
            {
                "variant": f"only {name}",
                "n_features": len(cols),
                "log_loss": round(met["log_loss"], 4),
                "d_log_loss": round(met["log_loss"] - base["log_loss"], 4),
                "auc": round(met["auc"], 4),
                "c_index": round(dur["c_index"], 4),
                "mae_median_pred": round(dur["mae_observed"], 3),
            }
        )

    abl = pl.DataFrame(rows)
    print(abl.sort("d_log_loss", descending=True))
    abl.write_csv(rep / "ablations.csv")

    # ------------------------------------------- permutation importance
    print()
    print("=" * 78)
    print("PERMUTATION IMPORTANCE (drop in log loss, test season, 5 repeats)")
    print("=" * 78)
    dyn_cols = ex.dynamic_feature_cols()
    dyn = fit_lgbm(dyn_cols, trva)
    base_ll = ev.hazard_row_metrics(y_te, dyn.hazard(te))["log_loss"]

    imp = []
    for col in dyn_cols:
        deltas = []
        for _ in range(N_REPEATS):
            shuffled = te.with_columns(pl.col(col).shuffle(seed=int(RNG.integers(1 << 30))))
            ll = ev.hazard_row_metrics(y_te, dyn.hazard(shuffled))["log_loss"]
            deltas.append(ll - base_ll)
        imp.append(
            {
                "feature": col,
                "d_log_loss": round(float(np.mean(deltas)), 5),
                "sd": round(float(np.std(deltas)), 5),
            }
        )
    impdf = pl.DataFrame(imp).sort("d_log_loss", descending=True)
    print(impdf.head(25))
    impdf.write_csv(rep / "permutation_importance.csv")
    print()
    print("features with no measurable contribution:")
    print(impdf.filter(pl.col("d_log_loss") <= 0.0001)["feature"].to_list())

    # ------------------------------------------------- partial dependence
    print()
    print("=" * 78)
    print("PARTIAL DEPENDENCE of the per-game return hazard")
    print("=" * 78)
    pdp_rows = []
    grids = {
        "ailment_class": te["ailment_class"].drop_nulls().unique().sort().to_list(),
        "body_region": te["body_region"].drop_nulls().unique().sort().to_list(),
        "age": [21, 24, 27, 30, 33, 36],
        "days_to_next_game": [1, 2, 3, 4, 6],
        "prior_spells_365d": [0, 1, 2, 4, 8],
        "min_roll5": [10, 20, 25, 30, 35],
        "team_win_pct_before": [0.2, 0.35, 0.5, 0.65, 0.8],
        "team_regular_remaining": [1, 5, 15, 40, 70],
    }
    sample = te.sample(n=min(3000, te.height), seed=0)

    # Elapsed time has to move as a group. games_missed_so_far,
    # days_missed_so_far and its log are three views of the same quantity, so
    # setting one to 50 while the others stay at 2 asks the model about a
    # combination that cannot happen and flattens the curve to nothing.
    for k in (1, 2, 3, 5, 8, 12, 20, 30, 50):
        patched = sample.with_columns(
            pl.lit(k).cast(sample.schema["games_missed_so_far"]).alias("games_missed_so_far"),
            pl.lit(round(k * hz.MEAN_DAYS_BETWEEN_GAMES))
            .cast(sample.schema["days_missed_so_far"])
            .alias("days_missed_so_far"),
            pl.lit(float(np.log(k + 1))).alias("log_games_missed_so_far"),
        )
        pdp_rows.append(
            {
                "feature": "elapsed (games, days, log together)",
                "value": str(k),
                "mean_hazard": round(float(dyn.hazard(patched).mean()), 4),
            }
        )

    for col, values in grids.items():
        if col not in te.columns:
            continue
        for v in values:
            patched = sample.with_columns(pl.lit(v).cast(sample.schema[col]).alias(col))
            pdp_rows.append(
                {
                    "feature": col,
                    "value": str(v),
                    "mean_hazard": round(float(dyn.hazard(patched).mean()), 4),
                }
            )
    pdp = pl.DataFrame(pdp_rows)
    for col in pdp["feature"].unique(maintain_order=True):
        sub = pdp.filter(pl.col("feature") == col)
        print(f"\n{col}:")
        for r in sub.iter_rows(named=True):
            bar = "#" * int(r["mean_hazard"] * 300)
            print(f"  {r['value']:>14}  {r['mean_hazard']:.4f}  {bar}")
    pdp.write_csv(rep / "partial_dependence.csv")

    # ---------------------------------------------------- calibration
    print()
    print("=" * 78)
    print("CALIBRATION of the per-game return probability (test season)")
    print("=" * 78)
    cal = ev.calibration_table(y_te, dyn.hazard(te), bins=10)
    print(cal.with_columns(pl.col("pred_mean").round(4), pl.col("actual_mean").round(4)))
    cal.write_csv(rep / "calibration.csv")

    # ------------------------------------------------------- parsimony
    print()
    print("=" * 78)
    print("HOW FEW FEATURES ARE ENOUGH")
    print("=" * 78)
    print("Features added in permutation-importance order, index-only set.\n")
    ranked = [f for f in impdf["feature"].to_list() if f in full_cols]
    par = []
    for n in (1, 2, 3, 5, 8, 12, 20, 30, len(full_cols)):
        cols = ranked[:n] if n < len(full_cols) else full_cols
        m = fit_lgbm(cols, trva)
        met = ev.hazard_row_metrics(y_te, m.hazard(te))
        pred = models.predict_from_grid(m, grid_te)
        joined = sp_te.select("spell_id", "games_missed", "event").join(
            pred, on="spell_id", how="left"
        )
        dur = ev.duration_metrics(joined, joined["pred_median_games"].to_numpy())
        par.append(
            {
                "n_features": len(cols),
                "log_loss": round(met["log_loss"], 4),
                "auc": round(met["auc"], 4),
                "c_index": round(dur["c_index"], 4),
                "mae_median_pred": round(dur["mae_observed"], 3),
                "newest_feature": cols[-1] if n < len(full_cols) else "(all)",
            }
        )
    pardf = pl.DataFrame(par)
    print(pardf)
    pardf.write_csv(rep / "parsimony.csv")

    # ------------------------------------------------------- robustness
    print()
    print("=" * 78)
    print("ROBUSTNESS: FRESH INJURIES vs ALREADY-RUNNING ABSENCES")
    print("=" * 78)
    print("days_since_last_played is the second most important feature, and")
    print("8% of spells have a gap of more than ten days -- those are absences")
    print("already under way when our spell window opens (an off-season injury")
    print("carried into game 1, or a spell that restarted after a G League or")
    print("trade break). They are much longer, so a headline number computed")
    print("over all spells partly measures the model's ability to spot them.")
    print("Scoring the fresh ones separately says how much is left.\n")

    best = fit_lgbm(full_cols, trva)
    pred_all = models.predict_from_grid(best, grid_te)
    joined_all = sp_te.select(
        "spell_id",
        "games_missed",
        "event",
        "days_since_last_played",
        "starts_at_season_open",
        "min_roll5",
    ).join(pred_all, on="spell_id", how="left")

    strata = [
        ("all test spells", pl.lit(True)),
        ("fresh (last played <= 4d ago)", pl.col("days_since_last_played") <= 4),
        ("gap 5-10 days", pl.col("days_since_last_played").is_between(5, 10)),
        ("gap > 10 days", pl.col("days_since_last_played") > 10),
        ("not at season open", ~pl.col("starts_at_season_open")),
        # Rotation players only. The model leans on recent minutes, which
        # separates injury severity from rotation status badly: a two-way
        # rookie playing six minutes a night looks, to the features, a lot
        # like someone with a serious problem. The question is mostly asked
        # about players who were actually playing.
        ("rotation players (>=15 min/game over last 5)", pl.col("min_roll5") >= 15),
        (
            "fresh AND rotation",
            (pl.col("days_since_last_played") <= 4) & (pl.col("min_roll5") >= 15),
        ),
    ]
    rob = []
    for label, cond in strata:
        sub = joined_all.filter(cond)
        if sub.height < 30:
            continue
        for col, point in (("pred_mean_games", "mean"), ("pred_median_games", "median")):
            met = ev.duration_metrics(sub, sub[col].to_numpy())
            rob.append(
                {
                    "stratum": label,
                    "point": point,
                    "n": sub.height,
                    "observed_frac": round(float(sub["event"].mean()), 3),
                    "km_mean_actual": round(
                        hz.km_mean(sub["games_missed"].to_numpy(), sub["event"].to_numpy()), 1
                    ),
                    "mae_observed": round(met["mae_observed"], 3),
                    "c_index": round(met["c_index"], 4),
                }
            )
    robdf = pl.DataFrame(rob)
    print(robdf)
    robdf.write_csv(rep / "robustness_by_gap.csv")
    print(f"\nwrote tables to {rep}")


if __name__ == "__main__":
    main()

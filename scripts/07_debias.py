"""Separate how long an injury keeps a player out from how long his team keeps him out.

The mechanism is counterfactual prediction, not classification. There is no
label for "tanking injury" and no way to validate one: the injury is almost
always real, and what the team's situation changes is the *return date*. So
instead of guessing a class, the hazard is fitted with the incentive features
in it and then evaluated twice for every spell:

* at the team's **actual** playoff position on the day, and
* at a **neutral** position — a team still in contention, neither eliminated
  nor clinched.

The first is a forecast. The second is what the injury would have cost at
ordinary urgency, which is the number a medical table should report. Their
difference is a per-spell *discretion score*: how many games of the absence
are attributable to the team's situation rather than to the injury.

This rests on one exclusion restriction — a team's playoff position does not
change how tissue heals — which is what scripts/06_incentive.py tests.

Run: uv run python scripts/07_debias.py
"""

from __future__ import annotations

import warnings

import polars as pl

from injury_duration import evaluate as ev
from injury_duration import experiment as ex
from injury_duration import models, paths

warnings.filterwarnings("ignore")

TRAIN = ["2021-22", "2022-23", "2023-24"]
VALID = ["2024-25"]
TEST = ["2025-26"]
FIT = TRAIN + VALID

BEST = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}

# A team in contention: hope near the league base rate, neither eliminated nor
# clinched, sitting on the play-in line.
NEUTRAL = {
    "hope_at_onset": 2 / 3,
    "stakes_at_onset": 4 * (2 / 3) * (1 / 3),
    "eliminated_at_onset": False,
    "clinched_at_onset": False,
    "games_since_elim_at_onset": None,
    "wins_vs_playin_at_onset": 0.0,
    "hope_now": 2 / 3,
    "eliminated_now": False,
}
INCENTIVE_COLS = tuple(NEUTRAL)


def neutralise(df: pl.DataFrame) -> pl.DataFrame:
    """Overwrite the incentive columns with the neutral reference."""
    return df.with_columns(
        [pl.lit(v).cast(df.schema[c]).alias(c) for c, v in NEUTRAL.items() if c in df.columns]
    )


def fit(cols: list[str], design: pl.DataFrame) -> models.HazardModel:
    num, cat = ex.split_numeric_categorical(cols)
    return models.HazardModel("lightgbm", "lightgbm", num, cat, BEST).fit(design)


def main() -> None:
    rep = paths.ensure_reports()
    d = ex.load()
    design = ex.hazard_design(d["rows"], d["features"])
    feats = d["features"]

    cols = ex.index_feature_cols()
    without = [c for c in cols if c not in INCENTIVE_COLS]

    fit_rows = design.filter(pl.col("season").is_in(FIT))
    te_rows = design.filter(pl.col("season").is_in(TEST))
    sp_te = feats.filter(pl.col("season").is_in(TEST))

    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_width_chars(195)

    # ---------------------------------------------------------------- 1
    print("=" * 78)
    print("1. DOES KNOWING THE TEAM'S SITUATION HELP PREDICT? (test 2025-26)")
    print("=" * 78)
    m_with = fit(cols, fit_rows)
    m_without = fit(without, fit_rows)

    grid = ex.grid_design(sp_te, d["team_context"], max_k=models.MAX_K)
    y = te_rows["returns_next"].to_numpy()

    late = sp_te.filter(pl.col("season_progress") > 0.80)["spell_id"]
    rows_late = te_rows.filter(pl.col("spell_id").is_in(late))

    comp = []
    for name, mdl, cs in [
        ("without incentive", m_without, without),
        ("with incentive", m_with, cols),
    ]:
        pred = models.predict_from_grid(mdl, grid)
        j = sp_te.select("spell_id", "games_missed", "event").join(pred, on="spell_id", how="left")
        met = ev.hazard_row_metrics(y, mdl.hazard(te_rows))
        dur = ev.duration_metrics(j, j["pred_median_games"].to_numpy())
        late_met = ev.hazard_row_metrics(
            rows_late["returns_next"].to_numpy(), mdl.hazard(rows_late)
        )
        comp.append(
            {
                "model": name,
                "n_features": len(cs),
                "log_loss": round(met["log_loss"], 4),
                "auc": round(met["auc"], 4),
                "log_loss_late": round(late_met["log_loss"], 4),
                "auc_late": round(late_met["auc"], 4),
                "n_rows_late": late_met["n_rows"],
                "c_index": round(dur["c_index"], 4),
                "mae_days": None,
            }
        )
    cf = pl.DataFrame(comp).drop("mae_days")
    print(cf)
    cf.write_csv(rep / "incentive_model_comparison.csv")
    print("\n  'late' = spells starting past 80% of the season, where the")
    print("  incentive gradient lives. A gain concentrated there is the")
    print("  expected shape; a flat gain would mean the feature is proxying")
    print("  for something else.")

    # ---------------------------------------------------------------- 2
    print()
    print("=" * 78)
    print("2. ACTUAL vs NEUTRAL-URGENCY PREDICTIONS")
    print("=" * 78)
    all_grid = ex.grid_design(feats, d["team_context"], max_k=models.MAX_K)
    m_full = fit(cols, design)  # every season, for describing the whole sample

    pred_actual = models.predict_from_grid(m_full, all_grid)
    pred_neutral = models.predict_from_grid(m_full, neutralise(all_grid))

    both = (
        feats.select(
            "spell_id",
            "player_name",
            "season",
            "start_date",
            "team_slug_start",
            "body_region",
            "ailment_class",
            "index_status",
            "hope_at_onset",
            "eliminated_at_onset",
            "season_progress",
            "min_roll5",
            "games_missed",
            "days_out",
            "event",
        )
        .join(
            pred_actual.select(
                "spell_id",
                pl.col("pred_mean_games").alias("games_actual"),
                pl.col("pred_median_games").alias("med_games_actual"),
                pl.col("pred_mean_days").alias("days_actual"),
            ),
            on="spell_id",
            how="left",
        )
        .join(
            pred_neutral.select(
                "spell_id",
                pl.col("pred_mean_games").alias("games_neutral"),
                pl.col("pred_median_games").alias("med_games_neutral"),
                pl.col("pred_mean_days").alias("days_neutral"),
            ),
            on="spell_id",
            how="left",
        )
        .with_columns(
            (pl.col("games_actual") - pl.col("games_neutral")).alias("discretion_games"),
            (pl.col("days_actual") - pl.col("days_neutral")).alias("discretion_days"),
        )
    )

    print("Mean predicted games missed, actual team situation vs neutral:")
    summ = (
        both.group_by(
            pl.when(pl.col("eliminated_at_onset"))
            .then(pl.lit("1 eliminated"))
            .when(pl.col("hope_at_onset") < 0.25)
            .then(pl.lit("2 hope < .25"))
            .when(pl.col("hope_at_onset") > 0.75)
            .then(pl.lit("4 hope > .75"))
            .otherwise(pl.lit("3 hope .25-.75"))
            .alias("situation_at_onset")
        )
        .agg(
            pl.len().alias("spells"),
            pl.col("games_actual").mean().round(2).alias("pred_games_actual"),
            pl.col("games_neutral").mean().round(2).alias("pred_games_neutral"),
            pl.col("discretion_games").mean().round(2).alias("discretion_games"),
            pl.col("discretion_days").mean().round(2).alias("discretion_days"),
        )
        .sort("situation_at_onset")
    )
    print(summ)
    summ.write_csv(rep / "discretion_by_situation.csv")

    # ---------------------------------------------------------------- 3
    print()
    print("=" * 78)
    print("3. DE-BIASED DURATION TABLES")
    print("=" * 78)
    print("What a published 'this injury costs N games' table says, against")
    print("what it says once team urgency is held at the league norm.\n")
    tab = (
        both.filter(pl.col("ailment_class").is_not_null())
        .group_by("ailment_class")
        .agg(
            pl.len().alias("n"),
            pl.col("games_actual").mean().round(2).alias("games_as_observed"),
            pl.col("games_neutral").mean().round(2).alias("games_neutral"),
            pl.col("discretion_games").mean().round(2).alias("delta"),
            pl.col("days_neutral").mean().round(1).alias("days_neutral"),
        )
        .filter(pl.col("n") >= 40)
        .sort("games_neutral", descending=True)
    )
    print(tab)
    tab.write_csv(rep / "km_debiased_by_ailment.csv")

    reg = (
        both.group_by("body_region")
        .agg(
            pl.len().alias("n"),
            pl.col("games_actual").mean().round(2).alias("games_as_observed"),
            pl.col("games_neutral").mean().round(2).alias("games_neutral"),
            pl.col("discretion_games").mean().round(2).alias("delta"),
        )
        .filter(pl.col("n") >= 40)
        .sort("games_neutral", descending=True)
    )
    print()
    print(reg)
    reg.write_csv(rep / "km_debiased_by_region.csv")

    # ---------------------------------------------------------------- 4
    print()
    print("=" * 78)
    print("4. PER-SPELL DISCRETION SCORES")
    print("=" * 78)
    out = both.select(
        "player_name",
        "season",
        "start_date",
        "team_slug_start",
        pl.col("body_region").alias("region"),
        pl.col("ailment_class").alias("ailment"),
        pl.col("hope_at_onset").round(2).alias("hope"),
        pl.col("games_actual").round(1).alias("pred_games"),
        pl.col("games_neutral").round(1).alias("pred_games_neutral"),
        pl.col("discretion_games").round(1).alias("discretion_games"),
        pl.col("discretion_days").round(1).alias("discretion_days"),
        "games_missed",
        "days_out",
        "event",
    ).sort("discretion_games", descending=True)
    # The full per-spell table is regenerable model output; reports/ keeps the
    # readable extremes, which is what anyone browsing it wants.
    out.write_csv(paths.ensure_build() / "discretion_scores_full.csv")
    pl.concat([out.head(200), out.tail(100)]).write_csv(rep / "discretion_scores.csv")

    print("Most team-attributable absences in the sample:")
    print(out.head(12))
    print("\nMost team-accelerated (came back sooner than the injury implies):")
    print(out.sort("discretion_games").head(8))

    print()
    share = both.filter(pl.col("games_actual") > 0)
    print(
        f"Across all {both.height:,} spells: mean predicted "
        f"{both['games_actual'].mean():.2f} games as observed against "
        f"{both['games_neutral'].mean():.2f} at neutral urgency "
        f"({both['discretion_games'].mean():+.2f})."
    )
    el = both.filter(pl.col("eliminated_at_onset"))
    print(
        f"Among the {el.height} spells beginning after the team was already "
        f"mathematically eliminated: {el['games_actual'].mean():.2f} against "
        f"{el['games_neutral'].mean():.2f} ({el['discretion_games'].mean():+.2f} games, "
        f"{el['discretion_days'].mean():+.1f} days)."
    )
    del share
    print(f"\nwrote tables to {rep}")


if __name__ == "__main__":
    main()

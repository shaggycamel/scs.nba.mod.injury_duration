"""Describe how long injuries actually keep players out.

Kaplan-Meier rather than plain averages throughout, because the long spells
are the censored ones and a plain average of observed durations understates
every category that contains a season-ender.

Run: uv run python scripts/02_describe.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from injury_duration import hazard as hz
from injury_duration import paths

MIN_N = 25


def km_summary(df: pl.DataFrame, by: list[str], min_n: int = MIN_N) -> pl.DataFrame:
    out = []
    for key, grp in df.group_by(by):
        if grp.height < min_n:
            continue
        d = grp["games_missed"].to_numpy()
        e = grp["event"].to_numpy()
        row = dict(zip(by, key))
        row.update(
            n=grp.height,
            censored=int((e == 0).sum()),
            km_mean=round(hz.km_mean(d, e), 2),
            km_median=hz.km_median(d, e),
            naive_mean_observed=round(float(d[e == 1].mean()) if (e == 1).any() else np.nan, 2),
            p_out_past_5=round(float(_surv_at(d, e, 5)), 3),
            p_out_past_20=round(float(_surv_at(d, e, 20)), 3),
        )
        out.append(row)
    return pl.DataFrame(out).sort("km_mean", descending=True)


def _surv_at(d: np.ndarray, e: np.ndarray, k: int) -> float:
    times, surv = hz.kaplan_meier(d, e)
    idx = np.searchsorted(times, k, side="right") - 1
    return surv[idx] if idx >= 0 else 1.0


def main() -> None:
    sp = pl.read_parquet(paths.BUILD / "spells.parquet")
    feats = pl.read_parquet(paths.BUILD / "spell_features.parquet")
    rep = paths.ensure_reports()

    inj = sp.filter(pl.col("index_category") == "injury")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(160)

    d, e = sp["games_missed"].to_numpy(), sp["event"].to_numpy()
    print("=" * 72)
    print("OVERALL")
    print("=" * 72)
    print(f"spells                {sp.height:>8,}")
    print(f"observed returns      {int(e.sum()):>8,} ({e.mean():.1%})")
    print(f"KM mean games missed  {hz.km_mean(d, e):>8.2f}")
    print(f"KM median             {hz.km_median(d, e):>8.0f}")
    print(
        f"mean of observed only {d[e == 1].mean():>8.2f}   "
        "<- what a censoring-blind average would say"
    )
    print(f"share of all games missed in spells of 10+: {d[d >= 10].sum() / d.sum():.1%}")
    print()
    print("survival curve, all spells (P(still out after k games)):")
    for k in (1, 2, 3, 5, 10, 20, 30, 50):
        print(f"  after {k:>3} games  {_surv_at(d, e, k):.3f}")

    print()
    print("=" * 72)
    print("BY AILMENT CLASS (injury spells only)")
    print("=" * 72)
    by_ail = km_summary(inj, ["ailment_class"])
    print(by_ail)
    by_ail.write_csv(rep / "km_by_ailment.csv")

    print()
    print("=" * 72)
    print("BY BODY REGION (injury spells only)")
    print("=" * 72)
    by_reg = km_summary(inj, ["body_region"])
    print(by_reg)
    by_reg.write_csv(rep / "km_by_region.csv")

    print()
    print("=" * 72)
    print("REGION x AILMENT, worst 20 by expected games missed")
    print("=" * 72)
    cross = km_summary(inj, ["body_region", "ailment_class"], min_n=30)
    print(cross.head(20))
    cross.write_csv(rep / "km_by_region_ailment.csv")

    print()
    print("=" * 72)
    print("MODIFIERS")
    print("=" * 72)
    # is_catastrophic is the conjunction of structure and pathology, so it is
    # derived in `features` rather than carried on the raw spell.
    inj_f = feats.filter(pl.col("index_category") == "injury")
    mods = []
    for col in [
        "is_surgical",
        "is_recovery_stage",
        "is_management",
        "is_bone_stress",
        "mentions_major_structure",
        "is_catastrophic",
    ]:
        src = inj_f if col == "is_catastrophic" else inj
        for val in (True, False):
            g = src.filter(pl.col(col) == val)
            if g.height < MIN_N:
                continue
            dd, ee = g["games_missed"].to_numpy(), g["event"].to_numpy()
            mods.append(
                {
                    "modifier": col,
                    "value": val,
                    "n": g.height,
                    "km_mean": round(hz.km_mean(dd, ee), 2),
                    "km_median": hz.km_median(dd, ee),
                }
            )
    mod_df = pl.DataFrame(mods)
    print(mod_df)
    mod_df.write_csv(rep / "km_by_modifier.csv")

    print()
    print("=" * 72)
    print("STATUS ON THE FIRST MISSED GAME")
    print("=" * 72)
    print(km_summary(sp, ["index_status"], min_n=20))

    print()
    print("=" * 72)
    print("RECURRENCE AND AGE")
    print("=" * 72)
    f = feats.filter(pl.col("index_category") == "injury")
    print(km_summary(f, ["is_quick_recurrence"], min_n=20))
    age_bins = f.with_columns(
        pl.when(pl.col("age") < 24)
        .then(pl.lit("1 under 24"))
        .when(pl.col("age") < 28)
        .then(pl.lit("2 24-27"))
        .when(pl.col("age") < 32)
        .then(pl.lit("3 28-31"))
        .otherwise(pl.lit("4 32+"))
        .alias("age_band")
    )
    print(km_summary(age_bins, ["age_band"], min_n=20).sort("age_band"))

    print()
    print("=" * 72)
    print("PER-GAME RETURN HAZARD BY GAMES ALREADY MISSED")
    print("=" * 72)
    rows = pl.read_parquet(paths.BUILD / "hazard_rows.parquet")
    hz_tab = (
        rows.with_columns(
            pl.when(pl.col("games_missed_so_far") <= 5)
            .then(pl.col("games_missed_so_far").cast(pl.String))
            .when(pl.col("games_missed_so_far") <= 10)
            .then(pl.lit("6-10"))
            .when(pl.col("games_missed_so_far") <= 20)
            .then(pl.lit("11-20"))
            .otherwise(pl.lit("21+"))
            .alias("missed_so_far")
        )
        .group_by("missed_so_far")
        .agg(pl.len().alias("rows"), pl.col("returns_next").mean().round(3).alias("p_return_next"))
        .sort("rows", descending=True)
    )
    print(hz_tab)
    print()
    print("per-game return hazard by report status on that game:")
    print(
        rows.group_by("status_clean")
        .agg(pl.len().alias("rows"), pl.col("returns_next").mean().round(3).alias("p_return_next"))
        .sort("rows", descending=True)
    )
    print(f"\nwrote tables to {rep}")


if __name__ == "__main__":
    main()

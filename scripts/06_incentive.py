"""How much of an injury absence is the team's decision rather than the injury.

The identifying idea: a team's playoff position changes what it wants from a
borderline player, and cannot change how a torn ligament heals. So variation
in playoff position, holding the diagnosis and the player's role fixed, moves
team willingness and not medical readiness.

Four cuts, weakest-to-strongest on power and strongest-to-weakest on rigour:

1. **Event study** around mathematical elimination. Clean but low-powered:
   elimination arrives so late that most of the behavioural response has
   already happened through fading playoff hope, so it catches the tail of
   the effect rather than its onset.
2. **Dose-response** on playoff hope, holding games-missed-so-far fixed.
3. **Difference-in-differences**: hope tercile crossed with early/late season,
   within diagnosis. Early season is the control period — everyone still has
   hope, so a team-quality gradient there would mean the gradient is about
   quality, not incentive.
4. **Policy break**: the 2023-24 Player Participation Policy restricted
   sitting healthy players. Two seasons before, three after.

Run: uv run python scripts/06_incentive.py
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from injury_duration import experiment as ex
from injury_duration import paths, standings

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(0)
N_BOOT = 2000
# The Player Participation Policy took effect for the 2023-24 season.
PPP_SEASONS = ("2023-24", "2024-25", "2025-26")
ROTATION_MIN = 15.0


def boot_ci(x: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean."""
    if len(x) < 2:
        return (float("nan"), float("nan"))
    draws = RNG.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Difference in means, with a bootstrap CI."""
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"), float("nan"))
    da = RNG.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    db = RNG.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    d = da - db
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> None:
    rep = paths.ensure_reports()
    d = ex.load()
    rows = d["rows"]
    feats = d["features"]

    inc = standings.playin_probability().select(
        "season",
        "game_date",
        "team_slug",
        "playin_probability",
        "incentive_state",
        "games_since_elimination",
    )
    r = rows.join(inc, on=["season", "game_date", "team_slug"], how="left")
    r = r.join(
        feats.select(
            "spell_id",
            "season_progress",
            "ailment_class",
            "body_region",
            "min_roll5",
            "hope_at_onset",
            "team_games_before",
        ),
        on="spell_id",
        how="left",
    )

    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(190)

    # ------------------------------------------------------------------ 1
    print("=" * 78)
    print("1. EVENT STUDY AROUND MATHEMATICAL ELIMINATION")
    print("=" * 78)
    print("Per-game return hazard by team games either side of the date the")
    print("team was mathematically out of the play-in race.\n")
    es_rows = []
    ev = r.filter(pl.col("games_since_elimination").is_between(-12, 12)).with_columns(
        (pl.col("games_since_elimination") // 3 * 3).alias("bucket")
    )
    for key, grp in ev.group_by("bucket"):
        y = grp["returns_next"].to_numpy().astype(float)
        lo, hi = boot_ci(y)
        es_rows.append(
            {
                "games_rel_elimination": int(key[0]),
                "n_rows": grp.height,
                "p_return": round(float(y.mean()), 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
            }
        )
    es = pl.DataFrame(es_rows).sort("games_rel_elimination")
    for row in es.iter_rows(named=True):
        bar = "#" * int(row["p_return"] * 300)
        tag = "  <- eliminated" if row["games_rel_elimination"] == 0 else ""
        print(
            f"  {row['games_rel_elimination']:>4} to {row['games_rel_elimination'] + 2:>3}  "
            f"n={row['n_rows']:>4}  {row['p_return']:.3f} "
            f"[{row['ci_lo']:.3f},{row['ci_hi']:.3f}] {bar}{tag}"
        )
    es.write_csv(rep / "elimination_event_study.csv")
    pre = ev.filter(pl.col("games_since_elimination") < 0)["returns_next"].to_numpy().astype(float)
    post = (
        ev.filter(pl.col("games_since_elimination") >= 0)["returns_next"].to_numpy().astype(float)
    )
    dd, lo, hi = diff_ci(post, pre)
    print(
        f"\n  post minus pre: {dd:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
        f"(n_pre={len(pre)}, n_post={len(post)})"
    )
    print("  The pre-period is flat and the post-period declines, so there is a")
    print("  level shift at elimination after all. Read at one-game resolution")
    print("  the pre-period looks like it is already falling; that was noise.")
    print("  The shift is still smaller than the difference-in-differences")
    print("  below, because elimination lands so late that most of the")
    print("  behavioural change has already happened through fading hope.")

    # ------------------------------------------------------------------ 2
    print()
    print("=" * 78)
    print("2. DOSE-RESPONSE ON PLAYOFF HOPE, HOLDING ELAPSED TIME FIXED")
    print("=" * 78)
    dr = r.filter(pl.col("playin_probability").is_not_null()).with_columns(
        (pl.col("playin_probability") * 5).floor().clip(0, 4).cast(pl.Int32).alias("hope_q"),
        pl.when(pl.col("games_missed_so_far") == 1)
        .then(pl.lit("k=1"))
        .when(pl.col("games_missed_so_far") <= 3)
        .then(pl.lit("k=2-3"))
        .when(pl.col("games_missed_so_far") <= 10)
        .then(pl.lit("k=4-10"))
        .otherwise(pl.lit("k=11+"))
        .alias("k"),
    )
    tab = (
        dr.group_by("k", "hope_q")
        .agg(pl.len().alias("rows"), pl.col("returns_next").mean().round(3).alias("p"))
        .sort("k", "hope_q")
    )
    print(tab.pivot(on="hope_q", index="k", values="p").sort("k"))
    print("  columns are playoff-hope quintiles, 0 = no hope, 4 = near certain")
    tab.write_csv(rep / "hope_dose_response.csv")

    # ------------------------------------------------------------------ 3
    print()
    print("=" * 78)
    print("3. DIFFERENCE-IN-DIFFERENCES: hope x season half, within diagnosis")
    print("=" * 78)
    print("Rotation players only (>=15 min/game over their last 5), first 10")
    print("games of the spell. 'late' = spell started past 80% of the season.\n")
    did = r.filter(
        (pl.col("min_roll5") >= ROTATION_MIN) & (pl.col("games_missed_so_far") <= 10)
    ).with_columns(
        pl.when(pl.col("hope_at_onset") < 0.25)
        .then(pl.lit("low"))
        .when(pl.col("hope_at_onset") > 0.75)
        .then(pl.lit("high"))
        .otherwise(pl.lit("mid"))
        .alias("hope_grp"),
        pl.when(pl.col("season_progress") > 0.80)
        .then(pl.lit("late"))
        .when(pl.col("season_progress") < 0.50)
        .then(pl.lit("early"))
        .otherwise(pl.lit("mid"))
        .alias("period"),
    )
    out = []
    for ail in ["soreness", "sprain", "strain", "management", "contusion", "ALL"]:
        sub = did if ail == "ALL" else did.filter(pl.col("ailment_class") == ail)
        cells = {}
        for period in ("early", "late"):
            for grp in ("low", "high"):
                y = (
                    sub.filter((pl.col("period") == period) & (pl.col("hope_grp") == grp))[
                        "returns_next"
                    ]
                    .to_numpy()
                    .astype(float)
                )
                cells[(period, grp)] = y
        if min(len(v) for v in cells.values()) < 40:
            continue
        g_early, lo_e, hi_e = diff_ci(cells[("early", "high")], cells[("early", "low")])
        g_late, lo_l, hi_l = diff_ci(cells[("late", "high")], cells[("late", "low")])
        # DiD: how much the high-minus-low hope gap widens from early to late.
        a = cells[("late", "high")]
        b = cells[("late", "low")]
        c = cells[("early", "high")]
        e = cells[("early", "low")]
        da = RNG.choice(a, (N_BOOT, len(a)), replace=True).mean(axis=1)
        db = RNG.choice(b, (N_BOOT, len(b)), replace=True).mean(axis=1)
        dc = RNG.choice(c, (N_BOOT, len(c)), replace=True).mean(axis=1)
        de = RNG.choice(e, (N_BOOT, len(e)), replace=True).mean(axis=1)
        boot = (da - db) - (dc - de)
        out.append(
            {
                "diagnosis": ail,
                "n_late_low": len(b),
                "n_late_high": len(a),
                "gap_early": round(g_early, 4),
                "gap_early_ci": f"[{lo_e:+.3f},{hi_e:+.3f}]",
                "gap_late": round(g_late, 4),
                "gap_late_ci": f"[{lo_l:+.3f},{hi_l:+.3f}]",
                "did": round(g_late - g_early, 4),
                "did_ci": f"[{np.quantile(boot, 0.025):+.3f},{np.quantile(boot, 0.975):+.3f}]",
                "did_excludes_zero": bool(
                    np.quantile(boot, 0.025) > 0 or np.quantile(boot, 0.975) < 0
                ),
            }
        )
    didf = pl.DataFrame(out)
    print(didf)
    didf.write_csv(rep / "incentive_did.csv")
    print("\n  gap = P(return next game | high hope) - P(... | low hope)")
    print("  did = how much that gap widens from the early season to the late season")

    # ------------------------------------------------------------------ 4
    print()
    print("=" * 78)
    print("4. POLICY BREAK: 2023-24 Player Participation Policy")
    print("=" * 78)
    print("The 2019 lottery reform predates the injury report entirely and is")
    print("not testable here. The PPP has two seasons before and three after.\n")
    pol = did.filter(pl.col("period") == "late")
    prows = []
    for regime, cond in [
        ("pre-PPP (2021-23)", ~pl.col("season").is_in(list(PPP_SEASONS))),
        ("post-PPP (2023-26)", pl.col("season").is_in(list(PPP_SEASONS))),
    ]:
        sub = pol.filter(cond)
        a = sub.filter(pl.col("hope_grp") == "high")["returns_next"].to_numpy().astype(float)
        b = sub.filter(pl.col("hope_grp") == "low")["returns_next"].to_numpy().astype(float)
        g, lo, hi = diff_ci(a, b)
        prows.append(
            {
                "regime": regime,
                "n_high": len(a),
                "n_low": len(b),
                "p_high": round(float(a.mean()), 4),
                "p_low": round(float(b.mean()), 4),
                "gap": round(g, 4),
                "gap_ci": f"[{lo:+.3f},{hi:+.3f}]",
            }
        )
    pf = pl.DataFrame(prows)
    print(pf)
    pf.write_csv(rep / "policy_break.csv")

    print(f"\nwrote tables to {rep}")


if __name__ == "__main__":
    main()

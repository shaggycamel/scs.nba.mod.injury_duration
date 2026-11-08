"""Rewrite the numeric tables in the Markdown report from the generated CSVs.

The report's prose is written by hand; its tables are not. Keeping them in
sync by editing both was the wrong call and it showed: a model refit shifted
every figure and the narrative quietly went stale, twice. This regenerates
each table in place from `reports/*.csv`, matching on the table's header row,
so the numbers in the write-up cannot drift from the numbers in the run.

Prose that quotes a specific figure is checked rather than rewritten: the
script prints anything that no longer matches its source so it can be fixed
deliberately.

Run: uv run python scripts/09_sync_report_numbers.py
"""

from __future__ import annotations

import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DOC = ROOT / "docs" / "injury_duration.md"


def load(name: str) -> list[dict]:
    with open(REPORTS / f"{name}.csv") as fh:
        return list(csv.DictReader(fh))


def n(v, dp: int = 4) -> str:
    return f"{float(v):.{dp}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join("---" for _ in headers) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def swap(doc: str, header_row: str, stop: str, new: str) -> str:
    """Replace the table starting at `header_row` with `new`."""
    i = doc.index(header_row)
    j = doc.index(stop, i)
    return doc[:i] + new + "\n" + doc[j:]


def swap_between(doc: str, after: str, before: str, new: str) -> str:
    """Replace whatever sits between two bits of prose.

    Used where the table's own header is not a usable anchor -- an empty first
    column renders as a double space and stops matching itself on the next run.
    """
    i = doc.index(after) + len(after)
    j = doc.index(before, i)
    return doc[:i] + "\n\n" + new + "\n" + doc[j:]


def main() -> None:
    doc = DOC.read_text()

    pg = {r["model"]: r for r in load("per_game_return_metrics")}
    doc = swap(
        doc,
        "| model | AUC | log loss | Brier skill |",
        "Boosted trees beat the forest",
        table(
            ["model", "AUC", "log loss", "Brier skill"],
            [
                [
                    "base rate",
                    n(pg["base rate (train mean)"]["auc"], 3),
                    n(pg["base rate (train mean)"]["log_loss"], 4),
                    "—",
                ],
                *[
                    [label, n(pg[k]["auc"], 3), n(pg[k]["log_loss"], 4), n(pg[k]["brier_skill"], 3)]
                    for label, k in [
                        ("logistic (L2)", "logistic / index_only"),
                        ("random forest", "forest / index_only"),
                        ("HistGradientBoosting", "histgb / index_only"),
                        ("LightGBM", "lightgbm / index_only"),
                    ]
                ],
                [
                    "LightGBM **+ live report state**",
                    f"**{n(pg['lightgbm / with_report_state']['auc'], 3)}**",
                    f"**{n(pg['lightgbm / with_report_state']['log_loss'], 4)}**",
                    f"**{n(pg['lightgbm / with_report_state']['brier_skill'], 3)}**",
                ],
            ],
        ),
    )

    dm = {r["model"]: r for r in load("duration_metrics") if r["point"] == "median"}
    rows = []
    for label, k in [
        ("KM global", "km global"),
        ("KM by region", "km by region"),
        ("KM by ailment", "km by ailment"),
        ("KM by region × ailment", "km by region x ailment"),
        ("KM by region × ailment × status", "km by region x ailment x status"),
        ("hazard, logistic", "hazard logistic"),
        ("LightGBM, observed spells only", "lgbm (observed spells only)"),
        ("hazard, random forest", "hazard forest"),
        ("hazard, LightGBM", "hazard lightgbm"),
        ("**hazard, HistGradientBoosting**", "hazard histgb"),
    ]:
        r = dm[k]
        mae = n(r["mae_observed"], 2)
        lng, ci = n(r["mae_observed_long"], 2), n(r["c_index"], 3)
        if k == "lgbm (observed spells only)":
            mae = f"**{mae}**"
        if k == "hazard histgb":
            lng, ci = f"**{lng}**", f"**{ci}**"
        rows.append([label, mae, lng, ci])
    doc = swap(
        doc,
        "| model | MAE | MAE (spells ≥5 games) | C-index |",
        "Calibration of the LightGBM hazard",
        table(["model", "MAE", "MAE (spells ≥5 games)", "C-index"], rows),
    )

    ab = {r["variant"]: r for r in load("ablations")}
    rows = []
    for label, k in [
        ("full model", "full model"),
        ("drop elapsed-time + schedule", "drop schedule_time"),
        ("drop injury taxonomy", "drop injury"),
        ("drop recent load", "drop load"),
        ("drop team incentive", "drop incentive"),
        ("drop season/team context", "drop context"),
        ("drop player attributes", "drop player"),
        ("drop injury history", "drop history"),
        ("*only* injury taxonomy", "only injury"),
        ("*only* recent load", "only load"),
        ("*only* elapsed + schedule", "only schedule_time"),
        ("*only* team incentive", "only incentive"),
        ("*only* player attributes", "only player"),
        ("*only* injury history", "only history"),
    ]:
        r = ab[k]
        d = float(r["d_log_loss"])
        ds = "—" if k == "full model" else (f"**{d:+.4f}**" if d >= 0.01 else f"{d:+.4f}")
        rows.append(
            [label, r["n_features"], n(r["log_loss"], 4), ds, n(r["auc"], 3), n(r["c_index"], 3)]
        )
    doc = swap(
        doc,
        "| variant | features | log loss | Δ | AUC | C-index |",
        "Note the split:",
        table(["variant", "features", "log loss", "Δ", "AUC", "C-index"], rows),
    )

    par = {r["n_features"]: r for r in load("parsimony")}
    keep = [k for k in ("1", "3", "8", "12", "20") if k in par] + [list(par)[-1]]
    rows = []
    for k in keep:
        r = par[k]
        b = (lambda x: f"**{x}**") if k == "12" else (lambda x: x)
        rows.append([b(k), b(n(r["log_loss"], 4)), b(n(r["auc"], 3)), b(n(r["c_index"], 3))])
    doc = swap(
        doc,
        "| features | log loss | AUC | C-index |",
        "Twelve features beat all",
        table(["features", "log loss", "AUC", "C-index"], rows),
    )

    rb = {r["stratum"]: r for r in load("robustness_by_gap") if r["point"] == "median"}
    rows = []
    for label, k in [
        ("all test spells", "all test spells"),
        ("fresh injuries (last played ≤4 days ago)", "fresh (last played <= 4d ago)"),
        ("rotation players (≥15 min/game)", "rotation players (>=15 min/game over last 5)"),
        ("**fresh AND rotation**", "fresh AND rotation"),
        ("already-running absences (gap >10 days)", "gap > 10 days"),
    ]:
        r = rb[k]
        b = (lambda x: f"**{x}**") if k == "fresh AND rotation" else (lambda x: x)
        rows.append(
            [
                label,
                f"{int(r['n']):,}",
                f"{float(r['observed_frac']) * 100:.0f}%",
                n(r["km_mean_actual"], 1),
                b(n(r["c_index"], 3)),
                b(n(r["mae_observed"], 2)),
            ]
        )
    doc = swap(
        doc,
        "| population | n | observed | actual KM mean | C-index | MAE (median pred) |",
        "The cleanest number",
        table(
            ["population", "n", "observed", "actual KM mean", "C-index", "MAE (median pred)"], rows
        ),
    )

    did = {r["diagnosis"]: r for r in load("incentive_did")}
    rows = []
    for k in ("soreness", "management", "contusion", "strain", "sprain"):
        r = did[k]
        d = f"{float(r['did']):+.3f}"
        rows.append(
            [
                {"management": "injury management"}.get(k, k),
                f"{float(r['gap_early']):+.3f}",
                f"{float(r['gap_late']):+.3f}",
                f"**{d}**" if k == "soreness" else d,
                r["did_ci"].replace(",", ", "),
            ]
        )
    doc = swap(
        doc,
        "| diagnosis | gap early | gap late | DiD | 95% CI |",
        "Worse medical staff, more fragile rosters",
        table(["diagnosis", "gap early", "gap late", "DiD", "95% CI"], rows),
    )

    pol = {r["regime"]: r for r in load("policy_break")}
    doc = swap(
        doc,
        "| regime | P(return \\| high hope) | P(return \\| low hope) | gap | 95% CI |",
        "Low-hope teams return players more often",
        table(
            ["regime", "P(return \\| high hope)", "P(return \\| low hope)", "gap", "95% CI"],
            [
                [
                    label,
                    n(pol[k]["p_high"], 3),
                    n(pol[k]["p_low"], 3),
                    n(pol[k]["gap"], 3),
                    pol[k]["gap_ci"].replace(",", ", "),
                ]
                for label, k in [
                    ("pre-policy, 2021-23", "pre-PPP (2021-23)"),
                    ("post-policy, 2023-26", "post-PPP (2023-26)"),
                ]
            ],
        ),
    )

    sit = {r["situation_at_onset"]: r for r in load("discretion_by_situation")}
    rows = []
    for label, k in [
        ("already eliminated", "1 eliminated"),
        ("hope < 0.25", "2 hope < .25"),
        ("hope 0.25-0.75", "3 hope .25-.75"),
        ("hope > 0.75", "4 hope > .75"),
    ]:
        r = sit[k]
        d = f"{float(r['discretion_games']):+.2f}"
        if k == "1 eliminated":
            d = f"**{d} games ({float(r['discretion_days']):+.1f} days)**"
        rows.append(
            [
                label,
                f"{int(r['spells']):,}",
                n(r["pred_games_actual"], 2),
                n(r["pred_games_neutral"], 2),
                d,
            ]
        )
    doc = swap(
        doc,
        "| situation when ruled out | spells | pred games, as observed |",
        "Across all 6,754 spells the de-biasing",
        table(
            [
                "situation when ruled out",
                "spells",
                "pred games, as observed",
                "at neutral urgency",
                "difference",
            ],
            rows,
        ),
    )

    deb = load("km_debiased_by_ailment")
    for r in deb:
        r["share"] = float(r["delta"]) / float(r["games_as_observed"])
    deb.sort(key=lambda r: -r["share"])
    rows = []
    for r in deb[:4] + deb[-4:]:
        sh = f"{r['share']:.1%}"
        if r is deb[0] or r is deb[-1]:
            sh = f"**{sh}**"
        rows.append(
            [
                r["ailment_class"].replace("_", " / "),
                n(r["games_as_observed"], 2),
                n(r["games_neutral"], 2),
                f"{float(r['delta']):+.2f}",
                sh,
            ]
        )
    doc = swap(
        doc,
        "| ailment | as observed | neutral | difference |",
        "Read as a share, the pattern",
        table(["ailment", "as observed", "neutral", "difference", "share of its own length"], rows),
    )

    comp = {r["model"]: r for r in load("incentive_model_comparison")}
    doc = swap_between(
        doc,
        "that is, by nothing:",
        "Barely more late in the season",
        table(
            ["model", "log loss", "AUC", "C-index", "log loss (late)", "AUC (late)"],
            [
                [
                    k,
                    n(comp[k]["log_loss"], 4),
                    n(comp[k]["auc"], 4),
                    n(comp[k]["c_index"], 4),
                    n(comp[k]["log_loss_late"], 4),
                    n(comp[k]["auc_late"], 4),
                ]
                for k in ("without incentive", "with incentive")
            ],
        ),
    )

    DOC.write_text(doc)
    print(f"synced 10 tables in {DOC.name}")

    # Prose figures are reported, not rewritten: a mismatch is a decision.
    hist, kmb = dm["hazard histgb"], dm["km by region x ailment x status"]
    fr, allsp = (rb["fresh AND rotation"], rb["all test spells"])
    ds = {
        r["player_name"]: r
        for r in load("discretion_scores")
        if r["team_slug_start"] == "DAL" and r["start_date"] == "2023-04-07"
    }
    checks = {
        "live next-game AUC": n(pg["lightgbm / with_report_state"]["auc"], 3),
        "best C-index": n(hist["c_index"], 3),
        "lookup-table C-index": n(kmb["c_index"], 3),
        "long-spell MAE, best": n(hist["mae_observed_long"], 2),
        "fresh+rotation C-index": n(fr["c_index"], 3),
        "all-spells C-index": n(allsp["c_index"], 3),
        "DiD, all diagnoses": f"{float(did['ALL']['did']):+.3f}",
        "Hardaway discretion": f"{float(ds['Tim Hardaway Jr.']['discretion_games']):.1f}",
    }
    missing = [k for k, v in checks.items() if v.lstrip("+") not in doc]
    print("prose figures found in the text:", f"{len(checks) - len(missing)}/{len(checks)}")
    for k in missing:
        print(f"  CHECK BY HAND — {k} is now {checks[k]}")


if __name__ == "__main__":
    main()

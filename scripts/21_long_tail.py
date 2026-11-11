"""Long-absence, season-end and competing-risk reporting.

Fits spell-level components on the onset features:
  * P(long or season-ending)
  * P(no return this season)
  * cause probabilities: returned / season_end / other censor

Then joins the rolling hazard model to produce the catastrophic-injury table
(P(out past 20), P(no return this season), median and mean games).

Usage:
    uv run python scripts/21_long_tail.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from injury_duration import experiment as ex
from injury_duration import long_tail, models, persist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hazard-model",
        type=Path,
        default=Path("artifacts/rolling/2025-26_rot/model.pkl"),
    )
    return parser.parse_args()


def evaluate(df: pl.DataFrame, long_model, season_model, cause_models) -> list[dict]:
    p_long = long_model.hazard(df)
    p_season = season_model.hazard(df)
    causes = long_tail.predict_causes(cause_models, df)
    rows = [
        {"split": df["_split"][0], **long_tail.binary_metrics(df["is_long"], p_long, "long")},
        {
            "split": df["_split"][0],
            **long_tail.binary_metrics(df["is_season_end"], p_season, "season_end"),
        },
        {"split": df["_split"][0], **long_tail.cause_metrics(df["cause"].to_list(), causes)},
    ]
    return rows


def main() -> None:
    args = parse_args()
    d = ex.load()
    data = long_tail.spell_targets(d["features"])
    cols = ex.spell_feature_cols()
    train = data.filter(pl.col("season") <= "2023-24")
    long_model = long_tail.fit_binary(data, cols, "is_long", train)
    season_model = long_tail.fit_binary(data, cols, "is_season_end", train)
    cause_models = long_tail.fit_causes(data, cols, train)

    rows = []
    for split, season in (("valid", "2024-25"), ("test", "2025-26")):
        df = data.filter(pl.col("season") == season).with_columns(pl.lit(split).alias("_split"))
        rows.extend(evaluate(df, long_model, season_model, cause_models))
    metrics = pl.DataFrame(rows)
    metrics.write_csv("reports/long_tail_metrics.csv")
    print(
        metrics.select(
            "split",
            "model",
            "n",
            "base_rate",
            "brier",
            "brier_skill",
            "log_loss",
            "auc",
            "accuracy",
            "auc_returned",
            "auc_season_end",
            "auc_other",
        )
    )

    test = data.filter(pl.col("season") == "2025-26")
    grid = ex.grid_design(
        d["features"].filter(pl.col("season") == "2025-26"),
        d["team_context"],
        max_k=models.MAX_K,
    )
    bundle = persist.load(args.hazard_model)
    pred = models.predict_from_grid(bundle.onset, grid)
    p_long = long_model.hazard(test)
    p_season = season_model.hazard(test)
    causes = long_tail.predict_causes(cause_models, test)
    report = (
        test.select(
            "spell_id",
            "player_name",
            "start_date",
            "body_region",
            "ailment_class",
            "games_missed",
            "event",
            "censor_reason",
            "is_catastrophic",
        )
        .join(pred, on="spell_id", how="left")
        .with_columns(
            pl.Series("p_long", p_long),
            pl.Series("p_no_return", p_season),
        )
        .join(causes, on="spell_id", how="left")
        .filter(pl.col("is_catastrophic"))
        .sort("p_no_return", descending=True)
    )
    report.write_csv("reports/catastrophic_report.csv")
    print()
    print("CATASTROPHIC INJURIES (2025-26)")
    print(
        report.select(
            "player_name",
            "start_date",
            "body_region",
            "ailment_class",
            "games_missed",
            "event",
            "p_out_past_20",
            "p_no_return",
            "pred_median_games",
            "pred_mean_games",
        ).head(20)
    )
    print("wrote reports/long_tail_metrics.csv")
    print("wrote reports/catastrophic_report.csv")


if __name__ == "__main__":
    main()

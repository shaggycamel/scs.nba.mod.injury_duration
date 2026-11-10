"""Daily as-of backtest and calibration for the rolling-origin forecasts.

Reads the parquet output of `14_backfill_daily.py` for several seasons, builds
the calendar-horizon outcomes, and reports:

* Brier / log loss / Brier skill by season and horizon;
* calibration by predicted-probability decile and elapsed bucket;
* isotonic and Platt calibrators fitted on 2024-25 and scored on 2025-26.

Usage:
    uv run python scripts/20_daily_backtest.py
    uv run python scripts/20_daily_backtest.py \
        --inputs artifacts/rolling/backfill_daily_2024_25.parquet \
                  artifacts/rolling/backfill_daily_2025_26.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

HORIZONS = (7, 14, 30)
FIT_SEASON = "2024-25"
TEST_SEASON = "2025-26"
DEFAULT_INPUTS = (
    "artifacts/rolling/backfill_daily_2023_24.parquet",
    "artifacts/rolling/backfill_daily_2024_25.parquet",
    "artifacts/rolling/backfill_daily_2025_26.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=list(DEFAULT_INPUTS))
    parser.add_argument("--out", type=Path, default=Path("reports/daily_backtest_metrics.csv"))
    parser.add_argument(
        "--calibration-out",
        type=Path,
        default=Path("reports/daily_calibration.csv"),
    )
    parser.add_argument(
        "--calibration-fit-out",
        type=Path,
        default=Path("reports/daily_calibration_fit.csv"),
    )
    return parser.parse_args()


def add_outcomes(df: pl.DataFrame) -> pl.DataFrame:
    max_pred = df.group_by("spell_id").agg(pl.col("prediction_date").max().alias("_max_prediction"))
    df = df.join(max_pred, on="spell_id", how="left").with_columns(
        pl.when(pl.col("return_seen"))
        .then(pl.col("actual_return_date"))
        .otherwise(pl.col("next_date"))
        .alias("_obs_end")
    )
    df = df.with_columns(
        pl.when(pl.col("_obs_end").is_null())
        .then(pl.col("_max_prediction"))
        .otherwise(pl.col("_obs_end"))
        .alias("_obs_end")
    )
    df = df.with_columns(
        pl.when(pl.col("days_missed_so_far") == 0)
        .then(pl.lit("onset"))
        .when(pl.col("days_missed_so_far") <= 3)
        .then(pl.lit("1-3 days"))
        .when(pl.col("days_missed_so_far") <= 7)
        .then(pl.lit("4-7 days"))
        .otherwise(pl.lit("8+ days"))
        .alias("elapsed_bucket")
    )
    for h in HORIZONS:
        df = df.with_columns(
            (
                pl.col("return_seen")
                & (pl.col("actual_return_date") <= pl.col("prediction_date") + pl.duration(days=h))
            )
            .fill_null(False)
            .cast(pl.Int8)
            .alias(f"y_{h}"),
            (
                pl.col("return_seen")
                | ((pl.col("_obs_end") - pl.col("prediction_date")).dt.total_days() >= h)
            )
            .fill_null(False)
            .alias(f"eval_{h}"),
        )
    return df


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _metrics(y: np.ndarray, p: np.ndarray, label: str) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base = y.mean()
    brier = float(brier_score_loss(y, p))
    base_brier = float(brier_score_loss(y, np.full_like(p, base)))
    return {
        "model": label,
        "n": len(y),
        "base_rate": float(base),
        "brier": brier,
        "brier_skill": 1 - brier / base_brier if base_brier else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if 0 < base < 1 else float("nan"),
    }


def calibration_rows(
    df: pl.DataFrame,
    season: str,
    horizon: int,
    p_col: str,
    y_col: str,
    eval_col: str,
) -> list[dict]:
    d = df.filter(pl.col(eval_col) & pl.col(p_col).is_not_null())
    if d.height < 20:
        return []
    out = []
    for bucket, part in zip(
        ["all", "onset", "1-3 days", "4-7 days", "8+ days"],
        [
            d,
            d.filter(pl.col("elapsed_bucket") == "onset"),
            d.filter(pl.col("elapsed_bucket") == "1-3 days"),
            d.filter(pl.col("elapsed_bucket") == "4-7 days"),
            d.filter(pl.col("elapsed_bucket") == "8+ days"),
        ],
        strict=True,
    ):
        if part.height < 20:
            continue
        py = part[y_col].to_numpy().astype(int)
        pp = np.clip(part[p_col].to_numpy(), 1e-6, 1 - 1e-6)
        ranked = np.argsort(np.argsort(pp))
        bins = (ranked * 10 // len(pp)).clip(0, 9)
        for b in range(10):
            mask = bins == b
            if mask.sum() == 0:
                continue
            out.append(
                {
                    "season": season,
                    "horizon": horizon,
                    "bucket": bucket,
                    "bin": b,
                    "n": int(mask.sum()),
                    "pred_mean": float(pp[mask].mean()),
                    "actual_mean": float(py[mask].mean()),
                }
            )
    return out


def fit_and_apply(
    train: pl.DataFrame,
    test: pl.DataFrame,
    horizon: int,
    p_col: str,
    y_col: str,
    eval_col: str,
    bucket: str = "all",
) -> list[dict]:
    """Fit isotonic and Platt calibrators on train, score test."""
    tr = train.filter(pl.col(eval_col) & pl.col(p_col).is_not_null())
    te = test.filter(pl.col(eval_col) & pl.col(p_col).is_not_null())
    if bucket != "all":
        tr = tr.filter(pl.col("elapsed_bucket") == bucket)
        te = te.filter(pl.col("elapsed_bucket") == bucket)
    if tr.height < 200 or te.height < 20:
        return []
    rows = []
    y_tr = tr[y_col].to_numpy().astype(int)
    p_tr = np.clip(tr[p_col].to_numpy(), 1e-6, 1 - 1e-6)
    y_te = te[y_col].to_numpy().astype(int)
    p_te = np.clip(te[p_col].to_numpy(), 1e-6, 1 - 1e-6)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_tr, y_tr)
    platt = LogisticRegression(max_iter=1000).fit(_logit(p_tr).reshape(-1, 1), y_tr)

    for label, p_cal in (
        ("raw", p_te),
        ("isotonic", iso.predict(p_te)),
        ("platt", platt.predict_proba(_logit(p_te).reshape(-1, 1))[:, 1]),
    ):
        rows.append(
            {
                "horizon": horizon,
                "bucket": bucket,
                "model": label,
                **_metrics(y_te, p_cal, label),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    frames = [pl.read_parquet(path) for path in args.inputs]
    df = pl.concat(frames, how="vertical_relaxed")
    df = add_outcomes(df)

    metric_rows: list[dict] = []
    calib_rows: list[dict] = []
    for season in sorted(df["season"].unique().to_list()):
        season_df = df.filter(pl.col("season") == season)
        for h in HORIZONS:
            p_col = f"p_return_within_{h}d"
            y_col = f"y_{h}"
            eval_col = f"eval_{h}"
            d = season_df.filter(pl.col(eval_col) & pl.col(p_col).is_not_null())
            if d.height < 20:
                continue
            y = d[y_col].to_numpy().astype(int)
            p = np.clip(d[p_col].to_numpy(), 1e-6, 1 - 1e-6)
            metric_rows.append(
                {
                    "season": season,
                    "horizon": h,
                    **_metrics(y, p, "raw"),
                }
            )
            calib_rows.extend(calibration_rows(season_df, season, h, p_col, y_col, eval_col))

    metrics = pl.DataFrame(metric_rows)
    calib = pl.DataFrame(calib_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_csv(args.out)
    calib.write_csv(args.calibration_out)

    print("DAILY BACKTEST")
    print(
        metrics.with_columns(
            pl.col("brier").round(4),
            pl.col("brier_skill").round(4),
            pl.col("log_loss").round(4),
            pl.col("auc").round(4),
        )
    )

    train = df.filter(pl.col("season") == FIT_SEASON)
    test = df.filter(pl.col("season") == TEST_SEASON)
    fit_rows: list[dict] = []
    for h in HORIZONS:
        for bucket in ("all", "onset", "1-3 days", "4-7 days", "8+ days"):
            fit_rows.extend(
                fit_and_apply(
                    train,
                    test,
                    h,
                    f"p_return_within_{h}d",
                    f"y_{h}",
                    f"eval_{h}",
                    bucket=bucket,
                )
            )
    if fit_rows:
        fit = pl.DataFrame(fit_rows)
        fit.write_csv(args.calibration_fit_out)
        print()
        print(f"CALIBRATION fit {FIT_SEASON} -> test {TEST_SEASON}")
        print(
            fit.filter(pl.col("bucket") == "all")
            .with_columns(
                pl.col("brier").round(4),
                pl.col("brier_skill").round(4),
                pl.col("log_loss").round(4),
            )
            .select("horizon", "model", "n", "base_rate", "brier", "brier_skill", "log_loss")
        )
        print(
            fit.filter(pl.col("bucket") != "all")
            .sort("horizon", "bucket", "model")
            .with_columns(
                pl.col("brier").round(4),
                pl.col("brier_skill").round(4),
                pl.col("log_loss").round(4),
            )
            .select("horizon", "bucket", "model", "n", "brier", "brier_skill", "log_loss")
            .head(30)
        )
    print()
    print(f"wrote {args.out}")
    print(f"wrote {args.calibration_out}")
    if fit_rows:
        print(f"wrote {args.calibration_fit_out}")


if __name__ == "__main__":
    main()

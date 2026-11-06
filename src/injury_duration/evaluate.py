"""Time-based splits and censoring-aware scoring.

Nothing is shuffled. A model trains on seasons that finished before the
season it is scored on, because every feature here is built from a player's
own past and a random split would let a model see a player's later injuries
while predicting their earlier ones.

Scoring a right-censored target needs care, so three complementary views:

* **MAE / RMSE on observed spells.** Honest but selective — censored spells
  are the long ones, so this measures accuracy on the short-to-medium range.
* **Harrell's C over every spell.** Uses the censored ones (a spell censored
  at 40 games is known to have outlasted one that ended at 3), and measures
  ranking rather than calibration.
* **Horizon accuracy.** "Still out after k games?" is a hard label for any
  spell that either returned by k or ran past k; spells censored at or before
  k are dropped. That gives a clean Brier score and AUC at each horizon with
  no inverse-probability weighting.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import hazard as hz

HORIZONS = (1, 3, 5, 10, 20)


def duration_metrics(
    spells: pl.DataFrame, pred: np.ndarray, label: str = ""
) -> dict:
    """Score a per-spell prediction of games missed."""
    d = spells["games_missed"].to_numpy().astype(float)
    e = spells["event"].to_numpy().astype(int)
    pred = np.asarray(pred, dtype=float)

    obs = e == 1
    err = pred[obs] - d[obs]
    out = {
        "model": label,
        "n_spells": len(d),
        "n_observed": int(obs.sum()),
        "mae_observed": float(np.mean(np.abs(err))),
        "rmse_observed": float(np.sqrt(np.mean(err**2))),
        "medae_observed": float(np.median(np.abs(err))),
        "bias_observed": float(np.mean(err)),
        # A model that only gets short spells right is not much use; the long
        # tail is where the cost is.
        "mae_observed_long": float(
            np.mean(np.abs(err[d[obs] >= 5])) if (d[obs] >= 5).any() else np.nan
        ),
        "spearman": float(spearmanr(pred[obs], d[obs]).statistic),
        # Risk is "expected to end sooner", so it is the negative duration.
        "c_index": hz.concordance_index(-pred, d, e),
    }

    for h in HORIZONS:
        mask = hz.horizon_table(d, e, h)
        if mask.sum() < 20:
            continue
        y = hz.horizon_label(d, e, h)[mask]
        # Monotone transform of the predicted mean into a score for "still out
        # after h games". Only the ranking matters for AUC; Brier needs an
        # actual probability, so it is reported separately from hazard models.
        out[f"auc_out_past_{h}"] = (
            float(roc_auc_score(y, pred[mask])) if 0 < y.mean() < 1 else float("nan")
        )
        out[f"base_rate_out_past_{h}"] = float(y.mean())
    return out


def horizon_brier(
    spells: pl.DataFrame, surv: dict[int, np.ndarray], label: str = ""
) -> list[dict]:
    """Brier score for P(still out after h games) from a survival model."""
    d = spells["games_missed"].to_numpy().astype(float)
    e = spells["event"].to_numpy().astype(int)
    rows = []
    for h, s in surv.items():
        mask = hz.horizon_table(d, e, h)
        if mask.sum() < 20:
            continue
        y = hz.horizon_label(d, e, h)[mask]
        p = np.clip(np.asarray(s, dtype=float)[mask], 1e-6, 1 - 1e-6)
        rows.append(
            {
                "model": label,
                "horizon": h,
                "n_evaluable": int(mask.sum()),
                "base_rate": float(y.mean()),
                "brier": float(brier_score_loss(y, p)),
                "brier_skill": float(
                    1 - brier_score_loss(y, p) / brier_score_loss(y, np.full_like(p, y.mean()))
                ),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
            }
        )
    return rows


def hazard_row_metrics(y: np.ndarray, p: np.ndarray, label: str = "") -> dict:
    """Scoring on the per-game return target itself."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base = np.full_like(p, y.mean())
    return {
        "model": label,
        "n_rows": len(y),
        "return_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "log_loss_skill": float(1 - log_loss(y, p, labels=[0, 1]) / log_loss(y, base, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "brier_skill": float(1 - brier_score_loss(y, p) / brier_score_loss(y, base)),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pl.DataFrame:
    """Predicted vs realised return rate, in equal-count bins."""
    df = pl.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df = df.with_columns(
        pl.col("p").rank("ordinal").alias("_r")
    ).with_columns(
        ((pl.col("_r") - 1) * bins // pl.len()).alias("bin")
    )
    return (
        df.group_by("bin")
        .agg(
            pl.len().alias("n"),
            pl.col("p").mean().alias("pred_mean"),
            pl.col("y").mean().alias("actual_mean"),
        )
        .sort("bin")
    )

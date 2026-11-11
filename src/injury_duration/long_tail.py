"""Long-absence and competing-risk components for the injury model.

The hazard answers "when do they return?"; these models answer two different
questions that a median hides:

* Is this spell long or season-ending? A binary long-tail component.
* If it ends without a return, why? Returned, season end, or another
  competing event (trade, G League, suspension, personal).

Both are spell-level, fitted on the onset features so they can be reported
alongside the survival curve.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import models

LONG_GAMES = 20
CAUSES = ("returned", "season_end", "other_censor")
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}


def spell_targets(spells: pl.DataFrame) -> pl.DataFrame:
    """Add the long/season-end and cause targets."""
    return spells.with_columns(
        (
            (pl.col("games_missed") >= LONG_GAMES)
            | ((pl.col("event") == 0) & (pl.col("censor_reason") == "season_end"))
        )
        .cast(pl.Int8)
        .alias("is_long"),
        ((pl.col("event") == 0) & (pl.col("censor_reason") == "season_end"))
        .cast(pl.Int8)
        .alias("is_season_end"),
        pl.when(pl.col("event") == 1)
        .then(pl.lit("returned"))
        .when(pl.col("censor_reason") == "season_end")
        .then(pl.lit("season_end"))
        .otherwise(pl.lit("other_censor"))
        .alias("cause"),
    )


def fit_binary(
    feats: pl.DataFrame,
    cols: list[str],
    target: str,
    train: pl.DataFrame,
) -> models.HazardModel:
    num, cat = _split(cols)
    return models.HazardModel("lightgbm", "lightgbm", num, cat, LGBM_PARAMS).fit(
        train.select(cols + [target]), target=target
    )


def fit_causes(
    feats: pl.DataFrame, cols: list[str], train: pl.DataFrame
) -> dict[str, models.HazardModel]:
    out = {}
    for cause in CAUSES:
        label = f"_is_{cause}"
        tr = train.with_columns((pl.col("cause") == cause).cast(pl.Int8).alias(label))
        num, cat = _split(cols)
        out[cause] = models.HazardModel("lightgbm", "lightgbm", num, cat, LGBM_PARAMS).fit(
            tr.select(cols + [label]), target=label
        )
    return out


def predict_causes(
    models_by_cause: dict[str, models.HazardModel], feats: pl.DataFrame
) -> pl.DataFrame:
    raw = pl.DataFrame(
        {
            "spell_id": feats["spell_id"],
            **{cause: model.hazard(feats) for cause, model in models_by_cause.items()},
        }
    )
    total = sum(pl.col(c) for c in CAUSES)
    return raw.with_columns((pl.col(c) / total).alias(c) for c in CAUSES)


def _split(cols: list[str]) -> tuple[list[str], list[str]]:
    from . import experiment as ex

    return ex.split_numeric_categorical(cols)


def binary_metrics(y: np.ndarray, p: np.ndarray, label: str) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base = y.mean()
    brier = float(brier_score_loss(y, p))
    return {
        "model": label,
        "n": len(y),
        "base_rate": float(base),
        "brier": brier,
        "brier_skill": 1 - brier / float(brier_score_loss(y, np.full_like(p, base))),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if 0 < base < 1 else float("nan"),
    }


def cause_metrics(y: list[str], probs: pl.DataFrame) -> dict:
    labels = list(CAUSES)
    y_idx = np.array([labels.index(v) for v in y])
    p = probs.select(labels).to_numpy()
    p = p / p.sum(axis=1, keepdims=True)
    onehot = np.eye(len(labels))[y_idx]
    return {
        "model": "cause_lightgbm",
        "n": len(y),
        "log_loss": float(log_loss(y_idx, p, labels=list(range(len(labels))))),
        "accuracy": float((p.argmax(axis=1) == y_idx).mean()),
        "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
        "auc_returned": float(roc_auc_score(onehot[:, 0], p[:, 0])),
        "auc_season_end": float(roc_auc_score(onehot[:, 1], p[:, 1])),
        "auc_other": float(roc_auc_score(onehot[:, 2], p[:, 2])),
    }

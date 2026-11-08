"""Turn a polars frame into the numeric matrices the estimators want.

scikit-learn and LightGBM both accept plain numpy, so there is no reason to
route through pandas: the encoders here fit their vocabularies and medians on
the training fold and apply them unchanged to later folds, which is also the
behaviour we want for a time-based split (a category first seen in the test
season must not silently become its own column).

Two encodings, because the estimators differ in what they can digest:

* `codes` — integer category codes and raw floats with NaN left in place, for
  LightGBM and `HistGradientBoosting*`, both of which handle categories and
  missing values natively and do it better than any preprocessing here.
* `onehot` — one-hot categories and median-imputed floats with missingness
  indicators, for the linear model and the random forest, which cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

# A category needs at least this many training rows to get its own level;
# rarer ones are pooled so the test season cannot be dominated by a level the
# model saw three times.
MIN_LEVEL_COUNT = 20
OTHER = "__other__"
MISSING = "__missing__"


@dataclass
class Design:
    numeric: list[str]
    categorical: list[str]
    levels: dict[str, list[str]] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    # Which numeric columns were ever missing in the training fold. Fixed at
    # fit time so the matrix has the same width on every later fold -- decided
    # per frame it would silently change shape between train and test.
    missing_cols: list[str] = field(default_factory=list)
    _fitted: bool = False

    # -------------------------------------------------------------- fitting
    def fit(self, df: pl.DataFrame) -> Design:
        self.levels = {}
        for c in self.categorical:
            counts = (
                df.select(pl.col(c).cast(pl.String).fill_null(MISSING))
                .group_by(c)
                .len()
                .filter(pl.col("len") >= MIN_LEVEL_COUNT)
                .sort(c)
            )
            self.levels[c] = counts[c].to_list() + [OTHER]
        self.medians = {}
        for c in self.numeric:
            # Take the median from the null-bearing column, not from the
            # NaN-filled one: polars' median treats NaN as a value and would
            # hand back NaN for any column with a missing entry.
            med = df.select(pl.col(c).cast(pl.Float64, strict=False).median()).item()
            self.medians[c] = 0.0 if med is None or math.isnan(med) else float(med)
        self.missing_cols = [c for c in self.numeric if df.select(pl.col(c).is_null().any()).item()]
        self._fitted = True
        return self

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() before transforming")

    # ----------------------------------------------------------- transforms
    def codes(self, df: pl.DataFrame) -> tuple[np.ndarray, list[int], list[str]]:
        """Floats with NaN, categories as integer codes.

        Returns `(X, categorical_indices, feature_names)`.
        """
        self._check()
        parts = [df.select(_as_float(c) for c in self.numeric)] if self.numeric else []
        names = list(self.numeric)
        cat_idx = []
        if self.categorical:
            exprs = []
            for c in self.categorical:
                lv = self.levels[c]
                mapping = {v: i for i, v in enumerate(lv)}
                other = mapping[OTHER]
                exprs.append(
                    pl.col(c)
                    .cast(pl.String)
                    .fill_null(MISSING)
                    .replace_strict(mapping, default=other)
                    .cast(pl.Float64)
                    .alias(c)
                )
            parts.append(df.select(exprs))
            cat_idx = [len(names) + i for i in range(len(self.categorical))]
            names += list(self.categorical)
        X = np.hstack([p.to_numpy().astype(np.float64) for p in parts])
        return X, cat_idx, names

    def onehot(self, df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Median-imputed floats with missingness flags, one-hot categories."""
        self._check()
        blocks, names = [], []
        if self.numeric:
            num = df.select(_as_float(c) for c in self.numeric).to_numpy().astype(np.float64)
            miss = np.isnan(num)
            med = np.array([self.medians[c] for c in self.numeric], dtype=np.float64)
            num = np.where(miss, med, num)
            blocks.append(num)
            names += list(self.numeric)
            keep = [i for i, c in enumerate(self.numeric) if c in self.missing_cols]
            if keep:
                blocks.append(miss[:, keep].astype(np.float64))
                names += [f"{self.numeric[i]}__is_missing" for i in keep]
        for c in self.categorical:
            lv = self.levels[c]
            col = df.select(pl.col(c).cast(pl.String).fill_null(MISSING)).to_series().to_list()
            idx = {v: i for i, v in enumerate(lv)}
            other = idx[OTHER]
            block = np.zeros((len(col), len(lv)), dtype=np.float64)
            for r, v in enumerate(col):
                block[r, idx.get(v, other)] = 1.0
            blocks.append(block)
            names += [f"{c}={v}" for v in lv]
        return np.hstack(blocks), names


def _as_float(col: str) -> pl.Expr:
    """Cast to float, mapping booleans to 0/1 and nulls to NaN."""
    return (
        pl.when(pl.col(col).is_null())
        .then(pl.lit(float("nan")))
        .otherwise(pl.col(col).cast(pl.Float64, strict=False))
        .alias(col)
    )

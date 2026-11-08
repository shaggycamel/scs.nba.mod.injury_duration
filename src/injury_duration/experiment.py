"""Wiring between the built tables and the models: feature sets and splits.

Kept out of the scripts so the modelling, the ablations and the explanation
pass all see exactly the same columns and the same fold boundaries.
"""

from __future__ import annotations

import polars as pl

from . import features, paths
from . import hazard as hz

# Per-game columns whose names collide with a spell-level feature. The
# per-game value is already summarised by the `*_now` and `*_changed` columns,
# so the spell-level one wins and these are dropped before the join.
_COLLIDING = [
    "body_region",
    "body_side",
    "ailment_class",
    "days_rest",
    "season_type",
    "is_management",
    "is_recovery_stage",
    "is_surgical",
    "team_games_next_14d",
]

# Grid-only scaffolding, not features and not wanted downstream.
_GRID_SCRATCH = [
    "game_date",
    "started_in_playoffs",
    "beyond_season",
    "last_idx",
    "last_date",
    "team_game_idx",
]


def spell_feature_cols() -> list[str]:
    return features.all_features()


def index_feature_cols() -> list[str]:
    """Everything knowable the moment the player is ruled out."""
    return spell_feature_cols() + hz.SCHEDULE_TIME_COLS


def dynamic_feature_cols() -> list[str]:
    """Index features plus how the report has moved since."""
    return index_feature_cols() + hz.REPORT_TIME_COLS


def split_numeric_categorical(cols: list[str]) -> tuple[list[str], list[str]]:
    cat = [c for c in features.CATEGORICAL if c in cols]
    num = [c for c in cols if c not in cat]
    return num, cat


def load() -> dict[str, pl.DataFrame]:
    b = paths.BUILD
    return {
        "spells": pl.read_parquet(b / "spells.parquet"),
        "features": pl.read_parquet(b / "spell_features.parquet"),
        "rows": pl.read_parquet(b / "hazard_rows.parquet"),
        "team_context": pl.read_parquet(b / "team_context.parquet"),
    }


def hazard_design(rows: pl.DataFrame, feats: pl.DataFrame) -> pl.DataFrame:
    """Per-missed-game rows joined to their spell's index features."""
    keep = [c for c in rows.columns if c not in _COLLIDING]
    spell_cols = ["spell_id", "season", "games_missed", "event"] + spell_feature_cols()
    spell_cols = list(dict.fromkeys(spell_cols))
    return (
        rows.select(keep)
        .drop(["season", "games_missed", "event"], strict=False)
        .join(feats.select(spell_cols), on="spell_id", how="inner")
    )


def grid_design(spells_feat: pl.DataFrame, tctx: pl.DataFrame, max_k: int = 100) -> pl.DataFrame:
    """Per-(spell, k) grid joined to index features, for duration prediction."""
    grid = hz.build_prediction_grid(spells_feat, tctx, max_k=max_k)
    spell_cols = ["spell_id", "season", "games_missed", "event"] + spell_feature_cols()
    spell_cols = list(dict.fromkeys(spell_cols))
    drop = [c for c in _COLLIDING + _GRID_SCRATCH if c in grid.columns]
    return grid.drop(drop).join(spells_feat.select(spell_cols), on="spell_id", how="inner")

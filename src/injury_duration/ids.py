"""Resolve the injury report's nullable `nba_id` to a player id.

`nba.injuries.nba_id` is null for 2-38% of rows depending on season, so the
report has to be re-keyed by name. The ladder below is exact id, then exact
name against the id crosswalk, then exact name against the box score, then
a punctuation- and suffix-stripped name against both. Roughly 0.02% of rows
survive all four, almost all of them rows where the scraper interleaved two
players ("Luka Reaves Doncic"); those are dropped and counted.
"""

from __future__ import annotations

import polars as pl

from . import paths

_SUFFIXES = r"\s+(jr|sr|ii|iii|iv|v)$"


def normalise_name(col: str | pl.Expr) -> pl.Expr:
    e = pl.col(col) if isinstance(col, str) else col
    return (
        e.str.to_lowercase()
        .str.replace_all(r"[.'`’\-]", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.replace(_SUFFIXES, "")
        .str.strip_chars()
    )


def _lookup(df: pl.DataFrame, name_col: str, id_col: str, out: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Exact and normalised name -> id lookups, dropping ambiguous names."""
    base = df.select(
        pl.col(name_col).alias("nm"), pl.col(id_col).cast(pl.Int64).alias(out)
    ).drop_nulls()
    exact = base.unique(subset=["nm", out]).group_by("nm").agg(
        pl.col(out).first(), pl.col(out).n_unique().alias("n")
    ).filter(pl.col("n") == 1).drop("n")
    norm = (
        base.with_columns(normalise_name("nm").alias("nm"))
        .unique(subset=["nm", out])
        .group_by("nm")
        .agg(pl.col(out).first(), pl.col(out).n_unique().alias("n"))
        .filter(pl.col("n") == 1)
        .drop("n")
        .rename({out: out + "_n"})
    )
    return exact, norm


def resolve_player_ids(report: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Add a non-null `player_id` to the report; return it plus a diagnostic."""
    pmap = pl.read_parquet(paths.PLAYER_ID_MAP)
    box = pl.read_parquet(paths.BOX, columns=["player_name", "player_id"])

    map_exact, map_norm = _lookup(pmap, "nba_name", "nba_id", "id_map")
    box_exact, box_norm = _lookup(box, "player_name", "player_id", "id_box")

    out = (
        report.with_columns(normalise_name("player_name").alias("_nm"))
        .join(map_exact, left_on="player_name", right_on="nm", how="left")
        .join(box_exact, left_on="player_name", right_on="nm", how="left")
        .join(map_norm, left_on="_nm", right_on="nm", how="left")
        .join(box_norm, left_on="_nm", right_on="nm", how="left")
        .with_columns(
            pl.coalesce("nba_id", "id_map", "id_box", "id_map_n", "id_box_n").alias("player_id")
        )
    )
    diag = {
        "rows": out.height,
        "unresolved_rows": out.filter(pl.col("player_id").is_null()).height,
        "unresolved_names": out.filter(pl.col("player_id").is_null())["player_name"]
        .n_unique(),
    }
    out = out.drop(["_nm", "id_map", "id_box", "id_map_n", "id_box_n"]).filter(
        pl.col("player_id").is_not_null()
    )
    return out, diag

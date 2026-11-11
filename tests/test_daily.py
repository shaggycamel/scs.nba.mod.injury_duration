from __future__ import annotations

from datetime import date

import polars as pl

from injury_duration import daily
from injury_duration import experiment as ex


def _frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    spells = pl.DataFrame(
        {
            "spell_id": ["capped", "returned", "season_end"],
            "player_name": ["Capped", "Returned", "Season End"],
            "season": ["2026-27"] * 3,
            "start_date": [date(2026, 11, 10), date(2026, 11, 9), date(2026, 11, 8)],
            "last_missed_date": [date(2026, 11, 11), date(2026, 11, 10), date(2026, 11, 9)],
            "next_date": [date(2026, 11, 13), date(2026, 11, 11), None],
            "event": [0, 1, 0],
            "games_missed": [2, 2, 2],
            "censor_reason": ["dropped_from_report", "returned", "season_end"],
            "body_region": ["ankle"] * 3,
            "ailment_class": ["sprain"] * 3,
        }
    )
    features = spells.select("spell_id", "season", "games_missed", "event").with_columns(
        pl.lit(None).alias(c) for c in ex.spell_feature_cols()
    )
    rows = pl.DataFrame(
        {
            "spell_id": ["capped", "capped", "returned", "returned", "season_end", "season_end"],
            "game_date": [
                date(2026, 11, 10),
                date(2026, 11, 11),
                date(2026, 11, 9),
                date(2026, 11, 10),
                date(2026, 11, 8),
                date(2026, 11, 9),
            ],
            "games_missed_so_far": [1, 2, 1, 2, 1, 2],
            "team_game_idx": [1, 2, 1, 2, 1, 2],
            "player_id": [1, 1, 2, 2, 3, 3],
        }
    )
    return spells, features, rows


def test_daily_states_stop_at_cutoff() -> None:
    spells, features, rows = _frames()
    cutoff = date(2026, 11, 11)

    states = daily.build_daily_states(spells, features, rows, cutoff)

    assert states["prediction_date"].max() == cutoff
    ends = states.group_by("spell_id").agg(pl.col("prediction_date").max())
    assert dict(ends.iter_rows()) == {
        "capped": cutoff,
        "returned": date(2026, 11, 10),
        "season_end": cutoff,
    }


def test_daily_states_respect_through() -> None:
    spells, features, rows = _frames()
    cutoff = date(2026, 11, 11)
    through = date(2026, 11, 10)

    states = daily.build_daily_states(spells, features, rows, cutoff, through=through)

    assert states["prediction_date"].max() == through

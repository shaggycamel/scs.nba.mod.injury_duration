from __future__ import annotations

import polars as pl

from injury_duration.spells import _bridge_report_gaps


def _row(state: str, status: str = "", category: str | None = None) -> dict:
    return {
        "state": state,
        "status_clean": status,
        "reason_category": category,
        "reason": None if category is None else f"Injury/Illness - {category}",
        "played": state == "played",
    }


def _panel(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [1] * len(rows),
            "season": ["2024-25"] * len(rows),
            "player_game_idx": list(range(1, len(rows) + 1)),
            "state": [r["state"] for r in rows],
            "status_clean": [r["status_clean"] for r in rows],
            "reason_category": [r["reason_category"] for r in rows],
            "reason": [r["reason"] for r in rows],
            "played": [r["played"] for r in rows],
        }
    )


OUT = _row("out", "Out", "injury")
GAP = _row("unlisted_absent")
AVAILABLE = _row("missed_Available", "Available", "roster")
PLAYED = _row("played")
CENSOR = _row("out", "Out", "roster")


def test_three_game_gap_before_a_return_is_bridged() -> None:
    out = _bridge_report_gaps(_panel([OUT, GAP, GAP, GAP, PLAYED]))
    assert out["is_bridged"].sum() == 3
    assert out.filter(pl.col("is_bridged"))["state"].unique().to_list() == ["out"]


def test_four_game_gap_before_a_return_is_not_bridged() -> None:
    out = _bridge_report_gaps(_panel([OUT, GAP, GAP, GAP, GAP, PLAYED]))
    assert out["is_bridged"].sum() == 0


def test_available_dnp_is_treated_as_a_gap() -> None:
    out = _bridge_report_gaps(_panel([OUT, AVAILABLE, GAP, PLAYED]))
    assert out["is_bridged"].sum() == 2
    bridged = out.filter(pl.col("is_bridged"))
    assert bridged["state"].unique().to_list() == ["out"]
    assert bridged["status_clean"].unique().to_list() == ["Out"]


def test_gap_after_a_played_game_is_not_bridged() -> None:
    out = _bridge_report_gaps(_panel([PLAYED, GAP, OUT]))
    assert out["is_bridged"].sum() == 0


def test_gap_before_a_censor_is_not_bridged() -> None:
    out = _bridge_report_gaps(_panel([OUT, GAP, CENSOR]))
    assert out["is_bridged"].sum() == 0


def test_gap_between_two_health_outs_is_bridged() -> None:
    out = _bridge_report_gaps(_panel([OUT, GAP, OUT]))
    assert out["is_bridged"].sum() == 1

from __future__ import annotations

import runpy
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/15_publish_daily.py"


class _DB:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame

    def read(self, _sql: str) -> pl.DataFrame:
        return self.frame


class _Hub:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.db = _DB(frame)


def _publish_module() -> dict:
    return runpy.run_path(str(SCRIPT))


def test_require_view_accepts_an_applied_view() -> None:
    ns = _publish_module()
    ns["_require_view"](_Hub(pl.DataFrame({"ok": [1]})))


def test_require_view_fails_with_the_db_repo_location() -> None:
    ns = _publish_module()
    with pytest.raises(SystemExit, match="scs.nba.database"):
        ns["_require_view"](_Hub(pl.DataFrame()))

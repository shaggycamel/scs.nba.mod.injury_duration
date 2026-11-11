from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/15_publish_daily.py"


class _Result:
    def __init__(self, row: tuple[int] | None) -> None:
        self._row = row

    def first(self):
        return self._row


class _Conn:
    def __init__(self, exists: bool) -> None:
        self.exists = exists
        self.statements: list[str] = []

    def execute(self, statement):
        sql = str(statement)
        self.statements.append(sql)
        if "information_schema.views" in sql:
            return _Result((1,) if self.exists else None)
        return _Result(None)


def _publish_module() -> dict:
    return runpy.run_path(str(SCRIPT))


def test_ensure_view_does_not_replace_an_existing_view() -> None:
    ns = _publish_module()
    conn = _Conn(exists=True)

    ns["_ensure_view"](conn)

    assert len(conn.statements) == 1
    assert "information_schema.views" in conn.statements[0]
    assert not any("CREATE" in sql.upper() for sql in conn.statements)


def test_ensure_view_installs_a_missing_view() -> None:
    ns = _publish_module()
    conn = _Conn(exists=False)

    ns["_ensure_view"](conn)

    assert any("CREATE OR REPLACE VIEW" in sql for sql in conn.statements)
    assert any("GRANT SELECT" in sql for sql in conn.statements)

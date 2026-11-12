"""Run the daily injury forecast end to end.

Called by `scs.nba.mod.runner` on the Jetson (through `__main__.py`), or
manually:

    uv run --group data python __main__.py --season 2026-27

Sequence:
  1. active spells from a current-season mini snapshot
  2. lean snapshot: target-player box history + all injuries + references
  3. build spells/features on that snapshot
  4. score the latest prediction date
  5. append that date to mod.injury_daily_forecast_base
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SOURCE = "scs.nba.mod.injury_duration"
TASK = "injury_duration.job"


def current_season(today: date | None = None) -> str:
    today = today or date.today()
    year = today.year if today.month >= 7 else today.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=os.environ.get("SCS_SEASON") or current_season())
    parser.add_argument("--cadence", default=os.environ.get("SCS_CADENCE", "daily"))
    parser.add_argument("--db-con", default="cockroach-write")
    parser.add_argument("--source", default=os.environ.get("UPDATE_LOG_SOURCE", SOURCE))
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/injury_duration/model.pkl")
    return parser.parse_args(argv)


def run(name: str, argv: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {name}", flush=True)
    subprocess.run([sys.executable, *argv], cwd=ROOT, env=env, check=True)


def parse_kickoff() -> datetime:
    raw = os.environ.get("UPDATE_LOG_KICKOFF")
    if raw:
        try:
            value = datetime.fromisoformat(raw)
        except ValueError:
            pass
        else:
            return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.now(UTC)


def log_run(
    kickoff: datetime,
    cadence: str,
    source: str,
    success: bool,
    error: str | None,
    row_count: int | None,
    db_con: str,
) -> None:
    import scs_hub

    hub = scs_hub.SportsHub(db_con=db_con)
    hub.db.write(
        pl.DataFrame(
            {
                "source": source,
                "job_kickoff": kickoff,
                "cadence": cadence,
                "task": TASK,
                "task_time": datetime.now(UTC),
                "success": success,
                "error_message": error,
                "row_count": row_count,
            },
            schema_overrides={"row_count": pl.Int64},
        ),
        "update_log",
        "util",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kickoff = parse_kickoff()
    result_file = os.environ.get("SCS_RESULT_FILE")
    season = args.season
    base_env = os.environ.copy()

    active_data = ROOT / "data/active"
    lean_data = ROOT / "data/lean_current"
    active_file = ROOT / "artifacts/active_spells.parquet"
    daily_file = ROOT / f"artifacts/daily_predictions_{season}.parquet"

    error = None
    rows = None
    try:
        if not args.model.exists():
            raise FileNotFoundError(f"missing model bundle: {args.model}")

        run(
            "active spells",
            [
                "scripts/18_active_spells.py",
                "--season",
                season,
                "--db-con",
                args.db_con,
                "--data",
                str(active_data),
                "--out",
                str(active_file),
            ],
        )
        run(
            "lean fetch",
            [
                "scripts/16_lean_fetch.py",
                "--players-file",
                str(active_file),
                "--season",
                season,
                "--out",
                str(lean_data),
                "--db-con",
                args.db_con,
            ],
        )
        lean_env = {**base_env, "INJURY_DURATION_DATA": str(lean_data)}
        run(
            "build",
            [
                "scripts/01_build.py",
                "--diagnostics-out",
                str(ROOT / "artifacts/lean_current_diagnostics.json"),
            ],
            env=lean_env,
        )
        run(
            "predict",
            [
                "scripts/19_daily_predict.py",
                "--season",
                season,
                "--model",
                str(args.model),
                "--active",
                str(active_file),
                "--out",
                str(daily_file),
            ],
            env=lean_env,
        )
        rows = pl.read_parquet(daily_file).height
        run(
            "publish",
            [
                "scripts/15_publish_daily.py",
                "--input",
                str(daily_file),
                "--season",
                season,
                "--append",
                "--db-con",
                args.db_con,
            ],
            env=base_env,
        )
    except Exception as exc:  # noqa: BLE001 - the runner needs one failure row
        error = f"{type(exc).__name__}: {exc}"
        print(f"failed: {error}", file=sys.stderr, flush=True)

    if result_file:
        Path(result_file).write_text(json.dumps({"row_count": rows if error is None else 0}) + "\n")
    else:
        log_run(kickoff, args.cadence, args.source, error is None, error, rows, args.db_con)
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Refresh `nba.injury_reason_map` from the current injury report.

The injury report's `reason` is free text; `taxonomy.reason_features()` parses
it into region / pathology / severity columns. That parse is stored at source
in `nba.injury_reason_map`. The public `nba.injuries_parsed_vw` that joins it to
the report is applied from `nba/database/sql/views`.

Run after the injury report has been refreshed, or whenever the taxonomy
changes:

    uv run --group data python scripts/11_refresh_reason_map.py
    uv run --group data python scripts/11_refresh_reason_map.py --db-con postgres

The rewrite is deterministic: every distinct reason is parsed again and the
table is replaced in one transaction.
"""

from __future__ import annotations

import argparse

import polars as pl
import scs_hub
from sqlalchemy import text

from injury_duration import taxonomy

COLUMNS = [
    "reason",
    "reason_category",
    "body_part_raw",
    "body_side",
    "body_region",
    "ailment_raw",
    "ailment_class",
    "is_surgical",
    "is_recovery_stage",
    "is_management",
    "is_bone_stress",
    "mentions_major_structure",
    "n_reason_separators",
    "reason_len",
]

TABLE_DDL = """
CREATE TABLE IF NOT EXISTS nba.injury_reason_map (
  reason text PRIMARY KEY,
  reason_category text,
  body_part_raw text,
  body_side text,
  body_region text,
  ailment_raw text,
  ailment_class text,
  is_surgical boolean,
  is_recovery_stage boolean,
  is_management boolean,
  is_bone_stress boolean,
  mentions_major_structure boolean,
  n_reason_separators bigint,
  reason_len bigint
)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-con",
        default="cockroach-write",
        help="credentials.ini section to write through (use 'postgres' locally)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hub = scs_hub.SportsHub(db_con=args.db_con)

    reasons = hub.db.read("SELECT DISTINCT coalesce(reason, '') AS reason FROM nba.injuries").sort(
        "reason"
    )
    parsed = (
        reasons.with_columns(taxonomy.reason_features())
        .with_columns(
            pl.col("n_reason_separators").cast(pl.Int64),
            pl.col("reason_len").cast(pl.Int64),
        )
        .select(COLUMNS)
    )

    with hub.db.engine.begin() as conn:
        conn.execute(text(TABLE_DDL))
        conn.execute(text("DELETE FROM nba.injury_reason_map"))
        parsed.write_database("nba.injury_reason_map", conn, if_table_exists="append")

    print(
        f"{args.db_con}: wrote {parsed.height:,} parsed reasons; nba.injury_reason_map is current"
    )


if __name__ == "__main__":
    main()

# View Wiring Instructions

> Archived 2026-11-13; this migration is complete. Current view rules live in
> `nba/database/sql/views/`.

Status: 2026-11-11. Branch: `views-fetch`.

This is the ordered checklist for switching the injury model from raw tables to
the read-only view layer. Do this **before** the label fixes.

## Already done on this branch

- `00_fetch.py`, `16_lean_fetch.py`, and `18_active_spells.py` now query:
  - `nba.injuries_vw`
  - `nba.league_game_schedule_vw`
  - `nba.player_box_score_vw`
  - `nba.player_info_vw`
  - `nba.team_roster_vw`
  - `nba.teams_vw`
  - `util.player_id_map_vw`
  - `util.unmatched_player_source_vw`
- The box query filters `min > 0`, deduplicates, aliases `inj_reason AS comment`,
  and casts `player_id` to `BIGINT`.
- `nba.injuries_vw` exposes `raw_name`; the fetch selects it and `load_report`
  uses `player_name` first, falling back to `raw_name` only when the crosswalk
  is missing.
- `build_panel` extends player windows with `team_roster_vw`, but only for
  players actually seen on that team. This covers DNP games and the season
  edges without treating every inactive roster row as an absence.
- `scs.nba.database` was fixed so `player_box_score_vw` and `team_roster_vw`
  use the full schedule window (preseason, regular season and playoffs).
  The previous regular-season-only window was dropping playoff box rows, which
  censored playoff returns.
- Ruff passes on all three scripts.

## Current parity result

Raw build is `main` @ `635c4e0` against the raw tables; view build is this
branch against the live views, both at the 2025-26 cutoff.

| Metric | Raw build | View build |
|---|---:|---:|
| Spells | 6,752 | 6,759 |
| Regular-season spells | 6,598 | 6,605 |
| Playoff spells | 154 | 154 |
| Hazard rows | 34,063 | 34,404 |
| Panel rows | 225,163 | 236,104 |
| Event rate | 77.8% | 77.5% |
| Features | 71 | 71 |

The panel is larger because the roster covers DNP games that the played-only
box view does not. That is the intended panel construction. At spell level,
6,369 spells match, 383 are raw-only, 390 are view-only, and only 14 outcomes
change. The 2025-26 test on the view build gives hazard histgb C-index 0.744
versus 0.743 raw, and MAE 2.578 versus 2.561; lightgbm is also 0.744. The view
snapshot is accepted.

## Ordered steps

1. **Review the fetch diff** — done.
2. **Confirm the local view layer is up to date** — Cockroach is current;
   local Postgres still needs the six views and the injury backfill.
3. **Run the view-only fetch** — done against `cockroach-read` into
   `data/views_check`.
4. **Build the view snapshot** — done.
5. **Compare against the raw build** — done; see the table above.
6. **Decide the panel construction** — use `team_roster_vw` plus schedule
   (done in `spells.py`).
7. **Accept the parity result** — accepted.
8. **Switch the production snapshot** — done. The Jetson is on `main`, the
   refit bundle is deployed, and the 2026-11-11 daily run published 99 rows
   through `mod.injury_daily_forecast`.
9. **Then** move to the label branch and finish the report-gap/trade work.
   Done: `injury-label-fix` has the report-gap / Available-DNP bridge and the
   roster-backed trade continuation, and is merged to `main`.

## Important

- No new `player_box_score_played_vw`. Use `nba.player_box_score_vw` with
  `min > 0`.
- `team_roster_vw` is required in all three fetch scripts; `paths.ROSTER`
  already exists.
- The 29 cross-team extras are accepted for training (0.007% of played keys).
- `injury-label-fix` is now on the remote. It contains the report-gap /
  Available-DNP bridge and the roster-backed trade continuation. Do not mix it
  into this branch until the view snapshot is deployed.
- 2026-27 remains out-of-sample; do not tune on it.

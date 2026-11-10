# View Wiring Instructions

Status: 2026-11-11. Branch: `views-fetch`.

This is the ordered checklist for switching the injury model from raw tables to
the read-only view layer. Do this **before** the label fixes.

## Already done on this branch

- `00_fetch.py`, `16_lean_fetch.py`, and `18_active_spells.py` now query:
  - `nba.injuries_vw`
  - `nba.league_game_schedule_vw`
  - `nba.player_box_score_vw`
  - `nba.player_info_vw`
  - `nba.teams_vw`
  - `util.player_id_map_vw`
  - `util.unmatched_player_source_vw`
- The box query filters `min > 0`, deduplicates, aliases `inj_reason AS comment`,
  and casts `player_id` to `BIGINT`.
- `nba.injuries_vw` now falls back to the raw report name when the crosswalk has
  no `nba_id`; this fixed the biggest parity break.
- Ruff passes on all three scripts.

## Current parity result

| Metric | Raw build | View build |
|---|---:|---:|
| Spells | 6,754 | 6,757 |
| Regular-season spells | 6,600 | 6,547 |
| Playoff spells | 154 | 210 |
| Hazard rows | 34,064 | 34,275 |
| Panel rows | 225,296 | 212,160 |
| Features | 71 | 71 |

The remaining difference is mainly the panel: the raw build uses raw **DNP box
rows** to extend player windows; the played-only view does not have them.
Playoff coverage also differs. The all-box-rows variant was worse (6,767 spells,
234,497 panel rows), so `min > 0` is the right box filter.

## Ordered steps

1. **Review the fetch diff** on this branch:
   ```bash
   git diff -- scripts/00_fetch.py scripts/16_lean_fetch.py scripts/18_active_spells.py
   ```
2. **Confirm the local view layer is up to date.** Required views:
   `injuries_vw`, `league_game_schedule_vw`, `player_box_score_vw`,
   `player_info_vw`, `teams_vw`, `team_roster_vw`, `daily_standings_vw`.
3. **Run the view-only fetch** to a scratch directory:
   ```bash
   uv run --group data python scripts/00_fetch.py \
     --db-con postgres --out data/views_check
   ```
4. **Build the view snapshot:**
   ```bash
   INJURY_DURATION_DATA=data/views_check uv run python scripts/01_build.py \
     --diagnostics-out artifacts/views_check_build_diagnostics.json
   ```
5. **Compare against `data/build`:** spells, regular-season spells, event rate,
   censor reasons, panel rows, hazard rows, feature columns.
6. **Decide the panel construction.** Two viable paths:
   - accept the ~0.8% regular-spell difference and document it; or
   - change panel construction to use the cleaned `nba.team_roster` and schedule
     instead of box DNP rows, then rerun parity. This matches the new model
     direction and is the preferred fix.
7. **Do not switch the production snapshot** until step 6 is decided and the
   parity result is accepted.
8. If accepted, replace the production snapshot with the view snapshot, rebuild,
   and re-run the 2025-26 test. Do not refit or publish until that passes.
9. **Then** move to the label branch and finish the report-gap/trade work.

## Important

- No new `player_box_score_played_vw`. Use `nba.player_box_score_vw` with
  `min > 0`.
- The 29 cross-team extras are accepted for training (0.007% of played keys).
- The WIP `spells.py` label edit is stashed on `injury-label-fix`; do not mix it
  into this branch.
- 2026-27 remains out-of-sample; do not tune on it.

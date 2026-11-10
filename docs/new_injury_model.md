# Injury Duration Model v2 — Proposal

Status: draft for discussion, 2026-11-10.

This document describes the proposed replacement for the current injury-duration
model. It changes the **data contract**, the **label construction**, and the
**modelling approach**. The goal is not a different booster; it is a cleaner
supervision target and a model that separates injury recovery from rotation and
team decisions.

The current model is the baseline, not the thing to preserve:

| Current result | Value |
|---|---:|
| 2025-26 held-out C-index | 0.744 |
| 2025-26 median MAE | 2.67 games |
| Fresh injury, rotation player C-index / MAE | 0.701 / 2.03 |
| Next-game AUC | 0.867 |
| 2026-27 live next-game AUC | 0.866 |
| 2026-27 live 7-day Brier skill | 0.317 |
| 2026-27 live resolved spells | 77 |

The live number is encouraging but early: 60% of the current-season spells are
still censored, and the model is slightly pessimistic on short absences.

---

## 1. Design principles

1. **Views are the data contract.** The model reads the read-only view layer,
   not raw tables. Raw tables stay private.
2. **Box score is authoritative for who played.** The injury report is
   authoritative for who was unavailable. Roster tables supply movement dates,
   not participation.
3. **Null dates have meaning.** In `nba.team_roster`, `entry_date IS NULL` means
   "already on the roster at the start of the season"; `exit_date IS NULL` means
   "still on the roster at the end of the season". Do not fill these in the raw
   table; the views coalesce them to the actual season window.
4. **No future information.** Every feature at prediction date *d* uses only
   games and reports whose date is on or before *d*.
5. **A distribution, not a point estimate.** Report P(return next game),
   P(return within 7/14/30 days), P(out past 20 games), and quantiles. A single
   median date is not enough for long absences.
6. **Evaluate the production path.** Daily as-of predictions, not only
   per-spell onset predictions.
7. **2026-27 remains out-of-sample.** It is used for monitoring, not for
   feature choice, label-rule choice, calibration fitting, or hyperparameters.

---

## 2. New data sources

### 2.1 Read-only view layer

| Source | Role |
|---|---|
| `nba.injuries_vw` | Per-game injury report: status, free-text reason, team, game. The only source of absence. Adds season/season type and conformed names. |
| `nba.league_game_schedule_vw` | Canonical team-game calendar. Used for the panel, schedule density, team context, and home/road. |
| `nba.player_box_score_vw` | Existing roster-expanded view. Use only with `WHERE min > 0` and deduplicate on `(game_id, player_id)` for the played source. No new played-only view. |
| `nba.player_info_vw` | Player attributes (birthdate, height, weight, position, draft). Mostly for context; the current model shows these are largely deletable. |
| `nba.teams_vw` | Team id/slug, conference, division, active flag. Conference is required for standings. |
| `nba.team_roster` | Clean player-team stints with the null-date convention. Used for rotation context and roster-movement history. |
| `nba.team_roster_vw` | View over `team_roster` with null dates coalesced to the actual schedule window. |
| `nba.daily_standings_vw` | Deterministic daily regular-season standings: wins, losses, games left, win pct, conference rank, games behind. Elimination/clinch and playoff-hope simulation stay in Python. |
| `util.player_id_map_vw` | NBA/Statyx/ESPN/Yahoo id crosswalk and conformed names. |
| `util.unmatched_player_source_vw` | Crosswalk health check. Must remain at zero rows. |

### 2.2 Supporting sources, not model features

| Source | Role |
|---|---|
| `nba.transaction_log` | Reconstruct player-team movement dates for the roster backfill. It has no `player_id`/`season`, so it is name-matched and used only for dates. |
| `statyx.team_roster` | Cross-check for the 2025-26/2026-27 open/closed spell pattern. Not a historical training source. |
| `nba.key_dates` | Season boundaries only. The view layer now uses the actual schedule minimum/maximum, because `key_dates` is one day late for some seasons. |

### 2.3 Sources deliberately retired

- Raw `nba.player_box_score`, `nba.league_game_schedule`, `nba.player_info`,
  `nba.teams`: not read by the model; the view layer replaces them.
- `nba.player_season_stats`: not used by the model. Dropped from the fetch and
  the data-quality audit.
- `nba.injuries_parsed`: not needed. The Python taxonomy is versioned with the
  model; the database map is an independent moving part.

### 2.4 Box-score decision

Do not create a separate played-only view. Use the existing
`nba.player_box_score_vw` with `WHERE min > 0` and deduplicate on
`(game_id, player_id)`.

The view currently has **29 cross-team extra rows** out of **421,584**
regular-season played player-games, or **0.007%**. They are historical
team-abbreviation/join artifacts. That is small enough not to affect model
metrics, so they are accepted for training. If exactness becomes important,
fix the team condition in the existing view rather than adding a new object.

The material risk is the roster-expanded DNP rows: they must be filtered with
`min > 0` so the model does not treat a DNP as a played game.

### 2.5 Remaining view wiring

The view layer is sufficient as a data contract, but the injury fetch still
reads raw tables. Required before switching the model:

1. Update `00_fetch.py`, `16_lean_fetch.py`, and `18_active_spells.py` to the
   view layer:
   - `nba.injuries` -> `nba.injuries_vw`
   - `nba.league_game_schedule` -> `nba.league_game_schedule_vw`
   - `nba.player_box_score` -> `nba.player_box_score_vw`
   - `nba.player_info` -> `nba.player_info_vw`
   - `nba.teams` -> `nba.teams_vw`
   - `nba.player_season_stats` -> removed
2. Map the box-score view correctly:
   - `WHERE min > 0`
   - deduplicate on `(game_id, player_id)`
   - `inj_reason AS comment`
   - cast `player_id` to `BIGINT`
   - keep `team_id` and `game_id`
3. Run a view-only fetch to a scratch directory, build with
   `INJURY_DURATION_DATA=<scratch>`, and compare spell count, event rate, censor
   reasons, panel rows, hazard rows, and feature columns with the raw build.
4. Only then switch the production snapshot and rebuild the model.

No new view is required. The 29 cross-team extras are accepted; the DNP filter
and the column mapping are the necessary work.

---

## 3. Clean roster contract

`nba.team_roster` is now rebuilt as player-team **stints**:

- One row per player-team stint.
- `entry_date NULL` = on the roster at season start.
- `exit_date NULL` = on the roster at season end/open.
- Midseason moves have both dates: old stint closes, new stint opens.
- No overlapping stints for a player-season.
- Box score is authoritative for any player-season with games; existing roster
  rows are kept only for player-seasons with no box appearance.
- Multiple legitimate same-team stints are kept.
- A stable `stint_id` exists on local Postgres.

Current Cockroach state: 10,949 stints, 0 missing box stints, 0 missing
injury-report stints, 0 invalid windows, 0 overlaps. The backfill now adds
stints evidenced only by the injury report, for players traded while injured
who never played for the new team.

The backfill and validation live in
`nba/database/scripts/backfill_team_roster.py`. It is idempotent and supports
`--db-con postgres` / `--db-con cockroach-write` and `--apply`.

### 3.1 View date windows

`team_roster_vw` and `player_box_score_vw` must coalesce null dates to:

- entry: first game date of the season,
- exit: last game date of the season **plus one day** (exclusive end).

Using `key_dates.end_date` with `<` drops the final day of the season. Using
`key_dates.begin_date` drops opening night when the schedule date is one day
earlier. Both are fixed.

### 3.2 Running the backfill

Dry run first; it writes only a scratch candidate and prints validation:

```bash
cd nba/database
uv run python scripts/backfill_team_roster.py --db-con postgres
uv run python scripts/backfill_team_roster.py --db-con cockroach-write
```

Apply only after the dry run reports zero missing box stints, zero invalid
windows, and zero overlaps:

```bash
uv run python scripts/backfill_team_roster.py --db-con postgres --apply
uv run python scripts/backfill_team_roster.py --db-con cockroach-write --apply
```

Then apply the view files and re-grant reads:

```bash
# Cockroach
uv run python scripts/apply_views.py

# Local Postgres: apply the same two files through Database('postgres')
# team_roster_vw.sql, player_box_score_vw.sql
```

Verify `nba.player_box_score_vw` against the played box score before switching
the model fetch.

---

## 4. Label construction

### 4.1 Target

The target is **return to play**: a game with minutes > 0. A player listed
Available but DNP is not a return.

### 4.2 Spell fixes

1. **Bridge report gaps to the next played game.** A short unreported run or an
   "Available" DNP between a health absence and the next played game is part of
   the same absence. Max gap: 3 games. This recovers roughly 405 spells in the
   historical build.
2. **Continue across trades.** A player traded while injured keeps one spell,
   not a censored old-team spell plus a new-team spell. Use player id, not team.
3. **Keep censoring.** Season end, G League/roster departure, suspension, and
   personal leave remain censored. They are competing events, not recoveries.
4. **Do not infer roster dates from the last game.** A player injured at the end
   of the season is still on the roster. Null exit is correct; a last-game date
   is not a departure.

### 4.3 Spell frame

- One row per missed game `k` within a spell.
- `returns_next = 1` only on the last missed game of an observed return.
- Censored spells contribute their "still out" rows and no positive row.
- Keep onset and in-progress views of the same spell.

---

## 5. Modelling approach

### 5.1 Two-stage: injury vs rotation

The current model conflates "injured" with "not in the rotation". The honest
fresh-and-rotation subset is C-index 0.701 vs 0.744 over all spells.

Proposed:

- **Stage 1 — availability/recovery:** P(available for the next game) from
  injury taxonomy, elapsed time, schedule, and report state.
- **Stage 2 — play conditional on availability:** P(plays | available) from
  rotation status, recent minutes, starts, and team context.
- **Combined:** P(plays) = P(available) × P(plays | available).

A cheaper first experiment is an explicit established-rotation indicator
(start rate over the last 10, minutes rank, games share) and fitting/reweighting
on rotation players.

### 5.2 Hazard model

Keep the discrete-time hazard framing; do not regress directly on observed
duration.

- Estimator: LightGBM hazard (plus HistGB/seed ensemble).
- Monotonic constraints: hazard non-increasing with games/days missed; increasing
  with days to the next game.
- Feature blocks:
  1. elapsed time and schedule,
  2. injury taxonomy (region × pathology × severity),
  3. recent workload and rotation,
  4. report-state trajectory,
  5. team context.
- Prune aggressively. The ablations show player attributes, prior injury
  history, team incentive, and season/team context are largely dead weight.
  Target 12–30 features, not 78.
- Keep team incentive out of the prediction model; if needed, use it for
  interpretation/de-biasing only.

### 5.3 Long tail and competing risks

The current hazard has to fit one-game absences and season-ending injuries in one
curve. Add:

- **Mixture cure / long-absence component:** estimate the probability that the
  spell is long or season-ending, plus a duration model conditional on that.
  This gives a plateau and a meaningful P(no return this season).
- **Competing risks:** return vs season end vs trade/G League/shutdown. Report
  cause-specific probabilities rather than treating all censoring as
  independent.
- For catastrophic injuries, never report only a median. Report P(out past 20),
  P(no return this season), and the survival curve.

### 5.4 Calibration

- Fit a calibration layer on out-of-time predictions, by horizon (1/7/14/30) and
  elapsed bucket.
- Use isotonic/Platt with shrinkage; fit on 2024-25, evaluate on 2025-26.
- For production, refit on 2021-26 and monitor 2026-27.
- Keep a frozen shadow model if an adaptive calibration is used, so the live
  benchmark remains clean.

### 5.5 Report-state features

The live report block is worth about 1.2 AUC on the next-game call. Extend it:

- consecutive games at the current status,
- number of status changes,
- days since the last change,
- upgrade/downgrade direction,
- whether the reason text or body part changed,
- whether surgical wording appeared later.

Optional: a TF-IDF/embedding feature over the free-text reason, tested as an
add-on to the rule-based taxonomy.

### 5.6 Outputs

- P(plays next game)
- P(return within 7 / 14 / 30 days)
- P(out past 20 games)
- P(no return this season)
- median/mean games and dates, with quantiles
- confidence/entropy or ambiguity flag

---

## 6. Evaluation

### 6.1 Splits

| Stage | Seasons | Use |
|---|---|---|
| Train | 2021-22 → 2023-24 | fit |
| Validation | 2024-25 | label rule, hyperparameters, calibration |
| Test | 2025-26 | final untouched historical test |
| Production refit | 2021-22 → 2025-26 | bundle deployed to 2026-27 |
| Live / out-of-sample | 2026-27 | daily backfill, monitoring only |

No random splits. No 2026-27 tuning.

### 6.2 Daily backtest

The current validation is per-spell at onset; production is daily. Build a
rolling-origin daily backtest for 2021-26 using the exact `daily` /
`forecast_daily` path. This gives five seasons of as-of predictions and is the
only fair way to tune the live model and calibration layer.

### 6.3 Metrics

- Severity ranking: Harrell C-index (all spells; fresh-and-rotation subset).
- Point accuracy: MAE/MedAE/bias on observed spells, plus long-spell MAE (≥5).
- Next-game: AUC, Brier, log loss, calibration.
- Horizons: Brier skill at 7/14/30 days; calibration by decile.
- Uncertainty: cluster bootstrap by spell.
- Baselines: KM global / by ailment / by region×ailment×status; current model.

### 6.4 Promotion gate

1. Better than the current model on the 2025-26 test: C-index, MAE, and 7/14-day
   Brier skill on the corrected labels.
2. No worse on long spells (5+ games), the fresh-and-rotation subset, or
   calibration slope.
3. Identical row coverage, no null keys, no duplicate keys, runtime within the
   daily window.
4. Shadow run on 2026-27 for 2–4 weeks. The shadow is a sanity check, not a
   selection metric; 77 resolved spells cannot choose a model.

---

## 7. Pipeline

### 7.1 Stages

1. **Fetch** — views → Parquet snapshot. Read-only role only.
2. **Build** — panel, spells, hazard rows, features, team context.
3. **Fit** — hazard model(s), rotation stage, calibration.
4. **Validate** — time splits, daily backtest, promotion gate.
5. **Publish** — daily forecasts to `mod.injury_daily_forecast` (or a versioned
   base table).
6. **Monitor** — rolling Brier/AUC/C-index by month, injury family, elapsed
   bucket; drift alerts.

### 7.2 Deployment

- Local Postgres first: rebuild, validate, and compare before touching Cockroach.
- Cockroach second: same idempotent backfill and view definitions.
- Archive the model bundle and predictions by version; keep a rollback path.
- No force-push; no in-place overwrite of live predictions without a versioned
  base table or a shadow view.

---

## 8. Work plan

### Phase 0 — Data layer

- [x] `league_game_schedule_vw` rename.
- [x] `teams_vw` conference/full mirror.
- [x] `daily_standings_vw` deterministic view.
- [x] Remove `player_season_stats` from the fetch/audit.
- [x] Use existing `player_box_score_vw` with `WHERE min > 0`; no new view.
- [x] Clean `nba.team_roster` locally and on Cockroach.
- [x] Fix `team_roster_vw` / `player_box_score_vw` season windows.
- [x] Apply view fixes on Cockroach and verify `kobe_public`.
- [x] Fix `scs_hub.nba.get_team_roster` (first snapshot, departures); no test
  suite exists in `scs_hub` yet.
- [x] Add `backfill_team_roster.py` to the repo and document it.
- [ ] Switch `00_fetch.py`, `16_lean_fetch.py`, and `18_active_spells.py` to the
  view layer.
- [ ] Map `player_box_score_vw`: `WHERE min > 0`, deduplicate, `inj_reason AS
  comment`, cast `player_id` to `BIGINT`.
- [ ] Run the view-only fetch/build parity check against the raw build.
- [ ] Switch the production snapshot only after parity passes.

### Phase 1 — Labels and rebuild

- [ ] Bridge report gaps / Available DNP to next played game.
- [ ] Continue spells across trades.
- [ ] Rebuild 2021-26 and compare old vs new labels.
- [ ] Re-run the 2025-26 test on corrected labels.

### Phase 2 — Daily backtest

- [ ] Backfill daily predictions for 2021-26.
- [ ] Measure as-of calibration and error by horizon/elapsed.
- [ ] Choose the calibration approach on 2024-25; test on 2025-26.

### Phase 3 — Rotation-aware model

- [ ] Two-stage availability/play model or explicit rotation feature.
- [ ] Evaluate on the fresh-and-rotation subset.
- [ ] Compare to the current hazard on the same labels.

### Phase 4 — Long tail

- [ ] Mixture-cure or competing-risk hazard.
- [ ] Long-spell MAE/C-index and P(no return this season).
- [ ] Catastrophic-injury reporting.

### Phase 5 — Calibration and ensemble

- [ ] Isotonic/Platt by horizon and elapsed bucket.
- [ ] LightGBM + HistGB/seed ensemble.
- [ ] Report quantiles and ambiguity.

### Phase 6 — Shadow and deploy

- [ ] Shadow predictions for 2–4 weeks.
- [ ] Promotion gate.
- [ ] Versioned publish and rollback.
- [ ] NUC image rebuild only if `scs_hub` changes.

---

## 9. Risks and open questions

- **Roster source conflicts.** Box score, transaction log, roster snapshots, and
  `how_acquired` do not always agree. The backfill uses box score as the
  authority for played player-seasons; this is a deliberate trade-off.
- **Transaction-log identity.** No `player_id` or `season`; name matching is
  imperfect, especially for historical and non-player rows.
- **Historical team abbreviations.** Old team names (NOH, NJN, etc.) can still
  produce cross-team view extras. The raw roster is clean; the view needs a
  separate team-normalization pass.
- **2026-27 schedule.** The current schedule has 80 regular-season games per
  team; `season_games` is pinned to 82 for 2021-22 onward. Do not infer the
  season length from the current schedule.
- **Small live sample.** 2026-27 has 77 resolved spells; it can detect drift,
  not select a model.
- **Dashboard impact.** Roster rows and `player_box_score_vw` coverage change;
  re-run and compare the dashboard data before deploying.

---

## 10. Definition of done

- Model reads only the view layer; no raw `nba.*` table outside views.
- All three fetch scripts use the view layer, and the box view is filtered to
  `min > 0` and deduplicated.
- The existing `player_box_score_vw`, filtered to `min > 0` and deduplicated,
  covers all played player-games.
- `team_roster` has clean stints, no overlaps, and the null-date pattern.
- 2025-26 test passes the promotion gate on corrected labels.
- Daily 2021-26 backtest shows calibrated as-of predictions.
- 2026-27 shadow run is stable; live predictions are versioned and reversible.
- A runbook documents refit, backfill, rollback, and monitoring.

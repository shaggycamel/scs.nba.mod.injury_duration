# Data defects in the `nba` Parquet snapshot, and how this repo works around them

Written as a handover. Everything below was found while building
`src/injury_duration/` (injury-duration modelling) against the `nba` Parquet
snapshot, produced by `scripts/00_fetch.py` from `[cockroach-read]` from the
`nba` schema on CockroachDB.

**Partly fixed at source (2026-11-07).** A1, A3, A5, A6, B2 and D1 were
corrected in the `nba` tables in both Postgres and CockroachDB; the code
workarounds remain and are now no-ops. A2 is partly fixed: the accent-spelled
crosswalk names now resolve, leaving 7 rows across 3 names. Everything else is
still worked around in code.

To see the current state of every item:

```sh
uv run python scripts/10_data_quality_check.py
```

That script is the executable version of this document. Re-run it after any
upstream change; the counts below are its output as of 2026-11-07.

---

## Severity summary

| # | Defect | Rows | Severity | Handled |
|---|---|---|---|---|
| A5 | 35 injury rows dated on the schedule's wrong clock | 0 | low | **fixed at source** |
| A6 | 17 injury rows 21 days from their own `game_id` | 0 | low | **fixed at source** |
| B2 | 2021-22 schedule dates are one day early | whole season | **critical** | **fixed at source** |
| C1 | Box score omits absent players entirely | — | **critical** | `spells.py` |
| A2 | `nba_id` null on 18.1% of injury rows | 11,408 | high | `ids.py` |
| A1 | Exact duplicate injury rows from 2026-02-20 | 0 | high | **fixed at source** |
| A7 | `reason` is free text, 2,858 distinct strings | all | high | `taxonomy.py` + `nba.injury_reason_map` |
| A4 | `game_id` null on 19.1% of injury rows | 12,042 | medium | `spells.py` |
| A3 | Concatenated `status` values | 0 | low | **fixed at source** |
| B1 | Non-NBA teams in the schedule | 736 | medium | `calendar.py` |
| C2 | `min` null for DNP rows | 105,323 | low | `spells.py` |
| D1 | `player_info` covered two seasons; string birthdate | — | medium | **fixed at source** |

---

## A. `nba.injuries`

### A1. Exact duplicate rows from 2026-02-20 onwards — 1,406 rows

**Fixed at source 2026-11-07:** 1,406 duplicates were deleted from both
Postgres and CockroachDB, and `get_team_injuries()` now deduplicates its frame
before writing. The code workaround remains as a guard.

Every row filed on or after 2026-02-20 appears two to six times, byte for
byte. 478 (team, date, player) keys are affected.

```python
inj = pl.read_parquet("data/nba/injuries.parquet")
inj.height - inj.unique().height          # 1406
```

They are true duplicates — identical `status`, `reason` and `matchup` — so
`.unique()` is lossless and leaves zero residual duplicate keys. Looks like a
re-ingestion that appended instead of upserting; worth checking the loader's
write mode around that date.

**Impact if ignored:** a player's absence is counted up to six times, which
inflates every per-game aggregate and double-counts spells.

**Handled in:** `spells.load_report()` — `.unique()` before anything else
(now a no-op; the source is clean).

### A2. `nba_id` is null on 18.1% of rows — 11,408 rows

Null rate varies by season, from 2% in 2024-25 to 38% in 2021-22, so a naive
`drop_nulls` would silently delete a third of the earliest season.

Worse, some `player_name` values are **two players' names interleaved**:

```
"Luka Reaves Doncic"        (Luka Doncic + Austin Reaves)
"Alex Mitchell Caruso"      (Alex Caruso + Davion Mitchell)
"Marvin Coulibaly Bagley III"
```

**Handled in:** `ids.resolve_player_ids()` — a four-step ladder: the existing
`nba_id`, then exact `player_name` against the `util` crosswalk, then against
the box score, then a punctuation- and suffix-stripped form of each. Names
that map to more than one id are dropped from the lookup rather than guessed.

**Residual:** 7 rows across 3 names survive all four steps and are dropped:
`Kyle Mangas` (5, G League), `Jeenathan Williams` (1, roster) and `David Jones`
(1, G League). The first two have no NBA id in any local source; David Jones is
the shortened form of "David Jones Garcia" and would need an alias. None of the
three starts an injury spell.

**At source 2026-11-07:** the accent-spelled players now resolve. The report
writes ASCII ("Nikola Djurisic") while the crosswalk keeps "Nikola Đurišić";
`get_team_injuries` folds both sides, and 130 existing rows were backfilled with
their `nba_id`. The 25 interleaved/doubled report rows were re-ingested from the
archived PDFs and are gone.

### A3. Concatenated `status` values — 21 rows

**Fixed at source 2026-11-07:** the merged rows were re-ingested from the
archived reports (2026-01-27..31) and are gone; `get_team_injuries()` now
deduplicates its frame. The parser still takes the first whitespace token as a
guard.

```
"Out Out", "Out Out Out", "Questionable Doubtful Out", "Available Out Out"
```

Same root cause as A2 — adjacent report rows merged. Note these are *not* the
same rows as A2: a merged row can have a clean name and a doubled status.

**Handled in:** `spells.load_report()` takes the first whitespace token.
Defensible because the first token is the one belonging to the named player.

### A4. `game_id` is null on 19.1% of rows — 12,042 rows

Nearly all of 2021-22. **This one is a trap**: the column has the same name
as `league_game_schedule.game_id`, so a naive join silently keeps the
injury-side null and drops the whole season. That cost a debugging session
here — see `spells.load_report()`, which renames it to `report_game_id` for
exactly this reason.

**Handled in:** `spells.build_panel()` maps report rows to games on
`(team_slug, report_date)` instead, which is unique because no team plays
twice in a day.

### A5. 35 rows dated on the schedule's wrong clock

**Fixed at source 2026-11-07:** all 35 rows were re-dated from their
`report_game_id` to the schedule's `game_date`; the 9 duplicate HOU@POR rows
that already sat on the correct date were removed. `scripts/10` reports 0.

Background: four 2021-22 games were filed one day early — the same off-by-one
that B2 described before the schedule was corrected — and they were precisely
the 2021-22 rows carrying a non-null `game_id`. The giveaway was HOU@POR
appearing under both dates:

```
HOU@POR  2022-03-25: 11 rows, all 11 carrying a game_id   <- wrong clock
         2022-03-26:  9 rows, none carrying a game_id     <- correct, subset
```

The other three (NOP@UTA 2021-11-26, MIN@SAC 2022-02-08, TOR@BKN 2022-02-28)
appeared only on the wrong date. The code-side alternative — preferring
`report_game_id` in `spells.build_panel()` — is still a good defensive change
but is no longer needed for these rows.

### A6. 17 rows 21 days away from their own `game_id`

**Fixed at source 2026-11-07:** the 17 stale rows for the postponed MIA@CHI
game (`22500529`) on 2026-01-08 were deleted; the 23 rows from the game's
actual report on 2026-01-29 remain. `scripts/10` reports 0.

### A7. `reason` is unvalidated free text

2,858 distinct strings for what is really a few dozen clinical categories.
The intended shape is

```
Injury/Illness - <Side> <Body part>; <Ailment>
```

but in practice:

* **926 rows** are blank — `null` or a bare `"-"`.
* **6 rows** have the team name glued to the front:
  `"Portland Trail Blazers Injury/Illness - Abdomen; Tendinopathy"`.
* Casing is inconsistent for the same concept: `Injury Recovery` /
  `Injury recovery`, `N/a` / `N/A`, `Bone Bruise` / `Bone bruise`.
* The same pathology is spelled many ways: `Tendinitis`, `Tendonitis`,
  `Tendinopathy`, `Tendinosis`.
* Compound entries are common: `"Abdominal; Strain/Left Ankle Sprain"`.

**Handled in:** `taxonomy.py`, which parses `reason` into `reason_category`,
`body_region`, `body_side`, `ailment_class` and a set of severity flags, with
ordered keyword rules. After parsing, 1.2% of injury rows have an unknown
body region and 2.7% an unknown ailment.

**At source (2026-11-07):** `nba.injury_reason_map` holds the parsed form of
every distinct reason and `nba.injuries_parsed_vw` joins it to the report, so
consumers can select structured columns without re-parsing. Refresh it with
`uv run --group data python scripts/11_refresh_reason_map.py`; pass
`--db-con postgres` for the local copy.

**One ordering trap worth knowing:** the literal prefix `"Injury/Illness"`
contains the word *illness*, so a rule list that tests for illness before
injury classifies every structured row as an illness. The injury/illness
split has to come from the payload, not the prefix.

---

## B. `nba.league_game_schedule`

### B1. Non-NBA teams — 736 rows, 43 team codes

Preseason fixtures against international and G-League sides: `ADL`, `ALB`,
`BAR`, `BAU`, `BJD`, `BNE` and others. Harmless but they break any assumption
that the file contains 30 teams.

**Handled in:** `calendar._raw_schedule()` filters to the 30 active franchises
from `nba.teams` and drops `season_type == "Pre Season"`.

### B2. 2021-22 dates are one day early — whole season, **critical**

**Fixed at source 2026-11-07:** every 2021-22 row in `league_game_schedule`
was shifted forward one day in both Postgres and CockroachDB. The measured
offset is now 0 for every season.

This was the big one. Every 2021-22 row in `league_game_schedule` was dated one
day before the game actually happened. Later seasons are correct.

Three independent confirmations:

1. **Christmas.** The NBA plays five games on 25 December and none on the
   24th. This file has **five games on 2021-12-24** and eight on the 25th.
   From 2022-23 on it correctly shows five on the 25th and none on the 24th.
   `nba.injuries` has it right for 2021-22: five matchups on 25 December.
2. **All-Star.** The 2022 All-Star Game was Sunday 20 February 2022. The file
   says 2022-02-19. The 2023 and 2024 entries are correct.
3. **Opening night.** The 2021-22 season opened on Tuesday 19 October 2021
   with BKN@MIL and GSW@LAL. `nba.injuries` says 2021-10-19;
   `league_game_schedule` says 2021-10-18 for both `game_id`s.

A fourth, incidentally: Klay Thompson returned on 2022-01-09 and the box
score dates that game 2022-01-08.

Also check pre-2021 seasons if you use them — 2009-10 opens on 2009-10-26 in
this file against a real 2009-10-27, so the defect is probably not confined
to 2021-22. It could not be verified here because the injury report only
starts in October 2021, and nothing else in the snapshot provides an
independent date anchor.

**Impact if ignored:** injury-report rows fail to join to games for an entire
season, and any day-level feature (rest days, back-to-backs, schedule
density) is wrong for that season.

**Handled in:** `calendar.report_date_offsets()` **measures** the offset per
season rather than hard-coding it — for each season it scores candidate
offsets of −1, 0 and +1 by how many `(team, date)` pairs align between the
report and the schedule, and takes the best. Current output:

```
2021-22: offset +0  matches 2476/2634   (alternatives {-1: 414, 1: 322})
2022-23: offset +0  matches 2465/2628
2023-24: offset +0  matches 2457/2624
2024-25: offset +0  matches 2447/2628
2025-26: offset +0  matches 2535/2630
```

The separation is wide enough that this is not a coin flip. The corrected
date lands in a **new** column, `report_date`; `game_date` is left untouched,
because a constant per-season offset cancels in any within-season difference
and only matters for joining.

**Because this was fixed at source**, the measured offset is 0 and the code
keeps working with no change. The pre-2021 seasons were not verified — 2009-10
also looks one day early, and nothing in the snapshot provides an independent
date anchor for it.

---

## C. `nba.player_box_score`

### C1. The box score does not record absence — **critical, semantic**

A player on a long-term absence is simply **missing from the file**. There is
no DNP row, no zero-minute row, nothing.

```
Golden State played 38 games while Klay Thompson was out in 2021-22.
Klay Thompson box-score rows for those 38 games: 0
```

His first 2021-22 regular-season row is his return game.

**Impact if ignored:** games missed cannot be counted from the box score, and
any attempt to do so silently produces zero for exactly the players who were
hurt longest. This is the single most important thing to know about the
snapshot.

**Handled in:** `spells.build_panel()` derives availability from the injury
report and uses the box score only to establish that a player *did* play
(`min > 0`).

### C2. `min` is null on 17.9% of rows — 105,394 rows

These are DNP rows for players who were active but did not play; `comment`
carries the reason (`"DNP - Coach's Decision"`, `"DND - Injury/Illness"`).
Null means did-not-play, not missing data.

**Handled in:** filled with `0.0`, and "played" is defined as `min > 0`.

---

## D. Other tables

### D1. `nba.player_info`

**Fixed at source 2026-11-07:** the table now covers all five analysis
seasons (2021-22: 603 rows, 2022-23: 540, 2023-24: 475, 2024-25: 5,065,
2025-26: 565) and `birthdate` is typed `date`, not text. Each player's latest
bio was replicated into the seasons they played; only `RaiQuan Gray` needed a
fresh fetch.

**Handled in:** `features.player_attributes()` takes the most recent row per
player and uses only genuinely time-invariant fields (birthdate, listed
height and weight, draft position), computing age per spell from the
birthdate. The column cast in that function tolerates either a date or a
string, so older snapshots still load.

### D3. `util.unmatched_player_source_vw`

Currently **0 rows**, the healthy state — a non-empty file means the
NBA/Statyx crosswalk has gaps that will silently drop rows from any
cross-source join. Worth asserting on, not just inspecting.

---

## E. Pitfalls in the processing code, not the data

Three bugs that cost real debugging time here and are easy to reintroduce.
All are fixed in this repo; they are listed because they are properties of
polars and of this data shape, not of this project.

1. **`group_by` order is not guaranteed.** `standings.playin_probability()`
   draws from a seeded generator inside a loop over groups. With an unstable
   group order the simulation was irreproducible, and so was every model
   downstream of it. Iterate over sorted keys.
2. **`sort` is not stable on ties.** `(player_id, season, game_date, game_id)`
   looks unique but is not: a player traded between two teams that then play
   each other is attached to **both sides of one game**. 44 rows in this
   snapshot. The whole per-player game index shifted between runs. Add
   `team_slug` as a final tiebreaker.
3. **`group_by().agg(pl.col(x).first())` does not mean "the first row".** It
   makes no promise about within-group order. Where it meant "as filed on the
   first missed game", it has to be written
   `pl.col(x).sort_by("player_game_idx").first()`.

With all three fixed the pipeline reproduces byte-identically, apart from one
float's last bit in a random forest metric, which is parallel summation order
in scikit-learn.

---

## F. What to do with this

If you are fixing the upstream extract, in priority order:

1. **B2** — correct the 2021-22 (and probably earlier) schedule dates. Check
   every season against the Christmas and All-Star anchors.
2. **A1** — stop the duplicate writes from 2026-02-20.
3. **A2 / A3** — fix the scraper's row splitting, which is producing both the
   interleaved names and the concatenated statuses.
4. **A5** — re-date the 35 rows, or just drop their `game_id` so they stop
   looking authoritative.
5. **A7** — a validated `reason` at source would retire most of
   `taxonomy.py`, but the parser is tested and in no hurry.

**C1 is not a defect to fix** — it is how NBA box scores work. It is listed
because any new work against this snapshot needs to know it.

After any change, run `scripts/10_data_quality_check.py` to see what moved,
then the full pipeline (`scripts/01_build.py` … `09_sync_report_numbers.py`),
which regenerates the reports and re-syncs their numbers automatically.

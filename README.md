# injury-duration

How long NBA injuries keep players out: injury spells, censoring-aware
survival models, and the team decisions that move a return date.

Extracted from the `nba_adhoc` injury-duration branch at commit
`d9f485a092016870cbc99c52e7a15d050afd63c3` (`injury-mod`). The source branch
is archived in that repository as tag `archive/injury-mod-2026-11-07`.

## What it does

Modelling the *duration* of an injury absence from the per-game injury report
(`nba.injuries`), 2021-22 to 2025-26. Two questions:

1. **At onset** — a player has just been ruled out. How many games will they
   miss?
2. **In progress** — they have missed *k* games. Do they play the next one?

A third section asks how much of an absence is the injury at all, and how much
is the team's situation.

Findings: [`docs/injury_duration.md`](docs/injury_duration.md) (narrative) and
[`docs/injury_duration.html`](docs/injury_duration.html) (self-contained,
charts, opens offline from a clone).

## Design rules

- **Absence comes from the report, not the box score.** A player on a long
  absence is simply missing from the box score, so games missed are counted
  from the per-game injury report instead.
- **Censored spells stay in.** 22% of spells never show a return because the
  season ends or the player is traded or sent down. They are kept as
  right-censored rather than dropped; dropping them cuts the mean absence from
  11.2 games to 3.6.
- **Time-based splits only.** Fit on 2021-22..2023-24, tune on 2024-25, refit,
  score 2025-26. Nothing is shuffled.
- **A distribution, not a point estimate.** One discrete-time hazard fit gives
  P(back next game), P(back within k games) and expected games missed.
- **Every feature is known before tip-off.**

## Install

```bash
uv sync --group dev
```

## Data

The Parquet snapshot is not committed. Point `INJURY_DURATION_DATA` at a
directory containing `nba/`, `statyx/` and `util/`:

```bash
export INJURY_DURATION_DATA=/path/to/nba_adhoc/data
```

Every loader also accepts an explicit `data=` argument. If the environment
variable is unset, the default is `data/` in the repository root. The snapshot
is produced by `data/gen.py` in the `scs.nba.mod.scratch` repository.

The model needs these files in particular:

```
nba/injuries.parquet
nba/league_game_schedule.parquet
nba/player_box_score.parquet
nba/team_box_score.parquet
nba/player_info.parquet
nba/teams.parquet
nba/team_roster.parquet
util/player_id_map_vw.parquet
```

`docs/data_quality.md` documents every defect found in the snapshot and where
each is worked around. Read it before doing new work against the data — in
particular, the box score does not record absence at all, and the 2021-22
schedule dates are one day early.

## Reproduce

```bash
uv run python scripts/01_build.py                # report + box score -> injury spells
uv run python scripts/02_describe.py             # Kaplan-Meier duration by injury type
uv run python scripts/03_models.py               # baselines vs hazard models, time splits
uv run python scripts/04_explain.py              # ablations, importance, partial dependence
uv run python scripts/05_forecast.py             # forecasts for the held-out season
uv run python scripts/06_incentive.py            # does the team's playoff position move returns?
uv run python scripts/07_debias.py               # separate injury duration from team decision
uv run python scripts/08_report_html.py          # standalone HTML report with charts
uv run python scripts/09_sync_report_numbers.py  # sync the Markdown tables to the CSVs
uv run python scripts/10_data_quality_check.py   # audit the source snapshot for known defects
```

`01_build.py` writes to `data/build/` (git-ignored); every later script reads
from there. Summary tables land in `reports/`; the rendered narrative is
written to `docs/`.

## Layout

```
src/injury_duration/
  taxonomy.py    # free-text reason -> region / pathology / severity flags
  ids.py         # resolve the report's nullable nba_id
  calendar.py    # canonical team-game calendar, report-date alignment
  spells.py      # availability panel -> spells, with censoring
  features.py    # pre-tipoff features, grouped into blocks for ablation
  hazard.py      # hazard rows, Kaplan-Meier, survival arithmetic, prediction grid
  design.py      # polars -> numpy encoding (no pandas)
  models.py      # baselines, hazard models, censoring-blind contrast
  evaluate.py    # C-index, horizon metrics, calibration
  experiment.py  # feature sets and design matrices, shared by every script
  standings.py   # daily standings, elimination dates, simulated playoff hope
  forecast.py    # usable forecasts from a fitted model
scripts/         # the pipeline, in order
reports/         # generated tables and diagnostics (CSV / JSON)
docs/            # rendered narrative and data-quality handover
```

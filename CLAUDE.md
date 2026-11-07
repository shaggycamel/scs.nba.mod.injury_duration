# scs.nba.mod.injury_duration

Injury-duration research against the `nba` database on CockroachDB Cloud.

## Purpose
Predict **how long an injury keeps an NBA player out**, using only information
known before tip-off.
- Games missed come from the per-game injury report, `nba.nba.injuries`. The box
  score does not record absence: a player on a long absence is simply missing
  from it.
- Build availability spells, keep censored spells (season end, trade,
  suspension, G League) rather than dropping them, and model duration with a
  discrete-time hazard so every spell is usable.
- Answer two questions: at onset ("just ruled out — how many games?") and in
  progress ("missed *k* games — do they play the next one?").
- Report distributions, not only point estimates. The distribution is brutally
  skewed: the median absence is 1-2 games, but spells of 10+ games account for
  most of the games missed.
- Separate medical duration from team discretion. Playoff position changes what
  a team wants from a borderline player; it cannot change how a torn ligament
  heals. Use that asymmetry.
- Evaluate with time-based splits only, never random shuffles, and compute
  every feature from prior games to avoid leakage.
- Always compare against simple baselines (Kaplan-Meier by region × ailment,
  league base rates) before claiming a model helps.
- C-index is the fairest single column for duration: it uses censored spells
  and does not depend on the choice of point estimate.

## Stack
- Python only, managed with `uv`. Python version is pinned in `.python-version`.
- Add dependencies with `uv add`; run code with `uv run`. Don't use `pip` directly.
- Don't add R. Do analysis in Python.
- Use `polars` for data analysis. Avoid `pandas` unless a library requires it;
  convert at the boundary and say why.

## Data
- The Parquet snapshot is not committed. Recreate it with
  `uv run --group data python scripts/00_fetch.py`; it reads
  `[cockroach-read]` and freezes the window at 2025-26. Point
  `INJURY_DURATION_DATA` at another copy to override the default `data/`.
- `docs/data_quality.md` documents every known defect in the snapshot and where
  it is worked around. Read it before new work; `scripts/10_data_quality_check.py`
  is the runnable version.
- Never commit raw data exports or large files.

## Persisted model
- `scripts/12_fit_model.py` fits the onset and live hazard models on every
  season in the build and writes `artifacts/injury_duration/model.pkl` plus
  `model.json`. The directory is git-ignored; rebuild it after a data refresh.
- `persist.load()` checks the package versions recorded at fit time and refuses
  a mismatch unless `strict=False`. Keep scikit-learn / LightGBM pinned in
  `uv.lock` when a bundle is in use.
- `scripts/14_backfill_daily.py` produces as-of daily return-date forecasts for
  one season. Fetch a separate current-season snapshot (`00_fetch --through
  2026-27 --out data/current`), build with
  `INJURY_DURATION_DATA=data/current` and pass
  `--diagnostics-out artifacts/current_build_diagnostics.json` so the frozen
  report diagnostics are not overwritten. Report rows are used only once their
  game date has arrived; no future report state is allowed into a prediction.
- The scheduled path is root `__main__.py` → `scripts/job.py`: active spells →
  lean fetch → build → `19_daily_predict.py` → `15_publish_daily.py --append`.
  It uses `data/active`, `data/lean_current` and `artifacts/` only; no tracked
  file is rewritten on the Jetson.

## Database (CockroachDB Cloud)
- Cluster: `nba-data-mgmt` (id `11af18b7-ef5e-41b2-b3b7-5db438b1d403`),
  database `nba`, schema `nba`.
- **Read-only, always.** Use only `select_query`, `list_*`,
  `get_table_schema`, `explain_query`, `show_statement` and
  `show_running_queries`.
- Never use `create_database`, `create_table`, `insert_rows`, or any DDL/DML,
  even if a task seems to need it. Ask the user to run writes themselves.
- Use only schema `nba.nba`. Ignore schemas `util`, `anl` and `fty`: don't query
  them or use their tables as features.
- Prefer the `*_vw` views (e.g. `nba_injuries_vw`) over raw tables when they
  cover the question.
- Check the schema (`get_table_schema`) before writing queries. Add explicit
  `LIMIT`s and filters on large tables (`player_box_score` is ~590k rows).
- In cloud sessions the SQL port (26257) is blocked. Query through the Cloud
  MCP server at `https://cockroachlabs.cloud/mcp`.
- Locally, connect with `COCKROACH_URL`.

## Secrets
- Never print, log or commit `COCKROACH_URL`, API keys or any credential.
  `credentials.ini` is git-ignored; keep it that way.

## Git
- Work on the session branch; never push to `main`.
- Make small, focused commits with descriptive messages.
- Don't open a pull request unless asked.

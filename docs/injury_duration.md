# How long do NBA injuries keep players out?

Modelling the *duration* of an injury absence from the per-game injury report
(`nba.nba.injuries`), 2021-22 to 2025-26. Two questions:

1. **At onset** — a player has just been ruled out. How many games will they
   miss?
2. **In progress** — they have missed *k* games. Do they play the next one?

Section 7 then asks how much of an absence is the injury at all, and how much
is the team's situation.

Everything below is scored on time-based splits: fit on 2021-22..2023-24, tune
on 2024-25, refit, score 2025-26. Reproduce with
`uv run python scripts/0{1..7}_*.py`.

---

## 1. The dataset, and why it has to be built this way

The box score is **not** a record of availability. A player on a long absence
is simply absent from it — Klay Thompson has no 2021-22 regular-season
box-score row until the night he came back. So games missed cannot be counted
from box scores; the injury report is the only source that says a player was
unavailable, and it says it one game at a time.

The build is therefore: report + box score → a per-(player, game) availability
panel → runs of consecutive missed games → **spells**.

**6,754 spells** from 62,885 report rows. Five data problems had to be fixed
to get there, each of which silently corrupts the result if missed:

| Problem | Effect if ignored |
|---|---|
| `nba_id` null for 2-38% of rows by season | a third of 2021-22 absences vanish |
| 2021-22 schedule dates sit one day early | report rows fail to map to games |
| the report's own `game_id` is null for most of 2021-22 | shadows the calendar's `game_id` on join and drops the season |
| rows from 2026-02-20 on are exactly duplicated | spells double-counted |
| a player listed **Doubtful** who still misses the game | scored as a *return*, truncating 896 spells |

### Censoring is the whole ballgame

**22.3% of spells never show a return**: the season ends, or the player is
traded, suspended or sent to the G League while still out. Those are the long
ones, so how they are handled decides everything:

| | games missed |
|---|---|
| mean of observed spells only (censoring-blind) | **3.6** |
| Kaplan-Meier mean (censoring-aware) | **11.2** |

A three-fold difference. For tears it is worse: 15.6 observed versus **57.6**
by KM. Any analysis that drops or truncates censored spells is describing a
league in which nobody misses 40 games.

The distribution is also brutally skewed. Among spells whose return was
observed: median **1** game, p75 3, p90 **8**, p95 14, p99 **30**, max 72.
Over all spells the KM median is 2. Meanwhile spells of 10 games or more
account for **60.6%** of every game missed. So the typical absence is one
night and essentially all of the cost lives in a thin tail — which is why
ranking severity matters more here than squeezing MAE.

---

## 2. What drives duration

Kaplan-Meier expected games missed, injury spells only (`reports/km_*.csv`):

**By pathology** — this is the single most informative variable:

| ailment | n | KM mean | KM median | P(out past 20) |
|---|---|---|---|---|
| rupture / tear | 54 | 57.6 | 58 | 0.68 |
| surgery | 124 | 38.9 | 21 | 0.51 |
| fracture | 126 | 29.9 | 20 | 0.45 |
| post-op recovery | 98 | 24.2 | 2 | 0.33 |
| dislocation | 44 | 19.6 | 4 | 0.31 |
| tendinopathy | 318 | 12.3 | 2 | 0.14 |
| sprain | 1308 | 11.9 | 3 | 0.14 |
| strain | 558 | 9.3 | 5 | 0.11 |
| soreness | 1872 | 7.5 | 2 | 0.09 |
| contusion | 648 | 5.7 | 2 | 0.05 |
| injury management | 623 | 4.0 | 1 | 0.06 |
| concussion | 71 | 3.9 | 3 | 0.00 |

**By body region**: knee ligament **44.9**, Achilles 20.2, shoulder 15.2, foot
12.3, hand/wrist 11.9, knee 11.6, lower leg 9.9, ankle 9.2, back/core 7.9,
hamstring 7.2, hip/groin 6.6, quad 6.5, head/face 4.8.

Region alone is a weak predictor (C-index 0.51) because it pools a sore knee
with a torn ACL. **Region × pathology is what matters** — knee+surgery 49
spells at a 18-game median, knee+soreness 382 at 2.

**Severity flags.** A named structure only matters alongside a pathology that
actually takes one out: "Achilles; Soreness" is a rest night, "Achilles;
Repair" is a season.

| flag | n | KM mean |
|---|---|---|
| ACL/Achilles **and** tear/surgery/fracture | 39 | **80.4** |
| surgical wording anywhere | 100 | 42.9 |
| post-op recovery wording | 127 | 26.5 |
| injury-management wording | 693 | 4.1 |
| everything else | 6,697 | 10.3 |

**The label on the first missed game carries real information**, over and
above the diagnosis: Out **14.0** games, Doubtful 6.1, Questionable 5.5,
Probable 3.4. Teams know, and the report shows it.

### Strong negative duration dependence

The per-game probability of playing the next game collapses with time already
missed, and that is the dominant single signal in the data:

| games missed so far | P(plays next game) |
|---|---|
| 1 | 0.41 |
| 2 | 0.23 |
| 3 | 0.17 |
| 5 | 0.11 |
| 6-10 | 0.075 |
| 11-20 | 0.047 |
| 21+ | **0.020** |

Once a player is 20 games into an absence, each additional game carries only a
2% chance of being the last — a hazard that flat implies a median of roughly
another 22 games *no matter how long they have already been out*.

### Two behavioural findings

Both come out of the partial-dependence pass on the fitted hazard
(`reports/partial_dependence.csv`), holding the injury constant:

- **Rest between games matters as much as the injury.** P(plays next game)
  rises from 0.139 when the next game is a back-to-back to **0.310** when it
  is six days away. A borderline player sitting out is partly a scheduling
  decision.
- **Good teams get players back sooner.** P(plays next game) runs 0.137 at a
  .200 win rate to **0.180** at .800. Same injury, different urgency.

### What does *not* drive duration

- **Age.** The raw KM gradient looks dramatic (17.0 games for under-24s
  against 9.2 for 32+), but it is a selection artefact: under-24s are censored
  at 30% against 18%, because fringe young players get shut down or sent down.
  Conditional on injury type and elapsed time the age partial dependence is
  **flat** (0.149 at 21, 0.152 at 36), and dropping the whole eight-feature
  player-attribute block moves held-out log loss by less than 0.001 — age,
  height, weight and draft position are all deletable.
- **Prior injury history.** Prior spell counts, games missed in the last 365
  days, same-region recurrence — none of it registers. Dropping the entire
  14-feature history block moves held-out log loss by less than 0.001 in
  either direction, which is inside run-to-run noise: the block is free to
  delete. History alone reaches AUC 0.617 against 0.854 for the full model.
  Whatever a player's record says is already implied by what they are
  currently diagnosed with.
- **Recent workload as a cause.** Minutes in the last game and 7/14/30-day
  minute loads do register — dropping the load block costs +0.0135 log loss —
  but see the confound in §5: most of that is "is this player in the rotation",
  not "was this player overworked".

### Block ablations

Held-out log loss on the per-game return target; positive Δ means the block
was carrying signal (`reports/ablations.csv`):

| variant | features | log loss | Δ | AUC | C-index |
|---|---|---|---|---|---|
| full model | 81 | 0.3386 | — | 0.843 | 0.732 |
| drop elapsed-time + schedule | 74 | 0.3911 | **+0.0525** | 0.781 | 0.731 |
| drop injury taxonomy | 70 | 0.3554 | **+0.0169** | 0.823 | 0.667 |
| drop recent load | 57 | 0.3459 | +0.0074 | 0.834 | 0.704 |
| drop team incentive | 75 | 0.3403 | +0.0017 | 0.841 | 0.730 |
| drop season/team context | 70 | 0.3392 | +0.0006 | 0.842 | 0.734 |
| drop player attributes | 73 | 0.3382 | -0.0004 | 0.844 | 0.734 |
| drop injury history | 67 | 0.3379 | -0.0007 | 0.844 | 0.731 |
| *only* injury taxonomy | 11 | 0.4259 | **+0.0874** | 0.728 | 0.677 |
| *only* recent load | 24 | 0.4424 | **+0.1039** | 0.696 | 0.671 |
| *only* elapsed + schedule | 7 | 0.3837 | **+0.0451** | 0.790 | 0.530 |
| *only* team incentive | 6 | 0.4545 | **+0.1160** | 0.612 | 0.561 |
| *only* player attributes | 8 | 0.4922 | **+0.1536** | 0.484 | 0.519 |
| *only* injury history | 14 | 0.4889 | **+0.1503** | 0.613 | 0.594 |

Note the split: **elapsed time dominates the next-game call** (AUC 0.800 on
its own) but is useless for *ranking* severity (C-index 0.506), because every
spell starts at k=1 and the block knows nothing else. **The injury taxonomy is
the mirror image** — mediocre at timing (AUC 0.726) but the best single block
for severity (C-index 0.684). You need both, and only those two plus recent
load earn their place. Team incentive, added in section 7, prices at +0.0006 —
nothing, for reasons section 7 goes into.

### Twelve features are enough

Adding features in permutation-importance order (`reports/parsimony.csv`):

| features | log loss | AUC | C-index |
|---|---|---|---|
| 1 | 0.3903 | 0.783 | 0.500 |
| 3 | 0.3617 | 0.819 | 0.677 |
| 8 | 0.3385 | 0.843 | 0.717 |
| **12** | **0.3374** | **0.844** | **0.726** |
| 20 | 0.3351 | 0.847 | 0.724 |
| 81 | 0.3386 | 0.843 | 0.732 |

Twelve features beat all 78 on every column. They are, in importance order:
`days_missed_so_far`, `days_since_last_played`, `ailment_class`,
`games_missed_so_far`, `days_to_next_game`, `team_games_next_14d_now`,
`min_last`, `body_region`, `is_catastrophic`, `team_win_pct_before`,
`is_management`, `season_progress`.

Five of the twelve are elapsed time and schedule, two are the diagnosis, two
are the team's situation, and one is the player's last workload. Nothing about
age, size, draft position, career minutes or injury record survives.

---

## 3. The algorithm

A **discrete-time hazard model**. For a spell that misses games 1..*m*, row
*k* asks "the player has now missed *k* games — do they play game *k*+1?". The
answer is known for every *k* < *m* and for *k* = *m* only when the return was
observed, so a censored spell contributes its *m* "still out" rows and no
"came back" row. That is exactly the information it carries, and it is why
every one of the 6,754 spells is usable: **34,064 training rows** from 6,754
spells.

Duration then falls out as a product over the per-game hazards, evaluated on a
grid of the team's next 100 games:

- P(still out after *k* games) = Π(1 − h_j)
- E[games missed] = 1 + Σ P(still out after *k*)
- P(back within 3 games), P(out past 20 games), and so on

This is the whole reason to prefer a hazard model to regression on duration:
one fit gives a calibrated distribution rather than a point estimate, which
matters enormously here (§5).

### Algorithm comparison

Per-game return prediction, 2025-26, 7,635 rows, base rate 0.166
(`reports/per_game_return_metrics.csv`):

| model | AUC | log loss | Brier skill |
|---|---|---|---|
| base rate | 0.500 | 0.4677 | — |
| logistic (L2) | 0.818 | 0.3656 | 0.233 |
| random forest | 0.826 | 0.3578 | 0.257 |
| HistGradientBoosting | 0.838 | 0.3421 | 0.279 |
| LightGBM | 0.843 | 0.3386 | 0.285 |
| LightGBM **+ live report state** | **0.858** | **0.3278** | **0.308** |

Boosted trees beat the forest and the linear model clearly and consistently;
the two boosters are indistinguishable. Adding how the report has *moved*
since onset (status softened from Out, diagnosis re-filed) is worth 1.2 AUC
points — real, but smaller than you would guess, because most long absences
stay filed as "Out" right up to the night the player returns.

No neural network. At 34k rows and ~70 tabular features, with the signal
concentrated in a handful of categorical interactions, an MLP has nothing to
exploit that the boosters do not already get, and would cost a large
dependency for it.

### Duration at onset

2025-26, 1,581 spells, 1,271 with an observed return
(`reports/duration_metrics.csv`). Both point estimates are shown because the
distribution's median is 1 game and its mean is 11 — the mean minimises
squared error, the median minimises absolute error, and reporting only one
makes a model look better or worse than it is.

Median point estimate:

| model | MAE | MAE (spells ≥5 games) | C-index |
|---|---|---|---|
| KM global | 2.82 | 10.68 | 0.500 |
| KM by region | 2.90 | 10.38 | 0.535 |
| KM by ailment | 2.57 | 8.72 | 0.679 |
| KM by region × ailment | 2.54 | 8.92 | 0.667 |
| KM by region × ailment × status | 2.59 | 8.27 | 0.680 |
| hazard, logistic | 3.71 | 9.99 | 0.706 |
| LightGBM, observed spells only | **2.39** | 8.38 | 0.732 |
| hazard, random forest | 2.57 | 9.30 | 0.722 |
| hazard, LightGBM | 2.65 | 8.31 | 0.732 |
| **hazard, HistGradientBoosting** | 2.69 | **8.27** | **0.729** |

Calibration of the LightGBM hazard survival curve, Brier skill against the
base rate (`reports/horizon_brier.csv`): **0.244** at 1 game, **0.261** at 3,
0.210 at 5, 0.183 at 10, 0.198 at 20. The per-game probabilities are well
calibrated across the whole range — top decile predicted 0.696, realised
0.699; bottom decile 0.007 against 0.004 (`reports/calibration.csv`).

**On the censoring-blind model.** It wins overall MAE (2.33) and loses
everywhere that matters: worse on long spells (8.51 against 7.94) and worse at
ranking severity (0.731 against 0.744). The MAE win is an artefact — MAE is
computed on observed spells, which is precisely the population it was trained
on, and that population is mostly one-game absences. This is the trap the
hazard framing exists to avoid, and it is why **C-index is the fairest single
column in the table**: it uses the censored spells (a spell censored at 40
games is known to have outlasted one that ended at 3) and does not depend on
the choice of point estimate.

---

## 4. What the model is actually good at

Compared against the best thing you could do without it — a Kaplan-Meier table
by region × ailment × reported status — the model adds:

- **Severity ranking**: C-index 0.744 against 0.688.
- **A distribution instead of a cell mean**: P(back next game), P(back within
  3), P(out past 20), per spell.
- **Long absences**: MAE on spells of 5+ games drops from 8.04 to
  7.94, and from 10.74 for a flat baseline.
- **A live next-game call**: AUC 0.867 for "does he play tomorrow", which the
  static table cannot answer at all.

Honest subpopulation figures, because the headline is flattered by easy cases
(`reports/robustness_by_gap.csv`):

| population | n | observed | actual KM mean | C-index | MAE (median pred) |
|---|---|---|---|---|---|
| all test spells | 1,586 | 88% | 7.7 | 0.732 | 2.65 |
| fresh injuries (last played ≤4 days ago) | 1,386 | 91% | 5.5 | 0.707 | 2.20 |
| rotation players (≥15 min/game) | 1,346 | 90% | 6.7 | 0.718 | 2.46 |
| **fresh AND rotation** | 1,240 | 92% | 5.3 | **0.696** | **2.13** |
| already-running absences (gap >10 days) | 103 | 65% | 23.9 | 0.750 | 8.61 |

The cleanest number — a genuinely new injury to a player who was in the
rotation — is **C-index 0.701 and a median-prediction MAE of 2.03 games**.
The all-spells 0.744 is partly the model recognising absences that were already
under way, which is easier and less useful.

---

## 5. Limitations

**Point forecasts for catastrophic injuries are close to meaningless, and this
is a property of the data, not a model defect.** Jayson Tatum's Achilles
repair: forecast median 25 games, mean 37, actual 62. Max Strus's Jones
fracture surgery: forecast median 21, mean 32.5, actual 67. Both look like bad
misses. But the model's per-game hazard for these cases sits at 0.02-0.03,
which the raw data confirms is right (empirical hazard past 20 games missed:
0.020). A calibrated hazard that flat *implies* a median in the low twenties
with an enormous right tail — so 62 and 67 both sit inside the predicted
distribution, and the model gave each only a ~0.2-0.3 chance of being back
within ten games. For injuries like these, report P(out past 20 games), not a
number of games.

The same model handles the easy end of the same players' seasons correctly:
Tatum's later knee-management nights get a median of 1 game and a 0.89-0.91
probability of playing the next one, which is what happened.

**The report gives no return warning for surgical absences.** The game-by-game
trace (`scripts/05_forecast.py`) shows P(plays next game) never rising above
5% for either Tatum or Strus, including the night before they played. They
were filed "Out" for 62 and 67 straight games. The live-report lift is real
but concentrated in cases where the status actually softens to Questionable
(per-game return rate 0.477 vs 0.107 for Out).

**Rotation status confounds injury severity.** The model's longest calls are
dominated by two-way and deep-bench rookies, because `min_last` and
`days_since_last_played` cannot distinguish "badly hurt" from "not in the
rotation". On genuinely fresh injuries to rotation players — the honest
subpopulation — the C-index is 0.701 against 0.744 over all spells. Restricting
the training population, or modelling absence and rotation separately, is the
obvious next step.

**Censoring is informative, not independent — and not in one direction.**
An earlier version of this section claimed season-end censoring was
"plausibly independent of the injury", so that the long tail was "if anything
still understated". That was wrong, and the correction matters because it
changes the sign of the bias rather than its size.

Season-end censoring is strongly related to the team's incentive. Among
spells starting in the last 20% of a season, 43% run to season end on teams
below .350 against 20% on teams at .500 or better — for the same diagnoses.
And the per-game return hazard splits three ways by the team's playoff
position: 0.262 once a play-in place is clinched, 0.152 in contention, 0.084
once mathematically eliminated. Some season-end-censored spells are therefore
shutdowns whose *medical* duration was shorter than the censoring time, which
pushes the Kaplan-Meier tail the other way.

So the tail is inflated by team decisions and deflated by independent
censoring at once, and the net is not knowable from the duration data alone.
Section 7 takes this apart.

**Only five seasons.** 39 spells carry the catastrophic flag. Any statement
about ACL and Achilles duration here rests on dozens of cases, not hundreds.

**Spell boundaries are a judgement call.** Gaps of up to 3 games with no
report row, bracketed by reported absences, are bridged; longer gaps are not.
Reconditioning filings continue a spell; G League assignment censors it.
Reasonable alternatives would move the numbers somewhat. The rules are in
`src/injury_duration/spells.py` with the reasoning for each.

---

## 7. How much of an absence is the injury, and how much is the team?

A team's playoff position changes what it wants from a borderline player and
cannot change how a torn ligament heals. That asymmetry is the whole design:
variation in playoff position, holding the diagnosis and the player's role
fixed, moves team *willingness* and not medical *readiness*.

Mathematical elimination and clinch dates come from daily conference
standings; playoff hope is simulated by playing out the rest of each
conference's season from the standings on the day (mean 0.667 against the
20/30 base rate).

### The effect is large, and it is not team quality

Per-game return hazard, by the team's position on the day:

| | rows | spells | P(plays next game) |
|---|---|---|---|
| clinched a play-in place | 2,980 | 1,031 | **0.262** |
| in contention | 27,017 | 5,283 | 0.152 |
| mathematically eliminated | 3,006 | 650 | **0.084** |

A 3.1× spread. The control that rules out the boring explanations is the
season half. For rotation players within the same diagnosis, the gap between
high-hope and low-hope teams is **+0.008 early in the season** — nothing,
over 3,635 rows — and **+0.271 late**. Difference-in-differences
**+0.263 [+0.228, +0.298]**, and it excludes zero for every diagnosis
separately (`reports/incentive_did.csv`):

| diagnosis | gap early | gap late | DiD | 95% CI |
|---|---|---|---|---|
| soreness | +0.056 | +0.377 | **+0.321** | [+0.231, +0.409] |
| injury management | +0.105 | +0.369 | +0.265 | [+0.113, +0.427] |
| contusion | +0.012 | +0.239 | +0.227 | [+0.113, +0.347] |
| strain | +0.006 | +0.125 | +0.118 | [+0.028, +0.208] |
| sprain | +0.049 | +0.153 | +0.104 | [+0.032, +0.177] |

Worse medical staff, more fragile rosters or a different injury mix on bad
teams would all show up in the early season too. None of them do.

The event study around mathematical elimination agrees but is low-powered:
flat at ~0.13 beforehand, falling to 0.06 after, a shift of
**−0.039 [−0.057, −0.023]**. It is smaller than the DiD because elimination
arrives so late that most of the response has already happened as hope
faded. (At one-game resolution the pre-period looks like it is already
declining; that is noise, and it disappears at a readable bucket width.)

### The 2023-24 Player Participation Policy narrowed it

| regime | P(return \| high hope) | P(return \| low hope) | gap | 95% CI |
|---|---|---|---|---|
| pre-policy, 2021-23 | 0.477 | 0.150 | 0.326 | [+0.279, +0.375] |
| post-policy, 2023-26 | 0.416 | 0.174 | 0.242 | [+0.206, +0.276] |

Low-hope teams return players more often after the policy (0.150 → 0.174),
which is the direction the rule intended. The gap narrowed by about a third
and did not close. The 2019 lottery reform is not testable here: it predates
the injury report entirely.

### De-biasing the duration tables

Rather than classify spells — there is no label for "tanking injury" and the
injury is almost always real — the hazard is fitted with the incentive
features and then evaluated twice per spell: at the team's actual position,
and at a neutral one (in contention, neither eliminated nor clinched). The
difference is a per-spell *discretion score*.

| situation when ruled out | spells | pred games, as observed | at neutral urgency | difference |
|---|---|---|---|---|
| already eliminated | 458 | 16.24 | 8.67 | **+7.57 games (+17.2 days)** |
| hope < 0.25 | 1,477 | 10.01 | 7.70 | +2.30 |
| hope 0.25-0.75 | 889 | 12.69 | 12.16 | +0.53 |
| hope > 0.75 | 3,930 | 7.81 | 8.06 | -0.25 |

Across all 6,754 spells the de-biasing is worth under a game on average, but
it is concentrated exactly where it should be — **you cannot tank an ACL**:

| ailment | as observed | neutral | difference | share of its own length |
|---|---|---|---|---|
| management | 3.91 | 3.21 | +0.70 | **17.9%** |
| soreness | 7.64 | 6.41 | +1.23 | 16.1% |
| tendinopathy | 10.66 | 9.03 | +1.63 | 15.3% |
| contusion | 6.19 | 5.25 | +0.94 | 15.2% |
| surgery | 34.19 | 33.52 | +0.67 | 2.0% |
| fracture | 26.78 | 26.61 | +0.18 | 0.7% |
| rupture / tear | 45.73 | 45.60 | +0.13 | 0.3% |
| dislocation | 20.99 | 20.95 | +0.04 | **0.2%** |

Read as a share, the pattern is a clean twenty-fold gradient: the
non-discretionary diagnoses give up under 2% of their length, the
discretionary ones 13-15%. A torn ligament loses 0.7% of its duration to
the team's situation; a rest night filed as injury management loses 17.8%.
That proportionality is the strongest sign the measure is picking up
decisions rather than noise.

### It found a known tanking episode unprompted

The highest discretion scores in the sample include Tim Hardaway Jr.
(+26.9 games), Kyrie Irving (+23.8) and Maxi Kleber
(+12.7), all filed out by
Dallas on **2023-04-07** — the night Dallas sat its starters with a draft
pick at stake, and was fined by the league for it. Nothing in the model knows
about that episode, or about tanking; it only knows the team was out of
contention and the diagnoses were soft.

### It does not improve prediction

This is the part worth being blunt about. Adding the incentive features moves held-out log loss by +0.0006
and AUC by +0.0009 — that is, by nothing:

| model | log loss | AUC | C-index | log loss (late) | AUC (late) |
|---|---|---|---|---|---|
| without incentive | 0.3166 | 0.8537 | 0.7426 | 0.3868 | 0.8754 |
| with incentive | 0.3156 | 0.8556 | 0.7438 | 0.3862 | 0.8749 |

Barely more late in the season, where the whole effect lives. The reason is that
`team_win_pct_before`, `season_progress` and `team_regular_remaining` were
already in the feature set and already proxy the incentive well enough for
forecasting. What the explicit measure buys is **interpretation** — the
ability to state what an injury costs at ordinary urgency, and to attribute
the remainder — not accuracy. I expected a modest gain here and got none.

### Limitations of this section

- **93 of the top 100 discretion scores are censored spells** with a mean of
  4.6 games missed. The measure is overwhelmingly detecting end-of-season
  shutdowns, which is the dominant form of the behaviour but a narrow window,
  and the counterfactual is never observed in those cases.
- **The neutral prediction is a model output, not a measurement.** It is only
  as good as the exclusion restriction, and the restriction is credible rather
  than proven.
- **High hope and low hope are not symmetric.** A contending team rushing a
  player back is as much a decision as a tanking team holding one out, and the
  "neutral" reference sits between them by construction, not by evidence about
  what is medically correct.
- **Elimination is defined on the play-in**, so it says nothing about teams
  manoeuvring for seeding inside the top ten.

---

## 6. Code

| file | role |
|---|---|
| `src/injury_duration/taxonomy.py` | parse the free-text reason into region / pathology / severity flags |
| `src/injury_duration/ids.py` | resolve the report's nullable `nba_id` |
| `src/injury_duration/calendar.py` | canonical team-game calendar, report-date alignment |
| `src/injury_duration/spells.py` | availability panel → spells, with censoring |
| `src/injury_duration/features.py` | pre-tipoff features, grouped into blocks for ablation |
| `src/injury_duration/hazard.py` | hazard rows, Kaplan-Meier, survival arithmetic, prediction grid |
| `src/injury_duration/design.py` | polars → numpy encoding (no pandas) |
| `src/injury_duration/models.py` | baselines, hazard models, censoring-blind contrast |
| `src/injury_duration/evaluate.py` | C-index, horizon metrics, calibration |
| `src/injury_duration/experiment.py` | feature sets and design matrices, shared by every script |
| `src/injury_duration/standings.py` | daily standings, elimination dates, simulated playoff hope |
| `src/injury_duration/forecast.py` | usable forecasts from a fitted model |
| `scripts/01_build.py` … `07_debias.py` | the pipeline, in order |

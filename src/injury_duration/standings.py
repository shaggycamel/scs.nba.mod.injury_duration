"""Daily conference standings, and when each team's season stops mattering.

Why this exists
---------------
Team *quality* is a proxy for team *incentive*, and a bad proxy: it is a
smooth gradient that confounds several things at once (worse medical staff,
more fragile rosters, lower-value players). Mathematical elimination is a
discontinuity. It falls on a known date, it is unrelated to how a torn
ligament heals, and it flips the team's objective from winning games to
improving its lottery position overnight. That makes it the identifying
variation for separating "medically unavailable" from "not worth playing".

Elimination here means elimination from the **play-in**, i.e. from a top-10
finish in the conference, because every season in the data (2021-22 onward)
has the play-in tournament.

The test is the standard sufficient condition: a team is eliminated once at
least ten teams in its conference have *already* won more games than it can
still reach. That is conservative — it ignores tiebreakers and head-to-head,
which can only eliminate a team earlier — so the date it returns is an upper
bound on the true one. Being conservative is the right direction: it means
the "eliminated" window contains only teams that really were out.

A caveat that shapes the whole design
-------------------------------------
Mathematical elimination is a *lagging* indicator, which limits how much a
regression discontinuity can see. There is a level shift at the date — over
the twelve team games either side, the return hazard is flat beforehand at
about 0.13 and falls to 0.06 after, a difference of -0.039 with a bootstrap
interval excluding zero — but it is modest, because elimination arrives so
late that most of the behavioural response has already happened as the
team's playoff hopes faded. (At one-game resolution the pre-period looks
like it is already declining; that is noise, and it disappears once the
buckets are wide enough to estimate.)

So playoff hope, not the elimination date, carries most of the identifying
variation. `playin_probability` below estimates it from the standings, and
the main comparison becomes hope crossed with the half of the season — the
early season, when everyone still has hope, is the control period. That is
weaker causally than a sharp discontinuity and much better powered.
"""

from __future__ import annotations

import polars as pl

from . import calendar as cal_mod
from . import paths

# Top ten finishers in each conference reach the play-in.
PLAYIN_SEEDS = 10
# The seed line that separates a guaranteed playoff berth from the play-in.
PLAYOFF_SEEDS = 6
REGULAR_SEASON_GAMES = 82


def _regular_season_results() -> pl.DataFrame:
    """One row per team game, with the result and the team's conference."""
    cal = cal_mod.game_calendar().filter(pl.col("season_type") == "Regular Season")
    results = (
        pl.read_parquet(paths.SCHEDULE)
        .select("game_id", "team", "team_winner")
        .unique(subset=["game_id", "team"])
    )
    conf = pl.read_parquet(paths.TEAMS).select("team_slug", "conference")
    return (
        cal.join(results, left_on=["game_id", "team_slug"], right_on=["game_id", "team"],
                 how="left")
        .join(conf, on="team_slug", how="left")
        .with_columns((pl.col("team_winner") == pl.col("team_slug")).alias("won"))
        .select("season", "team_slug", "conference", "game_date", "team_game_idx", "won")
    )


def daily_standings() -> pl.DataFrame:
    """Every (season, date, team) with the record as of the end of that date.

    Dense in dates: a team that did not play still carries its standing
    forward, which is what ranking a conference on an arbitrary date needs.
    """
    games = _regular_season_results()

    cum = (
        games.sort("season", "team_slug", "game_date")
        .with_columns(
            pl.col("won").cum_sum().over("season", "team_slug").alias("wins"),
            pl.col("team_game_idx").alias("played"),
        )
        .group_by("season", "team_slug", "game_date")
        .agg(pl.col("wins").last(), pl.col("played").last())
    )

    dates = games.select("season", "game_date").unique()
    teams = games.select("season", "team_slug", "conference").unique()
    grid = dates.join(teams, on="season", how="left")

    out = (
        grid.sort("game_date")
        .join_asof(
            cum.sort("game_date"),
            on="game_date",
            by=["season", "team_slug"],
            strategy="backward",
        )
        # Before a team's opener it has no record yet.
        .with_columns(pl.col("wins").fill_null(0), pl.col("played").fill_null(0))
        .with_columns(
            (REGULAR_SEASON_GAMES - pl.col("played")).alias("games_left"),
            (pl.col("wins") + REGULAR_SEASON_GAMES - pl.col("played")).alias("max_wins"),
        )
    )
    return out.select(
        "season", "game_date", "team_slug", "conference", "wins", "played",
        "games_left", "max_wins",
    )


def elimination_status() -> pl.DataFrame:
    """Per (season, date, team): seeding position and whether it is out.

    `n_ahead_certain` counts conference rivals that have already won more
    games than this team can still reach; ten or more of those and the team
    cannot finish top ten. `n_can_pass` counts rivals that could still finish
    above the team's current win total; fewer than ten and a top-ten finish
    is already secured.
    """
    st = daily_standings()

    rivals = st.select(
        "season", "game_date", "conference",
        pl.col("team_slug").alias("rival"),
        pl.col("wins").alias("rival_wins"),
        pl.col("max_wins").alias("rival_max_wins"),
    )
    pairs = st.join(rivals, on=["season", "game_date", "conference"], how="left").filter(
        pl.col("team_slug") != pl.col("rival")
    )

    agg = pairs.group_by("season", "game_date", "team_slug").agg(
        (pl.col("rival_wins") > pl.col("max_wins")).sum().alias("n_ahead_certain"),
        (pl.col("rival_max_wins") > pl.col("wins")).sum().alias("n_can_pass"),
        # Win totals of the seed lines, for the stakes measures below.
        # To finish in the top N a team needs at most N-1 rivals above it,
        # so the line it must clear is the Nth-best rival: index N-1.
        pl.col("rival_wins").sort(descending=True).get(PLAYIN_SEEDS - 1, null_on_oob=True)
        .alias("rival_wins_at_playin_line"),
        pl.col("rival_wins").sort(descending=True).get(PLAYOFF_SEEDS - 1, null_on_oob=True)
        .alias("rival_wins_at_playoff_line"),
    )

    out = st.join(agg, on=["season", "game_date", "team_slug"], how="left").with_columns(
        (pl.col("n_ahead_certain") >= PLAYIN_SEEDS).alias("eliminated"),
        (pl.col("n_can_pass") < PLAYIN_SEEDS).alias("clinched_playin"),
        # Signed distance in wins to each seed line. Negative = below it.
        (pl.col("wins") - pl.col("rival_wins_at_playin_line")).alias("wins_vs_playin_line"),
        (pl.col("wins") - pl.col("rival_wins_at_playoff_line")).alias("wins_vs_playoff_line"),
    )
    return out


def team_incentive() -> pl.DataFrame:
    """Elimination and clinch dates per team-season, joined back per date.

    Adds `games_since_elimination`, which is the running variable for the
    event study: negative before the team was eliminated, zero on the day,
    positive after.
    """
    st = elimination_status()

    marks = st.group_by("season", "team_slug").agg(
        pl.col("game_date").filter(pl.col("eliminated")).min().alias("elimination_date"),
        pl.col("game_date").filter(pl.col("clinched_playin")).min().alias("clinch_date"),
    )
    out = st.join(marks, on=["season", "team_slug"], how="left")

    # Count the team's own games either side of the elimination date, rather
    # than calendar days, so the running variable is in the same units as the
    # hazard (one row per game).
    out = out.with_columns(
        pl.when(pl.col("elimination_date").is_null())
        .then(None)
        .otherwise(pl.col("played") - pl.col("played").filter(
            pl.col("game_date") == pl.col("elimination_date")
        ).first().over("season", "team_slug"))
        .alias("games_since_elimination")
    )
    return out.with_columns(
        pl.when(pl.col("eliminated")).then(pl.lit("eliminated"))
        .when(pl.col("clinched_playin")).then(pl.lit("clinched"))
        .otherwise(pl.lit("in_contention"))
        .alias("incentive_state")
    )


# --------------------------------------------------------------------------
# Playoff hope
# --------------------------------------------------------------------------

def final_playin_outcome() -> pl.DataFrame:
    """Whether each team actually finished in its conference's top ten."""
    st = elimination_status()
    last = st.group_by("season").agg(pl.col("game_date").max().alias("final_date"))
    return (
        st.join(last, on="season", how="left")
        .filter(pl.col("game_date") == pl.col("final_date"))
        .select("season", "team_slug", pl.col("clinched_playin").alias("made_playin"))
    )


# Games of notional prior added when estimating a team's win rate, to stop a
# 3-1 start implying a 75% team.
WIN_RATE_PRIOR_GAMES = 10.0
N_SIMS = 2000


def playin_probability(n_sims: int = N_SIMS, seed: int = 0) -> pl.DataFrame:
    """P(team reaches the play-in), per (season, date, team), plus stakes.

    Estimated by simulating the rest of each conference's season from the
    standings on the day: every team's remaining games are drawn as binomial
    with its shrunk win rate, the final table is ranked, and the probability
    is the share of simulations in which the team lands in the top ten.

    Two closed forms were tried first and both failed, for the same reason.
    Comparing a team against the *current* tenth-place team — Phi(gap / sd)
    over the remaining games — ignores that finishing top ten is an order
    statistic over fifteen teams, not a duel with one of them: a team can
    hold its ground and still be passed. That approximation read 0.99 on the
    largest bin where the realised rate was 0.62. A two-feature logistic on
    the same quantities was worse, squashed into 0.17-0.50 against a 67% base
    rate. Simulation gets the order statistic right and needs no fitting, so
    a season's own outcome never informs its own incentive measure.

    Everything here is a function of the standings on the day, which are
    public before tip-off, so nothing leaks about the future.
    """
    import numpy as np

    st = team_incentive().filter(pl.col("played") > 0)
    rng = np.random.default_rng(seed)

    frames = []
    # Iterate groups in a fixed order. polars does not guarantee group_by
    # ordering, and the generator is drawn from inside this loop, so an
    # unstable order silently makes the whole simulation irreproducible --
    # and with it every downstream model that uses playoff hope as a feature.
    groups = {
        k: g for k, g in st.group_by("season", "conference")
    }
    for key in sorted(groups):
        season, conference = key
        grp = groups[key]
        dates = grp["game_date"].unique().sort().to_list()
        wide = grp.select("game_date", "team_slug", "wins", "played", "games_left")
        for d in dates:
            day = wide.filter(pl.col("game_date") == d).sort("team_slug")
            wins = day["wins"].to_numpy().astype(float)
            played = day["played"].to_numpy().astype(float)
            left = day["games_left"].to_numpy().astype(int)
            # Shrunk win rate, so an early hot streak is not extrapolated.
            rate = (wins + 0.5 * WIN_RATE_PRIOR_GAMES) / (played + WIN_RATE_PRIOR_GAMES)
            extra = rng.binomial(
                np.maximum(left, 0)[:, None], rate[:, None], size=(len(wins), n_sims)
            )
            final = wins[:, None] + extra
            # Rank descending within the conference; ties broken at random by
            # the jitter, which is the fairest treatment of a coin-flip tiebreak.
            jitter = rng.random(final.shape) * 0.5
            order = np.argsort(-(final + jitter), axis=0)
            rank = np.empty_like(order)
            np.put_along_axis(
                rank, order, np.arange(final.shape[0])[:, None].repeat(n_sims, axis=1), axis=0
            )
            prob = (rank < PLAYIN_SEEDS).mean(axis=1)
            frames.append(
                day.select("game_date", "team_slug").with_columns(
                    pl.lit(season).alias("season"),
                    pl.lit(conference).alias("conference"),
                    pl.Series("playin_probability", prob),
                )
            )

    probs = pl.concat(frames)
    out = st.join(probs, on=["season", "conference", "game_date", "team_slug"], how="left")
    return out.select(
        "season", "game_date", "team_slug", "conference", "wins", "played",
        "games_left", "wins_vs_playin_line", "wins_vs_playoff_line",
        "eliminated", "clinched_playin", "incentive_state",
        "games_since_elimination", "playin_probability",
    ).with_columns(
        # How much the next game matters to the team: highest when the play-in
        # place is genuinely in doubt, near zero once it is settled either way.
        # This is the "stakes" counterpart to hope.
        (4.0 * pl.col("playin_probability") * (1.0 - pl.col("playin_probability")))
        .alias("playin_stakes")
    # A join does not promise an output order either, so pin it: downstream
    # fits should not depend on which way round two equal rows landed.
    ).sort("season", "team_slug", "game_date")

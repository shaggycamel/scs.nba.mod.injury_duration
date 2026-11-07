"""Render the injury-duration findings as a single standalone HTML page.

Writes `docs/injury_duration.html`: one file, inline SVG and CSS, no CDN
and no build step, so it opens from a git clone with no network. Light and
dark are both selected from the same palette rather than one being an
automatic inversion of the other.

Every figure ships a data table underneath it. That is partly accessibility
and partly a hard requirement: one of the three series colours sits below 3:1
contrast on the light surface, which obliges visible labels or a table view.

Run: uv run python scripts/08_report_html.py
"""

from __future__ import annotations

import json
import warnings
from datetime import date

import numpy as np
import polars as pl

from injury_duration import hazard as hz
from injury_duration import paths, standings
from injury_duration.viz import (
    Box,
    esc,
    grid_x,
    grid_y,
    hbar,
    legend,
    marked,
    nice_ticks,
    vbar,
)

warnings.filterwarnings("ignore")

S1, S2, S3 = "var(--series-1)", "var(--series-2)", "var(--series-3)"
SEQ = "var(--seq-450)"
SEQ_DIM = "var(--seq-200)"
SEQ2_DIM = "var(--seq2-200)"
MUTED = "var(--text-muted)"


# --------------------------------------------------------------------------
# Page furniture
# --------------------------------------------------------------------------

def table(headers: list[str], rows: list[list[object]], caption: str = "") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f'<div class="t-wrap"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def details_table(headers, rows, label="Show the numbers") -> str:
    return f"<details><summary>{esc(label)}</summary>{table(headers, rows)}</details>"


def figure(title: str, subtitle: str, svg: str, note: str = "",
           lg: str = "", data: str = "") -> str:
    return f"""<figure>
  <h3>{esc(title)}</h3>
  <p class="sub">{subtitle}</p>
  {lg}
  <div class="plot">{svg}</div>
  {f'<figcaption>{note}</figcaption>' if note else ''}
  {data}
</figure>"""


def stat(value: str, label: str, note: str = "") -> str:
    return (
        f'<div class="stat"><div class="stat-v">{esc(value)}</div>'
        f'<div class="stat-l">{esc(label)}</div>'
        f'{f"<div class=\'stat-n\'>{esc(note)}</div>" if note else ""}</div>'
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_survival(sp: pl.DataFrame) -> str:
    """Survival curves: all spells against a severe and a mild diagnosis."""
    box = Box(720, 300, left=46, bottom=38)
    kmax = 40
    series = [
        ("All spells", S1, sp),
        ("Surgery / tear", S2, sp.filter(pl.col("ailment_class").is_in(["surgery", "rupture_tear"]))),
        ("Soreness", S3, sp.filter(pl.col("ailment_class") == "soreness")),
    ]
    sx = lambda k: box.x0 + (k / kmax) * box.iw
    sy = lambda p: box.y1 - p * box.ih

    parts = [grid_y(box, [0, 0.25, 0.5, 0.75, 1.0], sy, 2),
             grid_x(box, [0, 10, 20, 30, 40], sx)]
    rows, curves, ends = [], {}, []
    for label, colour, df in series:
        d, e = df["games_missed"].to_numpy(), df["event"].to_numpy()
        times, surv = hz.kaplan_meier(d, e)
        pts = []
        for k in range(kmax + 1):
            idx = np.searchsorted(times, k, side="right") - 1
            pts.append((k, float(surv[idx]) if idx >= 0 else 1.0))
        curves[label] = pts
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(k):.2f},{sy(p):.2f}" for i, (k, p) in enumerate(pts)
        )
        parts.append(f'<path class="line" d="{path}" stroke="{colour}"/>')
        ends.append((label, colour, pts[-1]))
        rows.append([label, f"{df.height:,}", f"{pts[1][1]:.0%}", f"{pts[3][1]:.0%}",
                     f"{pts[10][1]:.0%}", f"{pts[20][1]:.0%}", f"{pts[40][1]:.0%}"])

    # End labels, nudged apart where two curves finish close together: a label
    # that overlaps another is worse than no label.
    placed: list[float] = []
    for label, colour, (kx, ky) in sorted(ends, key=lambda e: -e[2][1]):
        y = sy(ky)
        while any(abs(y - q) < 15 for q in placed):
            y += 15
        placed.append(y)
        parts.append(
            f'<circle class="dot" cx="{sx(kx):.2f}" cy="{sy(ky):.2f}" r="4.5" fill="{colour}"/>'
        )
        parts.append(
            f'<text class="dlabel" x="{sx(kx) - 10:.2f}" y="{y - 9:.2f}" '
            f'text-anchor="end">{esc(label)} {ky:.0%}</text>'
        )

    # Crosshair hit columns, one per k, carrying every series' value.
    for k in range(kmax + 1):
        w = box.iw / kmax
        txt = f"After {k} games missed — " + ", ".join(
            f"{lab} {dict(pts)[k]:.0%}" for lab, pts in curves.items()
        )
        parts.append(marked(
            "rect",
            f'class="hit" x="{sx(k) - w / 2:.2f}" y="{box.y0:.2f}" '
            f'width="{w:.2f}" height="{box.ih:.2f}"',
            txt,
        ))
    parts.append(
        f'<text class="axis-t" x="{box.x0 + box.iw / 2:.2f}" y="{box.height - 4:.2f}" '
        f'text-anchor="middle">team games missed</text>'
    )
    svg = f'<svg viewBox="0 0 {box.width} {box.height}" role="img" '\
          f'aria-label="Survival curves by injury type">{"".join(parts)}</svg>'
    return figure(
        "Most absences are over in a week. A few never end.",
        "Kaplan-Meier probability a player is <em>still</em> out after k team games.",
        svg,
        "Half of all spells are done after two games, but the surgery and tear "
        "curve is still above 50% at twenty. A single average over these is not "
        "describing one population.",
        legend([(l, c) for l, c, _ in series]),
        details_table(
            ["series", "spells", "after 1", "after 3", "after 10", "after 20", "after 40"],
            rows,
        ),
    )


def fig_censoring(km: pl.DataFrame) -> str:
    """Dumbbell: what ignoring censoring costs, per ailment."""
    df = km.filter(pl.col("n") >= 40).sort("km_mean", descending=True)
    tear = df.filter(pl.col("ailment_class") == "rupture_tear").row(0, named=True)
    rowh, pad = 26, 10
    box = Box(720, pad * 2 + rowh * df.height + 34, left=112, right=58, top=pad, bottom=34)
    hi = float(df["km_mean"].max()) * 1.04
    sx = lambda v: box.x0 + (v / hi) * box.iw
    parts = [grid_x(box, nice_ticks(0, hi, 5), sx)]
    rows = []
    for i, r in enumerate(df.iter_rows(named=True)):
        y = box.y0 + i * rowh + rowh / 2
        a, b = float(r["naive_mean_observed"]), float(r["km_mean"])
        parts.append(
            f'<line class="dumb" x1="{sx(a):.2f}" y1="{y:.2f}" x2="{sx(b):.2f}" y2="{y:.2f}"/>'
        )
        t = (f'{r["ailment_class"]}: {a:.1f} games if censored spells are ignored, '
             f'{b:.1f} with them (n={r["n"]}, {r["censored"]} censored)')
        parts.append(marked("circle", f'class="dot" cx="{sx(a):.2f}" cy="{y:.2f}" r="4.5" fill="{SEQ_DIM}"', t))
        parts.append(marked("circle", f'class="dot" cx="{sx(b):.2f}" cy="{y:.2f}" r="5" fill="{S1}"', t))
        parts.append(
            f'<text class="cat" x="{box.x0 - 10:.2f}" y="{y + 4:.2f}" '
            f'text-anchor="end">{esc(r["ailment_class"].replace("_", " "))}</text>'
        )
        parts.append(
            f'<text class="dlabel" x="{sx(b) + 9:.2f}" y="{y + 4:.2f}">{b:.0f}</text>'
        )
        rows.append([r["ailment_class"], f"{r['n']:,}", r["censored"],
                     f"{a:.2f}", f"{b:.2f}", f"{b / a:.1f}x"])
    parts.append(
        f'<text class="axis-t" x="{box.x0 + box.iw / 2:.2f}" y="{box.height - 4:.2f}" '
        f'text-anchor="middle">mean team games missed</text>'
    )
    svg = f'<svg viewBox="0 0 {box.width} {box.height}" role="img" '\
          f'aria-label="Censoring effect by ailment">{"".join(parts)}</svg>'
    return figure(
        "Dropping unfinished absences understates every diagnosis",
        "Mean games missed counting only spells that were seen to end (pale) against "
        "the Kaplan-Meier estimate that uses the unfinished ones too (solid).",
        svg,
        f"{int(df['censored'].sum()) / int(df['n'].sum()):.0%} of spells never show a "
        "return — the season ends, or the player is traded or sent down — and they are "
        f"the long ones. For tears the gap is {float(tear['naive_mean_observed']):.1f} "
        f"against {float(tear['km_mean']):.1f} games.",
        legend([("seen to end only", SEQ_DIM), ("Kaplan-Meier", S1)]),
        details_table(
            ["ailment", "n", "censored", "observed-only mean", "KM mean", "ratio"], rows
        ),
    )


def fig_hazard(rows_df: pl.DataFrame) -> str:
    """Return hazard against games already missed: duration dependence."""
    t = (
        rows_df.filter(pl.col("games_missed_so_far") <= 12)
        .group_by("games_missed_so_far")
        .agg(pl.len().alias("n"), pl.col("returns_next").mean().alias("p"))
        .sort("games_missed_so_far")
    )
    box = Box(720, 260, left=50, bottom=38)
    hi = float(t["p"].max()) * 1.12
    band = box.iw / t.height
    thick = min(24.0, band - 2 * 2)
    sy = lambda p: box.y1 - (p / hi) * box.ih
    parts = [grid_y(box, nice_ticks(0, hi, 4), sy, 2)]
    out = []
    for i, r in enumerate(t.iter_rows(named=True)):
        x = box.x0 + i * band + (band - thick) / 2
        p = float(r["p"])
        parts.append(marked(
            "path",
            f'd="{vbar(x, box.y1, sy(p), thick)}" fill="{SEQ}"',
            f'after {r["games_missed_so_far"]} missed: {p:.1%} play the next game '
            f'(n={r["n"]:,})',
        ))
        parts.append(
            f'<text class="tick" x="{x + thick / 2:.2f}" y="{box.y1 + 16:.2f}" '
            f'text-anchor="middle">{r["games_missed_so_far"]}</text>'
        )
        if i in (0, t.height - 1):
            parts.append(
                f'<text class="dlabel" x="{x + thick / 2:.2f}" y="{sy(p) - 7:.2f}" '
                f'text-anchor="middle">{p:.0%}</text>'
            )
        out.append([r["games_missed_so_far"], f"{r['n']:,}", f"{p:.3f}"])
    parts.append(
        f'<text class="axis-t" x="{box.x0 + box.iw / 2:.2f}" y="{box.height - 4:.2f}" '
        f'text-anchor="middle">games already missed</text>'
    )
    svg = f'<svg viewBox="0 0 {box.width} {box.height}" role="img" '\
          f'aria-label="Return hazard by games already missed">{"".join(parts)}</svg>'
    return figure(
        "The longer a player has been out, the less likely he is back tomorrow",
        "Share of cases where the player suits up for the next game, by how many he "
        "has already missed.",
        svg,
        f"A {float(t['p'][0]):.0%} chance after one game missed, "
        f"{float(t['p'][-1]):.0%} after {int(t['games_missed_so_far'][-1])}. "
        "Once a hazard is this "
        "flat, a calibrated model implies a median of roughly another twenty games "
        "<em>however long</em> the player has already been out — which is why point "
        "forecasts for severe injuries are close to meaningless.",
        "",
        details_table(["games already missed", "rows", "P(plays next game)"], out),
    )


def fig_incentive(state: pl.DataFrame) -> str:
    """Return hazard by the team's playoff position."""
    order = ["clinched", "in_contention", "eliminated"]
    labels = {"clinched": "Play-in place clinched", "in_contention": "In contention",
              "eliminated": "Mathematically eliminated"}
    d = {r["incentive_state"]: r for r in state.iter_rows(named=True)}
    box = Box(720, 190, left=200, right=70, top=10, bottom=34)
    hi = max(float(d[k]["p"]) for k in order) * 1.18
    sx = lambda v: box.x0 + (v / hi) * box.iw
    parts = [grid_x(box, nice_ticks(0, hi, 4), sx)]
    rows, rowh = [], 44
    for i, k in enumerate(order):
        r = d[k]
        y = box.y0 + i * rowh + (rowh - 22) / 2
        p = float(r["p"])
        colour = S1 if k == "clinched" else (S2 if k == "eliminated" else SEQ_DIM)
        parts.append(marked(
            "path",
            f'd="{hbar(box.x0, sx(p), y, 22)}" fill="{colour}"',
            f'{labels[k]}: {p:.1%} ({r["rows"]:,} games across {r["spells"]:,} spells)',
        ))
        parts.append(
            f'<text class="cat" x="{box.x0 - 10:.2f}" y="{y + 15:.2f}" '
            f'text-anchor="end">{esc(labels[k])}</text>'
        )
        parts.append(
            f'<text class="dlabel" x="{sx(p) + 9:.2f}" y="{y + 15:.2f}">{p:.1%}</text>'
        )
        rows.append([labels[k], f"{r['rows']:,}", f"{r['spells']:,}", f"{p:.4f}"])
    parts.append(
        f'<text class="axis-t" x="{box.x0 + box.iw / 2:.2f}" y="{box.height - 4:.2f}" '
        f'text-anchor="middle">P(plays the next game)</text>'
    )
    svg = f'<svg viewBox="0 0 {box.width} {box.height}" role="img" '\
          f'aria-label="Return hazard by team playoff position">{"".join(parts)}</svg>'
    return figure(
        "A player on an eliminated team is a third as likely to play tomorrow",
        "Same question, split by what the team still has to play for.",
        svg,
        f"A {float(d['clinched']['p']) / float(d['eliminated']['p']):.1f}x spread. "
        "On its own this could just be that bad teams have worse "
        "players and worse medical staff — which is what the next figure rules out.",
        "",
        details_table(["team position", "games", "spells", "P(plays next)"], rows),
    )


def fig_did(did: pl.DataFrame) -> str:
    """Dumbbell: the high-minus-low-hope gap, early season against late."""
    d = did.filter(pl.col("diagnosis") != "ALL")
    allr = did.filter(pl.col("diagnosis") == "ALL").row(0, named=True)
    rowh = 30
    box = Box(720, rowh * (d.height + 1) + 54, left=118, right=70, top=10, bottom=40)
    lo = min(0.0, float(d["gap_early"].min()))
    hi = max(float(d["gap_late"].max()), float(allr["gap_late"])) * 1.08
    sx = lambda v: box.x0 + ((v - lo) / (hi - lo)) * box.iw
    parts = [grid_x(box, nice_ticks(lo, hi, 5), sx)]
    parts.append(
        f'<line class="zero" x1="{sx(0):.2f}" y1="{box.y0:.2f}" '
        f'x2="{sx(0):.2f}" y2="{box.y1:.2f}"/>'
    )
    rows = []
    for i, r in enumerate(list(d.iter_rows(named=True)) + [allr]):
        y = box.y0 + i * rowh + rowh / 2
        a, b = float(r["gap_early"]), float(r["gap_late"])
        is_all = r["diagnosis"] == "ALL"
        name = "all of the above" if is_all else r["diagnosis"]
        parts.append(
            f'<line class="dumb{" emph" if is_all else ""}" x1="{sx(a):.2f}" y1="{y:.2f}" '
            f'x2="{sx(b):.2f}" y2="{y:.2f}"/>'
        )
        t = (f'{name}: gap {a:+.3f} early, {b:+.3f} late; '
             f'difference-in-differences {r["did"]:+.3f} {r["did_ci"]}')
        parts.append(marked("circle", f'class="dot" cx="{sx(a):.2f}" cy="{y:.2f}" r="4.5" fill="{SEQ2_DIM}"', t))
        parts.append(marked("circle", f'class="dot" cx="{sx(b):.2f}" cy="{y:.2f}" r="5" fill="{S2}"', t))
        parts.append(
            f'<text class="cat{" emph" if is_all else ""}" x="{box.x0 - 10:.2f}" '
            f'y="{y + 4:.2f}" text-anchor="end">{esc(name)}</text>'
        )
        parts.append(
            f'<text class="dlabel" x="{sx(b) + 9:.2f}" y="{y + 4:.2f}">{b:+.2f}</text>'
        )
        rows.append([name, f"{a:+.4f}", f"{b:+.4f}", f"{r['did']:+.4f}", r["did_ci"]])
    parts.append(
        f'<text class="axis-t" x="{box.x0 + box.iw / 2:.2f}" y="{box.height - 4:.2f}" '
        f'text-anchor="middle">P(plays next | high hope) − P(plays next | low hope)</text>'
    )
    svg = f'<svg viewBox="0 0 {box.width} {box.height}" role="img" '\
          f'aria-label="Difference in differences by diagnosis">{"".join(parts)}</svg>'
    return figure(
        "The gap does not exist early in the season. It opens late.",
        "Difference in return probability between players on high-hope and low-hope "
        "teams, early in the season (pale) and late (solid). Rotation players, within "
        "the same diagnosis.",
        svg,
        f"Early in the season the gap is <strong>{float(allr['gap_early']):+.3f}</strong> "
        f"across {int(allr['n_late_low']) + int(allr['n_late_high']):,} observations — nothing. "
        f"Late it is <strong>{float(allr['gap_late']):+.3f}</strong>. "
        f"Difference-in-differences {float(allr['did']):+.3f} "
        f"{allr['did_ci'].replace(',', ', ')}, excluding zero for every diagnosis separately. "
        "Worse rosters or worse medical staff on bad teams would show up in both halves; "
        "they show up in neither.",
        legend([("early season", SEQ2_DIM), ("late season", S2)]),
        details_table(["diagnosis", "gap early", "gap late", "DiD", "95% CI"], rows),
    )


def fig_event_study(es: pl.DataFrame) -> str:
    """Hazard either side of mathematical elimination, with a CI band."""
    box = Box(720, 280, left=52, bottom=42)
    xs = es["games_rel_elimination"].to_list()
    lo_x, hi_x = min(xs), max(xs) + 3
    hi_y = float(es["ci_hi"].max()) * 1.1
    sx = lambda v: box.x0 + ((v - lo_x) / (hi_x - lo_x)) * box.iw
    sy = lambda p: box.y1 - (p / hi_y) * box.ih
    parts = [grid_y(box, nice_ticks(0, hi_y, 4), sy, 2),
             grid_x(box, [-12, -6, 0, 6, 12], sx)]
    up = " ".join(f"{'M' if i == 0 else 'L'}{sx(r['games_rel_elimination'] + 1.5):.2f},{sy(r['ci_hi']):.2f}"
                  for i, r in enumerate(es.iter_rows(named=True)))
    dn = " ".join(f"L{sx(r['games_rel_elimination'] + 1.5):.2f},{sy(r['ci_lo']):.2f}"
                  for r in reversed(list(es.iter_rows(named=True))))
    parts.append(f'<path class="band" d="{up} {dn} Z" fill="{S2}"/>')
    pts = [(r["games_rel_elimination"] + 1.5, r["p_return"]) for r in es.iter_rows(named=True)]
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{sx(x):.2f},{sy(p):.2f}"
        for i, (x, p) in enumerate(pts)
    )
    parts.append(f'<path class="line" stroke="{S2}" d="{path}"/>')
    parts.append(
        f'<line class="zero" x1="{sx(0):.2f}" y1="{box.y0:.2f}" x2="{sx(0):.2f}" y2="{box.y1:.2f}"/>'
    )
    parts.append(
        f'<text class="dlabel" x="{sx(0) + 6:.2f}" y="{box.y0 + 12:.2f}">eliminated</text>'
    )
    pre_mean = float(
        es.filter(pl.col("games_rel_elimination") < 0)["p_return"].mean()
    )
    rows = []
    for r in es.iter_rows(named=True):
        x, p = r["games_rel_elimination"] + 1.5, r["p_return"]
        t = (f'{r["games_rel_elimination"]} to {r["games_rel_elimination"] + 2} games: '
             f'{p:.1%} [{r["ci_lo"]:.1%}, {r["ci_hi"]:.1%}] (n={r["n_rows"]:,})')
        parts.append(marked("circle", f'class="dot" cx="{sx(x):.2f}" cy="{sy(p):.2f}" r="4.5" fill="{S2}"', t))
        rows.append([f"{r['games_rel_elimination']} to {r['games_rel_elimination'] + 2}",
                     f"{r['n_rows']:,}", f"{p:.4f}", f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"])
    parts.append(
        f'<text class="axis-t" x="{box.x0 + box.iw / 2:.2f}" y="{box.height - 4:.2f}" '
        f'text-anchor="middle">team games relative to mathematical elimination</text>'
    )
    svg = f'<svg viewBox="0 0 {box.width} {box.height}" role="img" '\
          f'aria-label="Event study around elimination">{"".join(parts)}</svg>'
    return figure(
        "Flat before elimination, falling after",
        "Return probability by team games either side of the date a team could no "
        "longer reach the play-in. Shaded band is a 95% bootstrap interval.",
        svg,
        f"Before elimination the hazard sits flat at about {pre_mean:.2f}; by ten games "
        f"after it is {float(es['p_return'][-1]):.2f}. It is a modest shift because "
        "elimination arrives so late that most of the response has already happened "
        "as the team's hopes faded — which is why playoff hope, not the elimination "
        "date, carries the identification.",
        "",
        details_table(["games relative to elimination", "rows", "P(plays next)", "95% CI"], rows),
    )


def fig_debias(deb: pl.DataFrame) -> str:
    """How much of each published duration is the team's decision.

    Plotted as a share of the diagnosis's own duration rather than in absolute
    games. The headline claim is about proportion — an ACL moves by almost
    none of its length, a sore knee by a sixth of its — and absolute games
    buries that, because a 1% slice of a 46-game tear is larger than a 16% one
    of a 4-game rest night.
    """
    d = (
        deb.filter(pl.col("n") >= 40)
        .with_columns((pl.col("delta") / pl.col("games_as_observed")).alias("share"))
        .sort("share", descending=True)
    )
    rowh = 24
    box = Box(720, rowh * d.height + 52, left=116, right=132, top=10, bottom=38)
    hi = float(d["share"].max()) * 1.14
    sx = lambda v: box.x0 + (v / hi) * box.iw
    parts = [grid_x(box, nice_ticks(0, hi, 5), sx, label=lambda t: f"{t:.0%}")]
    rows = []
    for i, r in enumerate(d.iter_rows(named=True)):
        y = box.y0 + i * rowh + (rowh - 16) / 2
        share, v = float(r["share"]), float(r["delta"])
        parts.append(marked(
            "path",
            f'd="{hbar(box.x0, sx(share), y, 16)}" fill="{SEQ}"',
            f'{r["ailment_class"]}: {r["games_as_observed"]:.2f} games as observed, '
            f'{r["games_neutral"]:.2f} at neutral urgency — {share:.1%} of its own '
            f'length is the team\'s situation (n={r["n"]:,})',
        ))
        parts.append(
            f'<text class="cat" x="{box.x0 - 10:.2f}" y="{y + 12:.2f}" '
            f'text-anchor="end">{esc(r["ailment_class"].replace("_", " "))}</text>'
        )
        parts.append(
            f'<text class="dlabel" x="{sx(share) + 9:.2f}" y="{y + 12:.2f}">'
            f'{share:.0%} <tspan class="cat">({v:+.2f} games)</tspan></text>'
        )
        rows.append([r["ailment_class"], f"{r['n']:,}", f"{r['games_as_observed']:.2f}",
                     f"{r['games_neutral']:.2f}", f"{v:+.2f}", f"{share:.1%}"])
    parts.append(
        f'<text class="axis-t" x="{box.x0 + box.iw / 2:.2f}" y="{box.height - 4:.2f}" '
        f'text-anchor="middle">share of the absence attributable to the team\'s situation</text>'
    )
    svg = f'<svg viewBox="0 0 {box.width} {box.height}" role="img" '\
          f'aria-label="De-biasing effect by ailment">{"".join(parts)}</svg>'
    top = d.row(0, named=True)
    bot = d.sort("share").row(0, named=True)
    return figure(
        "You cannot tank an ACL",
        "Share of each diagnosis's predicted absence that disappears when the team's "
        "playoff position is held at neutral urgency.",
        svg,
        f"The severe, non-discretionary diagnoses barely move — a tear loses "
        f"{float(deb.filter(pl.col('ailment_class') == 'rupture_tear')['delta'][0] / deb.filter(pl.col('ailment_class') == 'rupture_tear')['games_as_observed'][0]):.0%} "
        f"of its length, surgery "
        f"{float(deb.filter(pl.col('ailment_class') == 'surgery')['delta'][0] / deb.filter(pl.col('ailment_class') == 'surgery')['games_as_observed'][0]):.0%}. "
        f"The discretionary ones lose six to ten times that share: "
        f"{top['ailment_class'].replace('_', ' ')} {float(top['share']):.0%}, against "
        f"{bot['ailment_class'].replace('_', ' ')} {float(bot['share']):.0%} at the "
        "other end. That pattern — proportional to how much discretion the diagnosis "
        "allows — is the strongest sign the measure is picking up decisions rather "
        "than noise.",
        "",
        details_table(
            ["ailment", "n", "as observed", "at neutral urgency", "difference", "share"], rows
        ),
    )


def fig_day_calibration(cal: list[dict]) -> str:
    """Predicted against realised availability at each day horizon."""
    box = Box(720, 260, left=52, bottom=40)
    xs = [c["horizon_days"] for c in cal]
    sx = lambda v: box.x0 + (xs.index(v) / (len(xs) - 1)) * box.iw
    sy = lambda p: box.y1 - ((p - 0.5) / 0.55) * box.ih
    parts = [grid_y(box, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], sy, 2)]
    for label, key, colour in [("Predicted", "mean_pred", S1), ("Realised", "actual_rate", S2)]:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(c['horizon_days']):.2f},{sy(c[key]):.2f}"
            for i, c in enumerate(cal)
        )
        parts.append(f'<path class="line" d="{path}" stroke="{colour}"/>')
        for c in cal:
            parts.append(marked(
                "circle",
                f'class="dot" cx="{sx(c["horizon_days"]):.2f}" '
                f'cy="{sy(c[key]):.2f}" r="4.5" fill="{colour}"',
                f'within {c["horizon_days"]} days — {label.lower()} {c[key]:.1%} '
                f'(n={c["n_evaluable"]:,})',
            ))
        last = cal[-1]
        parts.append(
            f'<text class="dlabel" x="{sx(last["horizon_days"]) - 6:.2f}" '
            f'y="{sy(last[key]) + (14 if label == "Predicted" else -8):.2f}" '
            f'text-anchor="end">{esc(label)}</text>'
        )
    for c in cal:
        parts.append(
            f'<text class="tick" x="{sx(c["horizon_days"]):.2f}" y="{box.y1 + 16:.2f}" '
            f'text-anchor="middle">{c["horizon_days"]}</text>'
        )
    parts.append(
        f'<text class="axis-t" x="{box.x0 + box.iw / 2:.2f}" y="{box.height - 4:.2f}" '
        f'text-anchor="middle">days from being ruled out</text>'
    )
    svg = f'<svg viewBox="0 0 {box.width} {box.height}" role="img" '\
          f'aria-label="Calibration of availability by day">{"".join(parts)}</svg>'
    return figure(
        "“Will he be available by then?” is answerable. “How many days?” mostly is not.",
        "Predicted against realised probability the player is back within D days, "
        "held-out 2025-26 season.",
        svg,
        "Well calibrated, running 3–4 points pessimistic at every horizon. The matching "
        "point estimate is much weaker: a median-day forecast gives a mean absolute "
        "error of 5.7 days against 6.4 for predicting “five days” for everyone, because "
        "half of all absences are three days or less.",
        legend([("predicted", S1), ("realised", S2)]),
        details_table(
            ["horizon (days)", "evaluable spells", "predicted", "realised", "Brier skill"],
            [[c["horizon_days"], f"{c['n_evaluable']:,}", f"{c['mean_pred']:.3f}",
              f"{c['actual_rate']:.3f}", f"{c['brier_skill']:.3f}"] for c in cal],
        ),
    )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  color-scheme:light;
  --surface-0:#f7f7f4; --surface-1:#fcfcfb; --surface-2:#f0efec;
  --border:#e2e1dc; --grid:#e8e7e2;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#78776f;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --seq-200:#9ec5f4; --seq-450:#2a78d6;
  --seq2-200:#f8c0a2;
  --accent:#2a78d6;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --surface-0:#141413; --surface-1:#1a1a19; --surface-2:#232322;
  --border:#33332f; --grid:#2b2b29;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#9a998e;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --seq-200:#184f95; --seq-450:#3987e5;
  --seq2-200:#8a3a1c;
  --accent:#3987e5;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --surface-0:#141413; --surface-1:#1a1a19; --surface-2:#232322;
  --border:#33332f; --grid:#2b2b29;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#9a998e;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --seq-200:#184f95; --seq-450:#3987e5;
  --seq2-200:#8a3a1c;
  --accent:#3987e5;
}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
  font:16px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-variant-numeric:tabular-nums}
.wrap{max-width:860px;margin:0 auto;padding:48px 16px 96px}
header{border-bottom:1px solid var(--border);padding-bottom:28px;margin-bottom:36px}
h1{font-size:clamp(28px,5vw,40px);line-height:1.15;letter-spacing:-.02em;margin:0 0 12px}
.lede{font-size:18px;color:var(--text-secondary);margin:0 0 6px;max-width:62ch}
.meta{font-size:13px;color:var(--text-muted);margin-top:14px}
h2{font-size:22px;letter-spacing:-.01em;margin:56px 0 6px;padding-top:14px;
  border-top:1px solid var(--border)}
h2 .num{color:var(--text-muted);font-weight:400;margin-right:10px}
h3{font-size:17px;margin:0 0 4px;letter-spacing:-.01em}
p{max-width:68ch}
.sub{color:var(--text-secondary);font-size:14px;margin:0 0 12px;max-width:66ch}
figure{margin:28px 0 8px;padding:20px;background:var(--surface-1);
  border:1px solid var(--border);border-radius:12px}
.plot{margin:4px -6px 0;overflow:visible}
.plot svg{display:block;width:100%;height:auto;overflow:visible}
figcaption{font-size:14px;color:var(--text-secondary);margin-top:14px;max-width:66ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:28px 0}
.stat{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:16px}
.stat-v{font-size:30px;font-weight:600;letter-spacing:-.02em;line-height:1.1}
.stat-l{font-size:13px;color:var(--text-secondary);margin-top:5px}
.stat-n{font-size:12px;color:var(--text-muted);margin-top:5px}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:0 0 6px;font-size:13px;
  color:var(--text-secondary)}
.lg-item{display:inline-flex;align-items:center;gap:7px}
.lg-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.band{opacity:.12;stroke:none}
.dot{stroke:var(--surface-1);stroke-width:2}
.dumb{stroke:var(--grid);stroke-width:3;stroke-linecap:round}
.dumb.emph{stroke:var(--border)}
.grid{stroke:var(--grid);stroke-width:1}
.zero{stroke:var(--text-muted);stroke-width:1;opacity:.55}
.tick{fill:var(--text-muted);font-size:11px}
.cat{fill:var(--text-secondary);font-size:12.5px}
.cat.emph{fill:var(--text-primary);font-weight:600}
.dlabel{fill:var(--text-primary);font-size:12px;font-weight:600}
.axis-t{fill:var(--text-muted);font-size:11.5px}
.hit{fill:transparent}
details{margin-top:14px}
summary{cursor:pointer;font-size:13px;color:var(--accent);width:max-content}
.t-wrap{overflow-x:auto;margin-top:12px}
table{border-collapse:collapse;font-size:13px;width:100%}
caption{text-align:left;color:var(--text-secondary);font-size:13px;padding-bottom:8px}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--border);
  white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--text-secondary);font-weight:600;border-bottom:1px solid var(--text-muted)}
tbody tr:hover{background:var(--surface-2)}
#tip{position:fixed;z-index:50;pointer-events:none;opacity:0;transition:opacity .09s;
  background:var(--text-primary);color:var(--surface-1);font-size:12.5px;
  padding:7px 10px;border-radius:7px;max-width:290px;line-height:1.45;
  box-shadow:0 3px 14px rgba(0,0,0,.2)}
.callout{background:var(--surface-2);border-left:3px solid var(--accent);
  border-radius:0 8px 8px 0;padding:14px 18px;margin:22px 0;font-size:15px}
.callout strong{color:var(--text-primary)}
.toggle{position:fixed;top:14px;right:14px;z-index:40;background:var(--surface-1);
  color:var(--text-secondary);border:1px solid var(--border);border-radius:8px;
  padding:7px 11px;font-size:12.5px;cursor:pointer;font-family:inherit}
footer{margin-top:64px;padding-top:22px;border-top:1px solid var(--border);
  font-size:13px;color:var(--text-muted)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em;
  background:var(--surface-2);padding:1px 5px;border-radius:4px}
@media print{.toggle{display:none}figure{break-inside:avoid}details{display:none}}
"""

JS = """
(function(){
  var tip=document.getElementById('tip');
  function show(e,t){tip.textContent=t;tip.style.opacity='1';move(e)}
  function move(e){
    var r=tip.getBoundingClientRect(),x=e.clientX+14,y=e.clientY+14;
    if(x+r.width>innerWidth-8)x=e.clientX-r.width-14;
    if(y+r.height>innerHeight-8)y=e.clientY-r.height-14;
    tip.style.left=x+'px';tip.style.top=y+'px';
  }
  document.addEventListener('mouseover',function(e){
    var el=e.target.closest('[data-tip]');if(el)show(e,el.getAttribute('data-tip'));
  });
  document.addEventListener('mousemove',function(e){
    if(tip.style.opacity==='1')move(e);
  });
  document.addEventListener('mouseout',function(e){
    if(e.target.closest('[data-tip]'))tip.style.opacity='0';
  });
  var btn=document.querySelector('.toggle');
  btn.addEventListener('click',function(){
    var dark=document.documentElement.getAttribute('data-theme')==='dark'||
      (!document.documentElement.getAttribute('data-theme')&&
       matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.setAttribute('data-theme',dark?'light':'dark');
  });
})();
"""


def main() -> None:
    rep = paths.ensure_reports()
    docs = paths.ensure_docs()
    b = paths.BUILD

    sp = pl.read_parquet(b / "spells.parquet")
    rows_df = pl.read_parquet(b / "hazard_rows.parquet")
    diag = json.loads((rep / "build_diagnostics.json").read_text())

    km = pl.read_csv(rep / "km_by_ailment.csv")
    did = pl.read_csv(rep / "incentive_did.csv")
    es = pl.read_csv(rep / "elimination_event_study.csv")
    deb = pl.read_csv(rep / "km_debiased_by_ailment.csv")
    comp = pl.read_csv(rep / "incentive_model_comparison.csv")
    dur = pl.read_csv(rep / "duration_metrics.csv")
    pg = pl.read_csv(rep / "per_game_return_metrics.csv")

    inc = standings.playin_probability().select(
        "season", "game_date", "team_slug", "incentive_state"
    )
    state = (
        rows_df.join(inc, on=["season", "game_date", "team_slug"], how="inner")
        .group_by("incentive_state")
        .agg(pl.len().alias("rows"), pl.col("spell_id").n_unique().alias("spells"),
             pl.col("returns_next").mean().alias("p"))
    )

    d, e = sp["games_missed"].to_numpy(), sp["event"].to_numpy()
    km_mean, naive = hz.km_mean(d, e), float(d[e == 1].mean())

    day_cal = [
        {k: (int(v) if k in ("horizon_days", "n_evaluable") else float(v))
         for k, v in r.items()}
        for r in pl.read_csv(rep / "day_horizon_calibration.csv").iter_rows(named=True)
    ]

    med = dur.filter(pl.col("point") == "median")
    best = med.filter(pl.col("model").str.contains("histgb")).row(0, named=True)
    kmbase = med.filter(pl.col("model") == "km by region x ailment x status").row(0, named=True)
    blind = med.filter(pl.col("model").str.contains("observed spells only")).row(0, named=True)

    kpis = "".join([
        stat(f"{sp.height:,}", "injury spells", "2021-22 to 2025-26"),
        stat(f"{1 - float(sp['event'].mean()):.0%}", "never show a return",
             "season end, trade, G League"),
        stat(f"{km_mean:.1f}", "mean games missed",
             f"{naive:.1f} if censoring is ignored"),
        stat(f"{best['c_index']:.3f}", "C-index, held-out season",
             f"{kmbase['c_index']:.3f} for the best lookup table"),
    ])

    model_rows = [
        ["Flat Kaplan-Meier median", "—", f"{med.filter(pl.col('model') == 'km global').row(0, named=True)['mae_observed']:.2f}",
         f"{med.filter(pl.col('model') == 'km global').row(0, named=True)['mae_observed_long']:.2f}", "0.500"],
        ["Lookup: region x ailment x status", "—", f"{kmbase['mae_observed']:.2f}",
         f"{kmbase['mae_observed_long']:.2f}", f"{kmbase['c_index']:.3f}"],
        ["LightGBM, censoring-blind", "observed spells only",
         f"{blind['mae_observed']:.2f}", f"{blind['mae_observed_long']:.2f}",
         f"{blind['c_index']:.3f}"],
        ["Hazard, HistGradientBoosting", "all spells",
         f"{best['mae_observed']:.2f}", f"{best['mae_observed_long']:.2f}",
         f"{best['c_index']:.3f}"],
    ]
    pg_rows = [
        [r["model"], f"{r['auc']:.4f}", f"{r['log_loss']:.4f}", f"{r['brier_skill']:.3f}"]
        for r in pg.iter_rows(named=True)
    ]
    comp_rows = [
        [r["model"], r["n_features"], f"{r['log_loss']:.4f}", f"{r['auc']:.4f}",
         f"{r['log_loss_late']:.4f}", f"{r['auc_late']:.4f}", f"{r['c_index']:.4f}"]
        for r in comp.iter_rows(named=True)
    ]

    body = f"""<button class="toggle" type="button">Light / dark</button>
<div id="tip" role="status" aria-live="polite"></div>
<div class="wrap">
<header>
  <h1>How long do NBA injuries keep players out?</h1>
  <p class="lede">Five seasons of the per-game injury report, turned into
  {sp.height:,} player injury spells and modelled two ways: how many games a
  player will miss, and how much of the absence is the injury at all.</p>
  <p class="meta">Seasons 2021-22 to 2025-26 &middot; nba.nba.injuries &middot;
  every model scored on time-based splits, trained on seasons that finished
  before the one they score &middot; generated {date.today().isoformat()} from
  <code>scripts/08_report_html.py</code></p>
</header>

<div class="kpis">{kpis}</div>

<h2><span class="num">1</span>The data has to be built from the report, not the box score</h2>
<p>The box score is not a record of availability. A player on a long absence is
simply missing from it — Klay Thompson has no 2021-22 regular-season box-score
row until the night he came back. So games missed cannot be counted from box
scores; the injury report is the only source that says a player was unavailable,
and it says it one game at a time.</p>
<p>Five data faults had to be fixed before any of this was trustworthy: a null
player id on up to 38% of rows in a season, 2021-22 schedule dates sitting one
day early, the report's own <code>game_id</code> shadowing the calendar's on
join and silently dropping a whole season, exactly duplicated rows from
2026-02-20, and — the subtlest — a player listed <em>Doubtful</em> who still
misses the game being scored as a <em>return</em>, which truncated 896 spells.</p>

{fig_survival(sp)}

<div class="callout"><strong>Censoring is the whole ballgame.</strong>
{1 - float(sp['event'].mean()):.0%} of spells never show a return, and they are
the long ones. Any analysis that drops or truncates them is describing a league
in which nobody misses forty games.</div>

{fig_censoring(km)}

<h2><span class="num">2</span>What drives duration</h2>
<p>Pathology, not body part. Region alone ranks severity at a C-index of 0.51 —
it pools a sore knee with a torn ACL. Region crossed with pathology reaches
0.675. The label on the first missed game carries real information over and
above the diagnosis: a player listed <em>Out</em> misses 14.0 games on average,
<em>Questionable</em> 5.5.</p>

{fig_hazard(rows_df)}

<p>Two behavioural findings fall out of the fitted hazard, holding the injury
constant. Rest between games matters about as much as the injury: the chance of
playing the next game runs from 0.139 when it is a back-to-back to 0.310 when it
is six days away. And good teams get players back sooner — 0.137 at a .200 win
rate against 0.180 at .800. That second one is the thread the rest of this report
pulls.</p>
<p>What does <em>not</em> drive duration, and this surprised me: age, size, draft
position and prior injury history. Dropping either the player-attribute block or
the entire fourteen-feature injury-history block moves held-out log loss by less
than 0.001. The dramatic raw age gradient — 17.0 games for under-24s against 9.2
for over-32s — is a censoring artefact: fringe young players get shut down or
sent down, so they are censored at 30% against 18%. Twelve features match all
seventy-two.</p>

<h2><span class="num">3</span>The algorithm</h2>
<p>A discrete-time hazard model. For a spell that misses games 1..<em>m</em>, row
<em>k</em> asks “the player has now missed <em>k</em> games — does he play game
<em>k</em>+1?”. The answer is known for every <em>k</em> &lt; <em>m</em>, and for
<em>k</em> = <em>m</em> only when the return was observed, so a censored spell
contributes its <em>m</em> “still out” rows and no “came back” row. That is
exactly the information it carries, and it is why all {sp.height:,} spells are
usable: {diag['hazard_rows']:,} training rows.</p>
<p>Duration then falls out as a product over the per-game hazards. Days are
converted from the same hazards using the team's real calendar rather than
modelled directly — a player returns <em>for a game</em>, not on an arbitrary
Tuesday, and gaps run from one to seven days, so a daily hazard would be mostly
structural zeros dictated by the schedule.</p>

{table(["model", "AUC", "log loss", "Brier skill"], pg_rows,
       "Per-game return prediction, held-out 2025-26, 7,635 rows, base rate 0.166.")}

<p>Boosted trees beat the forest and the linear model consistently; the two
boosters are indistinguishable. Adding how the report has <em>moved</em> since
onset — a status softened from Out, a diagnosis re-filed — is worth about 1.3
AUC points, less than you would guess, because most long absences stay filed as
“Out” right up to the night the player returns.</p>

{table(["model", "trained on", "MAE (games)", "MAE on spells of 5+ games", "C-index"],
       model_rows,
       "Duration at the moment the player is ruled out, held-out 2025-26, median point estimate.")}

<div class="callout">The censoring-blind fit wins overall MAE and loses where it
matters: worse on long spells and worse at ranking severity. Its MAE win is an
artefact — MAE is scored on observed spells, which is precisely the population it
trained on, and that population is mostly one-game absences. <strong>C-index is
the fair column</strong>: it uses the censored spells and does not depend on the
choice of point estimate.</div>

{fig_day_calibration(day_cal)}

<h2><span class="num">4</span>How much of an absence is the injury, and how much is the team?</h2>
<p>A team's playoff position changes what it wants from a borderline player and
cannot change how a torn ligament heals. That asymmetry is the whole design:
variation in playoff position, holding the diagnosis and the player's role fixed,
moves team <em>willingness</em> and not medical <em>readiness</em>.</p>

{fig_incentive(state)}
{fig_did(did)}
{fig_event_study(es)}

<p>The 2023-24 Player Participation Policy, which restricted sitting healthy
players, narrowed the gap from 0.326 to 0.238 — with low-hope teams returning
players <em>more</em> often after it, 0.150 to 0.174, the direction the rule
intended. It did not close it. The 2019 lottery reform is not testable here: it
predates the injury report entirely.</p>

{fig_debias(deb)}

<p>Rather than classify spells — there is no label for “tanking injury”, and the
injury is almost always real — the hazard is fitted with the incentive features
and then evaluated twice per spell: once at the team's actual position, once at a
neutral one. For spells starting after the team was already eliminated, the
predicted absence is 18.09 games against 9.51 at neutral urgency: +8.6 games,
+19.5 days.</p>

<div class="callout">The measure recovered a known episode unprompted. Among the
highest discretion scores in the sample are Kyrie Irving (+24.4 games), Tim
Hardaway Jr. (+22.9) and Maxi Kleber (+14.7), all filed out by Dallas on
<strong>2023-04-07</strong> — the night Dallas sat its starters with a draft pick
at stake and was fined by the league for it. Nothing in the model knows about
that episode, or about tanking.</div>

<h2><span class="num">5</span>It does not improve prediction</h2>
<p>Worth being blunt about, because it was the one thing I expected to go the
other way. Adding the incentive features changes held-out accuracy by nothing —
not even late in the season, where the entire effect lives.</p>

{table(["model", "features", "log loss", "AUC", "log loss (late)", "AUC (late)", "C-index"],
       comp_rows,
       "“Late” means spells starting past 80% of the season.")}

<p><code>team_win_pct_before</code> and <code>season_progress</code> were already
in the feature set and already proxy the incentive well enough to forecast with.
What the explicit measure buys is <strong>interpretation</strong> — the ability to
say what an injury costs at ordinary urgency, and to attribute the remainder —
not accuracy.</p>

<h2><span class="num">6</span>What not to trust</h2>
<p><strong>Point forecasts for catastrophic injuries are close to meaningless, and
that is the data rather than the model.</strong> Jayson Tatum's Achilles repair
forecast a median of 25 games against an actual 62. But the model's per-game
hazard for those cases sits at 0.02–0.03, which the raw data confirms is right.
A hazard that flat <em>implies</em> a median in the low twenties with an enormous
right tail, so 62 sits inside the predicted distribution. Report
P(out past 20 games), not a number of games.</p>
<p><strong>Rotation status confounds injury severity.</strong> Recent minutes
cannot distinguish “badly hurt” from “not in the rotation”, so the model's
longest calls skew to two-way and deep-bench players. On genuinely fresh injuries
to rotation players — the honest subpopulation — the C-index is 0.701 and the
median-day error 2.03 games, against 0.744 over all spells.</p>
<p><strong>Censoring is informative in both directions.</strong> An earlier
version of this analysis claimed season-end censoring was plausibly independent
of the injury, so the long tail was if anything understated. That was wrong:
43% of late-season spells run to season end on teams below .350 against 20% on
winning teams, for the same diagnoses, so some season-end censoring is a shutdown
whose medical duration was shorter than the censoring time. The net is not
knowable from duration data alone.</p>
<p><strong>93 of the top 100 discretion scores are censored spells</strong> with a
mean of 4.6 games missed. The measure is overwhelmingly detecting end-of-season
shutdowns — the dominant form of the behaviour, but a narrow window, and the
counterfactual is never observed there. And the neutral prediction is a model
output, not a measurement: it is only as good as the assumption that playoff
position does not affect healing.</p>
<p><strong>Five seasons.</strong> 39 spells carry the catastrophic flag. Anything
here about ACL and Achilles duration rests on dozens of cases, not hundreds.</p>

<footer>
  Reproduce with <code>uv run python scripts/0{{1..8}}_*.py</code>. Narrative
  version in <code>docs/</code>, every underlying table in <code>reports/</code>;
  model code in <code>src/injury_duration/</code>. Charts are inline SVG with no
  external dependencies, so this file renders offline from a clone.
</footer>
</div>"""

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NBA injury duration</title>
<meta name="description" content="Modelling how long NBA injuries keep players out, and how much of an absence is the team's decision rather than the injury.">
<style>{CSS}</style>
</head>
<body data-palette="#2a78d6,#eb6834,#1baf7a">
{body}
<script>{JS}</script>
</body>
</html>"""

    out = docs / "injury_duration.html"
    out.write_text(page)
    print(f"wrote {out}  ({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

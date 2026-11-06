"""Minimal SVG chart primitives for the standalone HTML report.

Hand-rolled rather than matplotlib or a JS charting library for one reason:
the report has to be a single file that opens from a git clone with no
network and no build step. That rules out a CDN, and a PNG would lose the
hover values and the text selection.

The mark specs are fixed here so every figure in the report agrees: bars at
most 24px thick with a 4px rounded data-end squared off at the baseline, 2px
lines, markers at least 8px across with a 2px surface ring, hairline solid
gridlines, and a 2px surface gap between touching marks. Colours arrive as
CSS custom properties, so light and dark are one stylesheet swap rather than
two sets of figures.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

BAR_MAX = 24.0
BAR_RADIUS = 4.0
GAP = 2.0


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def fmt(v: float, dp: int = 2) -> str:
    if v is None:
        return "–"
    return f"{v:,.{dp}f}".rstrip("0").rstrip(".") if dp else f"{v:,.0f}"


@dataclass
class Box:
    """Plot geometry: outer size and the inset that leaves room for axes."""

    width: float
    height: float
    left: float = 54.0
    right: float = 18.0
    top: float = 12.0
    bottom: float = 34.0

    @property
    def x0(self) -> float:
        return self.left

    @property
    def x1(self) -> float:
        return self.width - self.right

    @property
    def y0(self) -> float:
        return self.top

    @property
    def y1(self) -> float:
        return self.height - self.bottom

    @property
    def iw(self) -> float:
        return self.x1 - self.x0

    @property
    def ih(self) -> float:
        return self.y1 - self.y0


def hbar(x_base: float, x_end: float, y: float, thick: float, r: float = BAR_RADIUS) -> str:
    """Horizontal bar: square at the baseline, rounded at the data end."""
    thick = min(thick, BAR_MAX)
    r = min(r, thick / 2, abs(x_end - x_base))
    if x_end >= x_base:
        return (
            f"M{x_base:.2f},{y:.2f} H{x_end - r:.2f} "
            f"a{r:.2f},{r:.2f} 0 0 1 {r:.2f},{r:.2f} "
            f"V{y + thick - r:.2f} a{r:.2f},{r:.2f} 0 0 1 {-r:.2f},{r:.2f} "
            f"H{x_base:.2f} Z"
        )
    return (
        f"M{x_base:.2f},{y:.2f} H{x_end + r:.2f} "
        f"a{r:.2f},{r:.2f} 0 0 0 {-r:.2f},{r:.2f} "
        f"V{y + thick - r:.2f} a{r:.2f},{r:.2f} 0 0 0 {r:.2f},{r:.2f} "
        f"H{x_base:.2f} Z"
    )


def vbar(x: float, y_base: float, y_end: float, thick: float, r: float = BAR_RADIUS) -> str:
    """Vertical column: square at the baseline, rounded at the cap."""
    thick = min(thick, BAR_MAX)
    r = min(r, thick / 2, abs(y_base - y_end))
    return (
        f"M{x:.2f},{y_base:.2f} V{y_end + r:.2f} "
        f"a{r:.2f},{r:.2f} 0 0 1 {r:.2f},{-r:.2f} "
        f"H{x + thick - r:.2f} a{r:.2f},{r:.2f} 0 0 1 {r:.2f},{r:.2f} "
        f"V{y_base:.2f} Z"
    )


def grid_x(box: Box, ticks: list[float], scale, label=None) -> str:
    """Vertical hairline gridlines with labels under the plot.

    `label` overrides the default number formatting, for axes whose units are
    not the raw values — a share axis labelled 0.05 when every mark beside it
    reads 5% makes the reader do arithmetic the chart should have done.
    """
    label = label or (lambda t: fmt(t, 0) if float(t).is_integer() else fmt(t))
    out = []
    for t in ticks:
        x = scale(t)
        out.append(
            f'<line class="grid" x1="{x:.2f}" y1="{box.y0:.2f}" '
            f'x2="{x:.2f}" y2="{box.y1:.2f}"/>'
        )
        out.append(
            f'<text class="tick" x="{x:.2f}" y="{box.y1 + 16:.2f}" '
            f'text-anchor="middle">{esc(label(t))}</text>'
        )
    return "".join(out)


def grid_y(box: Box, ticks: list[float], scale, dp: int = 2) -> str:
    """Horizontal hairline gridlines with labels left of the plot."""
    out = []
    for t in ticks:
        y = scale(t)
        out.append(
            f'<line class="grid" x1="{box.x0:.2f}" y1="{y:.2f}" '
            f'x2="{box.x1:.2f}" y2="{y:.2f}"/>'
        )
        out.append(
            f'<text class="tick" x="{box.x0 - 8:.2f}" y="{y + 4:.2f}" '
            f'text-anchor="end">{esc(fmt(t, dp))}</text>'
        )
    return "".join(out)


def marked(tag: str, attrs: str, text: str) -> str:
    """An SVG element carrying a hover payload and a native `<title>` child.

    The `<title>` is the fallback for no-JS, print and screen readers, so the
    element must be written with a real open and close tag — a self-closing
    mark cannot hold a child, and emitting one anyway silently breaks the
    nesting for everything that follows it in the SVG.
    """
    return (
        f"<{tag} {attrs} data-tip=\"{esc(text)}\">"
        f"<title>{esc(text)}</title></{tag}>"
    )


def legend(items: list[tuple[str, str]]) -> str:
    """Swatch + label per series. Always present for two or more series."""
    bits = [
        f'<span class="lg-item"><span class="lg-dot" style="background:{esc(c)}"></span>'
        f"{esc(label)}</span>"
        for label, c in items
    ]
    return f'<div class="legend">{"".join(bits)}</div>'


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values spanning [lo, hi]."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = step * int(lo / step)
    ticks, t = [], start
    while t <= hi + step * 0.5:
        if t >= lo - step * 0.001:
            ticks.append(round(t, 10))
        t += step
    return ticks

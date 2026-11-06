"""Injury-duration modelling for the NBA injury report.

The package turns the per-game injury report (`nba.nba.injuries`) into player
injury *spells*, attaches features that are knowable at the moment a player is
first ruled out, and fits models that predict how long the spell will last.

Read in this order: `taxonomy` (what the report says) -> `calendar` and `ids`
(making it join) -> `spells` (what a spell is, and when it is censored) ->
`features` -> `hazard` (the framing) -> `models` / `evaluate` -> `forecast`.
"""

__all__ = [
    "calendar",
    "design",
    "evaluate",
    "experiment",
    "features",
    "forecast",
    "hazard",
    "ids",
    "models",
    "paths",
    "spells",
    "taxonomy",
]

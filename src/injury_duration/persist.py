"""Persist and load the fitted onset/live hazard models.

The bundle holds the two models the forecast path uses: `onset` (everything
known when the player is ruled out) and `live` (adds how the report has moved
since). Each `HazardModel` carries its own `Design` encoder, so a loaded bundle
can score rows without rebuilding vocabularies or medians.

A pickle is only as portable as the libraries that produced it. The metadata
records the exact versions used at fit time and `load` refuses a mismatch by
default; pass `strict=False` to inspect an older artifact.
"""

from __future__ import annotations

import json
import pickle
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import models

BUNDLE_VERSION = 1
CHECKED_PACKAGES = ("scikit-learn", "lightgbm", "numpy", "polars")


@dataclass
class ModelBundle:
    onset: models.HazardModel
    live: models.HazardModel
    metadata: dict


def package_versions() -> dict[str, str]:
    out = {}
    for name in CHECKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def build_metadata(
    *,
    seasons: list[str],
    rows: int,
    spells: int,
    features: dict[str, list[str]],
    params: dict[str, dict],
    metrics: dict | None = None,
    git_commit: str | None = None,
) -> dict:
    return {
        "bundle_version": BUNDLE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "seasons": seasons,
        "rows": rows,
        "spells": spells,
        "features": features,
        "params": params,
        "metrics": metrics or {},
        "git_commit": git_commit,
    }


def save(bundle: ModelBundle, path: Path) -> Path:
    """Write the pickle atomically, plus a JSON sidecar for humans."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    sidecar = path.with_name("model.json")
    sidecar.write_text(json.dumps(bundle.metadata, indent=2) + "\n")
    return path


def load(path: Path, strict: bool = True) -> ModelBundle:
    """Load a bundle and check the library versions that built it."""
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} is not a ModelBundle")
    if bundle.metadata.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(
            f"{path}: bundle version {bundle.metadata.get('bundle_version')} != {BUNDLE_VERSION}"
        )
    trained = bundle.metadata.get("packages", {})
    current = package_versions()
    mismatched = {
        name: (trained.get(name), current.get(name))
        for name in CHECKED_PACKAGES
        if trained.get(name) != current.get(name)
    }
    if mismatched and strict:
        detail = ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in mismatched.items())
        raise RuntimeError(
            "model was fitted with different library versions "
            f"({detail}); reinstate them or load with strict=False"
        )
    return bundle

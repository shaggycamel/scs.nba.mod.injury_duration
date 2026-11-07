from __future__ import annotations

import polars as pl
import pytest

from injury_duration import models, persist


def _rows(n: int = 40) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": list(range(n)),
            "cat": ["a"] * (n // 2) + ["b"] * (n // 2),
            "returns_next": [0, 1] * (n // 2),
        }
    )


def _bundle() -> persist.ModelBundle:
    model = models.HazardModel("logistic", "logistic", ["x"], ["cat"], {}).fit(_rows())
    metadata = persist.build_metadata(
        seasons=["2025-26"],
        rows=40,
        spells=20,
        features={"onset": ["x", "cat"], "live": ["x", "cat"]},
        params={"onset": {}, "live": {}},
    )
    return persist.ModelBundle(onset=model, live=model, metadata=metadata)


def test_bundle_round_trip(tmp_path):
    path = persist.save(_bundle(), tmp_path / "model.pkl")
    loaded = persist.load(path)
    assert loaded.metadata["bundle_version"] == persist.BUNDLE_VERSION
    assert loaded.onset.hazard(_rows()).shape == (40,)
    assert loaded.live.hazard(_rows()).shape == (40,)
    assert (tmp_path / "model.json").exists()


def test_version_mismatch_is_loud(tmp_path):
    bundle = _bundle()
    bundle.metadata["packages"]["numpy"] = "0.0.0"
    path = persist.save(bundle, tmp_path / "model.pkl")
    with pytest.raises(RuntimeError, match="different library versions"):
        persist.load(path)
    assert persist.load(path, strict=False).metadata["packages"]["numpy"] == "0.0.0"

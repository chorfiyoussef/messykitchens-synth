"""Synthetic assets with the fields written by mksynth.assets.preprocess (upright frame, metres)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_asset(asset_id: str, *, height: float, radius: float, cavity_depth: float | None = None,
               cavity_radius: float | None = None, floor_radius: float | None = None, on_top: bool = True,
               support: bool = True, split: str = "train", bins: int = 24) -> dict:
    """An upright cylinder, optionally with a (possibly tapered) cylindrical cavity open at the top."""
    area = math.pi * radius ** 2
    starts = [height * i / bins for i in range(bins)]
    cavity = None
    if cavity_depth is not None:
        top_r = cavity_radius
        bottom_r = floor_radius if floor_radius is not None else cavity_radius
        floor_z = height - cavity_depth
        levels = [floor_z + cavity_depth * i / 10 for i in range(11)]
        radii = [bottom_r + (top_r - bottom_r) * i / 10 for i in range(11)]
        cavity = {"floor_z": floor_z, "rim_z": height, "depth": cavity_depth, "axis_xy": [0.0, 0.0], "levels": levels,
                  "radius": radii, "opening_radius": top_r,
                  "volume": math.pi * cavity_depth * (top_r ** 2 + top_r * bottom_r + bottom_r ** 2) / 3}
    return {
        "id": asset_id, "category": "test", "split": split, "source": "test", "licence": "CC-BY-4.0",
        "roles": {"container": cavity is not None, "support": support, "on_top": on_top},
        "z_min": 0.0, "z_max": height, "height": height,
        "com": [0.0, 0.0, 0.4 * height],
        "outer_profile_z": starts, "outer_profile_r": [radius] * bins,
        "footprint_area": area, "footprint_xy": [2 * radius, 2 * radius],
        "radius_xy": radius, "bounding_radius": math.hypot(radius, 0.6 * height),
        "top_bbox_area": (2 * radius) ** 2, "top_bbox_xy": [2 * radius, 2 * radius], "top_center_xy": [0.0, 0.0],
        "top_z": height, "volume_hull": area * height, "volume_solid": area * height * (0.3 if cavity else 1.0),
        "mass": 0.3, "cavity": cavity,
    }


@pytest.fixture
def kitchen() -> dict:
    """A small pool: containers that can host nests, stackable boxes and plates, a never-on-top pitcher."""
    assets = [
        make_asset("big_bowl", height=0.08, radius=0.11, cavity_depth=0.07, cavity_radius=0.10, floor_radius=0.06),
        make_asset("pot", height=0.12, radius=0.12, cavity_depth=0.11, cavity_radius=0.11),
        make_asset("mug", height=0.10, radius=0.045, cavity_depth=0.09, cavity_radius=0.038),
        make_asset("cup", height=0.06, radius=0.035),
        make_asset("ramekin", height=0.045, radius=0.05, cavity_depth=0.035, cavity_radius=0.042),
        make_asset("box", height=0.06, radius=0.08),
        make_asset("small_box", height=0.04, radius=0.04),
        make_asset("plate", height=0.02, radius=0.12),
        make_asset("pitcher", height=0.22, radius=0.07, on_top=False),
        make_asset("can", height=0.11, radius=0.033),
    ]
    return {a["id"]: a for a in assets}

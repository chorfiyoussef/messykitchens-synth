"""Stacking and nesting compatibility (paper App. C.3), numpy only.

The paper decides feasibility from a precomputed dictionary of upright top-face bounding-box
areas and object volumes, plus a list of objects that are never placed on top. We keep those
two rules and add a geometric fit test for nesting: the inner object's outer radius profile
must clear the container's cavity radius profile at the depth where it comes to rest. The
same fit gives the rest depth, so nested objects are released just above where they will
settle instead of being dropped from the rim (long drops tunnel through thin walls).

All lengths are metres in the asset's original (upright) mesh frame, as written by
``mksynth.assets.preprocess``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompatConfig:
    max_stack_area_ratio: float = 1.0      # top footprint area <= ratio * support top-face bbox area
    max_stack_volume_ratio: float = 1.0    # top hull volume <= ratio * support hull volume
    min_support_area: float = 0.004        # m^2; smaller top faces cannot carry anything
    max_nest_volume_ratio: float = 0.9     # inner hull volume <= ratio * cavity volume
    nest_clearance: float = 0.003          # radial clearance between inner object and cavity wall (m)
    min_inserted_fraction: float = 0.3     # share of the inner object's height that must sit below the rim
    nest_search_step: float = 0.001
    rim_overhang: float = 0.005            # stacking on a container: radius beyond the opening that rests on the rim
    rim_com_margin: float = 0.005          # ... and the part entering the opening stays this far below the COM

    @classmethod
    def from_dict(cls, values: dict | None) -> "CompatConfig":
        known = {key: value for key, value in (values or {}).items() if key in cls.__dataclass_fields__}
        return cls(**known)


def _profile_heights(asset: dict) -> tuple[np.ndarray, np.ndarray]:
    """Outer radius per height bin, measured from the object's bottom, at both edges of each bin
    (a cavity that narrows downwards is tightest at the lower edge)."""
    starts = np.asarray(asset["outer_profile_z"], dtype=np.float64) - asset["z_min"]
    radii = np.asarray(asset["outer_profile_r"], dtype=np.float64)
    width = (asset["z_max"] - asset["z_min"]) / max(len(starts), 1)
    return np.concatenate([starts, starts + width]), np.concatenate([radii, radii])


def cavity_radius_at(container: dict, height_above_floor: np.ndarray) -> np.ndarray:
    """Free radius of the cavity at heights above its floor; unbounded above the rim."""
    cav = container["cavity"]
    levels = np.asarray(cav["levels"], dtype=np.float64) - cav["floor_z"]
    radius = np.asarray(cav["radius"], dtype=np.float64)
    out = np.interp(height_above_floor, levels, radius)
    return np.where(height_above_floor >= levels[-1], np.inf, out)


def can_stack(top: dict, support: dict, cfg: CompatConfig = CompatConfig()) -> bool:
    """Paper rule: `top` may be dropped onto `support` when sizes and volumes are compatible."""
    if top["id"] == support["id"]:
        return False
    if not support["roles"]["support"] or not top["roles"]["on_top"]:
        return False
    if support["top_bbox_area"] < cfg.min_support_area:
        return False
    if top["footprint_area"] > cfg.max_stack_area_ratio * support["top_bbox_area"]:
        return False
    if top["volume_hull"] > cfg.max_stack_volume_ratio * support["volume_hull"]:
        return False
    if support["roles"]["container"] and support.get("cavity") is not None:
        if nest_fit(top, support, cfg) is not None:
            return False  # it would fall into the cavity: that is nesting, not stacking
        # It must bridge the opening: whatever enters it stays below the top's COM, so the COM
        # rests above the rim (otherwise it wedges in and is nested, see physics_checks.nested_inside).
        entry = rim_entry_height(top, support["cavity"]["opening_radius"] + cfg.rim_overhang)
        if entry > top["com"][2] - top["z_min"] - cfg.rim_com_margin:
            return False
    return True


def rim_entry_height(top: dict, radius: float) -> float:
    """Height above its bottom at which an upright object first gets wider than `radius`
    (how far it sinks into an opening of that radius; inf if it passes through)."""
    heights, radii = _profile_heights(top)
    wide = heights[radii >= radius]
    return float(wide.min()) if len(wide) else float("inf")


def nest_fit(inner: dict, container: dict, cfg: CompatConfig = CompatConfig()) -> dict | None:
    """Rest offset of an upright `inner` object inside `container`, or None if it does not fit."""
    if inner["id"] == container["id"]:
        return None
    if not container["roles"]["container"] or container.get("cavity") is None:
        return None
    if not inner["roles"]["on_top"]:
        return None
    cav = container["cavity"]
    if inner["volume_hull"] > cfg.max_nest_volume_ratio * cav["volume"]:
        return None
    heights, radii = _profile_heights(inner)
    inner_height = inner["z_max"] - inner["z_min"]
    com_height = inner["com"][2] - inner["z_min"]
    for offset in np.arange(0.0, cav["depth"], cfg.nest_search_step):
        free = cavity_radius_at(container, offset + heights)
        if np.all(radii + cfg.nest_clearance <= free):
            inserted = min(1.0, (cav["depth"] - offset) / inner_height)
            # Lids and saucers resting on a rim are stacking, not nesting: the inner object's
            # centre of mass must sit below the rim.
            if inserted < cfg.min_inserted_fraction or offset + com_height >= cav["depth"]:
                return None
            return {"rest_offset": float(offset), "inserted_fraction": float(inserted)}
    return None


def build_table(assets: dict[str, dict], cfg: CompatConfig = CompatConfig()) -> dict:
    """Precompute every feasible (top, support) and (inner, container) pair."""
    stack, nest = {}, {}
    for top_id, top in assets.items():
        for support_id, support in assets.items():
            if can_stack(top, support, cfg):
                stack.setdefault(support_id, []).append(top_id)
            fit = nest_fit(top, support, cfg)
            if fit is not None:
                nest.setdefault(support_id, {})[top_id] = fit
    return {"stack": stack, "nest": nest}

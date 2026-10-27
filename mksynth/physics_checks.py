"""Acceptance tests applied to simulated scenes (numpy only, used inside Blender and in tests).

Poses are 4x4 world transforms of each object's centre-of-mass frame, keyed by object index.
"""

from __future__ import annotations

import numpy as np

from mksynth.geometry import rotation_angle, tilt_angle


def motion(before: dict, after: dict) -> tuple[float, float]:
    """Largest COM translation (m) and rotation (deg) between two pose snapshots."""
    translation = max(np.linalg.norm(np.asarray(after[k])[:3, 3] - np.asarray(before[k])[:3, 3]) for k in after)
    rotation = max(np.degrees(rotation_angle(np.asarray(before[k])[:3, :3], np.asarray(after[k])[:3, :3])) for k in after)
    return float(translation), float(rotation)


def is_settled(before: dict, after: dict, max_translation: float, max_rotation_deg: float) -> bool:
    translation, rotation = motion(before, after)
    return translation <= max_translation and rotation <= max_rotation_deg


def peak_speed(trajectory: list[dict], dt: float) -> float:
    """Largest COM speed (m/s) along a list of consecutive pose snapshots spaced `dt` apart."""
    speed = 0.0
    for before, after in zip(trajectory[:-1], trajectory[1:]):
        speed = max(speed, motion(before, after)[0] / dt)
    return speed


def settle_state(trajectory: list[dict], check_frames: int, dt: float, max_translation: float,
                 max_rotation_deg: float, max_speed: float) -> tuple[str, dict]:
    """Classify a simulated trajectory (one pose snapshot per frame).

    "glitch":  some object moved faster than `max_speed` (the solver exploded);
    "settled": nothing moved more than the thresholds during the last `check_frames` frames;
    "moving":  otherwise.
    """
    speed = peak_speed(trajectory, dt)
    window = min(check_frames, len(trajectory) - 1)
    translation, rotation = motion(trajectory[-1 - window], trajectory[-1])
    info = {"peak_speed": speed, "translation": translation, "rotation_deg": rotation}
    if speed > max_speed:
        return "glitch", info
    if translation <= max_translation and rotation <= max_rotation_deg:
        return "settled", info
    return "moving", info


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def convex_hull_2d(points) -> np.ndarray:
    """Counter-clockwise convex hull of 2D points (Andrew's monotone chain)."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(pts) < 3:
        return pts

    def half(sequence):
        out = []
        for p in sequence:
            while len(out) >= 2 and _cross2(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.asarray(lower[:-1] + upper[:-1])


def support_margin(base_points_xy, com_xy) -> float:
    """Distance from `com_xy` to the edge of the support polygon (convex hull of the base contact
    points): positive inside, negative outside. A loaded support tips when this goes negative."""
    hull = convex_hull_2d(base_points_xy)
    if len(hull) < 3:
        return -np.inf
    com = np.asarray(com_xy, dtype=np.float64)
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 1e-12
    # For a CCW polygon the inside is to the left of every edge.
    signed = _cross2(edges[keep], com - hull[keep]) / lengths[keep]
    return float(signed.min())


def fell_off(poses: dict, centre_xy, radius: float, min_z: float = -0.02) -> list:
    """Objects whose COM left the disc of `radius` around the scene centre or sank below the plane."""
    centre_xy = np.asarray(centre_xy, dtype=np.float64)
    out = []
    for key, pose in poses.items():
        position = np.asarray(pose)[:3, 3]
        if np.linalg.norm(position[:2] - centre_xy) > radius or position[2] < min_z:
            out.append(key)
    return out


def is_upright(pose, max_tilt_deg: float) -> bool:
    return np.degrees(tilt_angle(np.asarray(pose)[:3, :3])) <= max_tilt_deg


def to_original_frame(pose_com, asset: dict, point_world) -> np.ndarray:
    """Express a world point in the asset's original mesh frame (the frame of its props)."""
    pose_com = np.asarray(pose_com, dtype=np.float64)
    local = pose_com[:3, :3].T @ (np.asarray(point_world, dtype=np.float64) - pose_com[:3, 3])
    return local + np.asarray(asset["com"])


def nested_inside(child_pose, parent_pose, parent: dict, radial_slack: float = 0.005) -> bool:
    """Child COM lies inside the parent's cavity cylinder, below its rim."""
    cavity = parent.get("cavity")
    if cavity is None:
        return False
    p = to_original_frame(parent_pose, parent, np.asarray(child_pose)[:3, 3])
    radial = np.linalg.norm(p[:2] - np.asarray(cavity["axis_xy"]))
    return bool(radial <= cavity["opening_radius"] + radial_slack and cavity["floor_z"] - 0.005 <= p[2] <= cavity["rim_z"])


def resting_on(child_pose, child_min_z: float, parent_pose, parent: dict, ground_z: float,
               ground_tol: float = 0.002, footprint_slack: float = 0.01) -> bool:
    """Child is carried by the parent: above its footprint, higher up, and not touching the table."""
    if child_min_z <= ground_z + ground_tol:
        return False
    child_com = np.asarray(child_pose)[:3, 3]
    if child_com[2] <= np.asarray(parent_pose)[2, 3]:
        return False
    p = to_original_frame(parent_pose, parent, child_com)
    half = np.asarray(parent["footprint_xy"]) / 2 + footprint_slack
    centre = np.asarray(parent["com"][:2])
    return bool(np.all(np.abs(p[:2] - centre) <= half))


def check_relations(plan: dict, poses: dict, min_z: dict, pool: dict, ground_z: float) -> list[str]:
    """Planned relations that no longer hold after settling (empty list = all hold)."""
    broken = []
    for obj in plan["objects"]:
        if obj["parent"] is None:
            continue
        parent_obj = plan["objects"][obj["parent"]]
        parent = pool[parent_obj["asset_id"]]
        child_pose, parent_pose = poses[obj["index"]], poses[parent_obj["index"]]
        if obj["relation"] == "in":
            ok = nested_inside(child_pose, parent_pose, parent)
        else:  # carried by the parent, and not wedged into its opening
            ok = resting_on(child_pose, min_z[obj["index"]], parent_pose, parent, ground_z) and not (
                parent.get("cavity") is not None and nested_inside(child_pose, parent_pose, parent))
        if not ok:
            broken.append(f"{obj['index']}:{obj['relation']}:{obj['parent']}")
    return broken


def clearance_lift(support_points, body_points, cell: float = 0.004) -> float | None:
    """Smallest upward shift of `body_points` that puts them above `support_points` wherever
    the two overlap in xy (heightmaps on a `cell` grid; support heights spread to neighbouring
    cells so points between vertices are covered). None if they do not overlap in xy."""
    support_points, body_points = np.asarray(support_points), np.asarray(body_points)
    s_keys = np.floor(support_points[:, :2] / cell).astype(np.int64)
    shifts = np.array([[dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    s_keys = (s_keys[None] + shifts[:, None]).reshape(-1, 2)
    s_z = np.tile(support_points[:, 2], len(shifts))
    b_keys = np.floor(body_points[:, :2] / cell).astype(np.int64)

    def per_cell(keys, z, reduce):
        cells, inverse = np.unique(keys, axis=0, return_inverse=True)
        out = np.full(len(cells), -np.inf if reduce is np.maximum else np.inf)
        reduce.at(out, inverse.ravel(), z)
        return cells, out

    s_cells, top = per_cell(s_keys, s_z, np.maximum)
    b_cells, bottom = per_cell(b_keys, body_points[:, 2], np.minimum)
    s_index = {tuple(c): i for i, c in enumerate(s_cells)}
    gaps = [top[s_index[tuple(c)]] - bottom[i] for i, c in enumerate(b_cells) if tuple(c) in s_index]
    return float(max(gaps)) if gaps else None

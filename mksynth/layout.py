"""Scene planning: which objects, where, and what rests on / inside what (numpy only).

A plan fixes everything random about a scene's construction before physics runs:

* base objects (stage 0): asset, orientation, (x, y) position and drop height;
* stacked / nested objects (stage >= 1): asset, parent object, relation ("on" or "in"),
  orientation relative to the parent's upright frame, and for stacks an (x, y) offset on the
  parent's top face. Their absolute drop pose depends on where the parent settled, so the
  Blender worker computes it after the previous stage has come to rest.

Plans are deterministic in the RNG they are given, so a scene is reproducible from its seed.
"""

from __future__ import annotations

import numpy as np

from mksynth.compat import CompatConfig, build_table
from mksynth.geometry import random_rotation, tilted_upright

LEVELS = ("easy", "medium", "hard")


class PlanError(RuntimeError):
    """No valid plan could be found with the given pool and settings."""


def scene_seed(global_seed: int, split: str, level: str, index: int, val_offset: int = 1000003) -> int:
    """Independent, stable 63-bit seed per (global seed, split, level, scene index)."""
    stream = global_seed + (val_offset if split == "val" else 0)
    sequence = np.random.SeedSequence([stream, LEVELS.index(level), index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def attempt_rng(seed: int, attempt: int, purpose: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, attempt, purpose])


def footprint_radius(asset: dict, upright: bool) -> float:
    """Radius of a disc around the COM axis that contains the object's (x, y) footprint."""
    return float(asset["radius_xy"] if upright else asset["bounding_radius"])


def place_separated(radii, scene_radius: float, gap: float, rng, tries: int = 400) -> np.ndarray:
    """Rejection-sample non-overlapping disc centres inside a disc (easy scenes)."""
    centres = []
    for radius in radii:
        limit = max(scene_radius - radius, 0.0)
        for _ in range(tries):
            r = limit * np.sqrt(rng.uniform())
            phi = rng.uniform(0, 2 * np.pi)
            point = np.array([r * np.cos(phi), r * np.sin(phi)])
            if all(np.linalg.norm(point - c) >= radius + rc + gap for c, rc in zip(centres, radii)):
                centres.append(point)
                break
        else:
            raise PlanError("could not place base objects without overlap")
    return np.asarray(centres)


def place_jittered(radii, jitter: float, gap: float, rng, iterations: int = 500) -> np.ndarray:
    """Perturb positions around the centre, then push discs apart until they only just clear.

    Small `jitter` gives tight, contact-rich clusters (medium / hard scenes).
    """
    radii = np.asarray(radii, dtype=np.float64)
    points = rng.uniform(-jitter, jitter, size=(len(radii), 2))
    for _ in range(iterations):
        moved = False
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                delta = points[j] - points[i]
                distance = np.linalg.norm(delta)
                needed = radii[i] + radii[j] + gap
                if distance < needed:
                    direction = delta / distance if distance > 1e-9 else np.array([np.cos(i + j), np.sin(i + j)])
                    shift = 0.5 * (needed - distance) + 1e-5
                    points[i] -= direction * shift
                    points[j] += direction * shift
                    moved = True
        if not moved:
            return points - points.mean(axis=0)
    raise PlanError("jittered placement did not converge")


def discs_overlap(centres, radii, gap: float = 0.0) -> bool:
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            if np.linalg.norm(np.asarray(centres[i]) - np.asarray(centres[j])) < radii[i] + radii[j] + gap - 1e-9:
                return True
    return False


def _choose_children(objects, pool, table, relation, count, p_chain, rng, used):
    """Attach `count` children with `relation` ("in" or "on") to compatible parents."""
    for _ in range(count):
        eligible_parents = [
            obj for obj in objects
            if obj["stage"] == 0 or (rng.uniform() < p_chain and obj["relation"] == relation)
        ]
        candidates = []
        for parent in eligible_parents:
            if any(child["parent"] == parent["index"] for child in objects):
                continue  # one child per parent keeps stacks stable and readable
            if relation == "in":
                options = table["nest"].get(parent["asset_id"], {})
            else:
                options = {child_id: None for child_id in table["stack"].get(parent["asset_id"], [])}
            candidates += [(parent, child_id, fit) for child_id, fit in options.items() if child_id not in used]
        if not candidates:
            return False
        parent, child_id, fit = candidates[rng.integers(len(candidates))]
        used.add(child_id)
        objects.append({
            "index": len(objects),
            "asset_id": child_id,
            "role": "nest" if relation == "in" else "stack",
            "relation": relation,
            "parent": parent["index"],
            "stage": parent["stage"] + 1,
            "fit": fit,
        })
    return True


def plan_scene(level_cfg: dict, placement_cfg: dict, pool: dict[str, dict], rng: np.random.Generator,
               compat_cfg: CompatConfig | None = None, table: dict | None = None,
               physics_cfg: dict | None = None) -> dict:
    """Sample a complete construction plan for one scene from assets in `pool`."""
    table = table or build_table(pool, compat_cfg or CompatConfig())
    ids = sorted(pool)
    n_base, n_stack, n_nest = level_cfg["n_base"], level_cfg.get("n_stack", 0), level_cfg.get("n_nest", 0)
    if n_base + n_stack + n_nest > len(ids):
        raise PlanError(f"pool has {len(ids)} assets but the level needs {n_base + n_stack + n_nest}")
    friction = (physics_cfg or {}).get("friction", [0.5, 0.5])
    hosts = sorted(asset_id for asset_id, options in table["nest"].items() if options)
    for _ in range(placement_cfg.get("max_plan_tries", 50)):
        # Seed the base set with containers that can host a nest, so hard scenes can always nest.
        n_hosts = min(n_nest, len(hosts), n_base)
        base = [str(a) for a in rng.choice(hosts, size=n_hosts, replace=False)] if n_hosts else []
        rest = [a for a in ids if a not in base]
        base += [str(a) for a in rng.choice(rest, size=n_base - n_hosts, replace=False)]
        base = [base[i] for i in rng.permutation(len(base))]
        objects = [
            {"index": i, "asset_id": a, "role": "base", "relation": None, "parent": None, "stage": 0, "fit": None}
            for i, a in enumerate(base)
        ]
        used = set(base)
        p_chain = level_cfg.get("p_chain", 0.0)
        if not _choose_children(objects, pool, table, "in", n_nest, p_chain, rng, used):
            continue
        if not _choose_children(objects, pool, table, "on", n_stack, p_chain, rng, used):
            continue

        parents = {obj["parent"] for obj in objects if obj["parent"] is not None}
        for obj in objects:
            asset = pool[obj["asset_id"]]
            obj["friction"] = float(rng.uniform(*friction))
            if obj["role"] == "base":
                upright = obj["index"] in parents or rng.uniform() < level_cfg.get("p_upright", 0.5)
                obj["upright"] = bool(upright)
                obj["R"] = (tilted_upright(rng, 0.0) if upright else random_rotation(rng)).tolist()
                obj["frame"] = "world"
                obj["drop_height"] = float(rng.uniform(*placement_cfg["drop_height"]))
            elif obj["role"] == "nest":
                obj["upright"] = True
                tilt = np.deg2rad(placement_cfg["nest_max_tilt_deg"])
                obj["R"] = tilted_upright(rng, tilt).tolist()
                obj["frame"] = "parent"
            else:
                upright = rng.uniform() < placement_cfg["p_stack_upright"]
                obj["upright"] = bool(upright)
                tilt = np.deg2rad(placement_cfg["stack_max_tilt_deg"])
                obj["R"] = (tilted_upright(rng, tilt) if upright else random_rotation(rng)).tolist()
                obj["frame"] = "world"
                parent = pool[objects[obj["parent"]]["asset_id"]]
                slack = np.maximum(np.asarray(parent["top_bbox_xy"]) - 2 * asset["radius_xy"], 0.0)
                obj["offset_xy"] = (rng.uniform(-0.25, 0.25, size=2) * slack).tolist()

        base_objs = [obj for obj in objects if obj["role"] == "base"]
        radii = [footprint_radius(pool[obj["asset_id"]], obj["upright"]) for obj in base_objs]
        try:
            if level_cfg["placement"] == "separated":
                gap = level_cfg["min_gap"]
                # Grow the disc for large objects so separated layouts stay feasible.
                radius = max(level_cfg["scene_radius"], 1.6 * float(np.sqrt(np.sum((np.asarray(radii) + gap / 2) ** 2))))
                centres = place_separated(radii, radius, gap, rng)
            else:
                centres = place_jittered(radii, level_cfg["jitter"], level_cfg["min_gap"], rng)
        except PlanError:
            continue
        for obj, centre in zip(base_objs, centres):
            obj["xy"] = centre.tolist()
        return {"objects": objects, "n_objects": len(objects), "stages": max(o["stage"] for o in objects) + 1}
    raise PlanError("no valid plan found; relax compatibility or enlarge the pool")


def split_pool(asset_table: dict[str, dict], split: str) -> dict[str, dict]:
    """Assets usable in scenes of `split`; train and val pools are disjoint by construction."""
    if split not in ("train", "val"):
        raise ValueError(f"unknown split {split!r}")
    return {asset_id: asset for asset_id, asset in asset_table.items() if asset["split"] == split}

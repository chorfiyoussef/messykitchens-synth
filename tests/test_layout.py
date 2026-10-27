import numpy as np
import pytest
from conftest import make_asset

from mksynth.compat import build_table
from mksynth.layout import (PlanError, attempt_rng, discs_overlap, place_jittered, place_separated, plan_scene,
                          scene_seed, split_pool)

PLACEMENT = {"drop_height": [0.003, 0.02], "p_stack_upright": 0.7, "nest_max_tilt_deg": 8, "stack_max_tilt_deg": 5,
             "max_plan_tries": 50}
EASY = {"n_base": 4, "n_stack": 0, "n_nest": 0, "p_upright": 0.5, "placement": "separated", "scene_radius": 0.22,
        "min_gap": 0.02}
MEDIUM = {"n_base": 4, "n_stack": 2, "n_nest": 0, "p_upright": 0.5, "placement": "jittered", "jitter": 0.08,
          "min_gap": 0.003, "p_chain": 0.0}
HARD = {"n_base": 4, "n_stack": 2, "n_nest": 2, "p_upright": 0.5, "placement": "jittered", "jitter": 0.07,
        "min_gap": 0.002, "p_chain": 0.3}


def test_scene_seeds_are_stable_and_distinct():
    assert scene_seed(0, "train", "easy", 3) == scene_seed(0, "train", "easy", 3)
    seeds = {scene_seed(g, s, lv, i) for g in (0, 1) for s in ("train", "val") for lv in ("easy", "hard") for i in range(50)}
    assert len(seeds) == 2 * 2 * 2 * 50


def test_separated_placement_has_gaps():
    rng = np.random.default_rng(0)
    radii = [0.05, 0.08, 0.04, 0.06]
    centres = place_separated(radii, 0.3, 0.02, rng)
    assert not discs_overlap(centres, radii, gap=0.02)
    assert all(np.linalg.norm(c) + r <= 0.3 + 1e-9 for c, r in zip(centres, radii))


def test_jittered_placement_resolves_overlaps():
    rng = np.random.default_rng(1)
    radii = [0.05, 0.08, 0.04, 0.06, 0.07]
    centres = place_jittered(radii, 0.01, 0.003, rng)  # start almost on top of each other
    assert not discs_overlap(centres, radii, gap=0.003)
    assert np.allclose(centres.mean(axis=0), 0.0)


def test_plans_are_deterministic(kitchen):
    a = plan_scene(HARD, PLACEMENT, kitchen, attempt_rng(123, 0))
    b = plan_scene(HARD, PLACEMENT, kitchen, attempt_rng(123, 0))
    c = plan_scene(HARD, PLACEMENT, kitchen, attempt_rng(123, 1))
    assert a == b
    assert a != c


def test_easy_plan(kitchen):
    for seed in range(20):
        plan = plan_scene(EASY, PLACEMENT, kitchen, attempt_rng(seed, 0))
        assert plan["n_objects"] == 4 and plan["stages"] == 1
        assert len({o["asset_id"] for o in plan["objects"]}) == 4
        assert all(o["role"] == "base" and o["parent"] is None for o in plan["objects"])


def test_medium_and_hard_children_are_compatible(kitchen):
    table = build_table(kitchen)
    nests = 0
    for level in (MEDIUM, HARD):
        for seed in range(30):
            plan = plan_scene(level, PLACEMENT, kitchen, attempt_rng(seed, 0), table=table)
            objects = plan["objects"]
            assert len(objects) == level["n_base"] + level["n_stack"] + level["n_nest"]
            assert len({o["asset_id"] for o in objects}) == len(objects)
            parents = [o["parent"] for o in objects if o["parent"] is not None]
            assert len(parents) == len(set(parents))  # one child per parent
            for obj in objects:
                if obj["parent"] is None:
                    continue
                parent = objects[obj["parent"]]
                assert obj["stage"] == parent["stage"] + 1
                assert parent["upright"]
                if obj["relation"] == "in":
                    nests += 1
                    assert obj["asset_id"] in table["nest"][parent["asset_id"]]
                    assert obj["fit"] == table["nest"][parent["asset_id"]][obj["asset_id"]]
                else:
                    assert obj["asset_id"] in table["stack"][parent["asset_id"]]
            base = [o for o in objects if o["role"] == "base"]
            radii = [o_r for o_r in (kitchen[o["asset_id"]]["radius_xy"] if o["upright"] else
                                     kitchen[o["asset_id"]]["bounding_radius"] for o in base)]
            assert not discs_overlap([o["xy"] for o in base], radii, gap=level["min_gap"] - 1e-6)
    assert nests >= 30 * 2  # every hard scene got both nests


def test_plan_error_when_pool_too_small(kitchen):
    pool = {k: kitchen[k] for k in list(kitchen)[:3]}
    with pytest.raises(PlanError):
        plan_scene(EASY, PLACEMENT, pool, attempt_rng(0, 0))


def test_split_pools_are_disjoint():
    table = {"a": make_asset("a", height=0.1, radius=0.05), "b": make_asset("b", height=0.1, radius=0.05, split="val")}
    train, val = split_pool(table, "train"), split_pool(table, "val")
    assert set(train) == {"a"} and set(val) == {"b"}
    with pytest.raises(ValueError):
        split_pool(table, "test")

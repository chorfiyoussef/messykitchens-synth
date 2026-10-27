"""Physics settling inside the pinned Blender (slow; opt in with MKSYNTH_BLENDER_TESTS=1).

Runs blender/scene_worker.py in physics-only mode on real assets and checks the accepted scenes:
everything settled on or above the table, stacked / nested objects kept their relation, and the
same seed reproduces the same poses.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mksynth.config import REPO_ROOT, load_config
from mksynth.layout import scene_seed

pytestmark = pytest.mark.skipif(
    os.environ.get("MKSYNTH_BLENDER_TESTS") != "1"
    or not (REPO_ROOT / ".blender").is_dir()
    or not (REPO_ROOT / "data/assets/processed/asset_table.json").is_file(),
    reason="set MKSYNTH_BLENDER_TESTS=1 (needs .blender/ and processed assets)",
)


def run_physics(tmp_path: Path, level: str, index: int, name: str) -> dict:
    cfg = load_config(REPO_ROOT / "configs/default.yaml")
    spec = {"scene_id": name, "level": level, "split": "train", "index": index,
            "seed": scene_seed(7, "train", level, index), "out_dir": str(tmp_path / name)}
    job = tmp_path / f"{name}.json"
    job.write_text(json.dumps({"config": cfg, "scenes": [spec], "physics_only": True}))
    command = [str(Path(sys.executable).parent / "blenderproc"), "run", "--blender-install-path",
               str(REPO_ROOT / ".blender"), str(REPO_ROOT / "blender/scene_worker.py"), str(job)]
    subprocess.run(command, check=True, capture_output=True, cwd=REPO_ROOT, timeout=900)
    raw = Path(spec["out_dir"]) / "raw"
    assert (raw / "DONE").is_file(), (raw / "FAILED.json").read_text() if (raw / "FAILED.json").exists() else "no output"
    return json.loads((raw / "physics.json").read_text())


@pytest.mark.parametrize("level", ["easy", "hard"])
def test_scene_settles_and_keeps_relations(tmp_path, level):
    result = run_physics(tmp_path, level, 0, f"{level}_a")
    accepted = result["attempts"][-1]
    assert accepted["status"] == "accepted"
    assert accepted["hold"]["translation_mm"] < 2.0
    assert accepted["peak_speed"] < 3.0
    for obj in result["plan"]["objects"]:
        pose = np.asarray(result["poses"][str(obj["index"])])
        assert pose[2, 3] > 0.0, f"{obj['asset_id']} sank through the table"
        assert np.linalg.norm(pose[:2, 3]) < 0.6, f"{obj['asset_id']} far from the scene centre"
    if level == "hard":
        assert any(o["relation"] == "in" for o in result["plan"]["objects"])


def test_same_seed_same_result(tmp_path):
    a = run_physics(tmp_path, "medium", 3, "medium_a")
    b = run_physics(tmp_path, "medium", 3, "medium_b")
    assert a["plan"] == b["plan"]
    for key, pose in a["poses"].items():
        assert np.allclose(pose, b["poses"][key], atol=1e-6)

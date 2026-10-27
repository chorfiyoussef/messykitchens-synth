"""Collect per-scene outputs into dataset manifests.

Writes, in the output directory:

  manifest.jsonl        one line per scene (status, split, objects, kept views, key statistics)
  mod_train.json        MOD raw manifest (one record per kept view), paths relative to this file
  mod_val.json          same for validation scenes
  failures.jsonl        scenes that could not be generated, with the reason
  stats.json            dataset statistics
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


def scene_dirs(out_dir: Path) -> list[Path]:
    return sorted(p for p in (out_dir / "scenes").glob("*") if p.is_dir())


def mod_records(scene: dict, scene_rel: str) -> list[dict]:
    records = []
    for record in scene["mod_records"]:
        records.append({
            **record,
            "image_path": f"{scene_rel}/{record['image_path']}",
            "instance_mask_path": f"{scene_rel}/{record['instance_mask_path']}",
            "mesh_path": f"{scene_rel}/{record['mesh_path']}",
            "dataset": f"mksynth_{scene['level']}",
            "level": scene["level"],
            "split": scene["split"],
            "scene_path": f"{scene_rel}/scene.json",
        })
    return records


def summarise(scene: dict) -> dict:
    contacts = scene["contacts"]
    return {
        "scene_id": scene["scene_id"], "level": scene["level"], "split": scene["split"], "status": "ok",
        "n_objects": len(scene["objects"]), "assets": [o["asset_id"] for o in scene["objects"]],
        "kept_views": scene["kept_views"], "mod_records": len(scene["mod_records"]),
        "object_contact_area_mm2": contacts["object_contact_area_mm2"],
        "object_penetration_area_mm2": contacts["object_penetration_area_mm2"],
        "penetration_contact_ratio": contacts["penetration_contact_ratio"],
        "max_penetration_depth_mm": contacts["max_penetration_depth_mm"],
        "support_edges": len(scene["support_graph"]["edges"]),
        "nest_edges": sum(e["relation"] == "in" for e in scene["support_graph"]["edges"]),
        "stack_edges": sum(e["relation"] == "on" and e["parent"] != "table" for e in scene["support_graph"]["edges"]),
        "physics_attempts": len(scene["physics"]["attempts"]),
        "physics_seconds": scene["physics"]["seconds"], "render_seconds": scene["render"]["seconds"],
        "annotate_seconds": scene["annotate_seconds"],
        "path": f"scenes/{scene['scene_id']}",
    }


def collect(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    rows, failures, records = [], [], {"train": [], "val": []}
    for directory in scene_dirs(out_dir):
        scene_path = directory / "scene.json"
        failed_path = directory / "raw" / "FAILED.json"
        if scene_path.is_file():
            scene = json.loads(scene_path.read_text())
            rows.append(summarise(scene))
            records[scene["split"]] += mod_records(scene, f"scenes/{directory.name}")
        elif failed_path.is_file():
            failure = json.loads(failed_path.read_text())
            failures.append({key: failure.get(key) for key in ("scene_id", "level", "split", "index", "seed", "reason", "retries")}
                            | {"detail": str(failure.get("detail", ""))[-2000:]})
    with open(out_dir / "manifest.jsonl", "w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    with open(out_dir / "failures.jsonl", "w") as stream:
        for row in failures:
            stream.write(json.dumps(row) + "\n")
    for split, items in records.items():
        (out_dir / f"mod_{split}.json").write_text(json.dumps(items, indent=1))
    stats = statistics(rows, failures)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def _describe(values) -> dict | None:
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return None
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


def statistics(rows: list[dict], failures: list[dict]) -> dict:
    out = {}
    for level in sorted({r["level"] for r in rows} | {f["level"] for f in failures if f.get("level")}):
        for split in ("train", "val"):
            sel = [r for r in rows if r["level"] == level and r["split"] == split]
            fail = [f for f in failures if f.get("level") == level and f.get("split") == split]
            if not sel and not fail:
                continue
            contact = sum(r["object_contact_area_mm2"] for r in sel)
            penetration = sum(r["object_penetration_area_mm2"] for r in sel)
            out[f"{level}/{split}"] = {
                "scenes": len(sel), "failed": len(fail), "failure_reasons": dict(Counter(f["reason"] for f in fail)),
                "mod_records": sum(r["mod_records"] for r in sel),
                "objects_per_scene": _describe([r["n_objects"] for r in sel]),
                "distinct_assets": len({a for r in sel for a in r["assets"]}),
                "penetration_contact_ratio_pooled": penetration / contact if contact else None,
                "penetration_contact_ratio_per_scene": _describe([r["penetration_contact_ratio"] for r in sel]),
                "object_contact_area_mm2": _describe([r["object_contact_area_mm2"] for r in sel]),
                "max_penetration_depth_mm": _describe([r["max_penetration_depth_mm"] for r in sel]),
                "scenes_with_nest": sum(r["nest_edges"] > 0 for r in sel),
                "scenes_with_stack": sum(r["stack_edges"] > 0 for r in sel),
                "physics_attempts": _describe([r["physics_attempts"] for r in sel]),
                "seconds": {key: _describe([r[f"{key}_seconds"] for r in sel]) for key in ("physics", "render", "annotate")},
            }
    return out

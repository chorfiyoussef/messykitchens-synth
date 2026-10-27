"""Validate a generated dataset directory.

    python validate.py --out out/easy                     # all scenes
    python validate.py --out out/easy --mod_prepare 10    # also run MOD's prepare step on 10 scenes (GPU pod)
    python validate.py --out out/easy --mod_prepare 20 --mod_prepare_views 1   # one kept view of each of 20 scenes

Per scene (thresholds under `validate:` in the config):
  * files       every output referenced by scene.json exists and has the right shape / dtype
  * render      no black RGB views (Cycles out of GPU memory renders black without failing)
  * floating    every object is supported (table or another object) and nothing sank into the table
  * penetration deepest object-object / object-table overlap is below the threshold
  * masks       visible masks re-derived from gt_scene.glb + saved cameras (independently of the
                annotation code) match mask_visible; visible pixels lie inside the amodal masks;
                rendered depth matches the raycast depth
  * poses       per-view T_cam_mesh = T_cam_world @ T_world_mesh, and the GLB node transforms equal T_world_mesh
  * relations   planned stack / nest relations appear in the contact-based support graph
  * mod         MOD records follow the rules mod-prepare enforces (labels present = gt_object_map keys + 1,
                num_parts, one-to-one geometry-bearing GLB nodes, image / mask shapes)
  * assets      allowed licences, no held-out sources, train scenes use train objects only (val likewise)
Dataset level: train and val scenes share no object; MOD manifests list exactly the kept views.
With --mod_prepare N: writes a sample manifest of N scenes (all kept views, or the first
--mod_prepare_views of each) and runs
third_party/MessyKitchens/scripts/prepare.sh --generate-training-gt on it (needs SAM3D_ROOT and
GAPS_MSHALIGN), then checks that every record produced target poses.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.ndimage import binary_dilation

from mksynth.annotate import _tensor_scene, camera_rays
from mksynth.config import REPO_ROOT, load_yaml

DEFAULTS = {
    "max_penetration_depth_mm": 2.0,
    "max_sink_mm": 1.0,
    "min_mask_agreement": 0.9,         # 1-px tolerant: a rim seen as a 2 px ring has a poor plain IoU
    "max_visible_outside_amodal": 0.05,
    "max_depth_error_mm": 2.0,
    "max_pose_error": 1e-6,
    "min_rgb_mean": 2.0,               # a black frame: Cycles ran out of GPU memory
    "require_planned_relations": True,
}


class Report:
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.errors: list[str] = []
        self.values: dict = {}

    def check(self, name: str, ok: bool, message: str = "") -> None:
        if not ok:
            self.errors.append(f"{name}: {message}")


def _within_1px(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of mask a that lies within 1 px of mask b."""
    return float((a & binary_dilation(b, np.ones((3, 3), bool))).sum() / a.sum())


def load_glb_nodes(path: Path) -> dict[str, trimesh.Trimesh]:
    """Posed meshes per geometry-bearing node, read the same way MOD's training_gt does."""
    scene = trimesh.load(path, force="scene", process=False)
    out = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        out[node] = (mesh, np.asarray(transform))
    return out


def check_mod_record(record: dict, scene_dir: Path, nodes: dict, report: Report) -> None:
    image = np.asarray(Image.open(scene_dir / record["image_path"]))
    mask = np.asarray(Image.open(scene_dir / record["instance_mask_path"]))
    report.check("mod", image.ndim == 3 and image.shape[-1] >= 3, f"{record['id']}: RGB shape {image.shape}")
    report.check("mod", mask.ndim == 2 and np.issubdtype(mask.dtype, np.integer), f"{record['id']}: mask {mask.shape} {mask.dtype}")
    report.check("mod", mask.shape == image.shape[:2], f"{record['id']}: mask {mask.shape} vs image {image.shape[:2]}")
    labels = sorted(int(v) for v in np.unique(mask) if v != 0)
    object_ids = sorted(int(k) for k in record["gt_object_map"])
    report.check("mod", [label - 1 for label in labels] == object_ids,
                 f"{record['id']}: labels {labels} vs gt_object_map keys {object_ids}")
    report.check("mod", record["num_parts"] == len(labels), f"{record['id']}: num_parts {record['num_parts']} vs {len(labels)} labels")
    values = list(record["gt_object_map"].values())
    report.check("mod", len(set(values)) == len(values), f"{record['id']}: gt_object_map not one-to-one")
    missing = [v for v in values if v not in nodes]
    report.check("mod", not missing, f"{record['id']}: GLB lacks geometry nodes {missing}")


def validate_scene(scene_dir: Path, cfg: dict, thresholds: dict, assets: dict) -> Report:
    scene = json.loads((scene_dir / "scene.json").read_text())
    report = Report(scene["scene_id"])
    n_obj, views, camera = len(scene["objects"]), scene["views"], scene["camera"]
    h, w = camera["height"], camera["width"]

    # files
    for view in views:
        for key in ("rgb", "depth", "mask_visible") + (("mod_mask",) if view["kept"] else ()):
            path = scene_dir / view[key]
            report.check("files", path.is_file(), f"missing {view[key]}")
    amodal = np.load(scene_dir / "masks_amodal.npz")["amodal"]
    report.check("files", amodal.shape == (len(views), n_obj, h, w) and amodal.dtype == bool, f"amodal {amodal.shape} {amodal.dtype}")
    contacts = np.load(scene_dir / "contacts.npz")
    report.check("files", len([k for k in contacts.files if k.startswith("face_contact_")]) == n_obj, "contacts.npz incomplete")
    report.check("files", len(scene["kept_views"]) == cfg["render"]["keep_views"], f"kept {len(scene['kept_views'])} views")
    dark = [v["view"] for v in views if np.asarray(Image.open(scene_dir / v["rgb"])).mean() < thresholds["min_rgb_mean"]]
    report.check("render", not dark, f"black views {dark}")

    # floating / sinking / penetration
    graph = scene["support_graph"]
    report.check("floating", not graph["unsupported"], f"unsupported objects {graph['unsupported']}")
    report.check("floating", scene["checks"]["min_object_z_mm"] >= -thresholds["max_sink_mm"],
                 f"object {scene['checks']['min_object_z_mm']:.2f} mm below the table")
    depth_mm = scene["contacts"]["max_penetration_depth_mm"]
    report.check("penetration", depth_mm <= thresholds["max_penetration_depth_mm"], f"max penetration {depth_mm:.2f} mm")
    report.values.update(max_penetration_mm=depth_mm, ratio=scene["contacts"]["penetration_contact_ratio"])

    # relations
    if thresholds["require_planned_relations"]:
        missing = [r for r in scene["planned_relations_hold"] if r["found"] != r["relation"]]
        report.check("relations", not missing, f"planned relations not in support graph: {missing}")

    # poses and GLB
    nodes = load_glb_nodes(scene_dir / "gt_scene.glb")
    for obj in scene["objects"]:
        node = obj["glb_node"]
        if node not in nodes:
            report.check("poses", False, f"GLB has no node {node}")
            continue
        error = np.abs(nodes[node][1] - np.asarray(obj["T_world_mesh"])).max()
        report.check("poses", error < 1e-5, f"{node} transform differs from T_world_mesh by {error:.2e}")
    for view in views:
        T_cam_world = np.asarray(view["T_cam_world"])
        report.check("poses", np.abs(T_cam_world @ np.asarray(view["T_world_cam"]) - np.eye(4)).max() < 1e-6, f"v{view['view']} camera inverse")
        for obj, per in zip(scene["objects"], view["objects"]):
            expected = T_cam_world @ np.asarray(obj["T_world_mesh"])
            error = np.abs(expected - np.asarray(per["T_cam_mesh"])).max()
            report.check("poses", error < thresholds["max_pose_error"], f"v{view['view']} obj {obj['index']} pose error {error:.2e}")

    # masks: independent re-derivation from the GLB and saved cameras (kept views)
    node_names = [obj["glb_node"] for obj in scene["objects"]]
    meshes = [nodes[name][0] for name in node_names if name in nodes]
    if len(meshes) == n_obj:
        raycaster, ids = _tensor_scene(meshes)
        size = scene["table"]["size"] / 2
        raycaster.add_triangles(  # the table occludes nothing above it but bounds the depth check
            o3d.core.Tensor(np.array([[-size, -size, 0], [size, -size, 0], [size, size, 0], [-size, size, 0]], np.float32)),
            o3d.core.Tensor(np.array([[0, 1, 2], [0, 2, 3]], np.uint32)))
        ious, agreement, outside, depth_errors = [], [], [], []
        for v in scene["kept_views"]:
            view = views[v]
            hit = raycaster.cast_rays(o3d.core.Tensor(camera_rays(camera["K"], view["T_world_cam"], w, h)))
            geometry, t_hit = hit["geometry_ids"].numpy(), hit["t_hit"].numpy()
            labels = np.zeros((h, w), np.int32)
            for k, gid in enumerate(ids):
                labels[geometry == gid] = k + 1
            visible = np.asarray(Image.open(scene_dir / view["mask_visible"]))
            depth = np.asarray(Image.open(scene_dir / view["depth"])).astype(np.float64) * camera["depth_unit_mm"]
            for k in range(n_obj):
                mine = visible == k + 1
                if mine.sum() < cfg["annotate"]["min_visible_px"]:
                    continue
                theirs = labels == k + 1
                ious.append(np.logical_and(mine, theirs).sum() / np.logical_or(mine, theirs).sum())
                agreement.append(min(_within_1px(mine, theirs), _within_1px(theirs, mine)))
                outside.append((mine & ~binary_dilation(amodal[v, k])).sum() / mine.sum())
            same = (labels > 0) & (labels == visible) & (depth > 0)
            if same.any():
                depth_errors.append(float(np.median(np.abs(depth[same] - t_hit[same] * 1e3))))
        if ious:
            report.check("masks", min(agreement) >= thresholds["min_mask_agreement"],
                         f"min 1-px agreement {min(agreement):.3f}")
            report.check("masks", max(outside) <= thresholds["max_visible_outside_amodal"], f"visible outside amodal {max(outside):.3f}")
            report.values.update(min_mask_agreement=float(min(agreement)), min_mask_iou=float(min(ious)),
                                 mean_mask_iou=float(np.mean(ious)))
        if depth_errors:
            report.check("masks", max(depth_errors) <= thresholds["max_depth_error_mm"], f"depth error {max(depth_errors):.2f} mm")
            report.values.update(depth_error_mm=max(depth_errors))
    else:
        report.check("masks", False, "GLB nodes missing; masks not checked")

    # MOD records
    report.check("mod", 0 < len(scene["mod_records"]) <= len(scene["kept_views"]),
                 f"{len(scene['mod_records'])} records for {len(scene['kept_views'])} kept views")
    for record in scene["mod_records"]:
        check_mod_record(record, scene_dir, nodes, report)

    # assets
    for obj in scene["objects"]:
        asset = assets.get(obj["asset_id"])
        if asset is None:
            report.check("assets", False, f"unknown asset {obj['asset_id']}")
            continue
        report.check("assets", asset["split"] == scene["split"], f"{obj['asset_id']} ({asset['split']}) in a {scene['split']} scene")
        report.check("assets", obj["licence"] in cfg["_allowed_licences"], f"{obj['asset_id']} licence {obj['licence']}")
        report.check("assets", asset["source"] not in cfg["_holdout_sources"], f"{obj['asset_id']} from a held-out source")
    return report


def run_mod_prepare(out_dir: Path, n_scenes: int, report: dict, views: int | None = None) -> bool:
    manifest = json.loads((out_dir / "mod_train.json").read_text())
    scene_ids = sorted({r["scene_id"] for r in manifest})[:n_scenes]
    sample, taken = [], {}
    for r in manifest:
        if r["scene_id"] in scene_ids and (views is None or taken.get(r["scene_id"], 0) < views):
            sample.append(r)
            taken[r["scene_id"]] = taken.get(r["scene_id"], 0) + 1
    sample_dir = out_dir / "mod_prepare_sample"
    sample_dir.mkdir(exist_ok=True)
    sample_path = sample_dir / "raw_manifest.json"
    # Paths stay relative: the sample manifest sits one level below the dataset root.
    sample_path.write_text(json.dumps([{**r, **{k: f"../{r[k]}" for k in ("image_path", "instance_mask_path", "mesh_path")}}
                                       for r in sample], indent=1))
    gaps = os.environ.get("GAPS_MSHALIGN")
    if not gaps or not os.environ.get("SAM3D_ROOT"):
        report["mod_prepare"] = {"status": "skipped", "reason": "set SAM3D_ROOT and GAPS_MSHALIGN (GPU pod only)",
                                 "sample_manifest": str(sample_path)}
        return True
    command = ["bash", str(REPO_ROOT / "third_party/MessyKitchens/scripts/prepare.sh"), "--data-json", str(sample_path),
               "--output-dir", str(sample_dir / "prepared"), "--generate-training-gt", "--gaps-binary", gaps, "--continue-on-error"]
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    seconds = time.perf_counter() - started
    (sample_dir / "prepare.log").write_text(result.stdout + "\n" + result.stderr)
    prepared_dir = sample_dir / "prepared"
    targets = sorted(prepared_dir.glob("*/target_poses.json"))
    # Without the model load: spacing between consecutive records' outputs.
    finished = sorted(t.stat().st_mtime for t in targets)
    steady = (finished[-1] - finished[0]) / (len(finished) - 1) if len(finished) > 1 else None
    report["mod_prepare"] = {"status": "ran", "exit_code": result.returncode, "records": len(sample), "scenes": len(scene_ids),
                             "views_per_scene": views, "target_poses": len(targets), "seconds": seconds,
                             "seconds_per_record": seconds / max(len(sample), 1), "seconds_per_record_steady": steady,
                             "log": str(sample_dir / "prepare.log")}
    return result.returncode == 0 and len(targets) == len(sample)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mod_prepare", type=int, default=0, help="run MOD prepare --generate-training-gt on N scenes")
    parser.add_argument("--mod_prepare_views", type=int, default=None, help="kept views per scene to prepare (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="validate at most this many scenes")
    args = parser.parse_args(argv)
    out_dir = Path(args.out).resolve()
    cfg = load_yaml(out_dir / "config.yaml")
    thresholds = {**DEFAULTS, **cfg.get("validate", {})}
    assets_cfg = load_yaml(REPO_ROOT / cfg["assets"]["config"])
    cfg["_allowed_licences"] = set(assets_cfg["allowed_licences"])
    cfg["_holdout_sources"] = set(assets_cfg["holdout"]["sources"])
    assets = json.loads((REPO_ROOT / cfg["assets"]["table"]).read_text())

    scene_dirs = sorted(p for p in (out_dir / "scenes").glob("*") if (p / "scene.json").is_file())[: args.limit]
    reports = [validate_scene(d, cfg, thresholds, assets) for d in scene_dirs]
    failed = [r for r in reports if r.errors]

    dataset_errors = []
    used = {"train": set(), "val": set()}
    for d in scene_dirs:
        scene = json.loads((d / "scene.json").read_text())
        used[scene["split"]] |= {o["asset_id"] for o in scene["objects"]}
    if used["train"] & used["val"]:
        dataset_errors.append(f"objects used in both splits: {sorted(used['train'] & used['val'])}")
    for split in ("train", "val"):
        manifest_path = out_dir / f"mod_{split}.json"
        if manifest_path.is_file():
            records = json.loads(manifest_path.read_text())
            ids = [r["id"] for r in records]
            if len(ids) != len(set(ids)):
                dataset_errors.append(f"mod_{split}.json has duplicate ids")
            for record in records:
                for key in ("image_path", "instance_mask_path", "mesh_path"):
                    if not (out_dir / record[key]).is_file():
                        dataset_errors.append(f"mod_{split}.json: missing {record[key]}")
                        break

    summary = {"scenes": len(reports), "failed": len(failed), "dataset_errors": dataset_errors,
               "failures": {r.scene_id: r.errors for r in failed},
               "values": {r.scene_id: r.values for r in reports}}
    ok = not failed and not dataset_errors
    if args.mod_prepare:
        ok = run_mod_prepare(out_dir, args.mod_prepare, summary, args.mod_prepare_views) and ok
    (out_dir / "validation.json").write_text(json.dumps(summary, indent=1, default=float))

    width = max([len(r.scene_id) for r in reports] + [8])
    print(f"{'scene':<{width}}  status  agree  mean IoU  depth err  max pen  pen/contact")
    for r in reports:
        v = r.values
        ratio = v.get("ratio")
        print(f"{r.scene_id:<{width}}  {'FAIL' if r.errors else 'ok':<6}  {v.get('min_mask_agreement', float('nan')):5.3f}  "
              f"{v.get('mean_mask_iou', float('nan')):8.3f}  "
              f"{v.get('depth_error_mm', float('nan')):6.2f} mm  {v.get('max_penetration_mm', 0):5.2f} mm  "
              f"{'-' if ratio is None else f'{ratio:.3f}'}")
        for error in r.errors:
            print(f"    {error}")
    for error in dataset_errors:
        print(f"dataset: {error}")
    if "mod_prepare" in summary:
        print("mod_prepare:", json.dumps(summary["mod_prepare"]))
    print(f"{len(reports) - len(failed)}/{len(reports)} scenes passed; dataset checks {'ok' if not dataset_errors else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

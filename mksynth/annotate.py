"""Final per-scene annotations, computed outside Blender from the worker's raw output.

Reads ``<scene>/raw/{scene_raw.json, views.npz}`` and writes, next to ``raw/``:

  rgb/vNN.png            the rendered views (all n_views)
  depth/vNN.png          uint16 z-depth in units of ``depth_unit_mm`` (0 = no surface / too far)
  mask_visible/vNN.png   uint8 instance labels from Blender: 0 = background, k = object k-1
  masks_amodal.npz       ``amodal`` bool [view, object, H, W], each object raycast on its own
  mod_mask/vNN.png       MOD instance mask for the kept views (objects below min_visible_px removed,
                         labels not renumbered)
  contacts.npz           per object, per face of its original mesh: contact / penetration area
                         fraction and the partner touched (see `contact_analysis`)
  gt_scene.glb           original meshes posed in the world frame, one node per object (obj_000, ...)
  scene.json             objects, cameras, per-view poses and visibility, support graph, contact
                         statistics, view selection, MOD records and self-checks

World frame: z up, metres, table top at z = 0. Cameras follow OpenCV (x right, y down, z forward);
``T_a_b`` maps points from frame b to frame a.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from mksynth.config import REPO_ROOT
from mksynth.geometry import matrix_to_quat
from mksynth.physics_checks import nested_inside

DEPTH_UNIT_MM = 0.1
GROUND = "table"
MIN_SUPPORT_AREA_MM2 = 4.0
SUPPORT_NORMAL_Z = -0.3
# Inside/outside by majority vote over this many rays: Open3D's default single ray misjudges
# points whose ray grazes a rim or an edge (seen as 15-40 mm phantom penetrations).
SIGN_RAYS = 7


@lru_cache(maxsize=128)
def load_mesh(path: str) -> trimesh.Trimesh:
    full = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    mesh = trimesh.load(full, force="mesh", process=False, skip_materials=True)
    return trimesh.Trimesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces), process=False)


def posed(mesh: trimesh.Trimesh, T) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_transform(np.asarray(T, dtype=np.float64))
    return out


def _tensor_scene(meshes) -> tuple[o3d.t.geometry.RaycastingScene, list[int]]:
    scene = o3d.t.geometry.RaycastingScene()
    ids = [scene.add_triangles(o3d.core.Tensor(np.asarray(m.vertices, dtype=np.float32)),
                               o3d.core.Tensor(np.asarray(m.faces, dtype=np.uint32))) for m in meshes]
    return scene, ids


def camera_rays(K, T_world_cam, width: int, height: int) -> np.ndarray:
    """Rays through pixel centres. Directions have unit z in the camera, so t_hit is z-depth."""
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T_world_cam, dtype=np.float64)
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    d_world = d_cam @ T[:3, :3].T
    origin = np.broadcast_to(T[:3, 3], d_world.shape)
    return np.concatenate([origin, d_world], axis=-1).astype(np.float32)


def table_mesh(size: float) -> trimesh.Trimesh:
    h = size / 2
    return trimesh.Trimesh([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], [[0, 1, 2], [0, 2, 3]], process=False)


def raycast_views(meshes_world, table_size: float, camera: dict) -> dict:
    """Amodal masks (each object alone), raycast visible labels and depth (all objects + table)."""
    width, height = camera["width"], camera["height"]
    n_views, n_obj = len(camera["views"]), len(meshes_world)
    scene_all, ids = _tensor_scene(list(meshes_world) + [table_mesh(table_size)])
    label_of = {gid: k + 1 for k, gid in enumerate(ids[:-1])}
    singles = [_tensor_scene([m])[0] for m in meshes_world]
    amodal = np.zeros((n_views, n_obj, height, width), dtype=bool)
    labels = np.zeros((n_views, height, width), dtype=np.uint8)
    depth = np.zeros((n_views, height, width), dtype=np.float32)
    for v, view in enumerate(camera["views"]):
        rays = o3d.core.Tensor(camera_rays(camera["K"], view["T_world_cam"], width, height))
        hit = scene_all.cast_rays(rays)
        geometry = hit["geometry_ids"].numpy()
        t_hit = hit["t_hit"].numpy()
        for gid, label in label_of.items():
            labels[v][geometry == gid] = label
        depth[v] = np.where(np.isfinite(t_hit), t_hit, 0.0)
        for k, single in enumerate(singles):
            amodal[v, k] = np.isfinite(single.cast_rays(rays)["t_hit"].numpy())
    return {"amodal": amodal, "labels": labels, "depth": depth}


def _voxel_centres(points: np.ndarray, pitch: float) -> np.ndarray:
    return (np.floor(points / pitch) + 0.5) * pitch


def contact_analysis(meshes_world, proxies_world, proxy_offset: float, cfg: dict, seed: int) -> dict:
    """Contact and penetration on the original meshes.

    Each original surface is sampled area-uniformly (``surface_spacing`` m between samples). A sample
    of object i is in contact with partner j (another object or the table) if it lies within
    ``contact_distance`` of j's original surface. It penetrates j if its voxel (``voxel_pitch``
    grid in world coordinates) is occupied by j's solid. Original scans are not watertight, so a
    solid is the watertight proxy shrunk back by the remeshing offset (proxy SDF < -offset); the
    table solid is z < 0. Penetration depth is measured without voxel quantisation.
    """
    n = len(meshes_world)
    cd, pitch = cfg["contact_distance"], cfg["voxel_pitch"]
    rng = np.random.default_rng(seed)
    distance_scenes = [_tensor_scene([m])[0] for m in meshes_world]
    sdf_scenes = [_tensor_scene([p])[0] for p in proxies_world]
    bounds = [m.bounds for m in meshes_world]
    per_object, pairs = [], {}
    for i, mesh in enumerate(meshes_world):
        count = int(np.clip(mesh.area / cfg["surface_spacing"] ** 2, 2000, cfg["max_samples_per_object"]))
        points, faces = trimesh.sample.sample_surface(mesh, count, seed=int(rng.integers(2**31)))
        normals = mesh.face_normals[faces]
        area = mesh.area / count
        distance = np.full((n + 1, count), np.inf)
        penetrating = np.zeros((n + 1, count), dtype=bool)
        depth = np.zeros(n + 1)
        voxels = _voxel_centres(points, pitch)
        for j in range(n):
            if j == i:
                continue
            lo, hi = bounds[j][0] - cd, bounds[j][1] + cd
            near = np.all((points >= lo) & (points <= hi), axis=1)
            if not near.any():
                continue
            query = o3d.core.Tensor(points[near].astype(np.float32))
            distance[j, near] = distance_scenes[j].compute_distance(query).numpy()
            inside = sdf_scenes[j].compute_signed_distance(query, nsamples=SIGN_RAYS).numpy() + proxy_offset
            depth[j] = max(0.0, float(-inside.min()))
            voxel_sdf = sdf_scenes[j].compute_signed_distance(o3d.core.Tensor(voxels[near].astype(np.float32)),
                                                              nsamples=SIGN_RAYS).numpy()
            penetrating[j, near] = voxel_sdf + proxy_offset < 0
        distance[n] = points[:, 2]
        penetrating[n] = voxels[:, 2] < 0
        depth[n] = max(0.0, float(-points[:, 2].min()))
        contact = distance <= cd
        contact[n] = points[:, 2] <= cfg["ground_contact_tol"]
        # Per-face labels on the original mesh.
        n_faces = len(mesh.faces)
        samples_per_face = np.bincount(faces, minlength=n_faces)
        safe = np.maximum(samples_per_face, 1)
        any_contact = contact.any(axis=0)
        any_penetration = penetrating.any(axis=0)
        partner = np.where(any_contact, np.argmin(np.where(contact, distance, np.inf), axis=0), -1)
        votes = np.bincount(faces[any_contact] * (n + 1) + partner[any_contact], minlength=n_faces * (n + 1))
        votes = votes.reshape(n_faces, n + 1)
        face_partner = np.where(votes.sum(axis=1) > 0, votes.argmax(axis=1), -1)
        per_object.append({
            "face_contact": (np.bincount(faces, weights=any_contact, minlength=n_faces) / safe).astype(np.float16),
            "face_penetration": (np.bincount(faces, weights=any_penetration, minlength=n_faces) / safe).astype(np.float16),
            "face_partner": face_partner.astype(np.int16),  # n = table, -1 = none
            "face_samples": samples_per_face.astype(np.int32),
            "stats": {
                "samples": count, "area_mm2": mesh.area * 1e6,
                "contact_area_mm2": {_partner_name(j, n): float(contact[j].sum() * area * 1e6) for j in range(n + 1) if contact[j].any()},
                "penetration_area_mm2": {_partner_name(j, n): float(penetrating[j].sum() * area * 1e6) for j in range(n + 1) if penetrating[j].any()},
                "penetration_depth_mm": {_partner_name(j, n): float(depth[j] * 1e3) for j in range(n + 1) if depth[j] > 0},
                "support_area_mm2": {_partner_name(j, n): float((contact[j] & (normals[:, 2] < SUPPORT_NORMAL_Z)).sum() * area * 1e6)
                                     for j in range(n + 1) if (contact[j] & (normals[:, 2] < SUPPORT_NORMAL_Z)).any()},
                "min_z_mm": float(points[:, 2].min() * 1e3),
                # Closest approach to each nearby object (surface to surface), for diagnosing gaps.
                "min_distance_mm": {str(j): float(distance[j].min() * 1e3) for j in range(n) if np.isfinite(distance[j]).any()},
            },
        })
        for j in range(n):
            if contact[j].any() or penetrating[j].any():
                key = (min(i, j), max(i, j))
                entry = pairs.setdefault(key, {"objects": list(key), "contact_area_mm2": 0.0, "penetration_area_mm2": 0.0,
                                               "max_penetration_depth_mm": 0.0})
                entry["contact_area_mm2"] += 0.5 * float(contact[j].sum() * area * 1e6)
                entry["penetration_area_mm2"] += 0.5 * float(penetrating[j].sum() * area * 1e6)
                entry["max_penetration_depth_mm"] = max(entry["max_penetration_depth_mm"], float(depth[j] * 1e3))
    return {"objects": per_object, "pairs": [pairs[k] for k in sorted(pairs)]}


def _partner_name(j: int, n: int) -> str:
    return GROUND if j == n else str(j)


def support_graph(objects: list[dict], contacts: dict, assets: dict) -> dict:
    """Directed edges parent -> child: the child rests on (or in) the parent."""
    edges = []
    for i, obj in enumerate(objects):
        for partner, area in contacts["objects"][i]["stats"]["support_area_mm2"].items():
            if area < MIN_SUPPORT_AREA_MM2:
                continue
            if partner == GROUND:
                edges.append({"parent": GROUND, "child": i, "relation": "on", "area_mm2": area})
                continue
            j = int(partner)
            parent_asset = assets[objects[j]["asset_id"]]
            inside = nested_inside(obj["T_world_com"], objects[j]["T_world_com"], parent_asset)
            edges.append({"parent": j, "child": i, "relation": "in" if inside else "on", "area_mm2": area})
    # A wedged object can touch its support with a second, small patch from below (a plate edge
    # under a bowl's lip), which reads as the reverse edge too. Keep the dominant direction.
    area_of = {(e["parent"], e["child"]): e["area_mm2"] for e in edges}
    secondary = [e for e in edges if e["parent"] != GROUND and area_of.get((e["child"], e["parent"]), -1) > e["area_mm2"]]
    edges = [e for e in edges if e not in secondary]
    supported = {e["child"] for e in edges}
    return {"nodes": [GROUND] + list(range(len(objects))), "edges": edges, "secondary_contacts": secondary,
            "unsupported": [i for i in range(len(objects)) if i not in supported]}


def select_views(visible_px: np.ndarray, amodal_px: np.ndarray, keep: int, min_px: int) -> list[int]:
    """Deterministic: most objects above `min_px`, then highest mean visible fraction."""
    usable = (visible_px >= min_px).sum(axis=1)
    fraction = np.where(amodal_px > 0, visible_px / np.maximum(amodal_px, 1), 0.0).mean(axis=1)
    order = sorted(range(len(usable)), key=lambda v: (-usable[v], -fraction[v], v))
    return sorted(order[:keep])


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def export_glb(meshes_local, transforms, path: Path) -> None:
    scene = trimesh.Scene()
    for k, (mesh, T) in enumerate(zip(meshes_local, transforms)):
        scene.add_geometry(mesh, node_name=f"obj_{k:03d}", geom_name=f"geom_{k:03d}", transform=np.asarray(T))
    path.write_bytes(scene.export(file_type="glb"))


def annotate_scene(scene_dir: Path, cfg: dict, asset_table: dict, keep_raw_views: bool = False) -> dict:
    started = time.perf_counter()
    scene_dir = Path(scene_dir)
    raw = json.loads((scene_dir / "raw" / "scene_raw.json").read_text())
    data = np.load(scene_dir / "raw" / "views.npz")
    rgb, depth_blender, seg = data["rgb"], data["depth"], data["seg"]
    acfg, rcfg = cfg["annotate"], cfg["render"]
    objects = raw["objects"]
    n_obj, n_views = len(objects), rgb.shape[0]
    camera = raw["camera"]
    for view in camera["views"]:
        view["T_world_cam"] = view["T_world_cam_opencv"]

    meshes_local = [load_mesh(o["mesh_path"]) for o in objects]
    meshes_world = [posed(m, o["T_world_mesh"]) for m, o in zip(meshes_local, objects)]
    proxies_world = [posed(load_mesh(asset_table[o["asset_id"]]["proxy_mesh"]), o["T_world_com"]) for o in objects]

    rays = raycast_views(meshes_world, raw["table"]["size"], camera)
    contacts = contact_analysis(meshes_world, proxies_world, acfg["proxy_offset"], acfg, raw["seed"] % 2**63)
    graph = support_graph(objects, contacts, asset_table)

    # Visibility, view selection and per-view poses.
    labels = np.arange(1, n_obj + 1)
    visible_px = np.stack([(seg == k).reshape(n_views, -1).sum(axis=1) for k in labels], axis=1)
    amodal_px = rays["amodal"].reshape(n_views, n_obj, -1).sum(axis=2)
    kept = select_views(visible_px, amodal_px, rcfg["keep_views"], acfg["min_visible_px"])

    for sub in ("rgb", "depth", "mask_visible", "mod_mask"):
        (scene_dir / sub).mkdir(exist_ok=True)
    views_out, mod_records, mask_ious, depth_errors, outside_amodal = [], [], [], [], []
    for v in range(n_views):
        name = f"v{v:02d}.png"
        Image.fromarray(rgb[v][..., :3]).save(scene_dir / "rgb" / name)
        depth_units = np.round(depth_blender[v] * 1e3 / DEPTH_UNIT_MM)
        depth_units[(depth_units > 65535) | ~np.isfinite(depth_units)] = 0
        Image.fromarray(depth_units.astype(np.uint16)).save(scene_dir / "depth" / name)
        Image.fromarray(seg[v].astype(np.uint8)).save(scene_dir / "mask_visible" / name)

        T_world_cam = np.asarray(camera["views"][v]["T_world_cam"])
        T_cam_world = np.linalg.inv(T_world_cam)
        per_object = []
        for k, obj in enumerate(objects):
            T_cam_mesh = T_cam_world @ np.asarray(obj["T_world_mesh"])
            visible = seg[v] == k + 1
            amodal = rays["amodal"][v, k]
            if visible_px[v, k] >= acfg["min_visible_px"]:
                mask_ious.append(_iou(visible, rays["labels"][v] == k + 1))
                outside_amodal.append(float((visible & ~amodal).sum() / max(visible.sum(), 1)))
            per_object.append({
                "index": k, "label": k + 1,
                "T_cam_mesh": T_cam_mesh.tolist(),
                "quaternion_wxyz": matrix_to_quat(T_cam_mesh[:3, :3]).tolist(),
                "translation": T_cam_mesh[:3, 3].tolist(), "scale": obj["scale"],
                "visible_px": int(visible_px[v, k]), "amodal_px": int(amodal_px[v, k]),
                "visible_fraction": float(visible_px[v, k] / amodal_px[v, k]) if amodal_px[v, k] else 0.0,
                "bbox_visible": _bbox(visible), "bbox_amodal": _bbox(amodal),
            })
        same = (rays["labels"][v] > 0) & (rays["labels"][v] == seg[v])
        if same.any():
            depth_errors.append(float(np.median(np.abs(depth_blender[v][same] - rays["depth"][v][same]))))
        view_out = {**{key: camera["views"][v][key] for key in ("view", "azimuth_deg", "elevation_deg", "distance")},
                    "T_world_cam": T_world_cam.tolist(), "T_cam_world": T_cam_world.tolist(),
                    "kept": v in kept, "rgb": f"rgb/{name}", "depth": f"depth/{name}", "mask_visible": f"mask_visible/{name}",
                    "objects": per_object}
        if v in kept:
            mod_mask = seg[v].astype(np.uint8).copy()
            for k in range(n_obj):
                if visible_px[v, k] < acfg["min_visible_px"]:
                    mod_mask[mod_mask == k + 1] = 0
            Image.fromarray(mod_mask).save(scene_dir / "mod_mask" / name)
            present = [int(x) for x in np.unique(mod_mask) if x]
            view_out["mod_mask"] = f"mod_mask/{name}"
            if present:
                mod_records.append({
                    "id": f"{raw['scene_id']}__v{v:02d}",
                    "scene_id": raw["scene_id"],
                    "image_path": f"rgb/{name}",
                    "instance_mask_path": f"mod_mask/{name}",
                    "num_parts": len(present),
                    "mesh_path": "gt_scene.glb",
                    "gt_object_map": {str(label - 1): f"obj_{label - 1:03d}" for label in present},
                })
        views_out.append(view_out)

    np.savez_compressed(scene_dir / "masks_amodal.npz", amodal=rays["amodal"])
    contact_arrays = {}
    for k, entry in enumerate(contacts["objects"]):
        for key in ("face_contact", "face_penetration", "face_partner", "face_samples"):
            contact_arrays[f"{key}_{k:03d}"] = entry[key]
    np.savez_compressed(scene_dir / "contacts.npz", **contact_arrays)
    export_glb(meshes_local, [o["T_world_mesh"] for o in objects], scene_dir / "gt_scene.glb")

    object_object = [s["stats"] for s in contacts["objects"]]
    contact_obj = sum(v for s in object_object for p, v in s["contact_area_mm2"].items() if p != GROUND)
    penetration_obj = sum(v for s in object_object for p, v in s["penetration_area_mm2"].items() if p != GROUND)
    contact_all = sum(v for s in object_object for v in s["contact_area_mm2"].values())
    penetration_all = sum(v for s in object_object for v in s["penetration_area_mm2"].values())
    max_depth = max([d for s in object_object for d in s["penetration_depth_mm"].values()] or [0.0])
    plan_relations = [{"child": o["index"], "parent": o["parent"], "relation": o["relation"]} for o in objects if o["parent"] is not None]
    edges = {(e["parent"], e["child"]): e["relation"] for e in graph["edges"]}
    scene = {
        "scene_id": raw["scene_id"], "level": raw["level"], "split": raw["split"], "index": raw["index"], "seed": raw["seed"],
        "world": {"up": "+z", "units": "m", "table_z": 0.0},
        "objects": [{
            **{key: o[key] for key in ("index", "label", "asset_id", "category", "role", "mesh_path", "licence",
                                       "T_world_mesh", "T_world_com", "com_mesh_frame", "scale")},
            "planned_parent": o["parent"], "planned_relation": o["relation"],
            "glb_node": f"obj_{o['index']:03d}",
            "attribution": asset_table[o["asset_id"]].get("attribution"),
            "contact": contacts["objects"][o["index"]]["stats"],
        } for o in objects],
        "camera": {"K": camera["K"], "width": camera["width"], "height": camera["height"], "fov_deg": camera["fov_deg"],
                   "convention": "opencv", "depth_unit_mm": DEPTH_UNIT_MM},
        "views": views_out,
        "kept_views": kept,
        "support_graph": graph,
        "planned_relations_hold": [{**r, "found": edges.get((r["parent"], r["child"]))} for r in plan_relations],
        "contacts": {
            "pairs": contacts["pairs"],
            "object_contact_area_mm2": contact_obj, "object_penetration_area_mm2": penetration_obj,
            "penetration_contact_ratio": penetration_obj / contact_obj if contact_obj else None,
            "contact_area_with_table_mm2": contact_all, "penetration_area_with_table_mm2": penetration_all,
            "max_penetration_depth_mm": max_depth,
            "settings": {k: acfg[k] for k in ("contact_distance", "voxel_pitch", "surface_spacing", "ground_contact_tol")},
        },
        "lighting": raw["lighting"], "table": raw["table"],
        "physics": {"attempts": raw["attempts"], "collision": raw["plan"].get("collision"), "seconds": raw["physics_seconds"]},
        "render": {**raw["render_settings"], "seconds": raw["render_seconds"]},
        "versions": raw["versions"],
        "checks": {
            "mask_iou_blender_vs_raycast_min": min(mask_ious) if mask_ious else None,
            "mask_iou_blender_vs_raycast_mean": float(np.mean(mask_ious)) if mask_ious else None,
            "visible_outside_amodal_max": max(outside_amodal) if outside_amodal else None,
            "depth_median_abs_error_mm_max": max(depth_errors) * 1e3 if depth_errors else None,
            "min_object_z_mm": min(s["min_z_mm"] for s in object_object),
        },
        "mod_records": mod_records,
        "annotate_seconds": time.perf_counter() - started,
    }
    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=1))
    if not keep_raw_views:
        (scene_dir / "raw" / "views.npz").unlink()
    return scene

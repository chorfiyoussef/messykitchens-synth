"""Turn raw scans into simulation-ready assets.

For every asset this writes ``<processed_root>/<id>/``:

* ``proxy.obj``: the collision mesh. It is rebuilt from a signed-distance field of the scan
  (majority-vote ray parity, robust to scan holes), extracted slightly *outside* the scan
  surface with marching cubes, smoothed, and decimated. The result is watertight with evenly
  sized triangles. Vertices are expressed relative to the centre of mass.
* ``props.json``: centre of mass (solid volume, uniform density) in the original mesh frame,
  volumes, mass, upright footprint and height, the top-face bounding-box area, the outer
  radius profile, and for containers the cavity profile used by the nesting planner. Also
  remesh quality statistics.
* ``coacd/part_*.obj`` (optional): convex decomposition of the proxy, the fallback
  collision shape when concave MESH collision glitches.
* ``collider.obj``: the scan decimated to at most ``collider.max_faces`` triangles, in the
  original mesh frame. Settled supports become static colliders of this surface (Bullet's cost
  grows with triangle count; some scans have 40k), with its deviation from the scan recorded.

The original ``meshes/model.obj`` stays the visual mesh and defines the pose frame.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh

from mksynth.config import REPO_ROOT, load_yaml


def load_scan(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    mesh = trimesh.Trimesh(np.asarray(loaded.vertices), np.asarray(loaded.faces), process=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def _raycasting_scene(mesh: trimesh.Trimesh):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
    )
    return scene


def signed_distance_grid(mesh: trimesh.Trimesh, pitch: float, pad: float, nsamples: int):
    import open3d as o3d

    lo = mesh.bounds[0] - pad
    hi = mesh.bounds[1] + pad
    shape = np.ceil((hi - lo) / pitch).astype(int) + 1
    axes = [lo[i] + pitch * np.arange(shape[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    scene = _raycasting_scene(mesh)
    sdf = scene.compute_signed_distance(o3d.core.Tensor(grid), nsamples=nsamples).numpy()
    return sdf.reshape(shape), lo


def build_proxy(scan: trimesh.Trimesh, cfg: dict) -> trimesh.Trimesh:
    import open3d as o3d
    from skimage.measure import marching_cubes

    extent = float(scan.extents.max())
    pitch = max(cfg["pitch"], extent / cfg["max_cells"])
    sdf, origin = signed_distance_grid(scan, pitch, pad=4 * pitch + cfg["offset"], nsamples=cfg["nsamples"])
    verts, faces, _, _ = marching_cubes(sdf, level=cfg["offset"], spacing=(pitch,) * 3)
    mesh = trimesh.Trimesh(verts + origin, faces, process=True)
    # Keep solid shells only: drop tiny islands and inverted internal voids from sign noise.
    parts = [part for part in mesh.split(only_watertight=False) if part.is_watertight]
    parts = [part for part in parts if part.volume > cfg["min_part_volume_fraction"] * mesh.convex_hull.volume]
    if not parts:
        raise RuntimeError("marching cubes produced no watertight solid shell")
    mesh = trimesh.util.concatenate(parts)

    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
    )
    legacy = legacy.filter_smooth_taubin(number_of_iterations=cfg["taubin_iterations"])
    best = None
    for target in (cfg["max_faces"], int(cfg["max_faces"] * 1.5), cfg["max_faces"] * 2):
        reduced = legacy.simplify_quadric_decimation(target) if len(mesh.faces) > target else legacy
        candidate = trimesh.Trimesh(np.asarray(reduced.vertices), np.asarray(reduced.triangles), process=True)
        if candidate.is_watertight and candidate.is_winding_consistent and candidate.volume > 0:
            best = candidate
            break
    if best is None:
        best = trimesh.Trimesh(np.asarray(legacy.vertices), np.asarray(legacy.triangles), process=True)
        if not best.is_watertight:
            raise RuntimeError("could not obtain a watertight proxy")
    if best.volume < 0:
        best.invert()
    return best


def surface_distance_stats(a: trimesh.Trimesh, b: trimesh.Trimesh, count: int = 20000, seed: int = 0):
    """Distances from samples on ``a`` to the surface of ``b`` (metres)."""
    import open3d as o3d

    points, _ = trimesh.sample.sample_surface(a, count, seed=seed)
    distance = _raycasting_scene(b).compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
    return {
        "mean_mm": float(distance.mean() * 1e3),
        "p99_mm": float(np.percentile(distance, 99) * 1e3),
        "max_mm": float(distance.max() * 1e3),
    }


def height_map(mesh: trimesh.Trimesh, step: float):
    """Top-down first-hit height and normal z over the XY bounding box."""
    import open3d as o3d

    lo, hi = mesh.bounds
    xs = np.arange(lo[0] + step / 2, hi[0], step)
    ys = np.arange(lo[1] + step / 2, hi[1], step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    origins = np.stack([gx, gy, np.full_like(gx, hi[2] + 0.01)], -1).reshape(-1, 3)
    rays = np.concatenate([origins, np.tile([0.0, 0.0, -1.0], (len(origins), 1))], 1).astype(np.float32)
    hit = _raycasting_scene(mesh).cast_rays(o3d.core.Tensor(rays))
    t = hit["t_hit"].numpy()
    normals = hit["primitive_normals"].numpy()
    z = np.where(np.isfinite(t), origins[:, 2] - t, np.nan).reshape(gx.shape)
    nz = np.abs(normals[:, 2]).reshape(gx.shape)
    return gx, gy, z, nz


def shape_properties(proxy: trimesh.Trimesh, category: dict, cfg: dict) -> dict:
    """Geometry used by the stacking/nesting planner, in the upright canonical frame."""
    step = cfg["heightmap_step"]
    lo, hi = proxy.bounds
    height = float(hi[2] - lo[2])
    com = proxy.center_mass
    radial = np.linalg.norm(proxy.vertices[:, :2] - com[:2], axis=1)
    edges = np.linspace(lo[2], hi[2], cfg["profile_bins"] + 1)
    outer_profile = []
    for z0, z1 in zip(edges[:-1], edges[1:]):
        inside = (proxy.vertices[:, 2] >= z0) & (proxy.vertices[:, 2] <= z1)
        outer_profile.append(float(radial[inside].max()) if inside.any() else 0.0)

    gx, gy, z, nz = height_map(proxy, step)
    hit = np.isfinite(z)
    top_cells = hit & (z >= lo[2] + 0.5 * height) & (nz > cfg["top_normal_z"])
    if top_cells.any():
        top_dx = gx[top_cells].max() - gx[top_cells].min() + step
        top_dy = gy[top_cells].max() - gy[top_cells].min() + step
        top_bbox_area = float(top_dx * top_dy)
        top_z = float(np.median(z[top_cells]))
        top_center = [float((gx[top_cells].max() + gx[top_cells].min()) / 2),
                      float((gy[top_cells].max() + gy[top_cells].min()) / 2)]
    else:
        top_dx = top_dy = top_bbox_area = 0.0
        top_z, top_center = float(hi[2]), [float(com[0]), float(com[1])]

    props = {
        "height": height,
        "z_min": float(lo[2]),
        "z_max": float(hi[2]),
        "footprint_xy": [float(hi[0] - lo[0]), float(hi[1] - lo[1])],
        "footprint_area": float((hi[0] - lo[0]) * (hi[1] - lo[1])),
        "radius_xy": float(radial.max()),
        "bounding_radius": float(np.linalg.norm(proxy.vertices - com, axis=1).max()),
        "outer_profile_z": edges[:-1].tolist(),
        "outer_profile_r": outer_profile,
        "top_bbox_area": top_bbox_area,
        "top_bbox_xy": [float(top_dx), float(top_dy)],
        "top_center_xy": top_center,
        "top_z": top_z,
        "cavity": None,
    }
    if category.get("container"):
        props["cavity"] = cavity_properties(gx, gy, z, com, height, lo[2], step, cfg)
    return props


def cavity_properties(gx, gy, z, com, height, z_min, step, cfg) -> dict | None:
    """Free space inside a container, measured from the top-down height map.

    r(level) is the radius of the largest disc around the cavity axis in which no surface
    rises above `level`, so an upright object whose outer radius at height h is below
    r(floor + h) fits without touching the walls.
    """
    hit = np.isfinite(z)
    if not hit.any():
        return None
    center = np.array([com[0], com[1]])
    dist = np.hypot(gx - center[0], gy - center[1])
    near = hit & (dist <= cfg["cavity_center_radius"])
    if not near.any():
        return None
    floor = float(np.nanmin(z[near]))
    rim = float(np.nanmax(z[hit]))
    depth = rim - floor
    if depth < cfg["min_cavity_depth"] or depth < cfg["min_cavity_depth_fraction"] * height:
        return None
    levels = np.linspace(floor, rim, cfg["profile_bins"] + 1)
    radii = []
    for level in levels:
        blocked = hit & (z > level + 1e-4)
        radii.append(float(dist[blocked].min()) if blocked.any() else float(dist.max()))
    radii = np.maximum.accumulate(np.asarray(radii))  # free radius can only grow with height
    open_radius = float(radii[-2])
    inside = hit & (dist < open_radius)
    volume = float(np.sum(np.clip(rim - z[inside], 0, None)) * step * step)
    return {
        "floor_z": floor,
        "rim_z": rim,
        "depth": depth,
        "axis_xy": center.tolist(),
        "levels": levels.tolist(),
        "radius": radii.tolist(),
        "opening_radius": open_radius,
        "volume": volume,
    }


def support_polygon_margin(proxy: trimesh.Trimesh, contact_tol: float) -> dict:
    from scipy.spatial import ConvexHull

    base = proxy.vertices[proxy.vertices[:, 2] < proxy.bounds[0][2] + contact_tol][:, :2]
    com = proxy.center_mass[:2]
    if len(base) < 3:
        return {"stable": False, "margin": 0.0}
    hull = ConvexHull(base)
    # Hull equations: n.x + d <= 0 inside; margin = min distance to the edges.
    signed = hull.equations[:, :2] @ com + hull.equations[:, 2]
    return {"stable": bool(np.all(signed <= 0)), "margin": float(-signed.max())}


def coacd_decompose(proxy: trimesh.Trimesh, out_dir: Path, cfg: dict) -> int:
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(proxy.vertices, proxy.faces),
        threshold=cfg["threshold"],
        max_convex_hull=cfg["max_convex_hull"],
        preprocess_mode="off",
        seed=0,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part_*.obj"):
        old.unlink()
    for index, (vertices, faces) in enumerate(parts):
        hull = shrink_convex(np.asarray(vertices), cfg["shrink"]) or trimesh.Trimesh(vertices, faces)
        hull.export(out_dir / f"part_{index:03d}.obj")
    return len(parts)


def shrink_convex(vertices: np.ndarray, distance: float) -> trimesh.Trimesh | None:
    """Move every face of a convex hull inward by `distance` (None if the part is too thin).

    The proxy sits `remesh.offset` outside the scan, so its convex parts are shrunk by the same
    amount to put the collision surface back on the scanned surface.
    """
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    if distance <= 0:
        return None
    hull = ConvexHull(vertices)
    equations = hull.equations.copy()  # n . x + c <= 0 inside
    equations[:, 3] += distance
    interior = vertices.mean(axis=0)
    if np.max(equations[:, :3] @ interior + equations[:, 3]) >= -1e-6:
        return None
    points = HalfspaceIntersection(equations, interior).intersections
    return trimesh.convex.convex_hull(points)


def build_collider(scan: trimesh.Trimesh, path: Path, cfg: dict) -> dict:
    """Quadric-decimate the scan for static collision; report the two-sided surface deviation."""
    import open3d as o3d

    collider = scan
    if len(scan.faces) > cfg["max_faces"]:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(scan.vertices),
                                         o3d.utility.Vector3iVector(scan.faces))
        mesh = mesh.simplify_quadric_decimation(cfg["max_faces"])
        collider = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=True)
    collider.export(path)
    there, back = surface_distance_stats(scan, collider), surface_distance_stats(collider, scan)
    return {"faces": int(len(collider.faces)), "p99_mm": max(there["p99_mm"], back["p99_mm"]),
            "max_mm": max(there["max_mm"], back["max_mm"])}


def add_colliders(out_root: Path, raw_root: Path, cfg: dict, ids: list[str]) -> dict:
    """Build only the colliders, updating existing records (no proxy / CoACD recomputation)."""
    table_path = out_root / "asset_table.json"
    table = json.loads(table_path.read_text())
    for asset_id in ids:
        record = table[asset_id]
        path = out_root / asset_id / "collider.obj"
        record["collider_mesh"] = str(path.relative_to(REPO_ROOT))
        record["collider"] = build_collider(load_scan(REPO_ROOT / record["visual_mesh"]), path, cfg["collider"])
        (out_root / asset_id / "props.json").write_text(json.dumps(record, indent=2) + "\n")
        c = record["collider"]
        print(f"{asset_id:30s} collider {record['remesh']['scan_faces']:6d} -> {c['faces']:5d} faces, "
              f"p99 {c['p99_mm']:.2f} max {c['max_mm']:.2f} mm")
    table_path.write_text(json.dumps(table, indent=2) + "\n")
    return table


def process_asset(asset: dict, assets_cfg: dict, cfg: dict, raw_root: Path, out_root: Path, coacd: bool) -> dict:
    started = time.perf_counter()
    raw_dir = raw_root / asset["id"]
    source = json.loads((raw_dir / "source.json").read_text())
    category = dict(assets_cfg["categories"][asset["category"]])
    scan = load_scan(raw_dir / source["mesh"])
    proxy = build_proxy(scan, cfg["remesh"])
    density = float(asset["mass"]) / proxy.volume if "mass" in asset else float(asset.get("density", category["density"]))
    com = proxy.center_mass + np.asarray(asset.get("com_offset", [0.0, 0.0, 0.0]))
    props = shape_properties(proxy, category, cfg["shape"])

    out_dir = out_root / asset["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    centred = proxy.copy()
    centred.apply_translation(-com)
    centred.export(out_dir / "proxy.obj")
    edges = proxy.edges_unique_length
    record = {
        "id": asset["id"],
        "source": asset["source"],
        "name": asset["name"],
        "category": asset["category"],
        "split": asset["split"],
        "licence": source["licence"],
        "attribution": source["attribution"],
        "visual_mesh": str((raw_dir / source["mesh"]).relative_to(REPO_ROOT)),
        "visual_mesh_sha256": source["files_sha256"][source["mesh"]],
        "proxy_mesh": str((out_dir / "proxy.obj").relative_to(REPO_ROOT)),
        "com": com.tolist(),
        "density": density,
        "volume_solid": float(proxy.volume),
        "volume_hull": float(proxy.convex_hull.volume),
        "mass": float(proxy.volume * density),
        "roles": {
            "container": bool(asset.get("container", category["container"])) and props["cavity"] is not None,
            "support": bool(asset.get("support", category["support"])) and not asset.get("never_support", False),
            "on_top": bool(asset.get("on_top", category["on_top"])) and not asset.get("never_on_top", False),
        },
        "upright": support_polygon_margin(proxy, cfg["shape"]["base_contact_tol"]),
        **props,
        "remesh": {
            "faces": int(len(proxy.faces)),
            "scan_faces": int(len(scan.faces)),
            "edge_mm": {
                "min": float(edges.min() * 1e3),
                "median": float(np.median(edges) * 1e3),
                "max": float(edges.max() * 1e3),
            },
            "watertight": bool(proxy.is_watertight),
            "scan_to_proxy": surface_distance_stats(scan, proxy),
            "proxy_to_scan": surface_distance_stats(proxy, scan),
        },
    }
    record["remesh"]["ok"] = record["remesh"]["scan_to_proxy"]["p99_mm"] <= cfg["remesh"]["max_scan_to_proxy_p99_mm"]
    record["collider_mesh"] = str((out_dir / "collider.obj").relative_to(REPO_ROOT))
    record["collider"] = build_collider(scan, out_dir / "collider.obj", cfg["collider"])
    if asset.get("note"):
        record["note"] = asset["note"]
    if coacd:
        record["coacd_parts"] = coacd_decompose(centred, out_dir / "coacd", cfg["coacd"])
    else:
        record["coacd_parts"] = len(list((out_dir / "coacd").glob("part_*.obj")))
    record["seconds"] = round(time.perf_counter() - started, 2)
    (out_dir / "props.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


DEFAULTS = {
    "remesh": {
        "pitch": 0.0015,          # SDF grid spacing (m)
        "max_cells": 256,         # cap per axis; pitch grows for large objects
        "offset": 0.0004,         # iso level outside the scan surface (m)
        "nsamples": 7,            # rays per inside/outside vote
        "taubin_iterations": 5,
        "max_faces": 6000,
        "min_part_volume_fraction": 0.002,
        "max_scan_to_proxy_p99_mm": 2.0,  # larger means scan surfaces were filled over (flagged)
    },
    "shape": {
        "heightmap_step": 0.002,
        "profile_bins": 24,
        "top_normal_z": 0.85,
        "cavity_center_radius": 0.012,
        "min_cavity_depth": 0.015,
        "min_cavity_depth_fraction": 0.25,
        "base_contact_tol": 0.004,
    },
    "coacd": {"threshold": 0.04, "max_convex_hull": 48, "shrink": 0.0004},  # shrink = remesh offset
    "collider": {"max_faces": 8000},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", default=str(REPO_ROOT / "configs/assets.yaml"))
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "data/assets/raw"))
    parser.add_argument("--out-root", default=str(REPO_ROOT / "data/assets/processed"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--coacd", action="store_true", help="Also compute CoACD convex decompositions")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--colliders_only", action="store_true", help="Only (re)build collider.obj for existing records")
    args = parser.parse_args()

    assets_cfg = load_yaml(args.assets)
    cfg = DEFAULTS
    raw_root, out_root = Path(args.raw_root), Path(args.out_root)
    selected = [
        a for a in assets_cfg["assets"]
        if (args.only and a["id"] in args.only) or (not args.only and a["split"] in args.splits)
    ]
    if args.colliders_only:
        add_colliders(out_root, raw_root, cfg, [a["id"] for a in selected])
        return
    records = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(process_asset, a, assets_cfg, cfg, raw_root, out_root, args.coacd): a["id"] for a in selected
        }
        for future in as_completed(futures):
            asset_id = futures[future]
            try:
                record = future.result()
                records[asset_id] = record
                r = record["remesh"]
                cav = record["cavity"]
                print(
                    f"{asset_id:30s} faces {r['faces']:5d} edge_med {r['edge_mm']['median']:.1f}mm "
                    f"scan->proxy p99 {r['scan_to_proxy']['p99_mm']:.2f}mm  mass {record['mass']*1e3:6.0f}g "
                    f"h {record['height']*100:4.1f}cm upright {record['upright']['stable']} remesh_ok {r['ok']} "
                    f"cavity {'%.1fcm r%.1fcm' % (cav['depth']*100, cav['opening_radius']*100) if cav else '-'} "
                    f"{record['seconds']}s"
                )
            except Exception as error:  # noqa: BLE001
                print(f"{asset_id:30s} FAILED {type(error).__name__}: {error}")
    table_path = out_root / "asset_table.json"
    existing = json.loads(table_path.read_text()) if table_path.is_file() else {}
    existing.update(records)
    table_path.write_text(json.dumps(dict(sorted(existing.items())), indent=2) + "\n")
    print(f"wrote {table_path} ({len(existing)} assets)")


if __name__ == "__main__":
    main()

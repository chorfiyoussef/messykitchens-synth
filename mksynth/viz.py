"""Overlays and contact sheets for checking generated scenes by eye.

    python -m mksynth.viz out/easy/scenes/train_easy_00000          # one scene -> <scene>/viz/*.png
    python -m mksynth.viz out/easy --sheet out/easy/contact_sheet.png  # thumbnail grid of all scenes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mksynth.annotate import load_mesh

PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
    [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [170, 110, 40],
], dtype=np.uint8)


def colour(k: int) -> tuple:
    return tuple(int(c) for c in PALETTE[k % len(PALETTE)])


def project(K, T_cam_obj, points) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T_cam_obj)
    cam = points @ T[:3, :3].T + T[:3, 3]
    uv = cam @ np.asarray(K).T
    return uv[:, :2] / uv[:, 2:3], cam[:, 2]


def boundary(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask)
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    return edge & mask


def mask_overlay(rgb: np.ndarray, labels: np.ndarray, amodal: np.ndarray) -> Image.Image:
    out = rgb.astype(np.float32).copy()
    for k in range(amodal.shape[0]):
        visible = labels == k + 1
        out[visible] = 0.45 * out[visible] + 0.55 * PALETTE[k % len(PALETTE)]
    for k in range(amodal.shape[0]):
        edge = boundary(amodal[k])
        out[edge] = 255  # amodal outline in white, includes occluded parts
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def pose_overlay(rgb: np.ndarray, scene: dict, view: dict) -> Image.Image:
    """Original meshes projected with the annotated pose (dots) plus COM axes (x red, y green, z blue)."""
    image = Image.fromarray((rgb * 0.6).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    K = scene["camera"]["K"]
    for obj, pose in zip(scene["objects"], view["objects"]):
        mesh = load_mesh(obj["mesh_path"])
        vertices = np.asarray(mesh.vertices)[:: max(1, len(mesh.vertices) // 1500)]
        uv, _ = project(K, pose["T_cam_mesh"], vertices)
        c = colour(obj["index"])
        for u, v in uv:
            draw.point((u, v), fill=c)
        com = np.asarray(obj["com_mesh_frame"])
        axes = np.vstack([com, com + [0.04, 0, 0], com + [0, 0.04, 0], com + [0, 0, 0.04]])
        uv, _ = project(K, pose["T_cam_mesh"], axes)
        for end, axis_colour in zip(uv[1:], [(255, 0, 0), (0, 255, 0), (40, 90, 255)]):
            draw.line([tuple(uv[0]), tuple(end)], fill=axis_colour, width=2)
    return image


def depth_image(depth_png: np.ndarray, unit_mm: float) -> Image.Image:
    depth = depth_png.astype(np.float32) * unit_mm
    valid = depth > 0
    if not valid.any():
        return Image.fromarray(np.zeros(depth.shape, np.uint8))
    lo, hi = np.percentile(depth[valid], [1, 99])
    scaled = np.where(valid, 1 - (depth - lo) / max(hi - lo, 1e-6), 0).clip(0, 1)
    return Image.fromarray((scaled * 255).astype(np.uint8)).convert("RGB")


def contact_overlay(rgb: np.ndarray, scene: dict, view: dict, contacts: dict, depth_png: np.ndarray) -> Image.Image:
    """Faces of the original meshes labelled in contact (object: orange, table: cyan) or penetrating
    (magenta), projected into the view; faces hidden behind other surfaces are drawn dimmer."""
    image = Image.fromarray((rgb * 0.45).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    K = scene["camera"]["K"]
    depth = depth_png.astype(np.float32) * scene["camera"]["depth_unit_mm"] / 1e3
    n = len(scene["objects"])
    for obj, pose in zip(scene["objects"], view["objects"]):
        mesh = load_mesh(obj["mesh_path"])
        k = obj["index"]
        fc = contacts[f"face_contact_{k:03d}"].astype(np.float32)
        fp = contacts[f"face_penetration_{k:03d}"].astype(np.float32)
        partner = contacts[f"face_partner_{k:03d}"]
        selected = np.nonzero((fc > 0.3) | (fp > 0.3))[0]
        if not len(selected):
            continue
        centres = mesh.triangles_center[selected]
        uv, z = project(K, pose["T_cam_mesh"], centres)
        for f, (u, v), zz in zip(selected, uv, z):
            iu, iv = int(u), int(v)
            if not (0 <= iu < depth.shape[1] and 0 <= iv < depth.shape[0]):
                continue
            front = depth[iv, iu] == 0 or zz <= depth[iv, iu] + 0.004
            if fp[f] > 0.3:
                c = (255, 0, 255)
            elif partner[f] == n:
                c = (0, 220, 255)
            else:
                c = (255, 150, 0)
            if not front:
                c = tuple(int(x * 0.4) for x in c)
            draw.ellipse([u - 1.5, v - 1.5, u + 1.5, v + 1.5], fill=c)
    return image


def scene_panels(scene_dir: Path) -> Image.Image:
    """Rows = kept views; columns = RGB, visible masks + amodal outlines, pose reprojection, contacts, depth."""
    scene_dir = Path(scene_dir)
    scene = json.loads((scene_dir / "scene.json").read_text())
    amodal = np.load(scene_dir / "masks_amodal.npz")["amodal"]
    contacts = np.load(scene_dir / "contacts.npz")
    size = 256
    rows = []
    for v in scene["kept_views"]:
        view = scene["views"][v]
        rgb = np.asarray(Image.open(scene_dir / view["rgb"]).convert("RGB"))
        labels = np.asarray(Image.open(scene_dir / view["mask_visible"]))
        depth = np.asarray(Image.open(scene_dir / view["depth"]))
        panels = [Image.fromarray(rgb), mask_overlay(rgb, labels, amodal[v]), pose_overlay(rgb, scene, view),
                  contact_overlay(rgb, scene, view, contacts, depth), depth_image(depth, scene["camera"]["depth_unit_mm"])]
        row = Image.new("RGB", (size * len(panels), size))
        for i, panel in enumerate(panels):
            row.paste(panel.resize((size, size), Image.BILINEAR), (i * size, 0))
        ImageDraw.Draw(row).text((4, 4), f"v{v:02d} az {view['azimuth_deg']:.0f} el {view['elevation_deg']:.0f}", fill=(255, 255, 255))
        rows.append(row)
    header = Image.new("RGB", (rows[0].width, 34), (20, 20, 20))
    draw = ImageDraw.Draw(header)
    draw.text((6, 4), f"{scene['scene_id']}  ratio pen/contact {scene['contacts']['penetration_contact_ratio']}  "
                      f"max pen {scene['contacts']['max_penetration_depth_mm']:.2f} mm", fill=(255, 255, 255))
    legend = "  ".join(f"{o['index']}:{o['asset_id'].removeprefix('gso_')}" for o in scene["objects"])
    draw.text((6, 18), legend, fill=(200, 200, 200))
    sheet = Image.new("RGB", (rows[0].width, header.height + size * len(rows)))
    sheet.paste(header, (0, 0))
    for i, row in enumerate(rows):
        sheet.paste(row, (0, header.height + i * size))
    return sheet


def contact_sheet(scene_dirs: list[Path], columns: int = 6, size: int = 224) -> Image.Image:
    """One thumbnail per scene: its first kept view with visible masks."""
    tiles = []
    for scene_dir in scene_dirs:
        scene = json.loads((scene_dir / "scene.json").read_text())
        v = scene["kept_views"][0]
        rgb = np.asarray(Image.open(scene_dir / scene["views"][v]["rgb"]).convert("RGB"))
        labels = np.asarray(Image.open(scene_dir / scene["views"][v]["mask_visible"]))
        amodal = np.load(scene_dir / "masks_amodal.npz")["amodal"][v]
        left = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
        right = mask_overlay(rgb, labels, amodal).resize((size, size), Image.BILINEAR)
        tile = Image.new("RGB", (2 * size, size))
        tile.paste(left, (0, 0))
        tile.paste(right, (size, 0))
        ImageDraw.Draw(tile).text((4, 4), scene["scene_id"], fill=(255, 255, 255))
        tiles.append(tile)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 2 * size, rows * size), (30, 30, 30))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * 2 * size, (i // columns) * size))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="a scene directory, or an output directory with scenes/")
    parser.add_argument("--sheet", help="write a thumbnail grid of all scenes to this file")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--max_scenes", type=int, default=48, help="evenly spaced sample for large datasets")
    args = parser.parse_args()
    path = Path(args.path)
    if (path / "scene.json").is_file():
        (path / "viz").mkdir(exist_ok=True)
        scene_panels(path).save(path / "viz" / "panels.png")
        print(path / "viz" / "panels.png")
        return
    dirs = sorted(p for p in (path / "scenes").glob("*") if (p / "scene.json").is_file())
    if len(dirs) > args.max_scenes:
        dirs = [dirs[i] for i in np.linspace(0, len(dirs) - 1, args.max_scenes).round().astype(int)]
    contact_sheet(dirs, args.columns).save(args.sheet or path / "contact_sheet.png")
    print(args.sheet or path / "contact_sheet.png")


if __name__ == "__main__":
    main()

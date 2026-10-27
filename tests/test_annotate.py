import numpy as np
import open3d as o3d
import trimesh
from conftest import make_asset

from mksynth.annotate import _tensor_scene, camera_rays, contact_analysis, select_views, support_graph

CFG = {"contact_distance": 0.0025, "voxel_pitch": 0.001, "surface_spacing": 0.002, "max_samples_per_object": 50000,
       "ground_contact_tol": 0.0025}


def box(size, centre):
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(centre)
    return mesh


def test_select_views_prefers_complete_visibility():
    visible = np.array([[500, 500, 0], [500, 500, 500], [300, 500, 500], [500, 100, 500]])
    amodal = np.full_like(visible, 500)
    assert select_views(visible, amodal, keep=2, min_px=200) == [1, 2]
    assert select_views(visible, amodal, keep=2, min_px=200) == select_views(visible, amodal, keep=2, min_px=200)


def test_camera_rays_give_z_depth():
    K = np.array([[500.0, 0, 64], [0, 500.0, 64], [0, 0, 1]])
    # Camera 1 m above the origin looking straight down (OpenCV: z forward = world -z).
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0, 0, 1.0]
    scene, _ = _tensor_scene([box([2, 2, 0.01], [0, 0, -0.005])])
    hit = scene.cast_rays(o3d.core.Tensor(camera_rays(K, T, 128, 128)))
    assert np.allclose(hit["t_hit"].numpy(), 1.0, atol=1e-5)  # z-depth, also at the image corners


def test_contacts_resting_and_penetrating():
    bottom = box([0.1, 0.1, 0.05], [0, 0, 0.025])
    resting = box([0.06, 0.06, 0.04], [0, 0, 0.05 + 0.001 + 0.02])     # 1 mm gap
    sunk = box([0.06, 0.06, 0.04], [0, 0, 0.05 - 0.0015 + 0.02])       # 1.5 mm overlap

    result = contact_analysis([bottom, resting], [bottom, resting], 0.0, CFG, seed=0)
    stats = result["objects"][1]["stats"]
    assert abs(stats["contact_area_mm2"]["0"] - 3600) < 400          # the 6 x 6 cm bottom face
    assert stats["support_area_mm2"]["0"] > 3000
    assert "0" not in stats["penetration_area_mm2"]
    assert result["objects"][0]["stats"]["contact_area_mm2"]["table"] > 9000
    assert result["pairs"][0]["max_penetration_depth_mm"] == 0.0

    result = contact_analysis([bottom, sunk], [bottom, sunk], 0.0, CFG, seed=0)
    stats = result["objects"][1]["stats"]
    assert abs(stats["penetration_depth_mm"]["0"] - 1.5) < 0.05
    assert stats["penetration_area_mm2"]["0"] > 3000


def test_contacts_far_apart():
    a = box([0.05, 0.05, 0.05], [0, 0, 0.025])
    b = box([0.05, 0.05, 0.05], [0.2, 0, 0.025])
    result = contact_analysis([a, b], [a, b], 0.0, CFG, seed=0)
    assert result["pairs"] == []
    assert set(result["objects"][0]["stats"]["contact_area_mm2"]) == {"table"}


def test_support_graph_keeps_dominant_direction():
    assets = {"plate": make_asset("plate", height=0.02, radius=0.1), "cup": make_asset("cup", height=0.06, radius=0.035)}
    objects = [{"asset_id": "plate", "T_world_com": np.eye(4).tolist()},
               {"asset_id": "cup", "T_world_com": np.eye(4).tolist()}]
    stats = lambda areas: {"stats": {"support_area_mm2": areas}}  # noqa: E731
    contacts = {"objects": [stats({"table": 900.0, "1": 6.0}), stats({"0": 300.0})]}
    graph = support_graph(objects, contacts, assets)
    pairs = {(e["parent"], e["child"]) for e in graph["edges"]}
    assert pairs == {("table", 0), (0, 1)}               # cup on plate; the small reverse patch is dropped
    assert [(e["parent"], e["child"]) for e in graph["secondary_contacts"]] == [(1, 0)]
    assert graph["unsupported"] == []

import blenderproc as bproc  # noqa: I001  (BlenderProc requires this to be the first line)

# Blender worker: simulate and render a batch of scenes.
#
# Run through BlenderProc so the pinned Blender (4.2.1, BlenderProc 2.8.0) is used:
#
#     blenderproc run blender/scene_worker.py <job.json>
#
# The job file (written by generate.py) holds the resolved config and a list of scenes. For
# each scene the worker plans a layout (mksynth.layout), simulates it stage by stage (base
# objects, then stacked / nested objects dropped onto where their supports settled), applies
# the acceptance checks (mksynth.physics_checks), retries with a new layout on failure, and
# renders the accepted scene. Raw outputs go to ``<scene>/raw/``; generate.py turns them into
# the final annotations. A ``raw/DONE`` or ``raw/FAILED.json`` marker records the outcome.

import json
import math
import sys
import time
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix
from mathutils.bvhtree import BVHTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mksynth import geometry as geo  # noqa: E402
from mksynth import layout  # noqa: E402
from mksynth import physics_checks as checks  # noqa: E402
from mksynth.compat import CompatConfig, build_table  # noqa: E402

GROUND_GROUP = 0      # the table plane
SUPPORT_GROUP = 1     # static scan colliders used by the refinement passes (never touch the table)
ANCHOR_GROUP = 19
OBJECT_GROUPS = [i in (GROUND_GROUP, SUPPORT_GROUP) for i in range(20)]


class SimFailure(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


# Cycles reports running out of GPU memory only in its log and returns black frames.
BLACK_FRAME_MEAN = 2.0


# --------------------------------------------------------------------------- helpers

def repo_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)


def import_obj(path: str) -> bpy.types.Object:
    # Assets are z-up already; keep their coordinates exactly.
    bpy.ops.wm.obj_import(filepath=repo_path(path), forward_axis="Y", up_axis="Z")
    objects = bpy.context.selected_objects
    if len(objects) != 1:
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.join()
    obj = bpy.context.selected_objects[0]
    obj.rotation_mode = "QUATERNION"
    return obj


def import_visual(asset: dict) -> bpy.types.Object:
    """Original scan with its texture. GSO .mtl files point to a texture next to the mesh, but it
    lives in materials/textures/, so the material is rebuilt explicitly."""
    obj = import_obj(asset["visual_mesh"])
    texture = REPO_ROOT / Path(asset["visual_mesh"]).parent.parent / "materials" / "textures" / "texture.png"
    material = bpy.data.materials.new(f"mat_{asset['id']}")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.6
    if texture.is_file():
        image = nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(str(texture), check_existing=True)
        links.new(image.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def mesh_vertices(obj) -> np.ndarray:
    count = len(obj.data.vertices)
    out = np.empty(count * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", out)
    return out.reshape(count, 3)


def mesh_polygons(obj) -> list:
    return [tuple(p.vertices) for p in obj.data.polygons]


def to_matrix(T) -> Matrix:
    return Matrix(np.asarray(T, dtype=np.float64).tolist())


def from_matrix(M) -> np.ndarray:
    return np.array(M, dtype=np.float64)


def set_rigid_body(obj, *, active: bool, shape: str, margin: float, mass: float, friction: float, phys: dict):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    rb = obj.rigid_body
    rb.type = "ACTIVE" if active else "PASSIVE"
    rb.collision_shape = shape
    rb.mesh_source = "BASE"
    rb.use_margin = True
    rb.collision_margin = margin
    rb.mass = mass
    rb.friction = friction
    rb.restitution = phys["restitution"]
    rb.use_deactivation = phys["use_deactivation"]
    rb.linear_damping = phys["linear_damping"]
    rb.angular_damping = phys["angular_damping"]
    return rb


# --------------------------------------------------------------------------- physics

class Simulation:
    def __init__(self, cfg: dict, pool: dict, plan: dict, collision: str):
        self.cfg = cfg
        self.phys = cfg["physics"]
        self.pool = pool
        self.plan = plan
        self.collision = collision
        self.bodies = {}           # object index -> proxy object
        self.local_vertices = {}   # object index -> collision vertices in the COM frame
        self.parts = {}            # object index -> convex part objects of its compound shape
        self.frozen = {}           # object index -> (static scan collider, its world matrix)
        self.poses = {}            # object index -> latest settled 4x4 COM pose
        self.log = {"stages": [], "peak_speed": 0.0, "sim_seconds": 0.0}
        scene = bpy.context.scene
        scene.render.fps = self.phys["fps"]
        scene.frame_start = 1
        if scene.rigidbody_world is None:
            bpy.ops.rigidbody.world_add()
        rbw = scene.rigidbody_world
        rbw.enabled = True
        rbw.substeps_per_frame = self.phys["substeps_per_frame"]
        rbw.solver_iterations = self.phys["solver_iterations"]
        rbw.point_cache.frame_start = 1
        self.rbw = rbw
        self._make_ground()

    def frames(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.phys["fps"])))

    def _make_ground(self):
        phys = self.phys
        margin = phys["margin_ground"]
        # A 3 x 3 m slab whose top face is the table plane z = 0. Bullet embeds a box's margin,
        # so the collision surface stays on the face.
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.05), scale=(3.0, 3.0, 0.1))
        ground = bpy.context.active_object
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        ground.name = "phys_ground"
        ground.hide_render = True
        active = phys["ground_mode"] == "active_fixed"
        rb = set_rigid_body(ground, active=active, shape="BOX", margin=margin, mass=1000.0, friction=0.6, phys=phys)
        rb.collision_collections = [i == GROUND_GROUP for i in range(20)]
        if active:
            # Paper App. C.2: an active plane with translation and rotation fully constrained.
            bpy.ops.mesh.primitive_cube_add(size=0.01, location=(0, 0, -2.0))
            anchor = bpy.context.active_object
            anchor.name = "phys_anchor"
            anchor.hide_render = True
            rb_anchor = set_rigid_body(anchor, active=False, shape="BOX", margin=0.0, mass=1.0, friction=0.0, phys=phys)
            rb_anchor.collision_collections = [i == ANCHOR_GROUP for i in range(20)]
            bpy.ops.object.empty_add(location=(0, 0, -1.0))
            lock = bpy.context.active_object
            lock.name = "phys_ground_lock"
            bpy.ops.rigidbody.constraint_add()
            lock.rigid_body_constraint.type = "FIXED"
            lock.rigid_body_constraint.object1 = anchor
            lock.rigid_body_constraint.object2 = ground

    def _add_body(self, obj: dict, pose: np.ndarray):
        asset = self.pool[obj["asset_id"]]
        proxy = import_obj(asset["proxy_mesh"])
        proxy.name = f"proxy_{obj['index']:03d}"
        proxy.hide_render = True
        proxy.matrix_world = to_matrix(pose)
        shape = "COMPOUND" if self.collision == "coacd" else "MESH"
        margin = self.phys["margin_object"]
        rb = set_rigid_body(proxy, active=True, shape=shape, margin=margin, mass=asset["mass"],
                            friction=obj["friction"], phys=self.phys)
        rb.collision_collections = OBJECT_GROUPS
        vertices = [mesh_vertices(proxy)]
        self.parts[obj["index"]] = []
        if shape == "COMPOUND":
            parts_dir = REPO_ROOT / Path(asset["proxy_mesh"]).parent / "coacd"
            for part_path in sorted(parts_dir.glob("part_*.obj")):
                part = import_obj(str(part_path))
                part.name = f"proxy_{obj['index']:03d}_part"
                part.hide_render = True
                part.parent = proxy
                part.matrix_parent_inverse = Matrix.Identity(4)
                # Parts are shrunk by part_margin and Bullet adds the margin back outside: the
                # shape is the true surface, and contacts are found before the cores overlap.
                part_rb = set_rigid_body(part, active=True, shape="CONVEX_HULL", margin=self.phys["part_margin"],
                                         mass=1e-3, friction=obj["friction"], phys=self.phys)
                part_rb.collision_collections = OBJECT_GROUPS
                vertices.append(mesh_vertices(part))
                self.parts[obj["index"]].append(part)
            bpy.context.view_layer.objects.active = proxy
        self.bodies[obj["index"]] = proxy
        self.local_vertices[obj["index"]] = np.concatenate(vertices)

    def _world_tree(self, index: int, pose: np.ndarray) -> BVHTree:
        proxy = self.bodies[index]
        world = geo.transform_points(pose, mesh_vertices(proxy))
        return BVHTree.FromPolygons(world.tolist(), mesh_polygons(proxy))

    def _overlaps(self, index: int, pose: np.ndarray) -> list:
        tree = self._world_tree(index, pose)
        hits = []
        for other, other_pose in self.poses.items():
            if other != index and tree.overlap(self._world_tree(other, other_pose)):
                hits.append(other)
        return hits

    def _min_z(self, index: int, pose: np.ndarray) -> float:
        return float(geo.transform_points(pose, self.local_vertices[index])[:, 2].min())

    def _current_poses(self) -> dict:
        return {i: from_matrix(b.matrix_world) for i, b in self.bodies.items()}

    def _restart(self):
        """Make the last settled state the initial state of the next bake (velocities reset)."""
        pc = self.rbw.point_cache
        with bpy.context.temp_override(point_cache=pc):
            bpy.ops.ptcache.free_bake()
        bpy.context.scene.frame_set(1)
        for index, body in self.bodies.items():
            body.matrix_world = to_matrix(self.poses[index])
        # Rigid bodies take their start transform from here; one assigned while the cache still held
        # the last baked frame is lost.
        for scan, T in self.frozen.values():
            scan.matrix_world = to_matrix(T)
        bpy.context.view_layer.update()

    # -- placement -------------------------------------------------------------------
    def base_pose(self, obj: dict) -> np.ndarray:
        R = np.asarray(obj["R"])
        asset = self.pool[obj["asset_id"]]
        proxy_local = geo.transform_points(geo.make_transform(R), np.asarray(self._asset_extremes(asset)))
        z = -proxy_local[:, 2].min() + obj["drop_height"]
        return geo.make_transform(R, [obj["xy"][0], obj["xy"][1], z])

    @staticmethod
    def _asset_extremes(asset: dict) -> list:
        # Bounding-box corners of the proxy in its COM frame (cheap pre-import estimate).
        com = np.asarray(asset["com"])
        lo = np.array([asset["com"][0] - asset["radius_xy"], asset["com"][1] - asset["radius_xy"], asset["z_min"]]) - com
        hi = np.array([asset["com"][0] + asset["radius_xy"], asset["com"][1] + asset["radius_xy"], asset["z_max"]]) - com
        return [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]

    def child_pose(self, obj: dict) -> np.ndarray:
        placement = self.cfg["placement"]
        parent_obj = self.plan["objects"][obj["parent"]]
        parent = self.pool[parent_obj["asset_id"]]
        child = self.pool[obj["asset_id"]]
        parent_pose = self.poses[parent_obj["index"]]
        if not checks.is_upright(parent_pose, placement["max_support_tilt_deg"]):
            raise SimFailure("support_tipped", f"object {parent_obj['index']} is not upright")
        R_rel = np.asarray(obj["R"])
        parent_com = np.asarray(parent["com"])
        if obj["relation"] == "in":
            cavity, fit = parent["cavity"], obj["fit"]
            tilt_lift = child["radius_xy"] * math.sin(geo.tilt_angle(R_rel))
            bottom = [cavity["axis_xy"][0], cavity["axis_xy"][1],
                      cavity["floor_z"] + fit["rest_offset"] + placement["nest_clearance_z"] + tilt_lift]
            com_local = np.asarray(bottom) + [0, 0, child["com"][2] - child["z_min"]] - parent_com
            position = geo.transform_points(parent_pose, com_local[None])[0]
            R = parent_pose[:3, :3] @ R_rel
        else:
            top = np.asarray(parent["top_center_xy"]) + np.asarray(obj["offset_xy"])
            above = geo.transform_points(parent_pose, (np.array([top[0], top[1], parent["z_max"]]) - parent_com)[None])[0]
            R = R_rel
            # Released just above the parent wherever their footprints overlap (heightmaps of the
            # collision vertices): a single ray down the middle would find a cavity floor or a plate well.
            body = geo.transform_points(geo.make_transform(R, [above[0], above[1], 0.0]), self.local_vertices_for(obj))
            support = geo.transform_points(parent_pose, self.local_vertices[parent_obj["index"]])
            lift = checks.clearance_lift(support, body)
            if lift is None:
                raise SimFailure("no_support_surface", f"object {obj['index']} is not above object {parent_obj['index']}")
            position = np.array([above[0], above[1], lift + placement["stack_clearance"]])
        return geo.make_transform(R, position)

    def local_vertices_for(self, obj: dict) -> np.ndarray:
        return self.local_vertices[obj["index"]]

    # -- simulation ---------------------------------------------------------------
    def _bake(self, frame_end: int) -> list:
        """Bake from the current initial state; return the COM poses of every frame."""
        pc = self.rbw.point_cache
        with bpy.context.temp_override(point_cache=pc):
            bpy.ops.ptcache.free_bake()
        pc.frame_end = frame_end
        with bpy.context.temp_override(point_cache=pc):
            bpy.ops.ptcache.bake(bake=True)
        trajectory = []
        for frame in range(1, frame_end + 1):
            bpy.context.scene.frame_set(frame)
            trajectory.append(self._current_poses())
        return trajectory

    def _bake_until_settled(self, stage: int) -> dict:
        """Simulate until nothing moved during the last check interval (doubling the horizon)."""
        phys = self.phys
        check = self.frames(phys["check_interval"])
        end = self.frames(phys["min_stage_time"])
        limit = self.frames(phys["max_stage_time"])
        started = time.perf_counter()
        while True:
            trajectory = self._bake(end + 1)
            state, info = checks.settle_state(trajectory, check, 1.0 / phys["fps"], phys["settle_translation"],
                                              phys["settle_rotation_deg"], phys["max_speed"])
            self.log["peak_speed"] = max(self.log["peak_speed"], info["peak_speed"])
            translation, rotation = info["translation"], info["rotation_deg"]
            if state == "glitch":
                raise SimFailure("glitch", f"stage {stage}: peak speed {info['peak_speed']:.2f} m/s")
            if state == "settled":
                break
            if end >= limit:
                raise SimFailure("not_settled", f"stage {stage}: still moving {translation*1e3:.2f} mm, {rotation:.2f} deg")
            end = min(2 * end, limit)
        self.log["sim_seconds"] += time.perf_counter() - started
        return {"frames": end, "final": trajectory[-1], "translation": translation, "rotation": rotation}

    def run(self) -> dict:
        phys, plan = self.phys, self.plan
        stages = plan["stages"]
        for stage in range(stages):
            members = [o for o in plan["objects"] if o["stage"] == stage]
            if stage > 0:
                # Settled objects become exact static meshes of their scans; convex parts would
                # bulge over cavities and plate wells and hold new objects above (or across) them.
                for obj in plan["objects"]:
                    if obj["stage"] < stage and obj["index"] not in self.frozen:
                        self._freeze(obj["index"])
                self._restart()
            for obj in members:
                if stage == 0:
                    pose = self.base_pose(obj)
                    self._add_body(obj, pose)
                    pose[2, 3] += -self._min_z(obj["index"], pose) + obj["drop_height"]
                    self.bodies[obj["index"]].matrix_world = to_matrix(pose)
                else:
                    self._add_body(obj, np.eye(4))
                    pose = self.child_pose(obj)
                    for _ in range(10):
                        if not self._overlaps(obj["index"], pose):
                            break
                        pose[2, 3] += 0.005
                    else:
                        raise SimFailure("no_free_drop_pose", f"object {obj['index']}")
                    self.bodies[obj["index"]].matrix_world = to_matrix(pose)
                self.poses[obj["index"]] = pose
            if stage == 0 and len(members) > 1:
                hits = [i for i in self.poses if self._overlaps(i, self.poses[i])]
                if hits:
                    raise SimFailure("initial_overlap", f"objects {hits}")
            bpy.context.view_layer.update()
            result = self._bake_until_settled(stage)
            self.poses = result["final"]
            self.log["stages"].append({"stage": stage, "frames": result["frames"],
                                       "residual_mm": result["translation"] * 1e3, "residual_deg": result["rotation"]})
            self._acceptance(upto_stage=stage)
        # Hold: restart the last configuration at rest; a stable scene must not move.
        self._restart()
        final = self._bake(self.frames(phys["hold_time"]) + 1)[-1]
        translation, rotation = checks.motion(self.poses, final)
        self.log["hold"] = {"translation_mm": translation * 1e3, "rotation_deg": rotation}
        if translation > 2 * phys["settle_translation"] or rotation > 2 * phys["settle_rotation_deg"]:
            raise SimFailure("unstable_hold", f"moved {translation*1e3:.2f} mm / {rotation:.2f} deg during hold")
        self.poses = final
        self._acceptance(upto_stage=stages - 1)
        self._check_contacts()
        self._check_tipping()
        return self.poses

    def _freeze(self, index: int):
        """Replace an object's convex compound by a static collider of its scanned surface.

        A static triangle mesh is exact and collides robustly with convex shapes. It only
        collides with active objects (SUPPORT_GROUP), never with the table or other frozen objects.
        """
        obj = self.plan["objects"][index]
        asset = self.pool[obj["asset_id"]]
        for body in [self.bodies[index]] + self.parts[index]:
            if body.name in self.rbw.collection.objects:
                self.rbw.collection.objects.unlink(body)
        scan = import_obj(asset.get("collider_mesh", asset["visual_mesh"]))  # decimated scan, same frame
        scan.name = f"phys_scan_{index:03d}"
        scan.hide_render = True
        rb = set_rigid_body(scan, active=False, shape="MESH", margin=self.phys["margin_object"], mass=1.0,
                            friction=obj["friction"], phys=self.phys)
        rb.collision_collections = [i == SUPPORT_GROUP for i in range(20)]
        # Placed by _restart, which always follows.
        self.frozen[index] = (scan, self.poses[index] @ geo.make_transform(t=-np.asarray(asset["com"])))

    def _proxy_distances(self, a: int, b: int, tree_b: BVHTree) -> tuple[float, float]:
        """(deepest vertex of proxy a inside proxy b, smallest gap from a to b), both in metres."""
        world = geo.transform_points(self.poses[a], mesh_vertices(self.bodies[a]))
        depth, gap = 0.0, math.inf
        for v in world:
            location, normal, _, distance = tree_b.find_nearest(v.tolist())
            if location is None:
                continue
            # The face normal is only a hint (wrong near edges); confirm with ray parity.
            if (v - np.asarray(location)) @ np.asarray(normal) < 0 and self._inside(tree_b, v):
                depth = max(depth, distance)
                gap = 0.0
            else:
                gap = min(gap, distance)
        return depth, gap

    @staticmethod
    def _inside(tree: BVHTree, point: np.ndarray) -> bool:
        """Majority vote of ray parity along three skewed directions (proxies are watertight)."""
        votes = 0
        for direction in ((0.13, 0.21, 0.97), (0.95, -0.17, 0.26), (-0.31, 0.93, -0.19)):
            origin, crossings = point.tolist(), 0
            for _ in range(64):
                hit, _, _, _ = tree.ray_cast(origin, direction)
                if hit is None:
                    break
                crossings += 1
                origin = [hit[i] + 1e-6 * direction[i] for i in range(3)]
            votes += crossings % 2
        return votes >= 2

    def _check_contacts(self):
        """Frozen supports are thin static shells, so a thin part (a handle) can be pushed through
        one. Reject interpenetration as validation measures it, and require every stacked or nested
        object to actually touch its planned parent (not just hover over it, resting on others)."""
        phys = self.phys
        objects = self.plan["objects"]
        trees = {i: self._world_tree(i, self.poses[i]) for i in self.poses}
        boxes = {i: (lambda w: (w.min(axis=0), w.max(axis=0)))(geo.transform_points(self.poses[i], mesh_vertices(self.bodies[i])))
                 for i in self.poses}
        reach = phys["relation_contact_distance"]
        deepest = 0.0
        for a in self.poses:
            for b in self.poses:
                if a == b or np.any(boxes[a][0] > boxes[b][1] + reach) or np.any(boxes[b][0] > boxes[a][1] + reach):
                    continue
                depth, _ = self._proxy_distances(a, b, trees[b])
                deepest = max(deepest, depth)
                if depth > phys["max_proxy_penetration"]:
                    raise SimFailure("interpenetration", f"object {a} is {depth*1e3:.1f} mm inside object {b}")
        self.log["max_proxy_penetration_mm"] = deepest * 1e3
        self.log["parent_gap_mm"] = {}
        for obj in objects:
            if obj["parent"] is None:
                continue
            _, gap = self._proxy_distances(obj["index"], obj["parent"], trees[obj["parent"]])
            self.log["parent_gap_mm"][str(obj["index"])] = gap * 1e3
            if gap > phys["max_parent_gap"]:
                raise SimFailure("relation_broken", f"{obj['index']}:{obj['relation']}:{obj['parent']} "
                                                    f"(no contact, gap {gap*1e3:.1f} mm)")

    def _check_tipping(self):
        """Frozen supports cannot react to their load, so check statically that each table-standing
        object with things on / in it keeps the combined centre of mass over its base."""
        objects = self.plan["objects"]
        children = {o["index"]: [c["index"] for c in objects if c["parent"] == o["index"]] for o in objects}
        tol = self.phys["tip_base_tolerance"]
        self.log["tipping_margin_mm"] = {}
        for root in (o for o in objects if o["stage"] == 0 and children[o["index"]]):
            group, todo = [], [root["index"]]
            while todo:
                index = todo.pop()
                group.append(index)
                todo += children[index]
            masses = np.array([self.pool[objects[i]["asset_id"]]["mass"] for i in group])
            com = (masses[:, None] * np.array([self.poses[i][:3, 3] for i in group])).sum(axis=0) / masses.sum()
            world = geo.transform_points(self.poses[root["index"]], self.local_vertices[root["index"]])
            base = world[world[:, 2] <= world[:, 2].min() + tol]
            margin = checks.support_margin(base[:, :2], com[:2])
            self.log["tipping_margin_mm"][str(root["index"])] = margin * 1e3
            if margin < self.phys["tip_margin"]:
                raise SimFailure("would_tip", f"object {root['index']} carrying {group[1:]}: margin {margin*1e3:.1f} mm")

    def _acceptance(self, upto_stage: int):
        phys = self.phys
        base = [o for o in self.plan["objects"] if o["role"] == "base"]
        reach = max(np.linalg.norm(o["xy"]) + self.pool[o["asset_id"]]["bounding_radius"] for o in base)
        radius = max(phys["fall_radius"], reach + 0.08)
        fallen = checks.fell_off(self.poses, [0.0, 0.0], radius)
        if fallen:
            raise SimFailure("fell_off", f"objects {fallen}")
        present = {o["index"] for o in self.plan["objects"] if o["stage"] <= upto_stage}
        sub_plan = {"objects": [o if o["index"] in present else {**o, "parent": None} for o in self.plan["objects"]]}
        min_z = {i: self._min_z(i, p) for i, p in self.poses.items()}
        broken = checks.check_relations(sub_plan, self.poses, min_z, self.pool, ground_z=0.0)
        if broken:
            raise SimFailure("relation_broken", ", ".join(self._describe(b, min_z) for b in broken))

    def _describe(self, broken: str, min_z: dict) -> str:
        child, _, parent = broken.split(":")
        c, p = self.poses[int(child)], self.poses[int(parent)]
        asset = self.pool[self.plan["objects"][int(parent)]["asset_id"]]
        local = checks.to_original_frame(p, asset, c[:3, 3])[:2] - np.asarray(asset["com"][:2])
        return (f"{broken} (child bottom {min_z[int(child)]*1e3:.1f} mm, COM {(c[2, 3]-p[2, 3])*1e3:+.1f} mm "
                f"vs parent, offset {np.abs(local).max()*1e3:.0f}/{max(asset['footprint_xy'])*500:.0f} mm)")

    def teardown(self):
        with bpy.context.temp_override(point_cache=self.rbw.point_cache):
            bpy.ops.ptcache.free_bake()
        self.rbw.enabled = False
        for obj in list(bpy.data.objects):
            if obj.name.startswith("phys_") or obj.name.startswith("proxy_"):
                bpy.data.objects.remove(obj, do_unlink=True)


def reset_blender():
    scene = bpy.context.scene
    if scene.rigidbody_world is not None:
        bpy.ops.rigidbody.world_remove()
    # Keeps the compositor and the registered render outputs; removes every object and data block.
    bproc.clean_up(clean_up_camera=True)


# --------------------------------------------------------------------------- rendering

def look_at_blender(position, target, azimuth) -> np.ndarray:
    """Camera-to-world (Blender convention: -Z forward, +Y up), continuous at top-down views."""
    forward = np.asarray(target) - np.asarray(position)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, up, -forward, position
    return T


BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
_OUTPUTS_ENABLED = False


def denoiser(render_cfg: dict) -> str:
    """auto: OptiX's denoiser when rendering with OptiX (GPU), else Intel OIDN."""
    if render_cfg["denoiser"] == "auto":
        return "OPTIX" if render_cfg.get("devices") == "OPTIX" else "INTEL"
    return render_cfg["denoiser"]


def enable_outputs():
    """Depth and segmentation outputs can only be registered once per process; later scenes
    only need fresh pass indices, which the segmentation loader maps back to `category_id`."""
    global _OUTPUTS_ENABLED
    if not _OUTPUTS_ENABLED:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})
        _OUTPUTS_ENABLED = True
    else:
        for index, obj in enumerate(bproc.object.get_all_mesh_objects()):
            obj.blender_obj.pass_index = index + 1


def make_table_material(env: dict, rng, render_cfg: dict):
    material = bpy.data.materials.new("table_material")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes["Principled BSDF"]
    if rng.uniform() < render_cfg["p_plain_table"] or not env["materials"]:
        hsv = (rng.uniform(), rng.uniform(0.0, 0.5), rng.uniform(0.2, 0.9))
        import colorsys
        bsdf.inputs["Base Color"].default_value = (*colorsys.hsv_to_rgb(*hsv), 1.0)
        bsdf.inputs["Roughness"].default_value = float(rng.uniform(0.2, 0.9))
        return material, {"type": "plain", "hsv": list(hsv)}
    choice = env["materials"][rng.integers(len(env["materials"]))]
    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    scale = float(rng.uniform(1.0, 3.0)) * render_cfg["table_size"] / 2.0
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, float(rng.uniform(0, 2 * math.pi)))
    links.new(coord.outputs["UV"], mapping.inputs["Vector"])

    def image_node(map_name, non_color):
        node = nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(repo_path(choice["maps"][map_name]["path"]), check_existing=True)
        if non_color:
            node.image.colorspace_settings.name = "Non-Color"
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    links.new(image_node("diffuse", False).outputs["Color"], bsdf.inputs["Base Color"])
    if "roughness" in choice["maps"]:
        links.new(image_node("roughness", True).outputs["Color"], bsdf.inputs["Roughness"])
    if "normal" in choice["maps"]:
        normal_map = nodes.new("ShaderNodeNormalMap")
        links.new(image_node("normal", True).outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    return material, {"type": "texture", "id": choice["id"], "uv_scale": scale}


def render_scene(job: dict, scene_spec: dict, plan: dict, poses: dict, pool: dict, raw_dir: Path) -> dict:
    cfg = job["config"]
    render_cfg = cfg["render"]
    env = json.loads(Path(repo_path(cfg["assets"]["environment"])).read_text())
    rng = layout.attempt_rng(scene_spec["seed"], 0, purpose=7)  # render randomness, independent of physics attempts
    started = time.perf_counter()

    objects_out = []
    world_points = []
    for obj in plan["objects"]:
        asset = pool[obj["asset_id"]]
        visual = import_visual(asset)
        visual.name = f"obj_{obj['index']:03d}"
        pose_com = poses[obj["index"]]
        pose_mesh = pose_com @ geo.make_transform(t=-np.asarray(asset["com"]))
        visual.matrix_world = to_matrix(pose_mesh)
        bproc.types.MeshObject(visual).set_cp("category_id", obj["index"] + 1)
        vertices = mesh_vertices(visual)
        world_points.append(geo.transform_points(pose_mesh, vertices[:: max(1, len(vertices) // 4000)]))
        objects_out.append({
            "index": obj["index"], "label": obj["index"] + 1, "asset_id": obj["asset_id"],
            "category": asset["category"], "role": obj["role"], "relation": obj["relation"], "parent": obj["parent"],
            "mesh_path": asset["visual_mesh"], "licence": asset["licence"],
            "T_world_mesh": pose_mesh.tolist(), "T_world_com": pose_com.tolist(), "com_mesh_frame": asset["com"],
            "scale": 1.0,
        })

    size = render_cfg["table_size"]
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    table = bpy.context.active_object
    table.name = "table"
    material, material_info = make_table_material(env, rng, render_cfg)
    table.data.materials.append(material)
    bproc.types.MeshObject(table).set_cp("category_id", 0)

    hdri = env["hdris"][rng.integers(len(env["hdris"]))]
    strength = float(rng.uniform(*render_cfg["hdri_strength"]))
    rotation = float(rng.uniform(0, 2 * math.pi))
    bpy.context.scene.world.use_nodes = True  # clean_up() leaves a world without a node tree
    bproc.world.set_world_background_hdr_img(repo_path(hdri["path"]), strength=strength, rotation_euler=[0.0, 0.0, rotation])
    lighting = {"hdri": hdri["id"], "strength": strength, "rotation_z": rotation, "extra_light": None}
    if rng.uniform() < render_cfg["p_extra_light"]:
        light = bproc.types.Light()
        light.set_type("AREA")
        azimuth, elevation = rng.uniform(0, 2 * math.pi), rng.uniform(math.radians(40), math.radians(85))
        distance = rng.uniform(0.8, 2.0)
        location = distance * np.array([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
        light.set_location(location.tolist())
        light.set_rotation_mat(look_at_blender(location, [0, 0, 0], azimuth)[:3, :3])
        energy = float(rng.uniform(20, 120))
        light.set_energy(energy)
        lighting["extra_light"] = {"type": "AREA", "location": location.tolist(), "energy": energy}

    res = render_cfg["resolution"]
    fov = math.radians(rng.uniform(*render_cfg["fov_deg"]))
    focal = (res / 2.0) / math.tan(fov / 2.0)
    K = np.array([[focal, 0.0, res / 2.0], [0.0, focal, res / 2.0], [0.0, 0.0, 1.0]])
    bproc.camera.set_intrinsics_from_K_matrix(K, res, res)
    points = np.concatenate(world_points)
    target = (points.min(axis=0) + points.max(axis=0)) / 2
    half = math.tan(fov / 2.0) * render_cfg["frame_fill"]
    views = []
    for view in range(render_cfg["n_views"]):
        azimuth = math.radians(rng.uniform(*render_cfg["azimuth_deg"]))
        elevation = math.radians(rng.uniform(*render_cfg["elevation_deg"]))
        direction = np.array([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
        axes = look_at_blender(target + direction, target, azimuth)
        q = points - target
        # Closest distance at which every object point projects inside `frame_fill` of the image.
        lateral = np.maximum(np.abs(q @ axes[:3, 0]), np.abs(q @ axes[:3, 1]))
        fit = float(np.max(q @ direction + lateral / half))
        distance = fit * rng.uniform(*render_cfg["distance_scale"])
        position = target + distance * direction
        cam2world = look_at_blender(position, target, azimuth)
        bproc.camera.add_camera_pose(cam2world)
        views.append({"view": view, "azimuth_deg": math.degrees(azimuth), "elevation_deg": math.degrees(elevation),
                      "distance": distance, "T_world_cam_blender": cam2world.tolist(),
                      "T_world_cam_opencv": (cam2world @ BLENDER_TO_OPENCV).tolist()})

    bproc.renderer.set_max_amount_of_samples(render_cfg["samples"])
    bproc.renderer.set_noise_threshold(render_cfg["noise_threshold"])
    bproc.renderer.set_denoiser(denoiser(render_cfg))
    if render_cfg.get("threads"):
        bpy.context.scene.render.threads_mode = "FIXED"
        bpy.context.scene.render.threads = render_cfg["threads"]
    enable_outputs()
    render_started = time.perf_counter()
    data = bproc.renderer.render()
    render_seconds = time.perf_counter() - render_started

    rgb = np.stack(data["colors"]).astype(np.uint8)
    black = [i for i, frame in enumerate(rgb) if frame.mean() < BLACK_FRAME_MEAN]
    if black:
        raise SimFailure("render_failed", f"black frames {black}")
    depth = np.stack(data["depth"]).astype(np.float32)
    seg = np.stack(data["category_id_segmaps"]).astype(np.uint8)
    np.savez_compressed(raw_dir / "views.npz", rgb=rgb, depth=depth, seg=seg)
    return {
        "objects": objects_out,
        "camera": {"K": K.tolist(), "width": res, "height": res, "fov_deg": math.degrees(fov),
                   "target": target.tolist(), "views": views},
        "lighting": lighting,
        "table": {"z": 0.0, "size": size, "material": material_info},
        "render_seconds": render_seconds,
        "scene_setup_seconds": render_started - started,
        "render_settings": {**{k: render_cfg[k] for k in ("samples", "noise_threshold", "resolution")},
                            "denoiser": denoiser(render_cfg), "devices": render_cfg.get("devices")},
    }


# --------------------------------------------------------------------------- driver

def simulate_scene(job: dict, scene_spec: dict, pool: dict, table: dict) -> tuple[dict, dict, list]:
    cfg = job["config"]
    level_cfg = cfg["levels"][scene_spec["level"]]
    compat_cfg = CompatConfig.from_dict(cfg["compat"])
    collision = cfg["physics"]["collision"]
    modes = ["mesh", "coacd"] if collision == "mesh_then_coacd" else [collision]
    attempts = []
    first = scene_spec.get("attempt_offset", 0)  # retries continue with fresh layouts
    for attempt in range(first, first + cfg["physics"]["max_attempts"]):
        rng = layout.attempt_rng(scene_spec["seed"], attempt)
        try:
            plan = layout.plan_scene(level_cfg, cfg["placement"], pool, rng, compat_cfg, table, cfg["physics"])
        except layout.PlanError as error:
            attempts.append({"attempt": attempt, "status": "plan_failed", "reason": "plan", "detail": str(error)})
            continue
        for mode in modes:
            reset_blender()
            started = time.perf_counter()
            sim = Simulation(cfg, pool, plan, mode)
            try:
                poses = sim.run()
            except SimFailure as failure:
                attempts.append({"attempt": attempt, "collision": mode, "status": "rejected", "reason": failure.reason,
                                 "detail": failure.detail, "seconds": time.perf_counter() - started, **sim.log})
                if failure.reason == "glitch":
                    continue  # try the next collision mode on the same plan
                break
            attempts.append({"attempt": attempt, "collision": mode, "status": "accepted",
                             "seconds": time.perf_counter() - started, **sim.log})
            sim.teardown()
            plan["attempt"], plan["collision"] = attempt, mode
            return plan, poses, attempts
    raise SimFailure("max_attempts", json.dumps(attempts))


def run_scene(job: dict, scene_spec: dict, asset_table: dict, tables: dict):
    out_dir = Path(scene_spec["out_dir"])
    raw_dir = out_dir / "raw"
    if (raw_dir / "DONE").exists():
        return "skipped"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "FAILED.json").unlink(missing_ok=True)
    started = time.perf_counter()
    pool = layout.split_pool(asset_table, scene_spec["split"])
    if scene_spec["split"] not in tables:
        tables[scene_spec["split"]] = build_table(pool, CompatConfig.from_dict(job["config"]["compat"]))

    def failed(failure: SimFailure) -> str:
        (raw_dir / "FAILED.json").write_text(json.dumps({**scene_spec, "reason": failure.reason, "detail": failure.detail,
                                                         "seconds": time.perf_counter() - started}, indent=2))
        return "failed"

    try:
        plan, poses, attempts = simulate_scene(job, scene_spec, pool, tables[scene_spec["split"]])
    except SimFailure as failure:
        return failed(failure)
    physics_seconds = time.perf_counter() - started
    if job.get("physics_only"):  # tests and physics timing: no rendering
        (raw_dir / "physics.json").write_text(json.dumps({**scene_spec, "plan": plan, "attempts": attempts,
                                                          "poses": {str(k): v.tolist() for k, v in poses.items()},
                                                          "physics_seconds": physics_seconds}, indent=1))
        (raw_dir / "DONE").write_text("physics only\n")
        return "done"
    try:
        render_info = render_scene(job, scene_spec, plan, poses, pool, raw_dir)
    except SimFailure as failure:
        return failed(failure)
    record = {
        **scene_spec,
        "plan": plan,
        "attempts": attempts,
        "physics_seconds": physics_seconds,
        "total_seconds": time.perf_counter() - started,
        "versions": {"blender": bpy.app.version_string, "blenderproc": bproc.__version__},
        **render_info,
    }
    (raw_dir / "scene_raw.json").write_text(json.dumps(record, indent=2))
    (raw_dir / "DONE").write_text("ok\n")
    return "done"


def main():
    job_path = Path(sys.argv[-1])
    job = json.loads(job_path.read_text())
    bproc.init()
    devices = job["config"]["render"].get("devices")
    if devices:
        bproc.renderer.set_render_devices(desired_gpu_device_type=devices)
        # GPU only: BlenderProc also enables the CPU, whose tiles compete with the physics and
        # with the other workers sharing the machine.
        for device in bpy.context.preferences.addons["cycles"].preferences.devices:
            device.use = device.type == devices
    asset_table = json.loads(Path(repo_path(job["config"]["assets"]["table"])).read_text())
    tables = {}
    for scene_spec in job["scenes"]:
        try:
            status = run_scene(job, scene_spec, asset_table, tables)
        except Exception as error:  # noqa: BLE001 - record and continue with the batch
            raw_dir = Path(scene_spec["out_dir"]) / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "FAILED.json").write_text(json.dumps({**scene_spec, "reason": "exception",
                                                             "detail": traceback.format_exc()}, indent=2))
            status = "error"
        print(f"[worker] {scene_spec['scene_id']}: {status}", flush=True)
        log_path = Path(scene_spec["out_dir"]) / "raw" / ("physics.json" if job.get("physics_only") else "scene_raw.json")
        if status == "done" and log_path.is_file():
            record = json.loads(log_path.read_text())
            for attempt in record["attempts"]:
                print(f"[worker]   attempt {attempt['attempt']} {attempt.get('collision', '')} {attempt['status']} "
                      f"{attempt.get('reason', '')} {attempt.get('detail', '')} sim={attempt.get('seconds', 0):.1f}s", flush=True)
            print(f"[worker]   physics {record['physics_seconds']:.1f}s render {record.get('render_seconds', 0):.1f}s "
                  f"total {record.get('total_seconds', record['physics_seconds']):.1f}s", flush=True)
        reset_blender()


main()

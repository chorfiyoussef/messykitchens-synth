import numpy as np
from conftest import make_asset

from mksynth.geometry import axis_angle_matrix, make_transform
from mksynth.physics_checks import (check_relations, fell_off, is_upright, motion, nested_inside, resting_on,
                                  settle_state)

DT = 1 / 24


def pose(x=0.0, y=0.0, z=0.0, R=None):
    return make_transform(R, [x, y, z])


def test_motion_measures_translation_and_rotation():
    before = {0: pose(), 1: pose(0.1)}
    after = {0: pose(0.003), 1: pose(0.1, R=axis_angle_matrix([0, 0, 1], np.radians(5)))}
    translation, rotation = motion(before, after)
    assert abs(translation - 0.003) < 1e-12
    assert abs(rotation - 5.0) < 1e-9


def trajectory(positions):
    return [{0: pose(z=z), 1: pose(x=0.2)} for z in positions]


def test_settle_state_resting():
    # Falls 2 cm, then stays put for half a second.
    traj = trajectory(list(np.linspace(0.02, 0.0, 6)) + [0.0] * 12)
    state, info = settle_state(traj, check_frames=12, dt=DT, max_translation=0.001, max_rotation_deg=1, max_speed=3)
    assert state == "settled"
    assert info["translation"] == 0.0


def test_settle_state_slow_slide_is_moving():
    traj = trajectory(np.linspace(0.0, -0.01, 24))  # 1 cm in one second: too much
    state, _ = settle_state(traj, check_frames=12, dt=DT, max_translation=0.001, max_rotation_deg=1, max_speed=3)
    assert state == "moving"


def test_settle_state_creep_below_threshold_counts_as_settled():
    traj = trajectory(np.linspace(0.0, 0.0005, 13))
    state, _ = settle_state(traj, check_frames=12, dt=DT, max_translation=0.001, max_rotation_deg=1, max_speed=3)
    assert state == "settled"


def test_settle_state_glitch():
    traj = trajectory([0.0, 0.0, 0.5, 0.5])  # 0.5 m in one frame = 12 m/s
    state, info = settle_state(traj, check_frames=2, dt=DT, max_translation=0.001, max_rotation_deg=1, max_speed=3)
    assert state == "glitch"
    assert info["peak_speed"] > 10


def test_settle_state_rotation_counts():
    R = [axis_angle_matrix([1, 0, 0], np.radians(a)) for a in np.linspace(0, 10, 13)]
    traj = [{0: pose(R=r)} for r in R]
    state, _ = settle_state(traj, check_frames=12, dt=DT, max_translation=0.001, max_rotation_deg=1, max_speed=3)
    assert state == "moving"


def test_fell_off():
    poses = {0: pose(0.1), 1: pose(0.6), 2: pose(0.0, z=-0.05)}
    assert fell_off(poses, [0, 0], 0.45) == [1, 2]


def test_upright():
    assert is_upright(pose(), 5)
    assert not is_upright(pose(R=axis_angle_matrix([1, 0, 0], np.radians(30))), 15)


BOWL = make_asset("bowl", height=0.08, radius=0.11, cavity_depth=0.07, cavity_radius=0.10)
CUP = make_asset("cup", height=0.06, radius=0.035)
BOX = make_asset("box", height=0.06, radius=0.08)


def com_pose(asset, x=0.0, y=0.0, bottom=0.0, R=None):
    """Pose of the COM frame for an object whose bottom is at height `bottom`."""
    return pose(x, y, bottom + asset["com"][2] - asset["z_min"], R)


def test_nested_inside():
    bowl = com_pose(BOWL)
    floor = BOWL["cavity"]["floor_z"]
    assert nested_inside(com_pose(CUP, bottom=floor + 0.001), bowl, BOWL)
    assert not nested_inside(com_pose(CUP, bottom=0.09), bowl, BOWL)            # above the rim
    assert not nested_inside(com_pose(CUP, x=0.2, bottom=floor), bowl, BOWL)    # beside the bowl
    assert not nested_inside(com_pose(CUP), com_pose(BOX), BOX)                 # no cavity


def test_nested_inside_follows_container_pose():
    R = axis_angle_matrix([0, 0, 1], np.radians(90))
    bowl = make_transform(R, [0.3, -0.1, BOWL["com"][2]])
    child_world = bowl @ make_transform(t=[0, 0, BOWL["cavity"]["floor_z"] - BOWL["com"][2] + 0.03])
    assert nested_inside(child_world, bowl, BOWL)


def test_resting_on():
    box = com_pose(BOX)
    on_top = com_pose(CUP, bottom=0.0605)
    assert resting_on(on_top, 0.0605, box, BOX, ground_z=0.0)
    assert not resting_on(com_pose(CUP, x=0.3), 0.0, box, BOX, ground_z=0.0)             # on the table
    assert not resting_on(com_pose(CUP, x=0.3, bottom=0.06), 0.06, box, BOX, ground_z=0.0)  # beside, mid-air


def test_check_relations():
    plan = {"objects": [
        {"index": 0, "asset_id": "bowl", "parent": None, "relation": None},
        {"index": 1, "asset_id": "cup", "parent": 0, "relation": "in"},
        {"index": 2, "asset_id": "box", "parent": None, "relation": None},
    ]}
    pool = {"bowl": BOWL, "cup": CUP, "box": BOX}
    floor = BOWL["cavity"]["floor_z"]
    good = {0: com_pose(BOWL), 1: com_pose(CUP, bottom=floor), 2: com_pose(BOX, x=0.3)}
    min_z = {0: 0.0, 1: floor, 2: 0.0}
    assert check_relations(plan, good, min_z, pool, ground_z=0.0) == []
    bad = {**good, 1: com_pose(CUP, x=-0.25)}
    assert check_relations(plan, bad, {**min_z, 1: 0.0}, pool, ground_z=0.0) == ["1:in:0"]


def test_support_margin():
    from mksynth.physics_checks import convex_hull_2d, support_margin

    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.2, 0.9]])
    assert len(convex_hull_2d(square)) == 4
    assert abs(support_margin(square, [0.5, 0.5]) - 0.5) < 1e-12
    assert abs(support_margin(square, [0.9, 0.5]) - 0.1) < 1e-12
    assert support_margin(square, [1.2, 0.5]) < 0          # outside: tips
    assert support_margin(square[:2], [0.5, 0.0]) == -np.inf  # degenerate base


def test_clearance_lift():
    from mksynth.physics_checks import clearance_lift

    g = np.linspace(-0.05, 0.05, 41)
    xx, yy = np.meshgrid(g, g)
    # A plate: 2 mm well floor, 15 mm rim beyond r = 40 mm.
    plate = np.c_[xx.ravel(), yy.ravel(), np.where(np.hypot(xx, yy).ravel() > 0.04, 0.015, 0.002)]
    small = np.c_[xx.ravel() * 0.3, yy.ravel() * 0.3, np.zeros(xx.size)]  # 30 mm wide, flat bottom at z = 0
    assert abs(clearance_lift(plate, small) - 0.002) < 1e-9               # sits in the well
    wide = np.c_[xx.ravel() * 1.2, yy.ravel() * 1.2, np.zeros(xx.size)]
    assert abs(clearance_lift(plate, wide) - 0.015) < 1e-9                # spans the rim
    assert clearance_lift(plate, small + [1.0, 0, 0]) is None             # not above it

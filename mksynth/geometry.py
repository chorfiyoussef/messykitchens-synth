"""Small rigid-transform helpers (numpy only, importable inside Blender).

Quaternions are [w, x, y, z]. Matrices act on column vectors: p_world = T @ p_local.
"""

from __future__ import annotations

import numpy as np


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q) / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def make_transform(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def transform_points(T, points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def yaw_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def axis_angle_matrix(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return quat_to_matrix(q / np.linalg.norm(q))


def tilted_upright(rng: np.random.Generator, max_tilt_rad: float) -> np.ndarray:
    """Random yaw, then a tilt of up to ``max_tilt_rad`` about a random horizontal axis."""
    yaw = yaw_matrix(rng.uniform(0, 2 * np.pi))
    if max_tilt_rad <= 0:
        return yaw
    phi = rng.uniform(0, 2 * np.pi)
    tilt = axis_angle_matrix([np.cos(phi), np.sin(phi), 0.0], rng.uniform(0, max_tilt_rad))
    return tilt @ yaw


def rotation_angle(R_a, R_b) -> float:
    """Geodesic angle (rad) between two rotations."""
    cos = (np.trace(np.asarray(R_a).T @ np.asarray(R_b)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def tilt_angle(R) -> float:
    """Angle (rad) between the object's local +z and world +z."""
    return float(np.arccos(np.clip(np.asarray(R)[2, 2], -1.0, 1.0)))

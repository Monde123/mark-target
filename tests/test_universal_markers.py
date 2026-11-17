"""Tests unitaires synthétiques pour le module `isolate`.
Vérifie la robustesse mathématique, l'algorithme Kabsch et le solveur hiérarchique
sans AUCUNE dépendance externe lourde (ni PyTorch, ni SMPL-X, ni Mixamo).
"""

import unittest
import numpy as np

from core.geometry import (
    quat_normalize,
    quat_identity,
    quat_inv,
    quat_mul,
    rotate_vector,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quat_from_two_vectors,
)
from core.kabsch import kabsch_rotation, weighted_kabsch_rotation
from retargeting.solver import MarkerRetargetSolver


def generate_tetrahedral_markers(center: np.ndarray, radius: float = 10.0) -> np.ndarray:
    return np.array([
        center + [radius, 0.0, 0.0],
        center + [-radius / 2.0, radius * (3.0**0.5) / 2.0, 0.0],
        center + [-radius / 2.0, -radius * (3.0**0.5) / 2.0, 0.0],
        center + [0.0, 0.0, radius],
    ], dtype=np.float64)


def generate_cuboid_markers(center: np.ndarray, radius: float = 5.0, length: float = 15.0) -> np.ndarray:
    return np.array([
        center + [radius, radius, 0.0],
        center + [-radius, radius, 0.0],
        center + [-radius, -radius, 0.0],
        center + [radius, -radius, 0.0],
        center + [0.0, 0.0, length],
    ], dtype=np.float64)


class TestGeometryPrimitives(unittest.TestCase):
    def test_quaternion_identity_and_norm(self):
        q = quat_identity()
        self.assertEqual(len(q), 4)
        self.assertTrue(np.allclose(q, [1, 0, 0, 0]))
        q_norm = quat_normalize(np.array([2.0, 0, 0, 0]))
        self.assertAlmostEqual(np.linalg.norm(q_norm), 1.0)

    def test_rotation_matrix_roundtrip(self):
        angle = np.pi / 3.0
        q = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0])
        q = quat_normalize(q)
        R = quaternion_to_matrix(q)
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-6))
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=5)
        q_back = matrix_to_quaternion(R)
        dot = abs(np.dot(q, q_back))
        self.assertAlmostEqual(dot, 1.0, places=5)

    def test_rotate_vector(self):
        v = np.array([1.0, 0.0, 0.0])
        angle = np.pi / 2.0
        q_z90 = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
        v_rot = rotate_vector(q_z90, v)
        self.assertTrue(np.allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-6))


class TestKabschAlgorithm(unittest.TestCase):
    def test_pure_rigid_rotation(self):
        np.random.seed(42)
        P_rest = np.random.uniform(-10.0, 10.0, size=(10, 3))
        angle = np.pi / 4.0
        q_true = quat_normalize(np.array([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0]))
        R_true = quaternion_to_matrix(q_true)
        t_true = np.array([5.0, -3.0, 2.0])
        P_curr = (P_rest @ R_true.T) + t_true
        R_calc, t_calc, P_aligned, rms, _ = kabsch_rotation(P_rest, P_curr)
        self.assertTrue(np.allclose(R_calc, R_true, atol=1e-5))
        self.assertAlmostEqual(rms, 0.0, places=5)

    def test_weighted_kabsch(self):
        P_rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        weights = np.array([10.0, 1.0, 1.0, 1.0])
        R_calc, t_calc, P_aligned, rms, _ = weighted_kabsch_rotation(P_rest, P_rest, weights=weights)
        self.assertTrue(np.allclose(R_calc, np.eye(3), atol=1e-5))


class TestMarkerRetargetSolver(unittest.TestCase):
    def setUp(self):
        self.rest_rotations = {
            "Hips": quat_identity(),
            "Spine": quat_identity(),
            "LeftArm": quat_identity(),
        }
        self.parents = {
            "Hips": None,
            "Spine": "Hips",
            "LeftArm": "Spine",
        }
        self.solver = MarkerRetargetSolver(
            rest_rotations=self.rest_rotations,
            parents=self.parents,
        )

    def test_multi_kabsch_identity(self):
        markers_rest = {
            "Hips": generate_tetrahedral_markers(np.array([0, 0, 0])),
            "Spine": generate_tetrahedral_markers(np.array([0, 10, 0])),
            "LeftArm": generate_cuboid_markers(np.array([5, 10, 0])),
        }
        Q_glob, diag = self.solver.solve_frame_multi_kabsch(markers_rest, markers_rest)
        for bone in self.rest_rotations:
            dot = abs(np.dot(Q_glob[bone], quat_identity()))
            self.assertAlmostEqual(dot, 1.0, places=5)
            self.assertEqual(diag[bone]["status"], "kabsch_success")


if __name__ == "__main__":
    unittest.main()

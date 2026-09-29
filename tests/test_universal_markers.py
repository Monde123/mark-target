"""
Tests unitaires synthétiques pour le module `isolate`.
Vérifie la robustesse mathématique, l'algorithme Kabsch et le solveur hiérarchique
sans AUCUNE dépendance externe lourde (ni PyTorch, ni SMPL-X, ni Mixamo).
"""
import unittest
import numpy as np

from isolate.core.geometry import (
    quat_normalize,
    quat_identity,
    quat_inv,
    quat_mul,
    rotate_vector,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quat_from_two_vectors,
)
from isolate.core.kabsch import kabsch_rotation, weighted_kabsch_rotation
from isolate.retargeting.solver import MarkerRetargetSolver
from isolate.retargeting.kinematics import global_to_local_hierarchy, local_to_global_hierarchy
from isolate.mappings.registry import MappingRegistry


class TestIsolateModule(unittest.TestCase):

    def test_geometry_quaternions(self):
        # 1. Normalisation & identité
        q_id = quat_identity()
        self.assertTrue(np.allclose(q_id, [1.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(quat_mul(q_id, q_id), q_id))

        # 2. Rotation 90° autour de Z
        theta = np.pi / 2.0
        q_z90 = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])
        v = np.array([1.0, 0.0, 0.0])
        v_rot = rotate_vector(q_z90, v)
        self.assertTrue(np.allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-6))

        # 3. Conversion Matrice <-> Quaternion
        R = quaternion_to_matrix(q_z90)
        q_rebuilt = matrix_to_quaternion(R)
        self.assertTrue(np.allclose(q_z90, q_rebuilt, atol=1e-6))

    def test_kabsch_reconstruction(self):
        # Création de 4 points non colinéaires (tétraèdre)
        P_rest = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Applique une rotation connue (45° autour de l'axe Y)
        theta = np.radians(45.0)
        c, s = np.cos(theta), np.sin(theta)
        R_true = np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ])
        translation = np.array([2.5, -1.0, 3.0])
        P_curr = (R_true @ P_rest.T).T + translation

        # Résolution Kabsch
        R_est, cent_r, cent_c, rms, _ = kabsch_rotation(P_rest, P_curr)

        self.assertTrue(np.allclose(R_est, R_true, atol=1e-6))
        self.assertLess(rms, 1e-6)

    def test_solver_and_kinematics(self):
        # Hiérarchie simple : Root -> Arm -> ForeArm
        parents = {
            "Root": None,
            "Arm": "Root",
            "ForeArm": "Arm",
        }
        rest_rotations = {
            "Root": quat_identity(),
            "Arm": quat_identity(),
            "ForeArm": quat_identity(),
        }

        solver = MarkerRetargetSolver(rest_rotations, parents)

        # Simulation de nuages de points de repos
        P_rest = {
            "Arm": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "ForeArm": np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0, 1]]),
        }

        # Simulation d'un déplacement à l'instant t (rotation de 90° sur Arm)
        R_arm = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        P_t = {
            "Arm": (R_arm @ P_rest["Arm"].T).T,
            "ForeArm": (R_arm @ P_rest["ForeArm"].T).T,
        }

        Q_global, diagnostics = solver.solve_frame_multi_kabsch(P_t, P_rest)
        self.assertEqual(diagnostics["Arm"]["status"], "kabsch_success")

        # Conversion globale -> locale
        Q_local = global_to_local_hierarchy(Q_global, parents)
        self.assertIn("Arm", Q_local)
        self.assertIn("ForeArm", Q_local)

        # Reconversion locale -> globale
        Q_rebuilt_global = local_to_global_hierarchy(Q_local, parents)
        self.assertTrue(np.allclose(Q_global["Arm"], Q_rebuilt_global["Arm"], atol=1e-6))

    def test_mapping_registry(self):
        # Test du pont automatique Mixamo -> VRM
        mixamo_to_vrm = MappingRegistry.get_mapping("mixamo", "vrm")
        self.assertIn("mixamorig:Hips", mixamo_to_vrm)
        self.assertEqual(mixamo_to_vrm["mixamorig:Hips"], "hips")
        self.assertIn("mixamorig:LeftArm", mixamo_to_vrm)
        self.assertEqual(mixamo_to_vrm["mixamorig:LeftArm"], "leftUpperArm")


if __name__ == "__main__":
    unittest.main()

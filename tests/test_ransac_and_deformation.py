"""Tests unitaires pour le filtre RANSAC et le tenseur de déformation Green-Lagrange (Phase 3)."""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from markers.ransac_filter import ransac_kabsch_alignment, compute_green_lagrange_strain
from core.kabsch import kabsch_rotation
from retargeting.solver import MarkerRetargetSolver


class TestRansacAndDeformation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Nuage de 5 marqueurs rigides sur un os
        self.P_rest = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ], dtype=np.float64)

    def test_green_lagrange_rigid_invariance(self):
        """Vérifie que pour une rotation et translation pures, le tenseur E est strictement nul."""
        # Rotation 45° autour de l'axe Y
        theta = np.pi / 4.0
        R_true = np.array([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ])
        t_true = np.array([5.0, -3.0, 2.0])

        P_current = (self.P_rest @ R_true.T) + t_true

        E, strain_energy = compute_green_lagrange_strain(self.P_rest, P_current)

        # Pour une transformation rigide, E = 0 et ||E||_F = 0
        self.assertLess(strain_energy, 1e-6)
        np.testing.assert_allclose(E, np.zeros((3, 3)), atol=1e-6)

    def test_green_lagrange_non_rigid_strain(self):
        """Vérifie que l'étirement non rigide (déformation de chair) est détecté."""
        # Élongation de 30% sur l'axe X (squash & stretch)
        stretch_matrix = np.diag([1.3, 1.0, 1.0])
        P_stretched = self.P_rest @ stretch_matrix.T

        E, strain_energy = compute_green_lagrange_strain(self.P_rest, P_stretched)

        # E_xx théorique = 0.5 * (1.3^2 - 1) = 0.5 * (1.69 - 1) = 0.345
        self.assertAlmostEqual(E[0, 0], 0.345, places=2)
        self.assertGreater(strain_energy, 0.3)

    def test_ransac_outlier_rejection(self):
        """Vérifie que RANSAC élimine un marqueur aberrant corrompu (ex: occlusion ou artefact)."""
        # Rotation pure de 90° autour de X
        R_true = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        P_current = self.P_rest @ R_true.T

        # Injection d'un outlier violent sur le marqueur #4 (+50.0 unités)
        P_corrupted = P_current.copy()
        P_corrupted[4] += np.array([50.0, -30.0, 80.0])

        # 1. Kabsch naïf sans RANSAC subit l'artefact
        _, _, _, rms_naive, _ = kabsch_rotation(self.P_rest, P_corrupted)
        self.assertGreater(rms_naive, 5.0)

        # 2. RANSAC isole les 4 inliers et rejette l'outlier #4
        R_ransac, inlier_mask, rms_ransac, n_inliers = ransac_kabsch_alignment(
            self.P_rest, P_corrupted, inlier_threshold=0.1
        )

        self.assertEqual(n_inliers, 4)
        self.assertFalse(inlier_mask[4])  # Le point corrompu a été rejeté
        self.assertLess(rms_ransac, 1e-4)
        np.testing.assert_allclose(R_ransac, R_true, atol=1e-4)

    def test_solver_integration_strain_and_ransac(self):
        """Vérifie l'intégration du seuil de déformation et de RANSAC dans le solveur."""
        rest_rots = {"LeftForeArm": np.array([1.0, 0.0, 0.0, 0.0])}
        parents = {"LeftForeArm": None}

        solver = MarkerRetargetSolver(rest_rotations=rest_rots, parents=parents)

        # Cas 1 : Déformation excessive au-delà du seuil max
        stretch_matrix = np.diag([2.5, 1.0, 1.0])
        pos_t_strained = {"LeftForeArm": self.P_rest @ stretch_matrix.T}
        pos_rest = {"LeftForeArm": self.P_rest}

        _, diag = solver.solve_frame_multi_kabsch(
            positions_t=pos_t_strained,
            positions_rest=pos_rest,
            max_strain_threshold=0.2,
        )
        self.assertEqual(diag["LeftForeArm"]["status"], "fallback_strain_exceeded")


if __name__ == "__main__":
    unittest.main()

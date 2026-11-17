"""Tests unitaires pour l'algorithme de Kabsch-Umeyama et le lissage Bézier/SQUAD."""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.kabsch import kabsch_umeyama_rotation, weighted_kabsch_umeyama_rotation
from retargeting.smoothing import (
    quat_slerp,
    quat_squad,
    compute_squad_control_point,
    de_casteljau_bezier_3d,
    smooth_quaternion_trajectory,
)


class TestUmeyamaAndSmoothing(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Nuage de test tétraédrique (non colinéaire)
        self.A = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def test_umeyama_pure_scale(self):
        """Vérifie qu'Umeyama retrouve le facteur d'échelle exact sans rotation parasite."""
        expected_scale = 2.5
        B = self.A * expected_scale

        R, c, t, cent_A, cent_B, rms, S = kabsch_umeyama_rotation(self.A, B)

        self.assertAlmostEqual(c, expected_scale, places=5)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-5)
        self.assertLess(rms, 1e-6)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=5)

    def test_umeyama_rotation_scale_and_translation(self):
        """Vérifie la résolution conjointe Rotation + Échelle (0.5) + Translation."""
        expected_scale = 0.5
        # Rotation 90° autour de Z
        R_true = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        t_true = np.array([10.0, -5.0, 2.0])

        B = expected_scale * (self.A @ R_true.T) + t_true

        R, c, t, _, _, rms, _ = kabsch_umeyama_rotation(self.A, B)

        self.assertAlmostEqual(c, expected_scale, places=5)
        np.testing.assert_allclose(R, R_true, atol=1e-5)
        np.testing.assert_allclose(t, t_true, atol=1e-5)
        self.assertLess(rms, 1e-6)

    def test_weighted_umeyama(self):
        """Vérifie la version pondérée d'Umeyama."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        expected_scale = 1.3
        B = self.A * expected_scale

        R, c, t, _, _, rms, _ = weighted_kabsch_umeyama_rotation(self.A, B, weights)

        self.assertAlmostEqual(c, expected_scale, places=5)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-5)
        self.assertLess(rms, 1e-6)

    def test_de_casteljau_bezier(self):
        """Vérifie que De Casteljau interpole les extrémités P0 (t=0) et Pn (t=1)."""
        pts = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 0.0]),
            np.array([2.0, 2.0, 0.0]),
            np.array([3.0, 0.0, 0.0]),
        ]
        start = de_casteljau_bezier_3d(pts, 0.0)
        end = de_casteljau_bezier_3d(pts, 1.0)
        mid = de_casteljau_bezier_3d(pts, 0.5)

        np.testing.assert_allclose(start, pts[0], atol=1e-6)
        np.testing.assert_allclose(end, pts[-1], atol=1e-6)
        self.assertTrue(mid[1] > 0.0)

    def test_quat_slerp_and_squad(self):
        """Vérifie les propriétés géométriques de SLERP et SQUAD."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])  # Identité
        q1 = np.array([0.70710678, 0.70710678, 0.0, 0.0])  # 90° autour de X

        slerp_mid = quat_slerp(q0, q1, 0.5)
        self.assertAlmostEqual(np.linalg.norm(slerp_mid), 1.0, places=6)

        # SQUAD
        s0 = compute_squad_control_point(q0, q0, q1)
        s1 = compute_squad_control_point(q0, q1, q1)
        squad_mid = quat_squad(q0, q1, s0, s1, 0.5)
        self.assertAlmostEqual(np.linalg.norm(squad_mid), 1.0, places=6)

    def test_jitter_reduction(self):
        """Vérifie que le lissage temporel réduit la variance d'accélération angulaire (jitter)."""
        # Génération d'une séquence de 30 quaternions avec bruit haute fréquence
        base_quats = []
        for i in range(30):
            angle = i * 0.05
            base_q = np.array([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0])
            noise = np.random.normal(0.0, 0.05, 4) if i % 2 == 1 else np.zeros(4)
            noisy_q = base_q + noise
            base_quats.append(noisy_q / np.linalg.norm(noisy_q))

        smoothed = smooth_quaternion_trajectory(base_quats, smoothing_factor=0.4)

        # Calcul des différences d'angles (vitesses)
        raw_diffs = [np.linalg.norm(base_quats[i] - base_quats[i - 1]) for i in range(1, len(base_quats))]
        smooth_diffs = [np.linalg.norm(smoothed[i] - smoothed[i - 1]) for i in range(1, len(smoothed))]

        self.assertLess(np.std(smooth_diffs), np.std(raw_diffs))


if __name__ == "__main__":
    unittest.main()

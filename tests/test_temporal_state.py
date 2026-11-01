"""Tests unitaires pour le moteur d'état temporel récursif (t-1 -> t).
"""

import unittest
import numpy as np

from core.geometry import quat_identity, quat_normalize, quat_mul
from retargeting.temporal_state import (
    TemporalRetargetState,
    align_quaternion_hemisphere,
    quaternion_log_map,
    quaternion_exp_map,
)
from retargeting.solver import MarkerRetargetSolver


class TestTemporalRetargetState(unittest.TestCase):

    def test_quaternion_hemisphere_alignment(self):
        """Vérifie que les quaternions opposés (-q) sont alignés sur le même hémisphère."""
        q1 = quat_normalize(np.array([1.0, 0.2, -0.3, 0.4]))
        q2_neg = -q1.copy()

        # Produit scalaire négatif initialement
        self.assertLess(float(np.dot(q1, q2_neg)), 0.0)

        # Après redressement, produit scalaire positif égal à 1.0
        q2_aligned = align_quaternion_hemisphere(q2_neg, q1)
        self.assertGreaterEqual(float(np.dot(q1, q2_aligned)), 0.9999)
        np.testing.assert_allclose(q2_aligned, q1, atol=1e-6)

    def test_quaternion_lie_algebra_exp_log(self):
        """Vérifie l'aller-retour entre S³ et l'algèbre de Lie so(3)."""
        omega = np.array([0.5, -0.2, 0.8])
        q = quaternion_exp_map(omega)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=6)

        omega_rec = quaternion_log_map(q)
        np.testing.assert_allclose(omega_rec, omega, atol=1e-6)

    def test_inertial_prediction(self):
        """Vérifie que la prédiction extrapole la rotation selon la vitesse angulaire."""
        state = TemporalRetargetState(fps=30.0)
        q_rest = quat_identity()

        # Frame 0 : pose neutre
        q0 = quat_identity()
        state.update_frame({"LeftArm": q0})

        # Frame 1 : rotation de 0.1 rad autour de Z
        omega_z = np.array([0.0, 0.0, 3.0])  # rad/s -> en 1/30 s, delta = 0.1 rad
        q1 = quaternion_exp_map(omega_z * (1.0 / 30.0))
        state.update_frame({"LeftArm": q1})

        # Frame 2 : marqueurs masqués -> prédiction
        q2_pred = state.predict_rotation("LeftArm", fallback_rest=q_rest)
        self.assertFalse(np.allclose(q2_pred, q_rest, atol=1e-3))
        # Vérifie que la prédiction s'éloigne de q0 dans la même direction que q1
        dot_0 = float(np.dot(q2_pred, q0))
        dot_1 = float(np.dot(q2_pred, q1))
        self.assertGreater(dot_1, dot_0)

        # Vérification quantitative : extrapolation d'angle à ±15% de tolérance
        angle_pred = 2.0 * np.arccos(np.clip(abs(float(q2_pred[0])), 0.0, 1.0))
        angle_q1 = 2.0 * np.arccos(np.clip(abs(float(q1[0])), 0.0, 1.0))
        self.assertAlmostEqual(angle_pred, angle_q1 * 2.0, delta=angle_q1 * 0.15,
                               msg="La prédiction inertielle doit extrapoler la vitesse angulaire à ±15%")

    def test_occlusion_fallback_in_solver(self):
        """Vérifie que le solveur utilise la mémoire temporelle plutôt que de sauter en T-pose."""
        rest_rotations = {"Spine": quat_identity(), "LeftArm": quat_identity()}
        parents = {"Spine": None, "LeftArm": "Spine"}
        solver = MarkerRetargetSolver(rest_rotations=rest_rotations, parents=parents)

        state = TemporalRetargetState(fps=30.0)

        # Points de repos
        pos_rest = {
            "LeftArm": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "Spine": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        }

        # Frame 1 : rotation de 90° autour de Z pour LeftArm
        rot_90_z = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        # Tourner les points
        from core.geometry import quaternion_to_matrix
        R90 = quaternion_to_matrix(rot_90_z)
        pos_t1 = {
            "LeftArm": (R90 @ pos_rest["LeftArm"].T).T,
            "Spine": pos_rest["Spine"],
        }

        q_t1, diag1 = solver.solve_frame_multi_kabsch(
            pos_t1, pos_rest, temporal_state=state
        )
        state.update_frame(q_t1)
        np.testing.assert_allclose(q_t1["LeftArm"], rot_90_z, atol=1e-4)

        # Frame 2 : Occlusion complète de LeftArm (aucun marqueur présent)
        pos_t2 = {"Spine": pos_rest["Spine"]}  # LeftArm absent !

        # 1. Sans état temporel : saute immédiatement en T-pose (quat_identity)
        q_t2_stateless, diag2_sl = solver.solve_frame_multi_kabsch(
            pos_t2, pos_rest, temporal_state=None
        )
        np.testing.assert_allclose(q_t2_stateless["LeftArm"], quat_identity(), atol=1e-6)
        self.assertEqual(diag2_sl["LeftArm"]["status"], "fallback_rest_no_markers")

        # 2. Avec état temporel : maintient la rotation de la frame précédente sans pop
        q_t2_stateful, diag2_sf = solver.solve_frame_multi_kabsch(
            pos_t2, pos_rest, temporal_state=state
        )
        self.assertEqual(diag2_sf["LeftArm"]["status"], "fallback_temporal_prediction")
        # Doit être très proche de rot_90_z (conservé de t-1)
        self.assertAlmostEqual(float(np.dot(q_t2_stateful["LeftArm"], rot_90_z)), 1.0, places=3)

    def test_scale_exponential_smoothing(self):
        """Vérifie que l'échelle est filtrée progressivement d'une frame à l'autre."""
        state = TemporalRetargetState(scale_alpha=0.8)
        s0 = state.filter_scale(1.0)
        self.assertEqual(s0, 1.0)

        # Saut soudain à 2.0 (bruit d'observation)
        s1 = state.filter_scale(2.0)
        # s1 = 0.8 * 1.0 + 0.2 * 2.0 = 1.2
        self.assertAlmostEqual(s1, 1.2, places=4)


if __name__ == "__main__":
    unittest.main()

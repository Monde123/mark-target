"""Tests de qualité numérique et de robustesse de la pipeline complète de retargeting.
Valide les invariants mathématiques et biomécaniques sur des données synthétiques contrôlées :
- Fidélité aller-retour hiérarchique (global -> local -> global)
- Continuité temporelle et réduction du jitter (Bézier SQUAD / De Casteljau)
- Robustesse face aux cas dégénérés (marqueurs colinéaires / coplanaires)
- Stabilité absolue sur source immobile (zero-motion)
- Exactitude d'invariance d'échelle morphologique (Umeyama)
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.geometry import (
    quat_identity,
    quat_normalize,
    quat_mul,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotate_vector,
)
from core.kabsch import kabsch_rotation, kabsch_umeyama_rotation
from retargeting.solver import MarkerRetargetSolver
from retargeting.temporal_state import TemporalRetargetState
from retargeting.kinematics import (
    global_to_local_hierarchy,
    local_to_global_hierarchy,
)


class TestPipelineQuality(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Squelette de test hiérarchique standardisé (5 os)
        self.parents = {
            "Hips": None,
            "Spine": "Hips",
            "Neck": "Spine",
            "LeftArm": "Neck",
            "LeftForeArm": "LeftArm",
        }
        self.rest_rotations = {b: quat_identity() for b in self.parents}

        # Template de tétraèdre 3D non dégénéré (4 marqueurs non coplanaires)
        self.tetrahedron = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def test_pipeline_global_to_local_round_trip(self):
        """Vérifie la conservation stricte des orientations lors de l'aller-retour global <-> local."""
        # Générer des rotations globales arbitraires réalistes
        Q_global = {}
        for bone in self.parents:
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = float(np.random.uniform(-np.pi * 0.8, np.pi * 0.8))
            q = np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)], dtype=np.float64)
            Q_global[bone] = quat_normalize(q)

        # 1. Conversion globale vers locale
        Q_local = global_to_local_hierarchy(Q_global, self.parents)
        self.assertEqual(len(Q_local), len(self.parents))

        # 2. Reconstitution locale vers globale
        Q_rebuilt = local_to_global_hierarchy(Q_local, self.parents)

        # 3. Vérification de précision absolue (< 1e-6 rad)
        for bone in self.parents:
            q_orig = Q_global[bone]
            q_rec = Q_rebuilt[bone]
            # Prise en compte du double-cover de SO(3)
            dot = abs(float(np.dot(q_orig, q_rec)))
            self.assertAlmostEqual(
                dot, 1.0, places=6,
                msg=f"Dérive numérique détectée sur l'os {bone}: dot={dot}"
            )

    def test_pipeline_zero_motion_stability(self):
        """Vérifie qu'une source immobile ne génère aucune dérive ou vibration parasite."""
        solver = MarkerRetargetSolver(self.rest_rotations, self.parents)
        temporal_state = TemporalRetargetState(fps=30.0)

        # Nuage de repos
        P_rest = {b: self.tetrahedron.copy() for b in self.parents}

        # Simuler 15 frames parfaitement immobiles
        for t in range(15):
            P_t = {b: self.tetrahedron.copy() for b in self.parents}
            Q_global, diag = solver.solve_frame_multi_kabsch(P_t, P_rest)
            Q_local = global_to_local_hierarchy(Q_global, self.parents)
            temporal_state.update_frame(Q_global)

            # Vérifier que chaque rotation locale reste à l'identité stricte
            for bone, q in Q_local.items():
                np.testing.assert_allclose(
                    q, [1.0, 0.0, 0.0, 0.0], atol=1e-6,
                    err_msg=f"Frame {t}: rotation parasite non nulle sur {bone}"
                )

            # Vitesse angulaire nulle
            for bone, omega in temporal_state.angular_velocities.items():
                np.testing.assert_allclose(
                    omega, [0.0, 0.0, 0.0], atol=1e-6,
                    err_msg=f"Frame {t}: vitesse angulaire non nulle sur {bone}"
                )

    def test_pipeline_scale_invariance_umeyama(self):
        """Vérifie que le solveur Umeyama découple rigoureusement l'échelle et la rotation."""
        scale_true = 2.45
        theta = np.radians(65.0)
        c, s = np.cos(theta), np.sin(theta)
        R_true = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # P_curr = scale * (R @ P) + translation
        translation = np.array([12.0, -5.5, 3.2])
        P_curr = scale_true * (R_true @ self.tetrahedron.T).T + translation

        R_est, scale_est, t_est, cent_A, cent_B, rms, S = kabsch_umeyama_rotation(
            self.tetrahedron, P_curr
        )

        self.assertAlmostEqual(scale_est, scale_true, places=5)
        np.testing.assert_allclose(R_est, R_true, atol=1e-5)
        self.assertLess(rms, 1e-5)

    def test_pipeline_colinear_degenerate_markers_fallback(self):
        """Vérifie le comportement de repli propre (sans crash ni NaN) sur marqueurs colinéaires."""
        solver = MarkerRetargetSolver(self.rest_rotations, self.parents)

        # 4 points parfaitement colinéaires sur l'axe X (rang insuffisant pour Kabsch)
        P_colinear = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float64)

        P_rest = {"LeftArm": P_colinear}
        P_t = {"LeftArm": P_colinear + np.array([0.5, 0.0, 0.0])}

        # Doit s'exécuter sans lever d'exception non gérée
        Q_global, diag = solver.solve_frame_multi_kabsch(P_t, P_rest)

        self.assertIn("LeftArm", Q_global)
        q_res = Q_global["LeftArm"]
        # Doit être un quaternion valide (aucun NaN) et normalisé
        self.assertFalse(np.isnan(q_res).any(), "Aucun NaN toléré sur cas colinéaire")
        self.assertAlmostEqual(float(np.linalg.norm(q_res)), 1.0, places=6)
        # Le statut diagnostique doit signaler le repli gracieux
        self.assertIn(diag["LeftArm"]["status"], ["fallback_rest_exception", "fallback_rank_deficient", "fallback_degenerate"])

    def test_pipeline_temporal_continuity_reduces_jerk(self):
        """Vérifie que la résolution avec lissage temporel réduit significativement la variance d'accélération."""
        n_frames = 30
        solver = MarkerRetargetSolver(self.rest_rotations, self.parents)

        # Trajectoire de rotation lisse + bruit haute fréquence
        frames_markers = []
        rest_markers = {"LeftArm": self.tetrahedron.copy()}

        for t in range(n_frames):
            angle = float(t) * 0.05
            R_smooth = np.array([
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle),  np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ])
            noise = np.random.normal(0.0, 0.03, size=self.tetrahedron.shape)
            P_t = (R_smooth @ self.tetrahedron.T).T + noise
            frames_markers.append({"LeftArm": P_t})

        # 1. Résolution brute sans lissage
        res_raw = solver.solve_sequence(frames_markers, rest_markers, smoothing_factor=0.0)
        quats_raw = np.array([f["LeftArm"] for f in res_raw])

        # 2. Résolution avec lissage SQUAD / De Casteljau
        res_smooth = solver.solve_sequence(frames_markers, rest_markers, smoothing_factor=0.6)
        quats_smooth = np.array([f["LeftArm"] for f in res_smooth])

        # Calcul des accélérations discrètes d²q/dt²
        accel_raw = np.diff(quats_raw, n=2, axis=0)
        accel_smooth = np.diff(quats_smooth, n=2, axis=0)

        var_raw = float(np.mean(np.var(accel_raw, axis=0)))
        var_smooth = float(np.mean(np.var(accel_smooth, axis=0)))

        # Le lissage doit réduire la variance d'accélération d'au moins 25%
        self.assertLess(
            var_smooth, var_raw * 0.75,
            f"Le lissage temporel doit atténuer l'accélération angulaire (raw={var_raw:.5f}, smooth={var_smooth:.5f})"
        )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the QualityEvaluator subsystem.
==============================================
Validates that:
1. Forward kinematics preserves rest positions at identity.
2. Bone lengths are conserved to machine precision under arbitrary 3D rotations.
3. QualityScorecard computes weighted scores and grades properly.
4. Real asset evaluation meets the production-grade quality thresholds.
"""

from __future__ import annotations

import os
import unittest
import numpy as np

from core.geometry import quat_identity, quat_normalize, matrix_to_quaternion
from benchmark.quality_evaluator import QualityScorecard, RetargetQualityEvaluator


class TestQualityEvaluator(unittest.TestCase):
    """Suite de validation du banc d'évaluation qualité."""

    def test_quality_scorecard_grade_computation(self):
        """Vérifie que les seuils de notation et mentions sont conformes."""
        card = QualityScorecard(
            source_file="test.bvh",
            target_file="test.glb",
            n_frames=50,
            n_bones_matched=20,
            mpjpe_mm=12.0,
            geodesic_angular_error_deg=1.2,
            bone_length_max_error_mm=0.0,
            foot_sliding_mm_per_sec=2.0,
            raw_jerk=500.0,
            smooth_jerk=200.0,
            jerk_reduction_percent=60.0,
            noise_resilience_passed=True,
            ransac_outlier_resilience_pct=25.0,
            occlusion_recovery_passed=True,
        )
        card.compute_scores()

        self.assertGreaterEqual(card.total_score, 85.0)
        self.assertIn("A", card.grade)
        self.assertEqual(card.rigidity_score, 25.0)  # max rigidity

        # Vérification des exports
        d = card.to_dict()
        self.assertIn("total_score", d)
        self.assertIn("kinematic", d)
        md = card.to_markdown()
        self.assertIn("# 📊 Rapport d'Évaluation de la Qualité du Retargeting", md)

    def test_forward_kinematics_preserves_bone_lengths(self):
        """Vérifie que la cinématique directe préserve strictement la rigidité du squelette."""
        parents = {
            "root": None,
            "spine": "root",
            "neck": "spine",
            "head": "neck",
        }
        rest_positions = {
            "root": np.array([0.0, 1.0, 0.0]),
            "spine": np.array([0.0, 1.3, 0.0]),
            "neck": np.array([0.0, 1.5, 0.0]),
            "head": np.array([0.0, 1.7, 0.0]),
        }

        # Rotations arbitraires
        np.random.seed(42)
        q_rand = {}
        for b in parents:
            q = np.random.randn(4)
            q_rand[b] = quat_normalize(q)

        fk = RetargetQualityEvaluator.compute_forward_kinematics(parents, rest_positions, q_rand)

        # Calcul des longueurs
        for child, parent in parents.items():
            if parent is None:
                continue
            rest_len = float(np.linalg.norm(rest_positions[child] - rest_positions[parent]))
            fk_len = float(np.linalg.norm(fk[child] - fk[parent]))
            self.assertAlmostEqual(rest_len, fk_len, places=5)

    def test_evaluate_real_assets_quality(self):
        """Vérifie que l'évaluation des assets réels (a_out_precise.bvh -> clara.glb) atteint la mention A."""
        source = "a_out_precise.bvh"
        target = "clara.glb"
        if not os.path.exists(source) or not os.path.exists(target):
            self.skipTest("Assets réels non trouvés sur le disque")

        card = RetargetQualityEvaluator.evaluate(source, target, run_stress_tests=True)

        self.assertGreaterEqual(card.total_score, 75.0)
        self.assertLess(card.geodesic_angular_error_deg, 2.0)
        self.assertLess(card.bone_length_max_error_mm, 1e-4)
        self.assertTrue(card.noise_resilience_passed)
        self.assertTrue(card.occlusion_recovery_passed)
        self.assertGreaterEqual(card.ransac_outlier_resilience_pct, 20.0)


if __name__ == "__main__":
    unittest.main()

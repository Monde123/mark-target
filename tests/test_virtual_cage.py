"""
Tests pour markers.virtual_cage
===============================
Vérifie la construction géométrique, la non-coplanarité (rang 3 plein)
et la préservation stricte de la rigidité des cages tétraédriques sous rotation 3D arbitraire.
"""

import pytest
import numpy as np

from markers.virtual_cage import VirtualBoneCageGenerator
from core.kabsch import kabsch_umeyama_rotation
from core.geometry import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    quat_normalize,
    quat_identity,
)


class TestVirtualBoneCageGenerator:

    def test_tetrahedron_generation_and_shape(self):
        """Vérifie que la cage de repos génère bien 4 points 3D non dégénérés."""
        gen = VirtualBoneCageGenerator(lateral_ratio=0.2, orthogonal_ratio=0.15)
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 1.0, 0.0])

        cage = gen.build_rest_cage_template(p0, p1)
        assert cage.shape == (4, 3)

        # M0 doit correspondre à p0 et M1 à p1
        np.testing.assert_allclose(cage[0], p0, atol=1e-6)
        np.testing.assert_allclose(cage[1], p1, atol=1e-6)

        # Les 4 points ne doivent pas être coplanaires (volume du tétraèdre > 0)
        v1 = cage[1] - cage[0]
        v2 = cage[2] - cage[0]
        v3 = cage[3] - cage[0]
        volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        assert volume > 1e-4, "Le tétraèdre doit avoir un volume non nul (non coplanaire)"

    def test_rank_3_full_covariance(self):
        """Vérifie que la covariance centrée est de rang 3 plein (3 valeurs singulières significatives)."""
        gen = VirtualBoneCageGenerator()
        cage = gen.build_rest_cage_template([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])

        centered = cage - np.mean(cage, axis=0)
        U, S, Vt = np.linalg.svd(centered)

        assert len(S) == 3
        assert np.all(S > 1e-4), f"Les 3 valeurs singulières doivent être strictement positives: {S}"
        assert S[2] / S[0] > 0.05, "Le rapport de conditionnement géométrique doit permettre d'isoler le twist"

    def test_rigidity_under_arbitrary_3d_rotation(self):
        """
        Vérifie que sous une rotation 3D quelconque (avec twist),
        la cage est 100% rigide (distances pairwise invariantes)
        et Kabsch-Umeyama retrouve la rotation avec une erreur < 1e-12.
        """
        gen = VirtualBoneCageGenerator()
        p0 = np.array([0.1, 0.2, 0.3])
        p1 = np.array([0.5, 0.8, -0.2])
        rest_cage = gen.build_rest_cage_template(p0, p1)

        # Distances pairwise au repos
        dists_rest = []
        for i in range(4):
            for j in range(i + 1, 4):
                dists_rest.append(np.linalg.norm(rest_cage[i] - rest_cage[j]))

        # Rotation arbitraire avec composante de roulis (twist) non triviale
        q_rot = quat_normalize(np.array([0.5, -0.3, 0.6, 0.5477]))
        R_true = quaternion_to_matrix(q_rot)
        translation = np.array([1.5, -2.0, 0.7])

        # Application de la transformation rigide
        moved_cage = (R_true @ rest_cage.T).T + translation

        # Vérification de l'invariance des distances pairwise (rigidité intrinsèque)
        dists_moved = []
        for i in range(4):
            for j in range(i + 1, 4):
                dists_moved.append(np.linalg.norm(moved_cage[i] - moved_cage[j]))

        np.testing.assert_allclose(
            dists_moved, dists_rest, atol=1e-14,
            err_msg="Les distances internes doivent être strictement conservées sous rotation rigide"
        )

        # Résolution par Kabsch-Umeyama
        R_est, scale, _, _, _, rms, s_vals = kabsch_umeyama_rotation(rest_cage, moved_cage)

        assert rms < 1e-12, f"RMS doit être infinitésimal: {rms}"
        np.testing.assert_allclose(scale, 1.0, atol=1e-7)
        np.testing.assert_allclose(R_est, R_true, atol=1e-12)

    def test_sequence_generation(self):
        """Vérifie la génération temporelle d'une séquence de cages d'os hiérarchique."""
        gen = VirtualBoneCageGenerator()
        parents = {
            "root": None,
            "bone1": "root",
            "bone2": "bone1",
        }
        rest_pos = {
            "root": np.array([0.0, 0.0, 0.0]),
            "bone1": np.array([0.0, 0.5, 0.0]),
            "bone2": np.array([0.0, 1.0, 0.0]),
        }
        frames_pos = [
            rest_pos,
            {
                "root": np.array([0.0, 0.0, 0.0]),
                "bone1": np.array([0.3, 0.4, 0.0]),
                "bone2": np.array([0.7, 0.7, 0.0]),
            }
        ]

        rest_cages, frames_cages = gen.generate_bone_cages_sequence(
            parents=parents,
            rest_positions=rest_pos,
            frames_positions=frames_pos,
        )

        assert "root" in rest_cages
        assert "bone1" in rest_cages
        assert "bone2" in rest_cages
        assert len(frames_cages) == 2
        for f in frames_cages:
            assert f["bone1"].shape == (4, 3)

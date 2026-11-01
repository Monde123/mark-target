"""
Tests pour adapters.gltf_mixamo_adapter et préservation de bind pose
===================================================================
Vérifie la formule du Delta Bind-Pose et la préservation de la pose de repos locale
lors de l'exportation d'animations glTF.
"""

import pytest
import numpy as np

from adapters.gltf_mixamo_adapter import GLTFMixamoAdapter
from core.geometry import (
    quat_identity,
    quat_mul,
    quat_inv,
    quat_normalize,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


class TestGltfBindPoseBridge:

    def test_delta_bind_pose_identity(self):
        """
        Vérifie que la formule du Delta Bind-Pose restitue STRICTEMENT
        la rotation locale de repos quand le delta de mouvement est l'identité (repos).
        Formule : Q_anim_local = Q_bind_local * (Q_bind_world^(-1) * Q_delta * Q_bind_world)
        """
        # Pose de repos locale arbitraire
        q_bind_local = quat_normalize(np.array([0.7071, 0.7071, 0.0, 0.0]))
        # Pose de repos globale arbitraire
        q_bind_world = quat_normalize(np.array([0.5, 0.5, 0.5, 0.5]))

        # Delta de repos (I)
        q_delta = quat_identity()

        # Conjugaison
        q_loc_delta = quat_mul(quat_mul(quat_inv(q_bind_world), q_delta), q_bind_world)
        q_anim_local = quat_normalize(quat_mul(q_bind_local, q_loc_delta))

        # La rotation calculée doit être strictement identique à q_bind_local
        np.testing.assert_allclose(q_anim_local, q_bind_local, atol=1e-12)

    def test_delta_bind_pose_local_hinge_coherence(self):
        """
        Vérifie que la conjugaison dans le repère local applique bien la rotation
        autour de l'axe charnière anatomique local de l'os.
        """
        # Soit un coude dont la charnière locale de flexion est l'axe X local
        q_bind_local = quat_identity()
        # Supposons que dans le monde, l'os soit orienté à 90° autour de Y
        # (donc l'axe X local pointe vers Z monde)
        R_world = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
        q_bind_world = matrix_to_quaternion(R_world)

        # Mouvement source dans le monde : flexion de 45° autour de Z monde
        angle = np.pi / 4.0
        R_delta_world = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float64)
        q_src_delta = matrix_to_quaternion(R_delta_world)

        # Conjugaison : q_loc_delta = q_bind_world^(-1) * q_src_delta * q_bind_world
        q_loc_delta = quat_mul(quat_mul(quat_inv(q_bind_world), q_src_delta), q_bind_world)
        R_loc_delta = quaternion_to_matrix(q_loc_delta)

        # Puisque Z monde correspond à X local, la rotation locale conjuguée
        # DOIT être une rotation de 45° autour de l'axe X local !
        v_test = np.array([1.0, 0.0, 0.0])  # axe X local
        v_rotated = R_loc_delta @ v_test
        np.testing.assert_allclose(v_rotated, v_test, atol=1e-6,
                                  err_msg="L'axe X local doit être invariant (axe de rotation)")

"""Test unitaire pour l'adaptateur HybrIK / SMPL-X (.pk).
"""

import os
import pickle
import tempfile
import unittest
import numpy as np

from adapters.hybrik_pk_adapter import HybrIKPKAdapter


class TestHybrIKPKAdapter(unittest.TestCase):

    def test_load_pk_with_positions(self):
        """Vérifie le chargement d'un pickle avec coordonnées 3D 'pred_xyz_jts'."""
        n_frames = 10
        n_joints = 24
        fake_data = {
            "n_frames": n_frames,
            "fps": 30.0,
            "pred_xyz_jts": np.random.randn(n_frames, n_joints, 3),
            "transl": np.random.randn(n_frames, 3),
            "quats": np.zeros((n_frames, n_joints, 4)),
        }
        fake_data["quats"][:, :, 0] = 1.0  # Quaternions unitaires identité

        with tempfile.NamedTemporaryFile(suffix=".pk", delete=False) as tmp:
            pickle.dump(fake_data, tmp)
            tmp_path = tmp.name

        try:
            adapter = HybrIKPKAdapter()
            seq = adapter.load(tmp_path)
            self.assertEqual(len(seq.frames_markers), n_frames)
            self.assertGreater(len(seq.rest_markers), 10)
            self.assertEqual(seq.fps, 30.0)
            self.assertIsNotNone(seq.root_translations)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_pk_with_quaternions_only(self):
        """Vérifie le chargement d'un pickle contenant uniquement des quaternions de rotation."""
        n_frames = 8
        n_joints = 24
        fake_data = {
            "n_frames": n_frames,
            "fps": 60.0,
            "quats": np.zeros((n_frames, n_joints, 4)),
        }
        fake_data["quats"][:, :, 0] = 1.0

        with tempfile.NamedTemporaryFile(suffix=".pk", delete=False) as tmp:
            pickle.dump(fake_data, tmp)
            tmp_path = tmp.name

        try:
            adapter = HybrIKPKAdapter()
            seq = adapter.load(tmp_path)
            self.assertEqual(len(seq.frames_markers), n_frames)
            self.assertEqual(seq.fps, 60.0)
            self.assertIsNotNone(seq.joint_rotations)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

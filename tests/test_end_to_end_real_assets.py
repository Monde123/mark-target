"""Test d'intégration bout-en-bout avec de vrais assets 3D du monde réel.
Vérifie qu'un fichier BVH réel (a_out_precise.bvh) est transféré avec succès
vers un vrai modèle Mixamo .glb (clara.glb) et produit des canaux d'animation
glTF valides avec des mouvements de rotations mesurables.
"""

import os
import unittest
import numpy as np
from pygltflib import GLTF2

from adapters.bvh_adapter import BVHAdapter
from adapters.gltf_mixamo_adapter import GLTFMixamoAdapter
from mappings.auto_mapper import BufferRigMapper
from retargeting.solver import MarkerRetargetSolver
from retargeting.temporal_state import TemporalRetargetState
from retargeting.kinematics import global_to_local_hierarchy


class TestEndToEndRealAssets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists("a_out_precise.bvh") or not os.path.exists("clara.glb"):
            raise unittest.SkipTest("Assets binaires réels absents (a_out_precise.bvh ou clara.glb)")

    def setUp(self):
        self.bvh_path = "a_out_precise.bvh"
        self.glb_path = "clara.glb"
        self.output_glb = "test_end_to_end_out.glb"

    def tearDown(self):
        if os.path.exists(self.output_glb):
            try:
                os.remove(self.output_glb)
            except OSError:
                pass

    def test_end_to_end_bvh_to_glb_transfer(self):
        """Vérifie le pipeline complet sur vrais modèles et l'intégrité de l'animation glTF exportée."""
        # 1. Chargement de la source réelle
        source_adapter = BVHAdapter()
        sequence = source_adapter.load(self.bvh_path)
        self.assertGreater(len(sequence.frames_markers), 50)

        # 2. Chargement du rig cible réel
        target_adapter = GLTFMixamoAdapter()
        skeleton = target_adapter.load_skeleton(self.glb_path)
        self.assertGreater(len(skeleton.rest_rotations), 50)

        # 3. Auto-Mapping via l'ossature tampon
        mapper = BufferRigMapper(
            source_bones=list(sequence.rest_markers.keys()),
            target_bones=list(skeleton.parents.keys()),
            source_rest_positions=sequence.rest_markers,
            source_parents=getattr(sequence, "parents", {}),
            target_parents=skeleton.parents,
        )
        mapping = mapper.get_source_to_target_bone_map()
        self.assertGreater(len(mapping), 20, "L'auto-mapper doit apparier au moins 20 os maîtres")

        # 4. Résolution cinématique avec état temporel t-1 -> t
        solver = MarkerRetargetSolver(
            rest_rotations=skeleton.rest_rotations,
            parents=skeleton.parents,
        )
        temporal_state = TemporalRetargetState(fps=sequence.fps or 30.0)

        # Préparation de la pose de repos (rest_named) pour orienter la source
        rest_named = {}
        for src_name, val in sequence.rest_markers.items():
            tgt_name = mapping.get(src_name)
            if tgt_name and tgt_name in skeleton.parents:
                rest_named[tgt_name] = np.asarray(val, dtype=np.float64).reshape(-1)

        animation_clip = []
        n_frames = len(sequence.frames_markers)

        for t in range(n_frames):
            frame_markers = sequence.frames_markers[t]
            # Liaison tolérante
            named = {}
            for src_name, val in frame_markers.items():
                tgt_name = mapping.get(src_name)
                if tgt_name and tgt_name in skeleton.parents:
                    named[tgt_name] = np.asarray(val, dtype=np.float64).reshape(-1)

            positions_t = {name: val for name, val in named.items()}

            Q_global, _ = solver.solve_frame_joint_markers(
                positions_t=positions_t,
                positions_rest=rest_named,
            )

            # Compléter les os non animés par leur pose au repos
            for b in skeleton.parents:
                if b not in Q_global:
                    Q_global[b] = skeleton.rest_rotations.get(b, np.array([1.0, 0.0, 0.0, 0.0])).copy()

            temporal_state.update_frame(Q_global)

            Q_local = global_to_local_hierarchy(Q_global, skeleton.parents)
            animation_clip.append({
                "frame": t,
                "bones": Q_local,
                "world": Q_global,
                "root": [0.0, 0.0, 0.0],
            })

        # 5. Export de la vraie animation glTF
        target_adapter.export_animation(
            target_model_path=self.glb_path,
            output_path=self.output_glb,
            animation_clip=animation_clip,
            skeleton=skeleton,
            animation_name="e2e_animation",
        )

        self.assertTrue(os.path.exists(self.output_glb), "Le fichier exporté doit exister sur disque")

        # 6. Rechargement et contrôle strict de l'intégrité glTF
        gltf_out = GLTF2().load(self.output_glb)

        # Au moins une animation présente
        self.assertEqual(len(gltf_out.animations), 1)
        anim = gltf_out.animations[0]
        self.assertEqual(anim.name, "e2e_animation")
        self.assertGreater(len(anim.channels), 20)

        # Vérification du flux binaire
        blob = gltf_out.binary_blob()
        self.assertIsNotNone(blob)
        self.assertGreater(len(blob), 1000000)

        # Vérification des mouvements réels (variance non nulle sur les os animés)
        moving_count = 0
        for ch in anim.channels:
            sampler = anim.samplers[ch.sampler]
            acc = gltf_out.accessors[sampler.output]
            bv = gltf_out.bufferViews[acc.bufferView]
            data = np.frombuffer(
                blob[bv.byteOffset:bv.byteOffset + bv.byteLength], dtype=np.float32
            ).reshape(-1, 4)
            # Pas de NaN et quaternions unitaires
            self.assertFalse(np.isnan(data).any(), "Aucun NaN toléré dans les quaternions")
            norms = np.linalg.norm(data, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-3, err_msg="Tous les quaternions glTF doivent être unitaires")
            var = float(np.sum(np.var(data, axis=0)))
            if var > 1e-4:
                moving_count += 1

        self.assertGreaterEqual(
            moving_count, 10,
            f"Au moins 10 os doivent présenter une variance de mouvement réelle (obtenu: {moving_count})"
        )


if __name__ == "__main__":
    unittest.main()

"""Tests unitaires pour l'ossature tampon canonique et l'auto-mapper d'identification.
"""

import unittest
import numpy as np

from mappings.buffer_armature import CanonicalBufferArmature
from mappings.auto_mapper import AutoBoneIdentifier, BufferRigMapper
from core.geometry import quat_identity


class TestAutoMapperAndBufferArmature(unittest.TestCase):

    def test_canonical_buffer_armature_geometry(self):
        """Vérifie la consistance de la géométrie de l'ossature tampon canonique."""
        positions = CanonicalBufferArmature.get_canonical_positions()
        parents = CanonicalBufferArmature.get_canonical_parents()

        # 55 os présents
        self.assertGreaterEqual(len(positions), 55)
        self.assertIn("Hips", positions)
        self.assertIn("LeftArm", positions)
        self.assertIn("RightArm", positions)
        self.assertIn("LeftUpLeg", positions)

        # Vérification de l'horizontalité pure des bras (+X pour gauche, -X pour droite)
        u_left_arm = CanonicalBufferArmature.get_canonical_direction("LeftArm")
        self.assertIsNotNone(u_left_arm)
        np.testing.assert_allclose(u_left_arm, np.array([1.0, 0.0, 0.0]), atol=1e-5)

        u_right_arm = CanonicalBufferArmature.get_canonical_direction("RightArm")
        self.assertIsNotNone(u_right_arm)
        np.testing.assert_allclose(u_right_arm, np.array([-1.0, 0.0, 0.0]), atol=1e-5)

        # Vérification de la verticalité des jambes (-Y)
        u_left_leg = CanonicalBufferArmature.get_canonical_direction("LeftUpLeg")
        self.assertIsNotNone(u_left_leg)
        np.testing.assert_allclose(u_left_leg, np.array([0.0, -1.0, 0.0]), atol=1e-5)

    def test_auto_bone_identifier_mixamo(self):
        """Vérifie la reconnaissance automatique d'un rig Mixamo."""
        mixamo_bones = [
            "mixamorig:Hips",
            "mixamorig:Spine",
            "mixamorig:Spine1",
            "mixamorig:Spine2",
            "mixamorig:Neck",
            "mixamorig:Head",
            "mixamorig:LeftShoulder",
            "mixamorig:LeftArm",
            "mixamorig:LeftForeArm",
            "mixamorig:LeftHand",
            "mixamorig:RightShoulder",
            "mixamorig:RightArm",
            "mixamorig:RightForeArm",
            "mixamorig:RightHand",
            "mixamorig:LeftUpLeg",
            "mixamorig:LeftLeg",
            "mixamorig:LeftFoot",
            "mixamorig:LeftToeBase",
        ]
        mapping = AutoBoneIdentifier.map_skeleton(mixamo_bones)
        self.assertEqual(mapping["mixamorig:Hips"], "Hips")
        self.assertEqual(mapping["mixamorig:LeftArm"], "LeftArm")
        self.assertEqual(mapping["mixamorig:RightForeArm"], "RightForeArm")
        self.assertEqual(mapping["mixamorig:LeftUpLeg"], "LeftUpLeg")
        self.assertEqual(mapping["mixamorig:Head"], "Head")

    def test_auto_bone_identifier_blender_rigify(self):
        """Vérifie la reconnaissance automatique d'un rig Blender Rigify (conventions DEF- et .L / .R)."""
        rigify_bones = [
            "DEF-spine",
            "DEF-chest",
            "DEF-head",
            "DEF-upper_arm.L",
            "DEF-forearm.L",
            "DEF-hand.L",
            "DEF-upper_arm.R",
            "DEF-forearm.R",
            "DEF-hand.R",
            "DEF-thigh.L",
            "DEF-shin.L",
            "DEF-foot.L",
            "DEF-thigh.R",
            "DEF-shin.R",
            "DEF-foot.R",
        ]
        mapping = AutoBoneIdentifier.map_skeleton(rigify_bones)
        self.assertEqual(mapping["DEF-spine"], "Spine")
        self.assertEqual(mapping["DEF-head"], "Head")
        self.assertEqual(mapping["DEF-upper_arm.L"], "LeftArm")
        self.assertEqual(mapping["DEF-forearm.L"], "LeftForeArm")
        self.assertEqual(mapping["DEF-hand.L"], "LeftHand")
        self.assertEqual(mapping["DEF-upper_arm.R"], "RightArm")
        self.assertEqual(mapping["DEF-thigh.L"], "LeftUpLeg")
        self.assertEqual(mapping["DEF-shin.L"], "LeftLeg")
        self.assertEqual(mapping["DEF-foot.L"], "LeftFoot")

    def test_auto_bone_identifier_vroid_and_cmu(self):
        """Vérifie la reconnaissance automatique de conventions VRoid/VRM et CMU/Biovision BVH."""
        vroid_bones = [
            "J_Bip_C_Hips",
            "J_Bip_L_UpperArm",
            "J_Bip_L_LowerArm",
            "J_Bip_L_Hand",
            "J_Bip_R_UpperArm",
        ]
        mapping_vroid = AutoBoneIdentifier.map_skeleton(vroid_bones)
        self.assertEqual(mapping_vroid["J_Bip_C_Hips"], "Hips")
        self.assertEqual(mapping_vroid["J_Bip_L_UpperArm"], "LeftArm")
        self.assertEqual(mapping_vroid["J_Bip_L_LowerArm"], "LeftForeArm")

        cmu_bones = [
            "Hips",
            "LeftCollar",
            "LeftUpLeg",
            "LeftLowLeg",
            "LeftFoot",
            "RightUpLeg",
        ]
        mapping_cmu = AutoBoneIdentifier.map_skeleton(cmu_bones)
        self.assertEqual(mapping_cmu["LeftCollar"], "LeftShoulder")
        self.assertEqual(mapping_cmu["LeftUpLeg"], "LeftUpLeg")
        self.assertEqual(mapping_cmu["LeftLowLeg"], "LeftLeg")

    def test_auxiliary_bones_filtering(self):
        """Vérifie que les os auxiliaires (twist, vêtements, cheveux, armes) sont ignorés."""
        aux_bones = [
            "mixamorig:LeftArmTwist",
            "DEF-arm_twist.L",
            "Hair_Strand_01",
            "Cloth_Skirt_L",
            "Weapon_Sword_Bone",
            "LeftLegIK",
        ]
        for b in aux_bones:
            self.assertIsNone(AutoBoneIdentifier.identify_bone(b))

    def test_buffer_rig_mapper_alignment_a_pose_to_t_pose(self):
        """Vérifie que le mapper calcule une pré-rotation pour aligner un bras en A-Pose vers la T-Pose."""
        # Bras en A-Pose (incliné à 45° vers le bas dans le plan XY : dir = (cos(45°), -sin(45°), 0))
        angle_a = np.radians(45.0)
        p_shoulder = np.array([0.1, 1.4, 0.0])
        p_arm = np.array([0.2, 1.4, 0.0])
        # ForeArm descend à 45°
        p_forearm = p_arm + np.array([np.cos(angle_a), -np.sin(angle_a), 0.0]) * 0.3

        source_bones = ["Hips", "LeftArm", "LeftForeArm"]
        target_bones = ["mixamorig:Hips", "mixamorig:LeftArm", "mixamorig:LeftForeArm"]
        source_parents = {"Hips": None, "LeftArm": "Hips", "LeftForeArm": "LeftArm"}
        source_rest = {"Hips": np.array([0.0, 1.0, 0.0]), "LeftArm": p_arm, "LeftForeArm": p_forearm}

        mapper = BufferRigMapper(
            source_bones=source_bones,
            target_bones=target_bones,
            source_rest_positions=source_rest,
            source_parents=source_parents,
        )

        # Mapping direct vérifié
        direct_map = mapper.get_source_to_target_bone_map()
        self.assertEqual(direct_map["LeftArm"], "mixamorig:LeftArm")

        # La pré-rotation pour LeftArm doit redresser la direction de 45° vers l'horizontale +X
        q_align = mapper.source_align_rotations.get("LeftArm")
        self.assertIsNotNone(q_align)
        # En tournant le vecteur A-Pose par q_align, on doit obtenir la direction canonique (1, 0, 0)
        from core.geometry import quaternion_to_matrix
        R_align = quaternion_to_matrix(q_align)
        u_src = (p_forearm - p_arm) / np.linalg.norm(p_forearm - p_arm)
        u_aligned = R_align @ u_src
        np.testing.assert_allclose(u_aligned, np.array([1.0, 0.0, 0.0]), atol=1e-4)


if __name__ == "__main__":
    unittest.main()

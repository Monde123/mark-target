"""Tests unitaires pour le sous-système de mapping universel (mappings/)."""

import unittest
import sys
import os

# Assurer que le module racine est dans le sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mappings.standard_humanoid import (
    STANDARD_HUMANOID_BONES,
    STANDARD_HUMANOID_PARENTS,
    MIXAMO_TO_STANDARD,
    VRM_TO_STANDARD,
    SMPLX_TO_STANDARD,
    BVH_TO_STANDARD,
)
from mappings.registry import MappingRegistry


class TestMappingSubsystem(unittest.TestCase):
    def test_standard_bones_count(self):
        """Vérifie la présence exacte des 55 os standards du corps humain."""
        self.assertEqual(len(STANDARD_HUMANOID_BONES), 55)
        self.assertIn("Hips", STANDARD_HUMANOID_BONES)
        self.assertIn("Head", STANDARD_HUMANOID_BONES)
        self.assertIn("LeftFoot", STANDARD_HUMANOID_BONES)
        self.assertIn("RightToeBase", STANDARD_HUMANOID_BONES)
        # Vérification des doigts
        self.assertIn("LeftHandThumb1", STANDARD_HUMANOID_BONES)
        self.assertIn("RightHandPinky3", STANDARD_HUMANOID_BONES)

    def test_hierarchy_consistency(self):
        """Vérifie que la hiérarchie parentale est acyclique et que la racine est Hips."""
        self.assertIsNone(STANDARD_HUMANOID_PARENTS["Hips"])
        self.assertEqual(STANDARD_HUMANOID_PARENTS["Spine"], "Hips")
        self.assertEqual(STANDARD_HUMANOID_PARENTS["LeftForeArm"], "LeftArm")
        self.assertEqual(STANDARD_HUMANOID_PARENTS["LeftHandThumb1"], "LeftHand")

    def test_mixamo_to_standard_has_legs(self):
        """Vérifie que les jambes et pieds Mixamo sont bien mappés."""
        self.assertIn("mixamorig:LeftUpLeg", MIXAMO_TO_STANDARD)
        self.assertEqual(MIXAMO_TO_STANDARD["mixamorig:LeftUpLeg"], "LeftUpLeg")
        self.assertEqual(MIXAMO_TO_STANDARD["mixamorig:RightFoot"], "RightFoot")

    def test_smplx_hybrik_to_mixamo_resolution(self):
        """Vérifie le flux critique HybrIK-X (SMPL-X) -> Mixamo."""
        mapping = MappingRegistry.get_mapping("smplx", "mixamo")
        self.assertTrue(len(mapping) > 50, f"Expected >50 mapped bones, got {len(mapping)}")
        self.assertEqual(mapping["pelvis"], "mixamorig:Hips")
        self.assertEqual(mapping["left_elbow"], "mixamorig:LeftForeArm")
        self.assertEqual(mapping["left_ankle"], "mixamorig:LeftFoot")
        self.assertEqual(mapping["left_thumb1"], "mixamorig:LeftHandThumb1")
        self.assertEqual(mapping["right_index3"], "mixamorig:RightHandIndex3")

    def test_smplx_hybrik_to_vrm_resolution(self):
        """Vérifie le flux critique HybrIK-X (SMPL-X) -> VRM."""
        mapping = MappingRegistry.get_mapping("hybrik", "vrm")
        self.assertTrue(len(mapping) > 50)
        self.assertEqual(mapping["pelvis"], "hips")
        self.assertEqual(mapping["left_shoulder"], "leftUpperArm")
        self.assertEqual(mapping["left_wrist"], "leftHand")
        self.assertEqual(mapping["left_thumb1"], "leftThumbMetacarpal")

    def test_bvh_to_mixamo_resolution(self):
        """Vérifie le flux MoCap BVH -> Mixamo."""
        mapping = MappingRegistry.get_mapping("bvh", "mixamo")
        self.assertTrue(len(mapping) >= 20)
        self.assertEqual(mapping["Hips"], "mixamorig:Hips")
        self.assertEqual(mapping["LeftUpArm"], "mixamorig:LeftArm")
        self.assertEqual(mapping["LeftLowArm"], "mixamorig:LeftForeArm")

    def test_custom_registration(self):
        """Vérifie l'enregistrement d'un mapping propriétaire sur mesure."""
        custom_map = {"custom_hip": "target_root", "custom_hand": "target_palm"}
        MappingRegistry.register("mon_rig_perso", "unreal", custom_map)
        resolved = MappingRegistry.get_mapping("mon_rig_perso", "unreal")
        self.assertEqual(resolved, custom_map)




    def test_json_registration_and_export(self):
        """Vérifie le chargement et l'exportation de mappings via JSON."""
        import tempfile
        import json

        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            temp_json_path = f.name
            json.dump({"custom_root": "mixamorig:Hips", "custom_arm": "mixamorig:LeftArm"}, f)

        try:
            MappingRegistry.register_from_json(temp_json_path, "custom_json_rig", "mixamo")
            resolved = MappingRegistry.get_mapping("custom_json_rig", "mixamo")
            self.assertEqual(resolved["custom_root"], "mixamorig:Hips")
            self.assertEqual(resolved["custom_arm"], "mixamorig:LeftArm")

            # Test export
            with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f_exp:
                export_path = f_exp.name
            MappingRegistry.export_to_json("custom_json_rig", "mixamo", export_path)
            with open(export_path, "r") as f_check:
                exp_data = json.load(f_check)
            self.assertEqual(exp_data["mapping"]["custom_root"], "mixamorig:Hips")
            os.remove(export_path)
        finally:
            if os.path.exists(temp_json_path):
                os.remove(temp_json_path)


if __name__ == "__main__":
    unittest.main()
"""
Module isolate.mappings.standard_humanoid
=========================================
Nomenclature standardisée pivot "Humanoid" inspirée des standards Unity / VRM / OpenXR.
Sert de pont universel entre les formats SMPL-X, Mixamo, VRM, BVH.
"""
from typing import Dict, List, Optional

# Noms normalisés pivot
STANDARD_HUMANOID_BONES = [
    "Hips",
    "Spine", "Spine1", "Spine2",
    "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    # Doigts main gauche
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
    # Doigts main droite
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
]

# Hiérarchie standard parent-enfant
STANDARD_HUMANOID_PARENTS: Dict[str, Optional[str]] = {
    "Hips": None,
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Head": "Neck",

    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",

    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",

    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",

    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
}

# Ajout des doigts
for side in ("Left", "Right"):
    hand = f"{side}Hand"
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        STANDARD_HUMANOID_PARENTS[f"{side}Hand{finger}1"] = hand
        STANDARD_HUMANOID_PARENTS[f"{side}Hand{finger}2"] = f"{side}Hand{finger}1"
        STANDARD_HUMANOID_PARENTS[f"{side}Hand{finger}3"] = f"{side}Hand{finger}2"


# Mappings de correspondance vers le standard Humanoid Pivot
# 1. Mixamo -> Standard
MIXAMO_TO_STANDARD: Dict[str, str] = {
    "mixamorig:Hips": "Hips",
    "mixamorig:Spine": "Spine",
    "mixamorig:Spine1": "Spine1",
    "mixamorig:Spine2": "Spine2",
    "mixamorig:Neck": "Neck",
    "mixamorig:Head": "Head",
    "mixamorig:LeftShoulder": "LeftShoulder",
    "mixamorig:LeftArm": "LeftArm",
    "mixamorig:LeftForeArm": "LeftForeArm",
    "mixamorig:LeftHand": "LeftHand",
    "mixamorig:RightShoulder": "RightShoulder",
    "mixamorig:RightArm": "RightArm",
    "mixamorig:RightForeArm": "RightForeArm",
    "mixamorig:RightHand": "RightHand",
}
for side in ("Left", "Right"):
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for seg in (1, 2, 3):
            MIXAMO_TO_STANDARD[f"mixamorig:{side}Hand{finger}{seg}"] = f"{side}Hand{finger}{seg}"

STANDARD_TO_MIXAMO: Dict[str, str] = {v: k for k, v in MIXAMO_TO_STANDARD.items()}


# 2. VRM Humanoid -> Standard
VRM_TO_STANDARD: Dict[str, str] = {
    "hips": "Hips",
    "spine": "Spine",
    "chest": "Spine1",
    "upperChest": "Spine2",
    "neck": "Neck",
    "head": "Head",
    "leftShoulder": "LeftShoulder",
    "leftUpperArm": "LeftArm",
    "leftLowerArm": "LeftForeArm",
    "leftHand": "LeftHand",
    "rightShoulder": "RightShoulder",
    "rightUpperArm": "RightArm",
    "rightLowerArm": "RightForeArm",
    "rightHand": "RightHand",
    "leftUpperLeg": "LeftUpLeg",
    "leftLowerLeg": "LeftLeg",
    "leftFoot": "LeftFoot",
    "leftToes": "LeftToeBase",
    "rightUpperLeg": "RightUpLeg",
    "rightLowerLeg": "RightLeg",
    "rightFoot": "RightFoot",
    "rightToes": "RightToeBase",
    # Doigts VRM
    "leftThumbMetacarpal": "LeftHandThumb1",
    "leftThumbProximal": "LeftHandThumb2",
    "leftThumbDistal": "LeftHandThumb3",
    "leftIndexProximal": "LeftHandIndex1",
    "leftIndexIntermediate": "LeftHandIndex2",
    "leftIndexDistal": "LeftHandIndex3",
    "leftMiddleProximal": "LeftHandMiddle1",
    "leftMiddleIntermediate": "LeftHandMiddle2",
    "leftMiddleDistal": "LeftHandMiddle3",
    "leftRingProximal": "LeftHandRing1",
    "leftRingIntermediate": "LeftHandRing2",
    "leftRingDistal": "LeftHandRing3",
    "leftLittleProximal": "LeftHandPinky1",
    "leftLittleIntermediate": "LeftHandPinky2",
    "leftLittleDistal": "LeftHandPinky3",
    # Main droite
    "rightThumbMetacarpal": "RightHandThumb1",
    "rightThumbProximal": "RightHandThumb2",
    "rightThumbDistal": "RightHandThumb3",
    "rightIndexProximal": "RightHandIndex1",
    "rightIndexIntermediate": "RightHandIndex2",
    "rightIndexDistal": "RightHandIndex3",
    "rightMiddleProximal": "RightHandMiddle1",
    "rightMiddleIntermediate": "RightHandMiddle2",
    "rightMiddleDistal": "RightHandMiddle3",
    "rightRingProximal": "RightHandRing1",
    "rightRingIntermediate": "RightHandRing2",
    "rightRingDistal": "RightHandRing3",
    "rightLittleProximal": "RightHandPinky1",
    "rightLittleIntermediate": "RightHandPinky2",
    "rightLittleDistal": "RightHandPinky3",
}

STANDARD_TO_VRM: Dict[str, str] = {v: k for k, v in VRM_TO_STANDARD.items()}

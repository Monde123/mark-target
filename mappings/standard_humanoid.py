"""Module isolate.mappings.standard_humanoid
=========================================
Nomenclature standardisée pivot "Humanoid" inspirée des standards Unity / VRM / OpenXR.
Sert de pont universel entre les formats SMPL-X (HybrIK-X), Mixamo, VRM, BVH (MoCap).
Total : 55 os standards (22 corps + 3 tête/visage + 30 doigts).
"""

from typing import Dict, List, Optional

# Noms normalisés pivot (55 os exactement)
STANDARD_HUMANOID_BONES = [
    # Corps principal (22)
    "Hips",
    "Spine", "Spine1", "Spine2",
    "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    # Visage / Tête (3)
    "Jaw", "LeftEye", "RightEye",
    # Doigts main gauche (15)
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
    # Doigts main droite (15)
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
    "Jaw": "Head",
    "LeftEye": "Head",
    "RightEye": "Head",
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

# Ajout des doigts dans la hiérarchie parentale
for side in ("Left", "Right"):
    hand = f"{side}Hand"
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        STANDARD_HUMANOID_PARENTS[f"{side}Hand{finger}1"] = hand
        STANDARD_HUMANOID_PARENTS[f"{side}Hand{finger}2"] = f"{side}Hand{finger}1"
        STANDARD_HUMANOID_PARENTS[f"{side}Hand{finger}3"] = f"{side}Hand{finger}2"

# ==============================================================================
# 1. MIXAMO <-> STANDARD
# ==============================================================================
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
    # Membres inférieurs (jambes & pieds)
    "mixamorig:LeftUpLeg": "LeftUpLeg",
    "mixamorig:LeftLeg": "LeftLeg",
    "mixamorig:LeftFoot": "LeftFoot",
    "mixamorig:LeftToeBase": "LeftToeBase",
    "mixamorig:RightUpLeg": "RightUpLeg",
    "mixamorig:RightLeg": "RightLeg",
    "mixamorig:RightFoot": "RightFoot",
    "mixamorig:RightToeBase": "RightToeBase",
}

for side in ("Left", "Right"):
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for seg in (1, 2, 3):
            MIXAMO_TO_STANDARD[f"mixamorig:{side}Hand{finger}{seg}"] = f"{side}Hand{finger}{seg}"

STANDARD_TO_MIXAMO: Dict[str, str] = {v: k for k, v in MIXAMO_TO_STANDARD.items()}

# ==============================================================================
# 2. VRM HUMANOID <-> STANDARD
# ==============================================================================
VRM_TO_STANDARD: Dict[str, str] = {
    "hips": "Hips",
    "spine": "Spine",
    "chest": "Spine1",
    "upperChest": "Spine2",
    "neck": "Neck",
    "head": "Head",
    "jaw": "Jaw",
    "leftEye": "LeftEye",
    "rightEye": "RightEye",
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
    # Doigts VRM main gauche
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
    # Doigts VRM main droite
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

# ==============================================================================
# 3. SMPL-X (HybrIK-X) <-> STANDARD
# ==============================================================================
SMPLX_TO_STANDARD: Dict[str, str] = {
    "pelvis": "Hips",
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "spine1": "Spine",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "spine2": "Spine1",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "spine3": "Spine2",
    "left_foot": "LeftToeBase",
    "right_foot": "RightToeBase",
    "neck": "Neck",
    "left_collar": "LeftShoulder",
    "right_collar": "RightShoulder",
    "head": "Head",
    "jaw": "Jaw",
    "left_eye_smplhf": "LeftEye",
    "right_eye_smplhf": "RightEye",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}

# Doigts SMPL-X (HybrIK-X 15 articulations par main)
_SMPLX_FINGER_NAMES = {
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "pinky": "Pinky",
}

for smpl_f, std_f in _SMPLX_FINGER_NAMES.items():
    for seg in (1, 2, 3):
        SMPLX_TO_STANDARD[f"left_{smpl_f}{seg}"] = f"LeftHand{std_f}{seg}"
        SMPLX_TO_STANDARD[f"right_{smpl_f}{seg}"] = f"RightHand{std_f}{seg}"

STANDARD_TO_SMPLX: Dict[str, str] = {v: k for k, v in SMPLX_TO_STANDARD.items()}

# ==============================================================================
# 4. BVH / BIOMECANIQUE MOCAP <-> STANDARD
# ==============================================================================
BVH_TO_STANDARD: Dict[str, str] = {
    "Hips": "Hips",
    "hip": "Hips",
    "Spine": "Spine",
    "Spine1": "Spine1",
    "Spine2": "Spine2",
    "Chest": "Spine1",
    "UpperChest": "Spine2",
    "Neck": "Neck",
    "Head": "Head",
    "Jaw": "Jaw",
    # Bras gauche
    "LeftCollar": "LeftShoulder",
    "LeftShoulder": "LeftShoulder",
    "LeftUpArm": "LeftArm",
    "LeftArm": "LeftArm",
    "LeftLowArm": "LeftForeArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    # Bras droit
    "RightCollar": "RightShoulder",
    "RightShoulder": "RightShoulder",
    "RightUpArm": "RightArm",
    "RightArm": "RightArm",
    "RightLowArm": "RightForeArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
    # Jambes
    "LeftUpLeg": "LeftUpLeg",
    "LeftLeg": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
    "RightUpLeg": "RightUpLeg",
    "RightLeg": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
}

# Doigts BVH standard
for side in ("Left", "Right"):
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for seg in (1, 2, 3):
            std_bone = f"{side}Hand{finger}{seg}"
            BVH_TO_STANDARD[std_bone] = std_bone
            BVH_TO_STANDARD[f"{side}_{finger}_{seg}"] = std_bone

STANDARD_TO_BVH: Dict[str, str] = {
    "Hips": "Hips",
    "Spine": "Spine",
    "Spine1": "Spine1",
    "Spine2": "Spine2",
    "Neck": "Neck",
    "Head": "Head",
    "Jaw": "Jaw",
    "LeftShoulder": "LeftCollar",
    "LeftArm": "LeftUpArm",
    "LeftForeArm": "LeftLowArm",
    "LeftHand": "LeftHand",
    "RightShoulder": "RightCollar",
    "RightArm": "RightUpArm",
    "RightForeArm": "RightLowArm",
    "RightHand": "RightHand",
    "LeftUpLeg": "LeftUpLeg",
    "LeftLeg": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
    "RightUpLeg": "RightUpLeg",
    "RightLeg": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
}

for side in ("Left", "Right"):
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for seg in (1, 2, 3):
            b_name = f"{side}Hand{finger}{seg}"
            STANDARD_TO_BVH[b_name] = b_name

"""Module mappings.buffer_armature
==============================
Ossature Tampon Canonique (Canonical Buffer Armature).
Définit un squelette pivot de 55 os dans une T-Pose géométrique absolue normalisée.
Sert d'espace de référence intermédiaire pour découpler n'importe quelle source
de n'importe quelle cible, éliminant les problèmes de A-Pose vs T-Pose et de repères locaux.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from mappings.standard_humanoid import (
    STANDARD_HUMANOID_PARENTS,
)


class CanonicalBufferArmature:
    """
    Ossature tampon intermédiaire universelle.
    Possède une géométrie métrique normalisée (hauteur adulte ~ 1.75m en T-Pose pure).
    """

    # Coordonnées 3D spatiales absolues (x, y, z) en mètres d'une T-Pose parfaite
    # Convention : X = Gauche (+X) / Droite (-X), Y = Haut (+Y), Z = Avant (+Z)
    CANONICAL_POSITIONS: Dict[str, np.ndarray] = {
        # Racine et colonne
        "Hips": np.array([0.0, 1.00, 0.0], dtype=np.float64),
        "Spine": np.array([0.0, 1.12, 0.0], dtype=np.float64),
        "Spine1": np.array([0.0, 1.25, 0.0], dtype=np.float64),
        "Spine2": np.array([0.0, 1.40, 0.0], dtype=np.float64),
        "Neck": np.array([0.0, 1.55, 0.0], dtype=np.float64),
        "Head": np.array([0.0, 1.65, 0.0], dtype=np.float64),
        "Jaw": np.array([0.0, 1.62, 0.08], dtype=np.float64),
        "LeftEye": np.array([0.035, 1.70, 0.08], dtype=np.float64),
        "RightEye": np.array([-0.035, 1.70, 0.08], dtype=np.float64),

        # Bras Gauche (déployé à l'horizontale pure selon +X)
        "LeftShoulder": np.array([0.06, 1.45, 0.0], dtype=np.float64),
        "LeftArm": np.array([0.18, 1.45, 0.0], dtype=np.float64),
        "LeftForeArm": np.array([0.46, 1.45, 0.0], dtype=np.float64),
        "LeftHand": np.array([0.72, 1.45, 0.0], dtype=np.float64),

        # Doigts Main Gauche (+X)
        "LeftHandThumb1": np.array([0.75, 1.43, 0.03], dtype=np.float64),
        "LeftHandThumb2": np.array([0.78, 1.42, 0.04], dtype=np.float64),
        "LeftHandThumb3": np.array([0.81, 1.41, 0.05], dtype=np.float64),
        "LeftHandIndex1": np.array([0.80, 1.46, 0.02], dtype=np.float64),
        "LeftHandIndex2": np.array([0.84, 1.46, 0.02], dtype=np.float64),
        "LeftHandIndex3": np.array([0.87, 1.46, 0.02], dtype=np.float64),
        "LeftHandMiddle1": np.array([0.80, 1.45, 0.0], dtype=np.float64),
        "LeftHandMiddle2": np.array([0.85, 1.45, 0.0], dtype=np.float64),
        "LeftHandMiddle3": np.array([0.88, 1.45, 0.0], dtype=np.float64),
        "LeftHandRing1": np.array([0.80, 1.44, -0.02], dtype=np.float64),
        "LeftHandRing2": np.array([0.84, 1.44, -0.02], dtype=np.float64),
        "LeftHandRing3": np.array([0.87, 1.44, -0.02], dtype=np.float64),
        "LeftHandPinky1": np.array([0.79, 1.43, -0.04], dtype=np.float64),
        "LeftHandPinky2": np.array([0.82, 1.43, -0.04], dtype=np.float64),
        "LeftHandPinky3": np.array([0.85, 1.43, -0.04], dtype=np.float64),

        # Bras Droit (déployé à l'horizontale pure selon -X)
        "RightShoulder": np.array([-0.06, 1.45, 0.0], dtype=np.float64),
        "RightArm": np.array([-0.18, 1.45, 0.0], dtype=np.float64),
        "RightForeArm": np.array([-0.46, 1.45, 0.0], dtype=np.float64),
        "RightHand": np.array([-0.72, 1.45, 0.0], dtype=np.float64),

        # Doigts Main Droite (-X)
        "RightHandThumb1": np.array([-0.75, 1.43, 0.03], dtype=np.float64),
        "RightHandThumb2": np.array([-0.78, 1.42, 0.04], dtype=np.float64),
        "RightHandThumb3": np.array([-0.81, 1.41, 0.05], dtype=np.float64),
        "RightHandIndex1": np.array([-0.80, 1.46, 0.02], dtype=np.float64),
        "RightHandIndex2": np.array([-0.84, 1.46, 0.02], dtype=np.float64),
        "RightHandIndex3": np.array([-0.87, 1.46, 0.02], dtype=np.float64),
        "RightHandMiddle1": np.array([-0.80, 1.45, 0.0], dtype=np.float64),
        "RightHandMiddle2": np.array([-0.85, 1.45, 0.0], dtype=np.float64),
        "RightHandMiddle3": np.array([-0.88, 1.45, 0.0], dtype=np.float64),
        "RightHandRing1": np.array([-0.80, 1.44, -0.02], dtype=np.float64),
        "RightHandRing2": np.array([-0.84, 1.44, -0.02], dtype=np.float64),
        "RightHandRing3": np.array([-0.87, 1.44, -0.02], dtype=np.float64),
        "RightHandPinky1": np.array([-0.79, 1.43, -0.04], dtype=np.float64),
        "RightHandPinky2": np.array([-0.82, 1.43, -0.04], dtype=np.float64),
        "RightHandPinky3": np.array([-0.85, 1.43, -0.04], dtype=np.float64),

        # Jambe Gauche (verticale le long de -Y)
        "LeftUpLeg": np.array([0.10, 0.95, 0.0], dtype=np.float64),
        "LeftLeg": np.array([0.10, 0.52, 0.0], dtype=np.float64),
        "LeftFoot": np.array([0.10, 0.10, 0.0], dtype=np.float64),
        "LeftToeBase": np.array([0.10, 0.02, 0.14], dtype=np.float64),

        # Jambe Droite (verticale le long de -Y)
        "RightUpLeg": np.array([-0.10, 0.95, 0.0], dtype=np.float64),
        "RightLeg": np.array([-0.10, 0.52, 0.0], dtype=np.float64),
        "RightFoot": np.array([-0.10, 0.10, 0.0], dtype=np.float64),
        "RightToeBase": np.array([-0.10, 0.02, 0.14], dtype=np.float64),
    }

    @classmethod
    def get_canonical_positions(cls) -> Dict[str, np.ndarray]:
        """Retourne une copie isolée des positions 3D canoniques."""
        return {k: v.copy() for k, v in cls.CANONICAL_POSITIONS.items()}

    @classmethod
    def get_canonical_parents(cls) -> Dict[str, Optional[str]]:
        """Retourne la hiérarchie parent-enfant standard."""
        return dict(STANDARD_HUMANOID_PARENTS)

    @classmethod
    def get_canonical_direction(cls, bone: str) -> Optional[np.ndarray]:
        """
        Calcule le vecteur unitaire directeur canonique d'un os vers son enfant principal.
        """
        parents = cls.get_canonical_parents()
        children = [b for b, p in parents.items() if p == bone]
        if not children:
            # Pour les extrémités (Head, ToeBase, Tips), direction par défaut
            if "Head" in bone or "Eye" in bone:
                return np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if "Toe" in bone or "Foot" in bone:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if "Left" in bone:
                return np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if "Right" in bone:
                return np.array([-1.0, 0.0, 0.0], dtype=np.float64)
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)

        child = children[0]
        pos_bone = cls.CANONICAL_POSITIONS[bone]
        pos_child = cls.CANONICAL_POSITIONS[child]
        diff = pos_child - pos_bone
        norm = float(np.linalg.norm(diff))
        if norm < 1e-6:
            return None
        return diff / norm

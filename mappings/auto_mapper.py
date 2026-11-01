"""Module mappings.auto_mapper
===========================
Moteur d'Auto-Identification et d'Alignement par Ossature Tampon.
Identifie automatiquement les os de rigs hétérogènes (Mixamo, Blender Rigify,
Unreal Engine, VRoid/VRM, CMU/Biovision BVH, SMPL-X) et calcule les pré-rotations
d'alignement de pose de repos (ex: A-Pose vers T-Pose).
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

from core.geometry import (
    quat_identity,
    quat_normalize,
    quat_mul,
    quat_inv,
    quat_from_two_vectors,
)
from mappings.standard_humanoid import (
    STANDARD_HUMANOID_BONES,
    STANDARD_HUMANOID_PARENTS,
)
from mappings.buffer_armature import CanonicalBufferArmature


class AutoBoneIdentifier:
    """
    Identificateur automatique sémantique et topologique d'os.
    Reconnaît les segments anatomiques maîtres et ignore intelligemment les os auxiliaires.
    """

    # Motifs d'extraction de côté (Gauche / Droite)
    LEFT_PATTERNS = [
        re.compile(r"left", re.IGNORECASE),
        re.compile(r"(?:^|[_\.\-:])l(?:[_\.\-:]|$)", re.IGNORECASE),
        re.compile(r"^l[A-Z]"),
        re.compile(r"[\._\-]l$", re.IGNORECASE),
    ]
    RIGHT_PATTERNS = [
        re.compile(r"right", re.IGNORECASE),
        re.compile(r"(?:^|[_\.\-:])r(?:[_\.\-:]|$)", re.IGNORECASE),
        re.compile(r"^r[A-Z]"),
        re.compile(r"[\._\-]r$", re.IGNORECASE),
    ]

    # Préfixes communs à nettoyer
    PREFIXES = [
        "mixamorig:", "mixamorig_", "mixamorig",
        "def-", "org-", "mch-", "j_bip_c_", "j_bip_l_", "j_bip_r_", "j_bip_",
        "bip01_", "bip01", "b_", "valvebiped_", "armature_", "root_"
    ]

    # Mots-clés pour le corps axial (ordre décroissant de spécificité)
    BODY_PATTERNS: List[Tuple[str, List[str]]] = [
        ("Hips", ["hips", "pelvis", "hip", "root_motion"]),
        ("Spine2", ["spine2", "upperchest", "upper_chest", "chest2", "mid_shoulder", "midshoulder"]),
        ("Spine1", ["spine1", "chest", "thorax"]),
        ("Spine", ["spine0", "spine", "abdomen", "torso"]),
        ("Neck", ["neck"]),
        ("Head", ["head"]),
        ("Jaw", ["jaw", "chin"]),
        ("LeftEye", ["lefteye", "eye_l", "eye.l"]),
        ("RightEye", ["righteye", "eye_r", "eye.r"]),
    ]

    # Ordre strict : os composés / spécifiques d'abord, os génériques ensuite
    LIMB_PATTERNS: List[Tuple[str, List[str]]] = [
        ("Shoulder", ["shoulder", "clavicle", "collar"]),
        # Avant-bras d'abord pour éviter que 'arm' ne capture 'forearm'
        ("ForeArm", ["forearm", "lowerarm", "lower_arm", "fore_arm", "ulna", "elbow"]),
        ("Arm", ["upperarm", "upper_arm", "up_arm", "bicep", "shldr", "arm"]),
        ("Hand", ["hand", "wrist", "palm"]),
        # Bas de jambe d'abord pour éviter que 'leg' ne capture 'lowerleg'
        ("Leg", ["lowerleg", "lower_leg", "lowleg", "low_leg", "calf", "shin", "knee"]),
        ("UpLeg", ["upleg", "upperleg", "upper_leg", "up_leg", "thigh"]),
        # Si 'leg' seul sans préfixe
        ("Leg", ["leg"]),
        ("ToeBase", ["toebase", "toe_base", "ball", "toes"]),
        ("Foot", ["foot", "ankle"]),
        ("ToeBase", ["toe"]),
    ]

    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    @classmethod
    def _strip_name(cls, raw_name: str) -> str:
        """Nettoie les préfixes et suffixes de nommage."""
        cleaned = raw_name.strip()
        lower = cleaned.lower()
        for p in cls.PREFIXES:
            if lower.startswith(p):
                cleaned = cleaned[len(p):]
                lower = cleaned.lower()
                break
        return cleaned

    @classmethod
    def _detect_side(cls, name: str) -> Optional[str]:
        """Détecte si un nom d'os appartient au côté 'Left' ou 'Right'."""
        for pat in cls.LEFT_PATTERNS:
            if pat.search(name):
                return "Left"
        for pat in cls.RIGHT_PATTERNS:
            if pat.search(name):
                return "Right"
        return None

    @classmethod
    def identify_bone(cls, raw_name: str) -> Optional[str]:
        """
        Associe un nom d'os arbitraire au nom standard pivot (55 os).
        Retourne None si l'os est un auxiliaire non standard (twist, vêtement, cheveu).
        """
        lower = raw_name.lower()
        # Exclusion immédiate des os de torsion et de physique
        if any(ign in lower for ign in ["twist", "cloth", "hair", "skirt", "prop", "weapon", "ik", "pole", "top_end", "_end", "nub"]):
            return None

        side = cls._detect_side(raw_name)

        # 1. Vérification des doigts si un côté est présent
        if side:
            for finger in cls.FINGER_NAMES:
                if finger.lower() in lower:
                    if "4" in lower or "tip" in lower:
                        return None
                    digit = None
                    for d in ["1", "2", "3"]:
                        if d in lower:
                            digit = d
                            break
                    if not digit:
                        if any(w in lower for w in ["proximal", "meta", "mcp", "cmc"]):
                            digit = "1"
                        elif any(w in lower for w in ["intermediate", "middle", "pip"]):
                            digit = "2"
                        elif any(w in lower for w in ["distal", "dip"]):
                            digit = "3"
                        else:
                            digit = "1"
                    return f"{side}Hand{finger}{digit}"

        # 2. Vérification des membres (bras / jambes) si un côté est présent
        if side:
            for std_limb, aliases in cls.LIMB_PATTERNS:
                for alias in aliases:
                    if alias in lower:
                        return f"{side}{std_limb}"

        # 3. Vérification des os du corps axial
        for std_name, aliases in cls.BODY_PATTERNS:
            for alias in aliases:
                if alias in lower:
                    if "eye" in alias and side:
                        return f"{side}Eye"
                    return std_name

        return None

    @classmethod
    def map_skeleton(cls, bone_names: List[str]) -> Dict[str, str]:
        """
        Analyse une liste de noms d'os et retourne le mapping {raw_bone_name: standard_bone_name}.
        """
        mapping: Dict[str, str] = {}
        used_standard: Set[str] = set()

        for raw in bone_names:
            std = cls.identify_bone(raw)
            if std and std not in used_standard:
                mapping[raw] = std
                used_standard.add(std)

        return mapping


class BufferRigMapper:
    """
    Gestionnaire de correspondance et d'alignement géométrique vers l'ossature tampon.
    Calcule les pré-rotations nécessaires pour absorber les différences de pose au repos
    (notamment la conversion A-Pose vers T-Pose).
    """

    def __init__(
        self,
        source_bones: List[str],
        target_bones: List[str],
        source_rest_positions: Optional[Dict[str, np.ndarray]] = None,
        target_rest_positions: Optional[Dict[str, np.ndarray]] = None,
        source_parents: Optional[Dict[str, Optional[str]]] = None,
        target_parents: Optional[Dict[str, Optional[str]]] = None,
    ):
        self.source_to_standard = AutoBoneIdentifier.map_skeleton(source_bones)
        self.target_to_standard = AutoBoneIdentifier.map_skeleton(target_bones)
        self.standard_to_target = {std: tgt for tgt, std in self.target_to_standard.items()}

        # Résolution anatomique : si la source a Shoulder mais pas Arm séparé (bras à 2 segments),
        # l'os 'Shoulder' source correspond à l'humérus (Arm)
        for side in ["Left", "Right"]:
            sh_std = f"{side}Shoulder"
            arm_std = f"{side}Arm"
            if arm_std not in self.source_to_standard.values():
                src_sh = [s for s, std in self.source_to_standard.items() if std == sh_std]
                if src_sh:
                    self.source_to_standard[src_sh[0]] = arm_std

        self.source_rest_pos = source_rest_positions or {}
        self.target_rest_pos = target_rest_positions or {}
        self.source_parents = source_parents or {}
        self.target_parents = target_parents or {}

        # Pré-rotations d'alignement de pose au repos
        self.source_align_rotations: Dict[str, np.ndarray] = {}
        self.target_align_rotations: Dict[str, np.ndarray] = {}

        self._compute_alignment_rotations()

    def _compute_alignment_rotations(self) -> None:
        """
        Calcule les quaternions Delta R d'alignement entre le vecteur repos de l'os
        et le vecteur directeur de l'ossature tampon canonique (T-Pose pure).
        """
        # Alignement Source -> Buffer
        for src_bone, std_bone in self.source_to_standard.items():
            u_canon = CanonicalBufferArmature.get_canonical_direction(std_bone)
            if u_canon is None:
                self.source_align_rotations[src_bone] = quat_identity()
                continue

            # Trouver le vecteur de l'os dans la source
            children = [b for b, p in self.source_parents.items() if p == src_bone]
            if children and src_bone in self.source_rest_pos and children[0] in self.source_rest_pos:
                p_b = np.asarray(self.source_rest_pos[src_bone], dtype=np.float64).reshape(-1)
                p_c = np.asarray(self.source_rest_pos[children[0]], dtype=np.float64).reshape(-1)
                diff = p_c - p_b
                norm = float(np.linalg.norm(diff))
                if norm > 1e-6:
                    u_src = (diff / norm).reshape(-1)
                    # Rotation amenant la direction source sur la direction canonique
                    self.source_align_rotations[src_bone] = quat_from_two_vectors(u_src, u_canon)
                    continue

            self.source_align_rotations[src_bone] = quat_identity()

    def get_source_to_target_bone_map(self) -> Dict[str, str]:
        """Retourne la table directe {source_bone: target_bone} résolue via le tampon."""
        direct: Dict[str, str] = {}
        for src_bone, std_bone in self.source_to_standard.items():
            if std_bone in self.standard_to_target:
                direct[src_bone] = self.standard_to_target[std_bone]
        return direct

    def align_source_rotation_to_buffer(self, src_bone: str, q_src: np.ndarray) -> np.ndarray:
        """Applique la pré-rotation de repos pour amener la rotation en espace tampon."""
        delta_r = self.source_align_rotations.get(src_bone, quat_identity())
        return quat_normalize(quat_mul(delta_r, q_src))

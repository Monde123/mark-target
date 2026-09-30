"""Module isolate.mappings.registry
================================
Registre et résolveur universel de mappings entre espaces d'os arbitraires.
Supporte nativement : SMPL-X (HybrIK-X), Mixamo (.glb), VRM (.vrm), BVH (MoCap) et Standard Humanoid.
"""

from __future__ import annotations
from typing import Dict, List, Optional

from mappings.standard_humanoid import (
    STANDARD_HUMANOID_BONES,
    MIXAMO_TO_STANDARD,
    STANDARD_TO_MIXAMO,
    VRM_TO_STANDARD,
    STANDARD_TO_VRM,
    SMPLX_TO_STANDARD,
    STANDARD_TO_SMPLX,
    BVH_TO_STANDARD,
    STANDARD_TO_BVH,
)


class MappingRegistry:
    """Registre de mappings configurables avec pont automatique via standard pivot."""

    _direct_mappings: Dict[str, Dict[str, str]] = {}

    @classmethod
    def register(cls, source_format: str, target_format: str, mapping: Dict[str, str]) -> None:
        """Enregistre un dictionnaire de mapping explicite prioritaire entre deux formats."""
        key = f"{source_format.lower()}->{target_format.lower()}"
        cls._direct_mappings[key] = mapping

    @classmethod
    def list_supported_formats(cls) -> List[str]:
        """Retourne la liste des formats supportés nativement."""
        return ["mixamo", "glb", "vrm", "smplx", "hybrik", "bvh", "standard"]

    @classmethod
    def _get_to_standard_map(cls, fmt: str) -> Optional[Dict[str, str]]:
        """Récupère la table de correspondance d'un format vers le standard pivot."""
        f = fmt.lower()
        if f in ("mixamo", "glb"):
            return MIXAMO_TO_STANDARD
        elif f == "vrm":
            return VRM_TO_STANDARD
        elif f in ("smplx", "hybrik", "pk"):
            return SMPLX_TO_STANDARD
        elif f in ("bvh", "cmu", "mocap"):
            return BVH_TO_STANDARD
        elif f == "standard":
            return {b: b for b in STANDARD_HUMANOID_BONES}
        return None

    @classmethod
    def _get_from_standard_map(cls, fmt: str) -> Optional[Dict[str, str]]:
        """Récupère la table de correspondance du standard pivot vers le format cible."""
        f = fmt.lower()
        if f in ("mixamo", "glb"):
            return STANDARD_TO_MIXAMO
        elif f == "vrm":
            return STANDARD_TO_VRM
        elif f in ("smplx", "hybrik", "pk"):
            return STANDARD_TO_SMPLX
        elif f in ("bvh", "cmu", "mocap"):
            return STANDARD_TO_BVH
        elif f == "standard":
            return {b: b for b in STANDARD_HUMANOID_BONES}
        return None

    @classmethod
    def get_mapping(cls, source_format: str, target_format: str) -> Dict[str, str]:
        """Résout le dictionnaire de mapping complet entre le format source et cible."""
        src = source_format.lower()
        tgt = target_format.lower()
        key = f"{src}->{tgt}"

        # 1. Correspondance directe enregistrée
        if key in cls._direct_mappings:
            return cls._direct_mappings[key]

        # 2. Cas trivial : même format
        if src == tgt:
            to_std = cls._get_to_standard_map(src)
            if to_std:
                return {k: k for k in to_std.keys()}
            return {}

        # 3. Résolution automatique via pivot standard (source -> standard -> cible)
        to_standard_map = cls._get_to_standard_map(src)
        from_standard_map = cls._get_from_standard_map(tgt)

        if to_standard_map and from_standard_map:
            resolved: Dict[str, str] = {}
            for src_bone, std_bone in to_standard_map.items():
                if std_bone in from_standard_map:
                    resolved[src_bone] = from_standard_map[std_bone]
            return resolved

        # Si formats non reconnus, retourne dictionnaire vide
        return {}

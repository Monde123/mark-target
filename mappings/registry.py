"""
Module isolate.mappings.registry
================================
Registre et résolveur universel de mappings entre espaces d'os arbitraires.
"""
from __future__ import annotations

from typing import Dict, Optional
from mappings.standard_humanoid import (
    MIXAMO_TO_STANDARD,
    STANDARD_TO_MIXAMO,
    VRM_TO_STANDARD,
    STANDARD_TO_VRM,
)


class MappingRegistry:
    """Registre de mappings configurables avec pont automatique via standard pivot."""

    _direct_mappings: Dict[str, Dict[str, str]] = {}

    @classmethod
    def register(cls, source_format: str, target_format: str, mapping: Dict[str, str]) -> None:
        key = f"{source_format.lower()}->{target_format.lower()}"
        cls._direct_mappings[key] = mapping

    @classmethod
    def get_mapping(cls, source_format: str, target_format: str) -> Dict[str, str]:
        src = source_format.lower()
        tgt = target_format.lower()
        key = f"{src}->{tgt}"

        # 1. Correspondance directe enregistrée
        if key in cls._direct_mappings:
            return cls._direct_mappings[key]

        # 2. Résolution automatique via pivot standard
        to_standard_map = None
        if src in ("mixamo", "glb"):
            to_standard_map = MIXAMO_TO_STANDARD
        elif src == "vrm":
            to_standard_map = VRM_TO_STANDARD

        from_standard_map = None
        if tgt in ("mixamo", "glb"):
            from_standard_map = STANDARD_TO_MIXAMO
        elif tgt == "vrm":
            from_standard_map = STANDARD_TO_VRM

        if to_standard_map and from_standard_map:
            resolved = {}
            for src_bone, std_bone in to_standard_map.items():
                if std_bone in from_standard_map:
                    resolved[src_bone] = from_standard_map[std_bone]
            return resolved

        # Si formats identiques ou non reconnus, identité
        return {}

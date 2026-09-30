"""Module isolate.mappings.registry
================================
Registre et résolveur universel de mappings entre espaces d'os arbitraires.
Supporte nativement : SMPL-X (HybrIK-X / Mimo), Mixamo (.glb), VRM (.vrm), BVH (MoCap) et Standard Humanoid.
Permet également le chargement dynamique de mappings personnalisés (JSON ou dictionnaires Python).
"""

from __future__ import annotations
import json
import os
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
        cls._direct_mappings[key] = dict(mapping)

    @classmethod
    def register_from_json(cls, json_path: str, source_format: str, target_format: str) -> Dict[str, str]:
        """
        Charge et enregistre un mapping personnalisé depuis un fichier JSON.
        
        Format attendu dans le fichier JSON :
        {
            "nom_os_source_1": "nom_os_cible_1",
            "nom_os_source_2": "nom_os_cible_2"
        }
        ou
        {
            "source_format": "custom_rig",
            "target_format": "mixamo",
            "mapping": { ... }
        }
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Fichier de mapping introuvable : {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mapping: Dict[str, str] = {}
        if isinstance(data, dict):
            if "mapping" in data and isinstance(data["mapping"], dict):
                mapping = data["mapping"]
                # Détection optionnelle des formats déclarés dans le JSON
                source_format = data.get("source_format", source_format)
                target_format = data.get("target_format", target_format)
            else:
                mapping = {str(k): str(v) for k, v in data.items() if not k.startswith("_")}
        else:
            raise ValueError(f"Le fichier JSON de mapping doit contenir un objet/dictionnaire, reçu {type(data)}")

        cls.register(source_format, target_format, mapping)
        return mapping

    @classmethod
    def export_to_json(cls, source_format: str, target_format: str, json_path: str) -> None:
        """Exporte le mapping actif résolu entre deux formats vers un fichier JSON."""
        resolved = cls.get_mapping(source_format, target_format)
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        payload = {
            "_description": f"Mapping résolu de {source_format} vers {target_format} (mark-target)",
            "source_format": source_format,
            "target_format": target_format,
            "bones_count": len(resolved),
            "mapping": resolved
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def list_supported_formats(cls) -> List[str]:
        """Retourne la liste des formats supportés nativement."""
        return ["mixamo", "glb", "vrm", "smplx", "hybrik", "pk", "bvh", "cmu", "standard"]

    @classmethod
    def _get_to_standard_map(cls, fmt: str) -> Optional[Dict[str, str]]:
        """Récupère la table de correspondance d'un format vers le standard pivot."""
        f = fmt.lower()
        if f in ("mixamo", "glb"):
            return MIXAMO_TO_STANDARD
        elif f == "vrm":
            return VRM_TO_STANDARD
        elif f in ("smplx", "hybrik", "pk", "mimo"):
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
        elif f in ("smplx", "hybrik", "pk", "mimo"):
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

        # 1. Correspondance directe enregistrée (priorité maximale, y compris customs)
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

"""
Module isolate.adapters.vrm_adapter
===================================
Adaptateur VRM (format basé sur glTF avec extension VRM Humanoid):
- Lit la bind pose et la hiérarchie humanoid VRM
- Permet le retargeting direct vers des avatars VRoid / VRM
"""
from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pygltflib import GLTF2

from adapters.base import TargetAdapter, TargetSkeleton
from adapters.gltf_animation_builder import export_rotation_animation_to_gltf
from core.geometry import quat_identity, quat_mul
from mappings.standard_humanoid import VRM_TO_STANDARD


class VRMAdapter(TargetAdapter):
    """Adaptateur pour les avatars 3D au format VRM."""

    def load_skeleton(self, target_model_path: str, **kwargs) -> TargetSkeleton:
        """Parse un avatar VRM et extrait les os humanoid standardisés."""
        gltf = GLTF2().load(target_model_path)
        nodes = gltf.nodes

        # Recherche de l'extension VRM dans les extensions du gltf
        vrm_ext = None
        if hasattr(gltf, "extensions") and gltf.extensions:
            vrm_ext = gltf.extensions.get("VRM") or gltf.extensions.get("VRMC_vrm")

        # Map index de noeud -> nom standard
        node_to_vrm_bone: Dict[int, str] = {}
        if vrm_ext and "humanoid" in vrm_ext:
            human_bones = vrm_ext["humanoid"].get("humanBones", [])
            for hb in human_bones:
                bone_type = hb.get("bone")
                node_idx = hb.get("node")
                if bone_type and node_idx is not None:
                    node_to_vrm_bone[node_idx] = bone_type

        # Parents et rotations locales
        node_local_quat: Dict[int, np.ndarray] = {}
        parent_of_idx: Dict[int, int] = {}
        for i, node in enumerate(nodes):
            for child in (node.children or []):
                parent_of_idx[child] = i
            if node.rotation is None:
                node_local_quat[i] = quat_identity()
            else:
                x, y, z, w = node.rotation
                node_local_quat[i] = np.array([w, x, y, z], dtype=np.float64)

        cache_q: Dict[int, np.ndarray] = {}

        def global_pose(i: int) -> np.ndarray:
            if i in cache_q:
                return cache_q[i]
            if i not in parent_of_idx:
                q = node_local_quat[i]
            else:
                q = quat_mul(global_pose(parent_of_idx[i]), node_local_quat[i])
            cache_q[i] = q
            return q

        Q_rest: Dict[str, np.ndarray] = {}
        parents: Dict[str, Optional[str]] = {}
        raw_names: Dict[str, str] = {}

        for n_idx, vrm_bone in node_to_vrm_bone.items():
            Q_rest[vrm_bone] = global_pose(n_idx)
            raw_names[vrm_bone] = nodes[n_idx].name or f"node_{n_idx}"

            p_idx = parent_of_idx.get(n_idx)
            while p_idx is not None and p_idx not in node_to_vrm_bone:
                p_idx = parent_of_idx.get(p_idx)

            if p_idx is not None and p_idx in node_to_vrm_bone:
                parents[vrm_bone] = node_to_vrm_bone[p_idx]
            else:
                parents[vrm_bone] = None

        return TargetSkeleton(
            rest_rotations=Q_rest,
            parents=parents,
            raw_names=raw_names,
            root_name="hips",
            fps=30.0,
        )

    def export_animation(
        self,
        target_model_path: str,
        output_path: str,
        animation_clip: List[Dict[str, Any]],
        skeleton: TargetSkeleton,
        **kwargs
    ) -> None:
        """Exporte l'animation sur le modèle VRM."""
        gltf = GLTF2().load(target_model_path)
        node_map: Dict[str, int] = {}
        for idx, node in enumerate(gltf.nodes or []):
            if node.name:
                node_map[node.name] = idx

        # Les clés d'animation pour VRM suivent les noms humanoid (ex: hips, spine...)
        # raw_names relie ces clés au vrai nom de node glTF.
        raw_names = skeleton.raw_names or {}

        def node_lookup(bone_name: str) -> Optional[int]:
            raw = raw_names.get(bone_name)
            if raw and raw in node_map:
                return node_map[raw]
            return node_map.get(bone_name)

        export_rotation_animation_to_gltf(
            gltf=gltf,
            output_path=output_path,
            animation_clip=animation_clip,
            skeleton_fps=skeleton.fps,
            bone_names=list(skeleton.parents.keys()),
            node_lookup=node_lookup,
            animation_name="marker_animation",
        )

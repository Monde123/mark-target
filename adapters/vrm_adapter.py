"""
Module isolate.adapters.vrm_adapter
===================================
Adaptateur VRM (format basé sur glTF avec extension VRM Humanoid):
- Lit la bind pose et la hiérarchie humanoid VRM
- Permet le retargeting direct vers des avatars VRoid / VRM
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Any, List
from pygltflib import GLTF2

from adapters.base import TargetAdapter, TargetSkeleton
from adapters.gltf_animation_utils import export_quaternion_animation_to_gltf
from core.geometry import quat_identity, quat_mul
from mappings.standard_humanoid import VRM_TO_STANDARD


class VRMAdapter(TargetAdapter):
    """Adaptateur pour les avatars 3D au format VRM."""

    def load_skeleton(self, target_model_path: str, **kwargs) -> TargetSkeleton:
        """Parse un avatar VRM et extrait les os humanoid standardisés."""
        gltf = GLTF2().load(target_model_path)
        nodes = gltf.nodes or []

        vrm_ext = None
        if hasattr(gltf, "extensions") and gltf.extensions:
            vrm_ext = gltf.extensions.get("VRM") or gltf.extensions.get("VRMC_vrm")

        node_to_vrm_bone: Dict[int, str] = {}
        if vrm_ext:
            humanoid = vrm_ext.get("humanoid", {})
            human_bones = humanoid.get("humanBones", {})

            if isinstance(human_bones, list):
                for hb in human_bones:
                    bone_type = hb.get("bone")
                    node_idx = hb.get("node")
                    if bone_type and node_idx is not None:
                        node_to_vrm_bone[int(node_idx)] = bone_type
            elif isinstance(human_bones, dict):
                for bone_type, payload in human_bones.items():
                    if not isinstance(payload, dict):
                        continue
                    node_idx = payload.get("node")
                    if node_idx is not None:
                        node_to_vrm_bone[int(node_idx)] = bone_type

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
                q = node_local_quat.get(i, quat_identity())
            else:
                q = quat_mul(global_pose(parent_of_idx[i]), node_local_quat.get(i, quat_identity()))
            cache_q[i] = q
            return q

        q_rest: Dict[str, np.ndarray] = {}
        parents: Dict[str, Optional[str]] = {}
        raw_names: Dict[str, str] = {}

        for n_idx, vrm_bone in node_to_vrm_bone.items():
            std_name = VRM_TO_STANDARD.get(vrm_bone, vrm_bone)
            q_rest[std_name] = global_pose(n_idx)
            raw_names[std_name] = nodes[n_idx].name or f"node_{n_idx}"

            p_idx = parent_of_idx.get(n_idx)
            while p_idx is not None and p_idx not in node_to_vrm_bone:
                p_idx = parent_of_idx.get(p_idx)

            if p_idx is not None and p_idx in node_to_vrm_bone:
                parent_vrm = node_to_vrm_bone[p_idx]
                parents[std_name] = VRM_TO_STANDARD.get(parent_vrm, parent_vrm)
            else:
                parents[std_name] = None

        root_name = "Hips"
        if root_name not in q_rest and q_rest:
            root_name = next((bone for bone, p in parents.items() if p is None), next(iter(q_rest.keys())))

        return TargetSkeleton(
            rest_rotations=q_rest,
            parents=parents,
            raw_names=raw_names,
            root_name=root_name,
            fps=30.0,
        )

    def export_animation(
        self,
        target_model_path: str,
        output_path: str,
        animation_clip: List[Dict[str, Any]],
        skeleton: TargetSkeleton,
        animation_name: str = "marker_animation",
        **kwargs
    ) -> None:
        """Exporte l'animation sur le modèle VRM en conservant les extensions existantes."""
        gltf = GLTF2().load(target_model_path)

        node_map: Dict[str, int] = {}
        std_to_raw = skeleton.raw_names or {}
        raw_to_std = {raw: std for std, raw in std_to_raw.items()}

        for idx, node in enumerate(gltf.nodes or []):
            if node.name:
                node_map[node.name] = idx
                if node.name in raw_to_std:
                    node_map[raw_to_std[node.name]] = idx

        for std_name, raw_name in std_to_raw.items():
            if raw_name in node_map:
                node_map[std_name] = node_map[raw_name]

        export_quaternion_animation_to_gltf(
            gltf=gltf,
            output_path=output_path,
            animation_clip=animation_clip,
            skeleton=skeleton,
            node_map=node_map,
            animation_name=animation_name,
        )

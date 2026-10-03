"""
Module isolate.adapters.gltf_mixamo_adapter
===========================================
Adaptateur glTF / GLB spécifique pour avatars Mixamo:
- Extrait la bind pose exacte et l'arbre squelettique sans ambiguïté
- Injecte l'animation résultante et génère le fichier .glb final
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pygltflib import GLTF2

from adapters.base import TargetAdapter, TargetSkeleton
from adapters.gltf_animation_builder import export_rotation_animation_to_gltf
from core.geometry import quat_identity, quat_mul


class GLTFMixamoAdapter(TargetAdapter):
    """Adaptateur pour modèles Mixamo (.glb / .gltf)."""

    @staticmethod
    def canonical_bone_name(raw_name: str) -> str:
        """Standardise les noms d'os Mixamo en retirant les préfixes variables."""
        if ":" in raw_name:
            return "mixamorig:" + raw_name.split(":")[-1]
        return raw_name

    def load_skeleton(self, target_model_path: str, **kwargs) -> TargetSkeleton:
        """Lit la bind pose et la hiérarchie directement depuis le fichier GLB."""
        gltf = GLTF2().load(target_model_path)
        nodes = gltf.nodes

        node_name: Dict[int, str] = {}
        node_local_quat: Dict[int, np.ndarray] = {}
        node_local_trans: Dict[int, np.ndarray] = {}

        for i, node in enumerate(nodes):
            name = node.name or f"node_{i}"
            canon = self.canonical_bone_name(name)
            node_name[i] = canon
            if node.rotation is None:
                node_local_quat[i] = quat_identity()
            else:
                x, y, z, w = node.rotation
                node_local_quat[i] = np.array([w, x, y, z], dtype=np.float64)
            node_local_trans[i] = np.array(node.translation) if node.translation else np.zeros(3)

        parent_of_idx: Dict[int, int] = {}
        for i, node in enumerate(nodes):
            for child in (node.children or []):
                parent_of_idx[child] = i

        if not gltf.skins:
            raise ValueError(f"Aucun skin trouvé dans le fichier {target_model_path}")

        joint_indices = sorted({j for skin in gltf.skins for j in skin.joints})

        cache_q: Dict[int, np.ndarray] = {}

        def global_pose(i: int) -> np.ndarray:
            if i in cache_q:
                return cache_q[i]
            if i not in parent_of_idx:
                q = node_local_quat[i]
            else:
                q_parent = global_pose(parent_of_idx[i])
                q = quat_mul(q_parent, node_local_quat[i])
            cache_q[i] = q
            return q

        Q_rest: Dict[str, np.ndarray] = {}
        parents: Dict[str, Optional[str]] = {}
        canonical_to_raw: Dict[str, str] = {}

        for j_idx in joint_indices:
            c_name = node_name[j_idx]
            raw_name = nodes[j_idx].name or c_name
            Q_rest[c_name] = global_pose(j_idx)
            canonical_to_raw[c_name] = raw_name

            p_idx = parent_of_idx.get(j_idx)
            if p_idx is not None and p_idx in node_name:
                parents[c_name] = node_name[p_idx]
            else:
                parents[c_name] = None

        return TargetSkeleton(
            rest_rotations=Q_rest,
            parents=parents,
            raw_names=canonical_to_raw,
            root_name="mixamorig:Hips",
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
        """Exporte l'animation résolue dans le GLB cible."""
        gltf = GLTF2().load(target_model_path)
        node_map: Dict[str, int] = {}
        for idx, node in enumerate(gltf.nodes or []):
            if not node.name:
                continue
            node_map[node.name] = idx
            node_map[self.canonical_bone_name(node.name)] = idx

        export_rotation_animation_to_gltf(
            gltf=gltf,
            output_path=output_path,
            animation_clip=animation_clip,
            skeleton_fps=skeleton.fps,
            bone_names=list(skeleton.parents.keys()),
            node_lookup=lambda b: node_map.get(b),
            animation_name=animation_name,
        )

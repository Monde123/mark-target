"""
Module adapters.vrm_adapter
===================================
Adaptateur VRM (format basé sur glTF avec extension VRM Humanoid):
- Lit la bind pose et la hiérarchie humanoid VRM
- Sérialise et injecte les canaux d'animation dans le conteneur VRM (glTF)
"""
from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pygltflib import (
    GLTF2,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Accessor,
    BufferView,
)

from adapters.base import TargetAdapter, TargetSkeleton
from core.geometry import quat_identity, quat_mul, quat_normalize
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
            std_name = VRM_TO_STANDARD.get(vrm_bone, vrm_bone)
            Q_rest[std_name] = global_pose(n_idx)
            raw_names[std_name] = nodes[n_idx].name or f"node_{n_idx}"

            p_idx = parent_of_idx.get(n_idx)
            while p_idx is not None and p_idx not in node_to_vrm_bone:
                p_idx = parent_of_idx.get(p_idx)

            if p_idx is not None and p_idx in node_to_vrm_bone:
                parent_vrm = node_to_vrm_bone[p_idx]
                parents[std_name] = VRM_TO_STANDARD.get(parent_vrm, parent_vrm)
            else:
                parents[std_name] = None

        return TargetSkeleton(
            rest_rotations=Q_rest,
            parents=parents,
            raw_names=raw_names,
            root_name="Hips",
            fps=30.0,
        )

    def export_animation(
        self,
        target_model_path: str,
        output_path: str,
        animation_clip: List[Dict[str, Any]],
        skeleton: TargetSkeleton,
        animation_name: str = "vrm_animation",
        frame_time: Optional[float] = None,
        **kwargs
    ) -> None:
        """Exporte l'animation sur le modèle VRM avec de vraies pistes d'animation glTF."""
        gltf = GLTF2().load(target_model_path)
        if not animation_clip:
            gltf.save(output_path)
            return

        fps = skeleton.fps if (skeleton.fps and skeleton.fps > 0) else 30.0
        if frame_time is not None and frame_time > 0:
            fps = 1.0 / frame_time
        n_frames = len(animation_clip)
        time_array = np.arange(n_frames, dtype=np.float32) / fps

        node_map: Dict[str, int] = {}
        for idx, node in enumerate(gltf.nodes):
            if node.name:
                node_map[node.name] = idx

        # Association via raw_names du squelette
        for std_name, raw_n in skeleton.raw_names.items():
            if raw_n in node_map:
                node_map[std_name] = node_map[raw_n]

        blob = bytearray(gltf.binary_blob() or b"")
        pad = (4 - (len(blob) % 4)) % 4
        blob.extend(b"\x00" * pad)

        time_bytes = time_array.tobytes()
        time_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            BufferView(buffer=0, byteOffset=len(blob), byteLength=len(time_bytes))
        )
        blob.extend(time_bytes)
        pad = (4 - (len(blob) % 4)) % 4
        blob.extend(b"\x00" * pad)

        time_acc_idx = len(gltf.accessors)
        gltf.accessors.append(
            Accessor(
                bufferView=time_bv_idx,
                byteOffset=0,
                componentType=5126,
                count=n_frames,
                type="SCALAR",
                min=[float(time_array[0])],
                max=[float(time_array[-1])],
            )
        )

        anim = Animation(name=animation_name)
        sampler_idx = 0

        all_bones = set()
        for f in animation_clip:
            all_bones.update(f.get("bones", {}).keys())

        for bone_name in sorted(all_bones):
            node_idx = node_map.get(bone_name)
            if node_idx is None:
                continue

            quats_xyzw = []
            for f in animation_clip:
                q = f["bones"].get(bone_name, quat_identity())
                quats_xyzw.append([q[1], q[2], q[3], q[0]])

            quats_np = np.array(quats_xyzw, dtype=np.float32)
            quat_bytes = quats_np.tobytes()

            quat_bv_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(
                BufferView(buffer=0, byteOffset=len(blob), byteLength=len(quat_bytes))
            )
            blob.extend(quat_bytes)
            pad = (4 - (len(blob) % 4)) % 4
            blob.extend(b"\x00" * pad)

            quat_acc_idx = len(gltf.accessors)
            gltf.accessors.append(
                Accessor(
                    bufferView=quat_bv_idx,
                    byteOffset=0,
                    componentType=5126,
                    count=n_frames,
                    type="VEC4",
                )
            )

            anim.samplers.append(
                AnimationSampler(
                    input=time_acc_idx,
                    output=quat_acc_idx,
                    interpolation="LINEAR",
                )
            )
            anim.channels.append(
                AnimationChannel(
                    sampler=sampler_idx,
                    target=AnimationChannelTarget(node=node_idx, path="rotation"),
                )
            )
            sampler_idx += 1

        if anim.channels:
            gltf.animations.append(anim)

        if gltf.buffers:
            gltf.buffers[0].byteLength = len(blob)
        gltf.set_binary_blob(bytes(blob))
        gltf.save(output_path)

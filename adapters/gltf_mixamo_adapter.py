"""
Module adapters.gltf_mixamo_adapter
===========================================
Adaptateur glTF / GLB spécifique pour avatars Mixamo:
- Extrait la bind pose exacte et l'arbre squelettique sans ambiguïté
- Sérialise et injecte les canaux d'animation réels dans le conteneur glTF binaire
- Génère un fichier .glb final entièrement jouable dans Blender, Unreal, Three.js
"""
from __future__ import annotations

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
from core.geometry import quat_identity, quat_mul, quat_normalize, rotate_vector


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
        bind_local: Dict[str, np.ndarray] = {}
        parents: Dict[str, Optional[str]] = {}
        canonical_to_raw: Dict[str, str] = {}

        for j_idx in joint_indices:
            c_name = node_name[j_idx]
            raw_name = nodes[j_idx].name or c_name
            Q_rest[c_name] = global_pose(j_idx)
            bind_local[c_name] = node_local_quat[j_idx]
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
            bind_local_rotations=bind_local,
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
        frame_time: Optional[float] = None,
        **kwargs
    ) -> None:
        """Exporte l'animation résolue dans le GLB cible sous forme de vraies pistes glTF."""
        gltf = GLTF2().load(target_model_path)
        if not animation_clip:
            gltf.save(output_path)
            return

        fps = skeleton.fps if (skeleton.fps and skeleton.fps > 0) else 30.0
        if frame_time is not None and frame_time > 0:
            fps = 1.0 / frame_time
        n_frames = len(animation_clip)
        time_array = np.arange(n_frames, dtype=np.float32) / fps

        # Résolution des noeuds par nom
        node_map: Dict[str, int] = {}
        for idx, node in enumerate(gltf.nodes):
            if node.name:
                node_map[node.name] = idx
                c_name = self.canonical_bone_name(node.name)
                node_map[c_name] = idx

        # Récupération et alignement du binary blob
        blob = bytearray(gltf.binary_blob() or b"")
        pad = (4 - (len(blob) % 4)) % 4
        blob.extend(b"\x00" * pad)

        # 1. Accessor des timestamps (SCALAR, float32)
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
                componentType=5126,  # FLOAT
                count=n_frames,
                type="SCALAR",
                min=[float(time_array[0])],
                max=[float(time_array[-1])],
            )
        )

        anim = Animation(name=animation_name)
        sampler_idx = 0

        # Identifier tous les os présents dans les frames
        all_bones = set()
        for f in animation_clip:
            all_bones.update(f.get("bones", {}).keys())

        for bone_name in sorted(all_bones):
            # Retrouver l'indice du noeud glTF
            node_idx = node_map.get(bone_name)
            if node_idx is None:
                raw_name = skeleton.raw_names.get(bone_name)
                if raw_name:
                    node_idx = node_map.get(raw_name)
            if node_idx is None:
                continue

            # Extraire la trajectoire de quaternions [x, y, z, w]
            # Dans animation_clip, quat est [w, x, y, z]
            # Pose de repos locale de secours pour préserver le bind pose
            node_obj = gltf.nodes[node_idx]
            if node_obj.rotation is not None:
                r = node_obj.rotation
                bind_q = np.array([r[3], r[0], r[1], r[2]], dtype=np.float64)
            elif skeleton.bind_local_rotations and bone_name in skeleton.bind_local_rotations:
                bind_q = skeleton.bind_local_rotations[bone_name]
            else:
                bind_q = quat_identity()

            quats_xyzw = []
            for f in animation_clip:
                q = f["bones"].get(bone_name)
                if q is None:
                    q = bind_q
                # Conversion [w, x, y, z] -> convention glTF [x, y, z, w]
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
                    componentType=5126,  # FLOAT
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

        # 2. Canal de translation pour la racine si mouvement
        root_node_idx = node_map.get(skeleton.root_name)
        if root_node_idx is not None:
            trans_list = []
            has_movement = False
            for f in animation_clip:
                r_pos = f.get("root", [0.0, 0.0, 0.0])
                trans_list.append(r_pos)
                if any(abs(v) > 1e-4 for v in r_pos):
                    has_movement = True

            if has_movement:
                trans_np = np.array(trans_list, dtype=np.float32)
                trans_bytes = trans_np.tobytes()
                trans_bv_idx = len(gltf.bufferViews)
                gltf.bufferViews.append(
                    BufferView(buffer=0, byteOffset=len(blob), byteLength=len(trans_bytes))
                )
                blob.extend(trans_bytes)
                pad = (4 - (len(blob) % 4)) % 4
                blob.extend(b"\x00" * pad)

                trans_acc_idx = len(gltf.accessors)
                gltf.accessors.append(
                    Accessor(
                        bufferView=trans_bv_idx,
                        byteOffset=0,
                        componentType=5126,
                        count=n_frames,
                        type="VEC3",
                        min=trans_np.min(axis=0).tolist(),
                        max=trans_np.max(axis=0).tolist(),
                    )
                )

                anim.samplers.append(
                    AnimationSampler(
                        input=time_acc_idx,
                        output=trans_acc_idx,
                        interpolation="LINEAR",
                    )
                )
                anim.channels.append(
                    AnimationChannel(
                        sampler=sampler_idx,
                        target=AnimationChannelTarget(node=root_node_idx, path="translation"),
                    )
                )
                sampler_idx += 1

        if anim.channels:
            gltf.animations.append(anim)

        # Mettre à jour la longueur du buffer 0 et le blob binaire
        if gltf.buffers:
            gltf.buffers[0].byteLength = len(blob)
        gltf.set_binary_blob(bytes(blob))

        gltf.save(output_path)

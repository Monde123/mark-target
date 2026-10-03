"""Script CLI Universel de Retargeting par Marqueurs
==================================================
Permet d'exécuter la méthode des marqueurs de façon découplée :
- Sources supportées : .pk (HybrIK/SMPL-X), .bvh, .glb
- Cibles supportées : .glb (Mixamo), .vrm (VRoid/VRM), .bvh

Exemples d'utilisation :
    python run_marker_retarget.py --source anim.bvh --source-type bvh --target out.bvh --target-type bvh --output res.bvh
    python run_marker_retarget.py --source anim.bvh --source-type bvh --target avatar.glb --target-type mixamo --output out.glb
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Any
import numpy as np

from core.geometry import quat_identity
from retargeting.solver import MarkerRetargetSolver
from retargeting.kinematics import global_to_local_hierarchy, apply_euler_correction_to_root
from mappings.registry import MappingRegistry
from adapters.bvh_adapter import BVHAdapter


def get_target_adapter(target_type: str):
    if target_type == "bvh":
        return BVHAdapter()
    elif target_type == "mixamo":
        try:
            from adapters.gltf_mixamo_adapter import GLTFMixamoAdapter
            return GLTFMixamoAdapter()
        except ImportError:
            raise ImportError("Le support Mixamo (.glb) nécessite pygltflib : installez-le via `pip install pygltflib`")
    elif target_type == "vrm":
        try:
            from adapters.vrm_adapter import VRMAdapter
            return VRMAdapter()
        except ImportError:
            raise ImportError("Le support VRM (.vrm) nécessite pygltflib : installez-le via `pip install pygltflib`")
    else:
        raise ValueError(f"Format cible non supporté: {target_type}")


def get_source_adapter(source_type: str):
    if source_type == "bvh":
        return BVHAdapter()
    elif source_type == "pk":
        try:
            from adapters.hybrik_pk_adapter import HybrIKPKAdapter
            return HybrIKPKAdapter()
        except ImportError:
            raise ImportError("Le format source .pk nécessite numpy et pickle.")
    else:
        raise ValueError(f"Format source non supporté: {source_type}")


def main():
    parser = argparse.ArgumentParser(description="Retargeting universel par marqueurs 3D.")
    parser.add_argument("--source", required=True, help="Chemin vers le fichier source (.pk, .bvh, .glb)")
    parser.add_argument("--source-type", choices=["pk", "bvh", "mixamo", "vrm"], required=True, help="Format source")
    parser.add_argument("--target", required=True, help="Chemin vers le modèle cible (.glb, .vrm, .bvh)")
    parser.add_argument("--target-type", choices=["mixamo", "vrm", "bvh"], required=True, help="Format cible")
    parser.add_argument("--output", required=True, help="Chemin de sortie pour l'animation")
    parser.add_argument("--root-x-degrees", type=float, default=0.0, help="Correction axiale X sur le root")
    parser.add_argument("--custom-mapping", type=str, default=None, help="Chemin vers un fichier JSON de mapping personnalise")
    parser.add_argument("--use-umeyama", action="store_true", default=False, help="Active le solveur Kabsch-Umeyama avec absorption d echelle uniforme")
    parser.add_argument("--smoothing-factor", type=float, default=0.0, help="Facteur de lissage temporel Bezier/SLERP (0.0=aucun, 0.3=recommande)")
    parser.add_argument("--use-ransac", action="store_true", default=False, help="Active le filtrage robuste RANSAC contre les occlusions et marqueurs aberrants")
    parser.add_argument("--max-strain", type=float, default=None, help="Seuil d energie de deformation Green-Lagrange max tolerable avant fallback")

    args = parser.parse_args()

    if args.custom_mapping:
        MappingRegistry.register_from_json(args.custom_mapping, args.source_type, args.target_type)
        print(f"[mark-target] Mapping personnalisé appliqué depuis {args.custom_mapping}")

    print(f"[mark-target] Chargement du squelette cible ({args.target_type}): {args.target}")
    target_adapter = get_target_adapter(args.target_type)
    skeleton = target_adapter.load_skeleton(args.target)
    print(f"[mark-target] {len(skeleton.rest_rotations)} os détectés dans le squelette cible.")

    print(f"[mark-target] Chargement de la source ({args.source_type}): {args.source}")
    source_adapter = get_source_adapter(args.source_type)
    sequence = source_adapter.load(args.source)
    print(f"[mark-target] Source chargée : {len(sequence.frames_markers)} frames de marqueurs.")

    mapping = MappingRegistry.get_mapping(args.source_type, args.target_type)
    print(f"[mark-target] Table de mapping résolue : {len(mapping)} correspondances.")

    solver = MarkerRetargetSolver(
        rest_rotations=skeleton.rest_rotations,
        parents=skeleton.parents,
    )

    animation_clip = []
    root_rotations = sequence.root_rotations or []
    root_translations = None
    if sequence.root_translations is not None:
        arr = np.asarray(sequence.root_translations, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size == 3:
                arr = arr.reshape(1, 3)
            elif arr.size % 3 == 0:
                arr = arr.reshape(-1, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"[mark-target] root_translations invalide: attendu (n_frames, 3), reçu {arr.shape}"
            )
        root_translations = arr

    n_frames = max(
        len(sequence.frames_markers),
        len(root_rotations),
        0 if root_translations is None else int(root_translations.shape[0]),
    )
    if n_frames <= 0:
        raise ValueError("[mark-target] Aucune frame exploitable dans la source.")

    for frame_idx in range(n_frames):
        frame_markers = sequence.frames_markers[frame_idx] if frame_idx < len(sequence.frames_markers) else {}
        global_rotations = solver.solve_frame(
            frame_markers,
            sequence.rest_markers or {},
            mapping,
            use_umeyama=args.use_umeyama,
            use_ransac=args.use_ransac,
            max_strain=args.max_strain,
        )

        root_rot = root_rotations[frame_idx] if frame_idx < len(root_rotations) else None
        if root_rot is not None:
            global_rotations[skeleton.root_name] = root_rot

        local_rotations = global_to_local_hierarchy(
            global_rotations,
            skeleton.parents,
        )
        if args.root_x_degrees != 0.0:
            local_rotations = apply_euler_correction_to_root(
                local_rotations,
                root_name=skeleton.root_name,
                axis="x",
                degrees=args.root_x_degrees,
            )

        frame_data: Dict[str, Any] = {"rotations": local_rotations}
        if root_translations is not None:
            idx = frame_idx if frame_idx < root_translations.shape[0] else root_translations.shape[0] - 1
            frame_data["root"] = root_translations[idx].copy()
        animation_clip.append(frame_data)

    if args.smoothing_factor > 0.0:
        from retargeting.smoothing import smooth_quaternion_trajectory, smooth_translation_trajectory

        for bone_name in skeleton.parents.keys():
            bone_trajectory = [f["rotations"].get(bone_name, quat_identity()) for f in animation_clip]
            smoothed = smooth_quaternion_trajectory(bone_trajectory, args.smoothing_factor)
            for f_idx, s_rot in enumerate(smoothed):
                animation_clip[f_idx]["rotations"][bone_name] = s_rot

        if root_translations is not None:
            root_traj = [f.get("root", np.zeros(3, dtype=np.float64)) for f in animation_clip]
            window_size = 3 if args.smoothing_factor < 0.5 else 5
            smoothed_root = smooth_translation_trajectory(root_traj, window_size=window_size)
            for f_idx, t_root in enumerate(smoothed_root):
                animation_clip[f_idx]["root"] = np.asarray(t_root, dtype=np.float64)
        print(f"[mark-target] Lissage temporel SQUAD appliqué (facteur={args.smoothing_factor}).")

    target_adapter.export_animation(
        target_model_path=args.target,
        output_path=args.output,
        animation_clip=animation_clip,
        skeleton=skeleton,
    )
    print(f"[mark-target] Animation exportée avec succès -> {args.output}")


if __name__ == "__main__":
    main()

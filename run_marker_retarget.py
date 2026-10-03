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

from core.geometry import quat_identity
from retargeting.solver import MarkerRetargetSolver
from retargeting.kinematics import global_to_local_hierarchy, apply_axis_correction_to_quaternion
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
    root_translations = sequence.root_translations
    n_root_trans = len(root_translations) if root_translations is not None else 0
    n_frames = max(len(sequence.frames_markers), len(root_rotations), n_root_trans)

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

        root_rot = root_rotations[min(frame_idx, len(root_rotations) - 1)] if root_rotations else None
        if root_rot is not None:
            if args.root_x_degrees != 0.0:
                root_rot = apply_axis_correction_to_quaternion(root_rot, axis="x", degrees=args.root_x_degrees)
            global_rotations[skeleton.root_name] = root_rot
        elif args.root_x_degrees != 0.0:
            current_root = global_rotations.get(skeleton.root_name, quat_identity())
            global_rotations[skeleton.root_name] = apply_axis_correction_to_quaternion(
                current_root,
                axis="x",
                degrees=args.root_x_degrees,
            )

        local_rotations = global_to_local_hierarchy(
            global_rotations,
            skeleton.parents,
        )

        frame_data: Dict[str, Any] = {"rotations": local_rotations}
        if root_translations is not None and len(root_translations) > 0:
            root_t_idx = min(frame_idx, len(root_translations) - 1)
            frame_data["root"] = root_translations[root_t_idx]
        animation_clip.append(frame_data)

    if args.smoothing_factor > 0.0:
        from retargeting.smoothing import smooth_quaternion_trajectory
        for bone_name in skeleton.parents.keys():
            bone_trajectory = [f["rotations"].get(bone_name, quat_identity()) for f in animation_clip]
            smoothed = smooth_quaternion_trajectory(bone_trajectory, args.smoothing_factor)
            for f_idx, s_rot in enumerate(smoothed):
                animation_clip[f_idx]["rotations"][bone_name] = s_rot
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

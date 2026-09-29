"""
Script CLI Universel de Retargeting par Marqueurs
==================================================
Permet d'exécuter la méthode des marqueurs de façon découplée :
- Sources supportées : .pk (HybrIK/SMPL-X), .bvh, .glb
- Cibles supportées : .glb (Mixamo), .vrm (VRoid/VRM), .bvh

Exemples d'utilisation :
    python -m isolate.run_marker_retarget --source output/res.pk --source-type pk --target clara.glb --target-type mixamo --output out_markers.glb
    python -m isolate.run_marker_retarget --source anim.bvh --source-type bvh --target avatar.vrm --target-type vrm --output out_avatar.vrm
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, Any

from core.geometry import quat_identity
from retargeting.solver import MarkerRetargetSolver
from retargeting.kinematics import global_to_local_hierarchy, apply_euler_correction_to_root
from mappings.registry import MappingRegistry
from adapters.hybrik_pk_adapter import HybrIKPKAdapter
from adapters.bvh_adapter import BVHAdapter
from adapters.gltf_mixamo_adapter import GLTFMixamoAdapter
from adapters.vrm_adapter import VRMAdapter


def main():
    parser = argparse.ArgumentParser(description="Retargeting universel par marqueurs 3D.")
    parser.add_argument("--source", required=True, help="Chemin vers le fichier source (.pk, .bvh, .glb)")
    parser.add_argument("--source-type", choices=["pk", "bvh", "mixamo", "vrm"], required=True, help="Format source")
    parser.add_argument("--target", required=True, help="Chemin vers le modèle cible (.glb, .vrm, .bvh)")
    parser.add_argument("--target-type", choices=["mixamo", "vrm", "bvh"], required=True, help="Format cible")
    parser.add_argument("--output", required=True, help="Chemin de sortie pour l'animation")
    parser.add_argument("--root-x-degrees", type=float, default=0.0, help="Correction axiale X sur le root")
    args = parser.parse_args()

    print(f"[isolate] Chargement du squelette cible ({args.target_type}): {args.target}")
    if args.target_type == "mixamo":
        target_adapter = GLTFMixamoAdapter()
    elif args.target_type == "vrm":
        target_adapter = VRMAdapter()
    elif args.target_type == "bvh":
        target_adapter = BVHAdapter()
    else:
        raise ValueError(f"Format cible non supporté: {args.target_type}")

    skeleton = target_adapter.load_skeleton(args.target)
    print(f"[isolate] {len(skeleton.rest_rotations)} os détectés dans le squelette cible.")

    print(f"[isolate] Chargement de la source ({args.source_type}): {args.source}")
    if args.source_type == "pk":
        source_adapter = HybrIKPKAdapter()
    elif args.source_type == "bvh":
        source_adapter = BVHAdapter()
    else:
        raise ValueError(f"Format source non supporté: {args.source_type}")

    sequence = source_adapter.load(args.source)
    print(f"[isolate] Source chargée : {len(sequence.frames_markers)} frames de marqueurs.")

    mapping = MappingRegistry.get_mapping(args.source_type, args.target_type)
    print(f"[isolate] Table de mapping résolue : {len(mapping)} correspondances.")

    solver = MarkerRetargetSolver(
        rest_rotations=skeleton.rest_rotations,
        parents=skeleton.parents,
    )

    animation_clip = []
    n_frames = max(len(sequence.frames_markers), len(sequence.root_rotations or []))

    for t in range(n_frames):
        pts_t = sequence.frames_markers[t] if t < len(sequence.frames_markers) else {}
        root_override = {}
        if sequence.root_rotations and t < len(sequence.root_rotations):
            root_override[skeleton.root_name] = sequence.root_rotations[t]

        Q_global, _ = solver.solve_frame_multi_kabsch(
            positions_t=pts_t,
            positions_rest=sequence.rest_markers,
            weights=sequence.weights,
            root_rot_override=root_override,
        )

        Q_local = global_to_local_hierarchy(Q_global, skeleton.parents)
        if args.root_x_degrees != 0.0:
            Q_local = apply_euler_correction_to_root(
                Q_local,
                root_name=skeleton.root_name,
                axis="x",
                degrees=args.root_x_degrees,
            )

        root_trans = (
            sequence.root_translations[t]
            if sequence.root_translations is not None and t < len(sequence.root_translations)
            else [0.0, 0.0, 0.0]
        )

        animation_clip.append({
            "frame": t,
            "bones": Q_local,
            "root": root_trans,
        })

    print(f"[isolate] Exportation vers {args.output}...")
    target_adapter.export_animation(
        target_model_path=args.target,
        output_path=args.output,
        animation_clip=animation_clip,
        skeleton=skeleton,
    )
    print(f"[isolate] Succès ! Fichier écrit : {args.output}")


if __name__ == "__main__":
    main()

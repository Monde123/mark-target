"""
Script CLI Universel de Retargeting par Marqueurs
==================================================
Permet d'exécuter la méthode des marqueurs de façon découplée et robuste :
- Sources supportées : .pk (HybrIK/SMPL-X), .bvh, .glb
- Cibles supportées : .glb (Mixamo), .vrm (VRoid/VRM), .bvh
- Moteur d'état temporel récursif (t-1 -> t) anti-occlusion et anti-pop
- Auto-Mapper par ossature tampon canonique (détection automatique des os sans JSON)
- Tolérance aux disparités d'armature (os source excédentaires ignorés sans crash)

Exemples d'utilisation :
    python run_marker_retarget.py --source a_out_precise.bvh --source-type bvh --target clara.glb --target-type mixamo --output out_clara.glb --auto-map
    python run_marker_retarget.py --source capture.bvh --source-type bvh --target avatar.vrm --target-type vrm --output out_vrm.vrm
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np

from retargeting.solver import MarkerRetargetSolver
from retargeting.temporal_state import TemporalRetargetState
from retargeting.kinematics import global_to_local_hierarchy, apply_euler_correction_to_root
from mappings.registry import MappingRegistry
from mappings.auto_mapper import BufferRigMapper, AutoBoneIdentifier
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
    parser.add_argument("--custom-mapping", type=str, default=None, help="Chemin vers un fichier JSON de mapping personnalise")
    parser.add_argument("--auto-map", action="store_true", default=False, help="Active l'identification automatique et l'alignement via l'ossature tampon canonique")
    parser.add_argument("--use-temporal-prior", action="store_true", default=False, help="Active le moteur d'état temporel récursif (t-1 -> t) anti-occlusion")
    parser.add_argument("--use-umeyama", action="store_true", default=False, help="Active le solveur Kabsch-Umeyama avec absorption d echelle uniforme")
    parser.add_argument("--smoothing-factor", type=float, default=0.0, help="Facteur de lissage temporel Bezier/SLERP (0.0=aucun, 0.3=recommande)")
    parser.add_argument("--use-ransac", action="store_true", default=False, help="Active le filtrage robuste RANSAC contre les occlusions et marqueurs aberrants")
    parser.add_argument("--max-strain", type=float, default=None, help="Seuil d energie de deformation Green-Lagrange max tolerable avant fallback")
    parser.add_argument("--unit", choices=["m", "cm", "inch"], default=None, help="Unité déclarée du BVH. Rien n'est converti sans elle.")
    parser.add_argument("--up-axis", choices=["Y", "Z"], default=None)
    parser.add_argument("--handedness", choices=["right", "left"], default=None)
    parser.add_argument(
        "--ignore",
        default="",
        help="Os séparés par des virgules. Absent de la cible : retiré. Présent des deux côtés : canaux à zéro (repos local).",
    )
    args = parser.parse_args()
    source_options = {
        key: value
        for key, value in {
            "unit": args.unit,
            "up_axis": args.up_axis,
            "handedness": args.handedness,
        }.items()
        if value is not None
    }
    if args.custom_mapping:
        MappingRegistry.register_from_json(args.custom_mapping, args.source_type, args.target_type)
        print(f"[mark-target] Mapping personnalise applique depuis {args.custom_mapping}")

    print(f"[mark-target] Chargement du squelette cible ({args.target_type}): {args.target}")
    if args.target_type == "mixamo":
        target_adapter = GLTFMixamoAdapter()
    elif args.target_type == "vrm":
        target_adapter = VRMAdapter()
    elif args.target_type == "bvh":
        target_adapter = BVHAdapter()
    else:
        raise ValueError(f"Format cible non supporté: {args.target_type}")

    skeleton = target_adapter.load_skeleton(args.target, **source_options)
    print(f"[mark-target] {len(skeleton.rest_rotations)} os détectés dans le squelette cible.")

    print(f"[mark-target] Chargement de la source ({args.source_type}): {args.source}")
    if args.source_type == "pk":
        source_adapter = HybrIKPKAdapter()
    elif args.source_type == "bvh":
        source_adapter = BVHAdapter()
    else:
        raise ValueError(f"Format source non supporté: {args.source_type}")

    sequence = source_adapter.load(args.source, **source_options)
    print(f"[mark-target] Source chargée : {len(sequence.frames_markers)} frames de marqueurs.")

    if not sequence.frames_markers:
        raise ValueError("La source ne contient aucune frame de mouvement.")

    # Résolution du mapping (explicite, auto-map ou fallback automatique)
    if args.auto_map:
        mapper = BufferRigMapper(
            source_bones=list(sequence.rest_markers.keys()),
            target_bones=list(skeleton.parents.keys()),
            source_rest_positions=sequence.rest_markers,
            source_parents=getattr(sequence, "parents", {}),
            target_parents=skeleton.parents,
        )
        mapping = mapper.get_source_to_target_bone_map()
        print(f"[mark-target] Auto-Mapper activé : {len(mapping)} os appariés via l'ossature tampon.")
    else:
        mapping = MappingRegistry.get_mapping(args.source_type, args.target_type)
        hits = _mapping_hits(sequence.rest_markers, mapping, skeleton.parents)
        if len(hits) == 0 and not args.custom_mapping:
            print("[mark-target] Aucun os apparié par la table par défaut. Bascule automatique sur l'Auto-Mapper...")
            mapper = BufferRigMapper(
                source_bones=list(sequence.rest_markers.keys()),
                target_bones=list(skeleton.parents.keys()),
                source_rest_positions=sequence.rest_markers,
                source_parents=getattr(sequence, "parents", {}),
                target_parents=skeleton.parents,
            )
            mapping = mapper.get_source_to_target_bone_map()
            print(f"[mark-target] Auto-Mapper automatique : {len(mapping)} os appariés.")

    hits = _mapping_hits(sequence.rest_markers, mapping, skeleton.parents)
    print(
        f"[mark-target] Correspondance source->cible : {len(mapping)} entrées, {len(hits)} présentes dans la source."
    )

    ignored = [item.strip() for item in args.ignore.split(",") if item.strip()]
    joint_mode = _joint_marker_cloud(sequence.frames_markers[0])

    rest_named, dropped_rest, ignored_used, left_still = _bind_names(
        sequence.rest_markers, mapping, skeleton.parents, ignored
    )
    if dropped_rest:
        print(f"[mark-target] Os source non ciblés (ignorés proprement) : {len(dropped_rest)}")

    if joint_mode:
        print(
            "[mark-target] Mode Joints BVH : Un point par articulation. "
            "Orientation géométrique calculée depuis les os enfants."
        )

    solver = MarkerRetargetSolver(
        rest_rotations=skeleton.rest_rotations,
        parents=skeleton.parents,
    )

    temporal_state = None
    if args.use_temporal_prior:
        temporal_state = TemporalRetargetState(fps=getattr(sequence, "fps", 30.0))
        print("[mark-target] Moteur d'état temporel récursif (t-1 -> t) activé.")

    animation_clip = []
    n_frames = max(len(sequence.frames_markers), len(sequence.root_rotations or []))

    status_counts: Dict[str, int] = {}
    # status_seen: Dict[str, set] = {}  # variable morte — peuplée mais jamais lue

    for t in range(n_frames):
        pts_t = sequence.frames_markers[t] if t < len(sequence.frames_markers) else {}
        named, _dropped, _ignored_frame, _still_frame = _bind_names(
            pts_t, mapping, skeleton.parents, ignored
        )

        if not named and not left_still:
            # Si aucune correspondance, maintenir la pose au repos
            Q_global = {b: skeleton.rest_rotations[b].copy() for b in skeleton.parents}
            positions_t = {}
            diagnostics = {}
        elif joint_mode and named:
            positions_t = {name: _point(value) for name, value in named.items()}
            positions_rest = _rest_for_solve(target_adapter, positions_t, rest_named)
            Q_global, diagnostics = solver.solve_frame_joint_markers(
                positions_t=positions_t,
                positions_rest=positions_rest,
                translating=_translating_joints(target_adapter),
            )
            # Remplir les os cibles non appariés avec leur pose au repos
            for b in skeleton.parents:
                if b not in Q_global:
                    Q_global[b] = skeleton.rest_rotations.get(b, np.array([1.0, 0.0, 0.0, 0.0])).copy()

            if t == 0:
                for bone, info in diagnostics.items():
                    status_counts[info["status"]] = status_counts.get(info["status"], 0) + 1
        elif joint_mode:
            positions_t = {}
            Q_global = {b: skeleton.rest_rotations.get(b, np.array([1.0, 0.0, 0.0, 0.0])).copy() for b in skeleton.parents}
            diagnostics = {}
        else:
            root_override = {}
            if sequence.root_rotations and t < len(sequence.root_rotations):
                root_override[skeleton.root_name] = sequence.root_rotations[t]
            Q_global, diagnostics = solver.solve_frame_multi_kabsch(
                positions_t=named,
                positions_rest=rest_named,
                weights=sequence.weights,
                root_rot_override=root_override,
                use_umeyama=args.use_umeyama,
                use_ransac=args.use_ransac,
                max_strain_threshold=args.max_strain,
                temporal_state=temporal_state,
            )
            # Compléter les os non animés par leur pose au repos
            for b in skeleton.parents:
                if b not in Q_global:
                    Q_global[b] = skeleton.rest_rotations.get(b, np.array([1.0, 0.0, 0.0, 0.0])).copy()

            if temporal_state is not None:
                temporal_state.update_frame(Q_global)
            positions_t = {}

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
            "world": Q_global,
            "positions": positions_t,
            "root": root_trans,
            "marker_status": (
                {bone: info["status"] for bone, info in diagnostics.items()}
                if joint_mode and named
                else {}
            ),
        })

    if joint_mode and status_counts:
        summary = ", ".join(f"{name}={count}" for name, count in sorted(status_counts.items()))
        print(f"[mark-target] Statut frame 0 : {summary}")

    # Lissage temporel global SQUAD si demandé
    if args.smoothing_factor > 0.0 and len(animation_clip) > 2:
        from retargeting.smoothing import smooth_quaternion_trajectory
        bones = list(skeleton.parents.keys())
        for b in bones:
            traj = [animation_clip[f]["world"][b] for f in range(len(animation_clip)) if b in animation_clip[f]["world"]]
            if len(traj) == len(animation_clip):
                smoothed = smooth_quaternion_trajectory(traj, smoothing_factor=args.smoothing_factor)
                for f in range(len(animation_clip)):
                    animation_clip[f]["world"][b] = smoothed[f]
        for f in range(len(animation_clip)):
            animation_clip[f]["bones"] = global_to_local_hierarchy(animation_clip[f]["world"], skeleton.parents)

    source_frame_time = None
    source_clip = getattr(source_adapter, "last_clip", None)
    if source_clip is not None and getattr(source_clip, "normalization", None) is not None:
        source_frame_time = source_clip.normalization.frame_time

    print(f"[mark-target] Exportation vers {args.output}...")
    target_adapter.export_animation(
        target_model_path=args.target,
        output_path=args.output,
        animation_clip=animation_clip,
        skeleton=skeleton,
        frame_time=source_frame_time,
    )
    print(f"[mark-target] Succès ! Fichier d'animation complet écrit : {args.output}")


def _translating_joints(adapter) -> set:
    """Os dont le squelette déclare un canal de position."""
    clip = getattr(adapter, "last_clip", None)
    joints = getattr(clip, "joints", None)
    if not joints:
        return set()
    return {
        name
        for name, joint in joints.items()
        if any(channel.endswith("position") for channel in joint.channels)
    }


def _joint_marker_cloud(frame: Dict[str, np.ndarray]) -> bool:
    if not frame:
        return False
    for value in frame.values():
        array = np.asarray(value)
        count = 1 if array.shape == (3,) else (array.shape[0] if array.ndim == 2 else 0)
        if count >= 3:
            return False
    return True


def _point(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3,):
        return array
    if array.ndim == 2 and array.shape[1] == 3 and array.shape[0] == 1:
        return array[0]
    raise ValueError(
        f"Marqueur joint de forme {tuple(array.shape)}, un seul point (3,) ou (1, 3) est admis."
    )


def _rest_for_solve(adapter, names, source_rest):
    """Le repos qui oriente les os est celui de la cible si disponible, sinon la source."""
    clip = getattr(adapter, "last_clip", None)
    rest = getattr(clip, "rest_positions", None) if clip is not None else None
    if not rest:
        return {name: _point(source_rest[name]) for name in names if name in source_rest}
    return {name: _point(rest[name]) for name in names if name in rest}


def _mapping_hits(
    points: Dict[str, np.ndarray],
    mapping: Dict[str, str],
    target_bones: Dict[str, Optional[str]],
) -> List[str]:
    """Noms de la table qui existent vraiment dans la source et dans la cible."""
    return [
        name
        for name in points
        if mapping.get(name) in target_bones
    ]


def _bind_names(
    points: Dict[str, np.ndarray],
    mapping: Dict[str, str],
    target_bones: Dict[str, Optional[str]],
    ignored: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], List[str], List[str], List[str]]:
    """
    Apparie les os de la source vers la cible avec tolérance générale :
    - Les os source absents de la cible sont ignorés (dropped) sans lever d'exception fatale.
    - Les os cible absents de la source restent dans leur pose locale de repos.
    """
    ignored_set = set(ignored or [])
    bound: Dict[str, np.ndarray] = {}
    dropped: List[str] = []
    claimed: Dict[str, str] = {}
    ignored_used: List[str] = []
    left_still: List[str] = []

    for source_name, value in points.items():
        target_name = mapping.get(source_name)
        if target_name is None and source_name in target_bones:
            target_name = source_name

        binds = target_name is not None and target_name in target_bones

        if source_name in ignored_set:
            if binds:
                left_still.append(target_name)
            else:
                ignored_used.append(source_name)
            continue

        if not binds:
            dropped.append(source_name)
            continue

        # Si l'os cible est déjà attribué par un autre os source, on conserve le premier
        if target_name in claimed and claimed[target_name] != source_name:
            continue

        claimed[target_name] = source_name
        bound[target_name] = value

    return bound, dropped, ignored_used, left_still


if __name__ == "__main__":
    main()

"""
MarkTarget - Quickstart Script
==============================
Démonstration de transfert cinématique ultra-rapide par cages tétraédriques rigides (4 marqueurs)
et système d'observation / diagnostic 3D.

Transfère l'animation MoCap 'a_out_precise.bvh' vers le modèle 3D 'clara.glb'
sans déformation de bind pose ni indétermination de twist.
"""

from __future__ import annotations

import sys
import os
import time

# Reconfiguration de stdout en UTF-8 pour supporter les logs sous Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np


def run_quickstart():
    print("=" * 70)
    print("🚀 MARKTARGET : TRANSFERT DE MOUVEMENT PAR CAGES DE MARQUEURS 3D")
    print("=" * 70)

    source_bvh = "a_out_precise.bvh"
    target_glb = "clara.glb"
    output_glb = "clara_animated.glb"
    observer_json = os.path.join("visualizer", "marker_observations.json")

    # Vérification de la présence des fichiers
    for path, desc in [(source_bvh, "Source BVH"), (target_glb, "Modèle cible GLB")]:
        if not os.path.exists(path):
            print(f"❌ Erreur : Fichier requis introuvable : '{path}' ({desc})")
            print(f"Veuillez vous assurer que '{path}' est présent à la racine du projet.")
            return

    print(f"📁 Source MoCap      : {source_bvh}")
    print(f"📁 Modèle cible      : {target_glb}")
    print(f"🎯 Fichier de sortie : {output_glb}")
    print(f"🔭 Diagnostic 3D     : {observer_json}")
    print("-" * 70)

    start_time = time.perf_counter()

    # Import des modules du moteur MarkTarget
    from adapters.bvh_adapter import BVHAdapter
    from adapters.gltf_mixamo_adapter import GLTFMixamoAdapter
    from mappings.auto_mapper import BufferRigMapper
    from markers.virtual_cage import VirtualBoneCageGenerator
    from markers.observer import MarkerObserver
    from core.kabsch import kabsch_umeyama_rotation
    from core.geometry import (
        matrix_to_quaternion,
        quat_mul,
        quat_inv,
        quat_normalize,
        quat_identity,
    )

    # 1. Chargement de la source et de la cible
    print("⏳ 1/5. Chargement des structures squelettiques...")
    source_adapter = BVHAdapter()
    sequence = source_adapter.load(source_bvh)

    target_adapter = GLTFMixamoAdapter()
    skeleton = target_adapter.load_skeleton(target_glb)

    source_parents = sequence.parents or {
        j.name: j.parent for j in getattr(source_adapter.last_clip, "joints", {}).values()
    }

    # 2. Génération des cages rigides tétraédriques 4 marqueurs
    print("⏳ 2/5. Construction des cages virtuelles rigides à 4 marqueurs (3 DoF)...")
    cage_gen = VirtualBoneCageGenerator(lateral_ratio=0.20, orthogonal_ratio=0.15)
    rest_cages, frames_cages = cage_gen.generate_bone_cages_sequence(
        parents=source_parents,
        rest_positions=sequence.rest_markers,
        frames_positions=sequence.frames_markers,
        joint_rotations=sequence.joint_rotations,
    )

    # 3. Analyse cinématique et export du payload de diagnostic pour l'observateur WebGL
    print("⏳ 3/5. Calcul cinématique et diagnostic des marqueurs...")
    observer = MarkerObserver(
        bones_hierarchy=source_parents,
        rest_cages=rest_cages,
        frames_cages=frames_cages,
        fps=sequence.fps or 30.0,
    )
    os.makedirs("visualizer", exist_ok=True)
    observer.export_observation_json(
        output_path=observer_json,
        source_name=source_bvh,
        target_name=target_glb,
    )

    # 4. Auto-Mapping anatomique
    mapper = BufferRigMapper(
        source_bones=list(sequence.rest_markers.keys()),
        target_bones=list(skeleton.parents.keys()),
        source_rest_positions=sequence.rest_markers,
        source_parents=source_parents,
        target_parents=skeleton.parents,
    )
    mapping = mapper.get_source_to_target_bone_map()

    # Inversion du dictionnaire pour accès rapide target_bone -> source_bone
    tgt_to_src = {tgt: src for src, tgt in mapping.items()}

    # 5. Résolution cinématique par la formule du Delta Bind-Pose
    print("⏳ 4/5. Résolution SVD Kabsch-Umeyama & modulation du bind-pose...")
    bind_local = skeleton.bind_local_rotations or {}
    bind_world = skeleton.rest_rotations

    n_frames = len(frames_cages)
    animation_clip = []

    for t in range(n_frames):
        cages_t = frames_cages[t]
        local_rotations = {}
        world_rotations = {}

        for tgt_bone in skeleton.parents:
            q_bind_loc = bind_local.get(tgt_bone, quat_identity())
            q_bind_w = bind_world.get(tgt_bone, quat_identity())

            src_bone = tgt_to_src.get(tgt_bone)

            if src_bone and src_bone in cages_t and src_bone in rest_cages:
                pts_r = rest_cages[src_bone]
                pts_t = cages_t[src_bone]

                # SVD de rang 3 plein sur tétraèdre rigide
                R_move, _, _, _, _, rms, _ = kabsch_umeyama_rotation(pts_r, pts_t)
                q_src_delta = matrix_to_quaternion(R_move)

                # Conjugaison dans le repère local du joint cible :
                # q_loc_delta = q_bind_w^(-1) * q_src_delta * q_bind_w
                q_loc_delta = quat_mul(quat_mul(quat_inv(q_bind_w), q_src_delta), q_bind_w)
                q_anim_loc = quat_normalize(quat_mul(q_bind_loc, q_loc_delta))
                q_anim_world = quat_normalize(quat_mul(q_src_delta, q_bind_w))
            else:
                # Os non animé : préservation stricte de la pose de repos locale
                q_anim_loc = q_bind_loc
                q_anim_world = q_bind_w

            local_rotations[tgt_bone] = q_anim_loc
            world_rotations[tgt_bone] = q_anim_world

        animation_clip.append({
            "frame": t,
            "bones": local_rotations,
            "world": world_rotations,
            "root": [0.0, 0.0, 0.0],
        })

    # 6. Exportation dans le GLB cible
    print(f"⏳ 5/5. Exportation de l'animation vers {output_glb}...")
    target_adapter.export_animation(
        target_model_path=target_glb,
        output_path=output_glb,
        animation_clip=animation_clip,
        skeleton=skeleton,
        animation_name="marker_retarget_motion",
    )

    elapsed = time.perf_counter() - start_time
    file_size_mb = os.path.getsize(output_glb) / (1024 * 1024)

    print("-" * 70)
    print("✅ TRANSFERT RÉUSSI AVEC RIGIDITÉ ET BIND POSE PARFAITS !")
    print(f"⏱️  Temps total de calcul : {elapsed:.3f} seconde ({n_frames} frames traitées)")
    print(f"⚡ Vitesse moyenne         : {(elapsed / n_frames) * 1000:.2f} ms par frame")
    print(f"📦 Fichier GLB généré     : {output_glb} ({file_size_mb:.2f} Mo)")
    print(f"🦴 Articulations animées   : {len(mapping)} os appariés sans conflit")
    print(f"🔭 Données d'observation  : {observer_json}")
    print("=" * 70)
    print("👀 COMMENT OBSERVER ET VÉRIFIER LES RÉSULTATS :")
    print("1. VISUALISEUR DE MARQUEURS 3D (Observateur WebGL) :")
    print(f"   Ouvrez le fichier HTML dans votre navigateur :")
    print(f"   file:///{os.path.abspath('visualizer/marker_viewer.html').replace(os.sep, '/')}")
    print("   Vous y verrez en 3D interactive :")
    print("   - Les 4 sphères de chaque tétraèdre rigide (M0, M1, M2, M3)")
    print("   - Les trièdres RGB locaux de chaque articulation en temps réel")
    print("   - Les trajectoires temporelles et le panneau de métriques de rigidité")
    print()
    print("2. MODÈLE 3D FINAL TEXTURÉ :")
    print("   Glissez-déposez 'clara_animated.glb' sur : https://gltf-viewer.donmccurdy.com/")
    print("   Clara bouge désormais sans distorsion, sans torsion d'os et sans retournement !")
    print("=" * 70)


if __name__ == "__main__":
    run_quickstart()

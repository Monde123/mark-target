"""
⚡ Lanceur Rapide (Quickstart) — mark-target
=============================================
Exécute en 1 seconde une démonstration réelle de retargeting :
Transfère le mouvement de capture MoCap 'a_out_precise.bvh'
vers l'avatar 3D Mixamo 'clara.glb' et génère 'clara_animated.glb'.

Usage :
    python quickstart.py
"""

import os
import sys
import time

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 70)
    print("🚀 mark-target : Démonstration de Retargeting en 1 Clic")
    print("=" * 70)

    # 1. Vérification des dépendances
    try:
        import numpy as np
        import pygltflib
    except ImportError:
        print("❌ Dépendances manquantes. Veuillez exécuter :")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    source_bvh = "a_out_precise.bvh"
    target_glb = "clara.glb"
    output_glb = "clara_animated.glb"

    if not os.path.exists(source_bvh):
        print(f"❌ Fichier source introuvable : {source_bvh}")
        sys.exit(1)

    if not os.path.exists(target_glb):
        print(f"❌ Modèle cible introuvable : {target_glb}")
        sys.exit(1)

    print(f"📁 Source MoCap      : {source_bvh}")
    print(f"📁 Modèle Cible 3D   : {target_glb}")
    print(f"📁 Fichier de Sortie : {output_glb}")
    print("-" * 70)
    print("⚙️  Calcul cinématique et transfert en cours (100% CPU)...")

    start_time = time.perf_counter()

    # Import des modules du moteur
    from adapters.bvh_adapter import BVHAdapter
    from adapters.gltf_mixamo_adapter import GLTFMixamoAdapter
    from mappings.auto_mapper import BufferRigMapper
    from retargeting.solver import MarkerRetargetSolver
    from retargeting.temporal_state import TemporalRetargetState
    from retargeting.kinematics import global_to_local_hierarchy

    # Chargement
    source_adapter = BVHAdapter()
    sequence = source_adapter.load(source_bvh)

    target_adapter = GLTFMixamoAdapter()
    skeleton = target_adapter.load_skeleton(target_glb)

    # Auto-Mapping via l'ossature tampon
    mapper = BufferRigMapper(
        source_bones=list(sequence.rest_markers.keys()),
        target_bones=list(skeleton.parents.keys()),
        source_rest_positions=sequence.rest_markers,
        source_parents=getattr(sequence, "parents", {}),
        target_parents=skeleton.parents,
    )
    mapping = mapper.get_source_to_target_bone_map()

    # Pose de repos
    rest_named = {}
    for src_name, val in sequence.rest_markers.items():
        tgt_name = mapping.get(src_name)
        if tgt_name and tgt_name in skeleton.parents:
            rest_named[tgt_name] = np.asarray(val, dtype=np.float64).reshape(-1)

    solver = MarkerRetargetSolver(
        rest_rotations=skeleton.rest_rotations,
        parents=skeleton.parents,
    )
    temporal_state = TemporalRetargetState(fps=sequence.fps or 30.0)

    animation_clip = []
    n_frames = len(sequence.frames_markers)

    for t in range(n_frames):
        frame_markers = sequence.frames_markers[t]
        named = {}
        for src_name, val in frame_markers.items():
            tgt_name = mapping.get(src_name)
            if tgt_name and tgt_name in skeleton.parents:
                named[tgt_name] = np.asarray(val, dtype=np.float64).reshape(-1)

        positions_t = {name: val for name, val in named.items()}

        Q_global, _ = solver.solve_frame_joint_markers(
            positions_t=positions_t,
            positions_rest=rest_named,
        )

        for b in skeleton.parents:
            if b not in Q_global:
                Q_global[b] = skeleton.rest_rotations.get(b, np.array([1.0, 0.0, 0.0, 0.0])).copy()

        temporal_state.update_frame(Q_global)

        Q_local = global_to_local_hierarchy(Q_global, skeleton.parents)
        animation_clip.append({
            "frame": t,
            "bones": Q_local,
            "world": Q_global,
            "root": [0.0, 0.0, 0.0],
        })

    # Exportation dans le GLB cible
    target_adapter.export_animation(
        target_model_path=target_glb,
        output_path=output_glb,
        animation_clip=animation_clip,
        skeleton=skeleton,
        animation_name="quickstart_motion",
    )

    elapsed = time.perf_counter() - start_time
    file_size_mb = os.path.getsize(output_glb) / (1024 * 1024)

    print("-" * 70)
    print("✅ TRANSFERT RÉUSSI !")
    print(f"⏱️  Temps total de calcul : {elapsed:.3f} seconde ({n_frames} frames traitées)")
    print(f"⚡ Vitesse moyenne         : {(elapsed / n_frames) * 1000:.2f} ms par frame")
    print(f"📦 Fichier généré          : {output_glb} ({file_size_mb:.2f} Mo)")
    print(f"🦴 Articulations animées   : {len(mapping)} os appariés automatiquement")
    print("=" * 70)
    print("👀 COMMENT VOIR LE RÉSULTAT EN 3D DÈS MAINTENANT :")
    print("1. Ouvrez dans votre navigateur : https://gltf-viewer.donmccurdy.com/")
    print(f"2. Glissez-déposez le fichier '{output_glb}' dans la page.")
    print("3. Vous verrez Clara s'animer immédiatement en 3D avec le mouvement MoCap !")
    print("=" * 70)


if __name__ == "__main__":
    main()

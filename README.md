# Module `isolate` : Retargeting par Marqueurs 3D Autonome et Universel

Ce dossier contient l'isolation complète et agnostique de la méthode par marqueurs (Kabsch multi-points / Aim mono-marqueur).

---

## 🌟 Caractéristiques clés

- **100% Découplé** : Ne dépend ni d'un modèle 3D spécifique, ni de la topologie SMPL-X, ni des préfixes Mixamo.
- **Support Universel Multi-Formats** :
  - **Sources d'entrée** : `.pk` (HybrIK / SMPL-X), `.bvh`, `.glb` / Mixamo, nuages de marqueurs optiques.
  - **Cibles de sortie** : Avatars Mixamo (`.glb`), Avatars VRoid / Humanoid (`.vrm`), squelettes `.bvh`.
- **Pivot Standardisé "Humanoid"** : Intègre un mapping basé sur la norme Humanoid pour convertir dynamiquement les nomenclatures d'os.
- **Tolérance aux pannes et fallbacks** : Détection des colinéarités et singularités via SVD avec bascule propre en pose de repos.

---

## 📁 Architecture du package

```text
isolate/
├── core/                       # Algèbre géométrique pure
│   ├── geometry.py             # Quaternions [w, x, y, z], conversions matrices 3x3, rotations
│   └── kabsch.py               # Solveurs Kabsch (standard SVD et pondéré par LBS)
├── markers/                    # Sélection et suivi de marqueurs
│   ├── surface_sampling.py     # Découpage sous-maillages, FPS (Farthest Point Sampling)
│   └── tracker.py              # Suivi temporel et pondération de stabilité
├── retargeting/                # Solveurs cinématiques
│   ├── solver.py               # Moteur d'estimation de rotations (MarkerRetargetSolver)
│   └── kinematics.py           # Propagation hiérarchique (Global <-> Local)
├── adapters/                   # Adaptateurs de formats d'E/S
│   ├── base.py                 # Protocoles abstraits SourceAdapter et TargetAdapter
│   ├── hybrik_pk_adapter.py    # Prise en charge des fichiers .pk HybrIK
│   ├── gltf_mixamo_adapter.py  # Prise en charge des rigs Mixamo (.glb)
│   ├── vrm_adapter.py          # Prise en charge des avatars VRM (.vrm)
│   └── bvh_adapter.py          # Prise en charge du format BVH
├── mappings/                   # Mappings et ponts d'os
│   ├── standard_humanoid.py    # Définition du standard pivot Humanoid
│   └── registry.py             # Résolveur dynamique (ex: Mixamo <-> VRM)
├── run_marker_retarget.py      # CLI universel de retargeting
└── tests/
    └── test_universal_markers.py # Tests unitaires
```

---

## 🚀 Utilisation en ligne de commande

```bash
# Retargeting depuis un .pk HybrIK vers un avatar Mixamo .glb
python -m isolate.run_marker_retarget --source output/hybrik_a/res.pk --source-type pk --target clara.glb --target-type mixamo --output out_isolated.glb

# Retargeting depuis un fichier BVH vers un avatar VRM
python -m isolate.run_marker_retarget --source anim.bvh --source-type bvh --target model.vrm --target-type vrm --output out_vrm.vrm
```

---

## 🧪 Exécution des tests unitaires

```bash
python -m unittest isolate.tests.test_universal_markers
```

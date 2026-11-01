# mark-target : Moteur Universel de Retargeting Déterministe Temps-Réel

Moteur géométrique de retargeting d'animation 3D ultra-rapide ($< 0.4\text{ ms}$/frame), 100% CPU en forme fermée (SVD), sans dépendance GPU.

---

## 🌟 Points Forts & Capacités Majeures

- **Moteur Temporel Récursif ($t-1 \to t$)** :
  - **Zéro Pop d'Occlusion :** Extrapolation cinématique fluide par vitesse angulaire $\vec{\omega}_{t-1}$ au lieu de sauter en T-Pose lors de pertes de marqueurs.
  - **Verrouillage d'Hémisphère $\mathbb{S}^3$ :** Éradication totale des sauts de $360^\circ$ ($\langle q_t, q_{t-1} \rangle \ge 0$).
  - **Échelle Morphologique Stabilisée :** Filtrage exponentiel de l'échelle d'Umeyama $c$.
- **Ossature Tampon Canonique & Auto-Mapper Intelligente (`--auto-map`)** :
  - **Reconnaissance Automatique des Rigs :** Détection sémantique et topologique multi-logiciels (Mixamo, Blender Rigify `DEF-`, VRoid/VRM, BVH CMU/Biovision, SMPL-X).
  - **Alignement Automatique A-Pose $\leftrightarrow$ T-Pose :** Pré-rotation vectorielle automatique $\Delta R_{\text{align}}$ vers le squelette pivot 55 os.
  - **Aucun fichier JSON manuel requis.**
- **Solveur Hybride Robuste** :
  - **Kabsch-Umeyama SVD :** Résolution conjointe rotation + échelle ($c$).
  - **Filtre RANSAC :** Rejet des occlusions partielles et marqueurs aberrants.
  - **Tenseur de Green-Lagrange :** Surveillance d'élongation élastique de chair ($E = \frac{1}{2}(F^T F - I_3)$).
  - **Bézier SQUAD :** Lissage temporel sphérique continu $C^2$ anti-tremblement.
- **Agnosticisme Multi-Formats** :
  - Sources : `.pk` (HybrIK / SMPL-X), `.bvh` (MoCap), `.glb` (Mixamo).
  - Cibles : `.glb` (Mixamo/gltf), `.vrm` (VRoid), `.bvh`.

---

## 🚀 Utilisation en Ligne de Commande

### 1. Retargeting Intelligent Automatique (Auto-Map & Mémoire Temporelle)
```bash
python run_marker_retarget.py \
    --source animation.bvh --source-type bvh \
    --target avatar.glb --target-type mixamo \
    --output resultat_anime.glb \
    --auto-map \
    --use-temporal-prior
```

### 2. Retargeting MoCap avec Absorption d'Échelle & Anti-Jitter
```bash
python run_marker_retarget.py \
    --source capture.bvh --source-type bvh \
    --target personnage.vrm --target-type vrm \
    --output personnage_anime.vrm \
    --use-umeyama \
    --smoothing-factor 0.3 \
    --use-temporal-prior
```

### 3. Retargeting Robuste avec Filtrage RANSAC & Contrôle de Déformation
```bash
python run_marker_retarget.py \
    --source res.pk --source-type pk \
    --target clara.glb --target-type mixamo \
    --output out_markers.glb \
    --use-ransac \
    --max-strain 0.15 \
    --use-temporal-prior
```

---

## 📁 Architecture Modulaire

```text
mark_target/
├── core/                       # Algèbre géométrique fermée
│   ├── geometry.py             # Quaternions, so(3), conjugués, rotations
│   └── kabsch.py               # Kabsch SVD et Kabsch-Umeyama (avec scale c)
├── mappings/                   # Ponts d'os et armature pivot
│   ├── buffer_armature.py      # Ossature Tampon Canonique 55 os en T-Pose
│   ├── auto_mapper.py          # Auto-Identifier sémantique et alignement A->T Pose
│   ├── standard_humanoid.py    # Nomenclature et hiérarchie pivot 55 os
│   └── registry.py             # Résolveur dynamique de formats
├── retargeting/                # Solveurs cinématiques et temporels
│   ├── temporal_state.py       # Moteur récursif à mémoire d'état (t-1 -> t)
│   ├── solver.py               # MarkerRetargetSolver (Kabsch, Umeyama, RANSAC, Aim)
│   ├── joint_markers.py        # Solveur analytique direct pour points de joints BVH
│   ├── smoothing.py            # Bézier SQUAD sur S³ et De Casteljau
│   └── kinematics.py           # Hiérarchie globale <-> locale
├── markers/                    # Traitement de surface et robustesse
│   ├── ransac_filter.py        # Consensus RANSAC et tenseur de Green-Lagrange
│   └── surface_sampling.py     # FPS (Farthest Point Sampling) et poids LBS
├── adapters/                   # Adaptateurs d'entrée/sortie
│   ├── bvh_adapter.py          # Adaptateur bidirectionnel BVH
│   ├── bvh_parse.py            # Parseur et exportateur BVH certifié sub-angulaire
│   ├── gltf_mixamo_adapter.py  # Rigs Mixamo (.glb)
│   ├── vrm_adapter.py          # Avatars VRoid (.vrm)
│   └── hybrik_pk_adapter.py    # Inférence vision amont (.pk)
├── docs/                       # Spécifications et analyses d'ingénierie
│   ├── TEMPORAL_STATE_RECURSIVE_RETARGETING.md
│   ├── BUFFER_ARMATURE_AND_AUTO_MAPPING.md
│   ├── COMMERCIALIZATION_AND_MATURITY_AUDIT.md
│   └── CAGE_VS_SKELETON_RISK_ANALYSIS.md
└── tests/                      # Suite de tests unitaires automatisés (33 tests OK)
```

---

## 🧪 Exécution des Tests

```bash
python -m unittest discover -s tests
```
*33 tests unitaires couvrant l'auto-identification, les géodésiques de quaternions, l'absorption d'échelle et la non-régression.*

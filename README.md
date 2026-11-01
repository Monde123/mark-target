# mark-target : Moteur Universel de Retargeting Déterministe Temps-Réel

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-40%20passed%20(100%25)-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-CPU%20pure%20(NumPy)-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

Moteur géométrique de retargeting d'animation 3D ultra-rapide ($< 0.4\text{ ms}$/frame), 100% CPU en forme fermée (SVD), déterministe et sans dépendance GPU.

---

## ⚡ Démarrage Rapide en 1 Clic (Quickstart)

Pour tester immédiatement le moteur sur un cas réel de production (transfert d'un clip MoCap vers un avatar 3D humanoïde) :

```bash
# 1. Installer les dépendances légères
pip install -r requirements.txt

# 2. Lancer la démonstration instantanée
python quickstart.py
```

### Ce qui est exécuté :
- Transfert cinématique de 102 frames MoCap vers l'avatar Mixamo `clara.glb`.
- Durée moyenne : **~1.8 seconde** pour l'intégralité du clip (**17 ms/frame** incluant l'E/S binaire glTF).
- Génération du fichier `clara_animated.glb` prêt pour le Web, Blender ou Unreal Engine.
- **Visualisation 3D immédiate :** Glissez-déposez le fichier généré dans [gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com/) pour observer l'avatar en mouvement.

> Pour une démo purement MoCap BVH vers BVH :
> ```bash
> python examples/run_bvh_example.py
> ```

---

## 🌟 Points Forts & Capacités Majeures

- **Moteur Temporel Récursif ($t-1 \to t$)** :
  - **Zéro Pop d'Occlusion :** Extrapolation cinématique fluide par vitesse angulaire dans l'algèbre de Lie $\mathfrak{so}(3)$ ($\vec{\omega}_{t-1}$) au lieu de sauter en T-Pose lors de pertes de marqueurs.
  - **Verrouillage d'Hémisphère $\mathbb{S}^3$ :** Éradication totale des sauts de $360^\circ$ dus au double-cover ($\langle q_t, q_{t-1} \rangle \ge 0$).
  - **Échelle Morphologique Stabilisée :** Filtrage exponentiel du facteur d'échelle d'Umeyama $c$.

- **Ossature Tampon Canonique & Auto-Mapper Intelligente (`--auto-map`)** :
  - **Reconnaissance Automatique des Rigs :** Détection sémantique et topologique multi-logiciels (Mixamo, Blender Rigify `DEF-`, VRoid/VRM, BVH CMU/Biovision, SMPL-X).
  - **Alignement Automatique A-Pose $\leftrightarrow$ T-Pose :** Pré-rotation vectorielle automatique $\Delta R_{\text{align}}$ vers le squelette pivot 55 os.
  - **Aucun fichier JSON manuel requis.**

- **Solveur Hybride Robuste** :
  - **Kabsch-Umeyama SVD :** Résolution conjointe rotation propre $R \in \text{SO}(3)$ + facteur d'échelle morphologique $c$.
  - **Filtre RANSAC :** Consensus robuste rejetant les occlusions partielles et marqueurs aberrants (glissements de peau).
  - **Tenseur de Green-Lagrange :** Surveillance d'élongation élastique ($E = \frac{1}{2}(F^T F - I_3)$) et repli gracieux automatique.
  - **Bézier SQUAD :** Lissage temporel sphérique continu $C^2$ anti-tremblement (suppression du jitter haute fréquence).

- **Agnosticisme Multi-Formats** :
  - **Sources supportées :** `.pk` (HybrIK / SMPL-X), `.bvh` (MoCap), `.glb` (Mixamo).
  - **Cibles supportées :** `.glb` (Mixamo/glTF), `.vrm` (VRoid), `.bvh` (Biovision).

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

### 2. Retargeting MoCap avec Absorption d'Échelle & Anti-Jitter SQUAD
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
│   ├── geometry.py             # Quaternions, so(3), conjugués, exp/log maps
│   └── kabsch.py               # Kabsch SVD et Kabsch-Umeyama (échelle c)
├── mappings/                   # Ponts d'os et armature pivot
│   ├── buffer_armature.py      # Ossature Tampon Canonique 55 os en T-Pose
│   ├── auto_mapper.py          # Auto-Identifier sémantique et alignement A->T Pose
│   ├── standard_humanoid.py    # Nomenclature et hiérarchie pivot 55 os
│   └── registry.py             # Résolveur dynamique multi-formats
├── retargeting/                # Solveurs cinématiques et temporels
│   ├── temporal_state.py       # Moteur récursif à mémoire d'état (t-1 -> t)
│   ├── solver.py               # MarkerRetargetSolver (Kabsch, Umeyama, RANSAC, Aim)
│   ├── joint_markers.py        # Solveur direct pour points de joints BVH
│   ├── smoothing.py            # Bézier SQUAD sur S³ et De Casteljau
│   └── kinematics.py           # Hiérarchie globale <-> locale
├── markers/                    # Traitement de surface et robustesse
│   ├── ransac_filter.py        # Consensus RANSAC et tenseur de Green-Lagrange
│   ├── surface_sampling.py     # FPS (Farthest Point Sampling) et poids LBS
│   └── tracker.py              # Extraction temporelle des marqueurs
├── adapters/                   # Adaptateurs d'entrée/sortie
│   ├── bvh_adapter.py          # Adaptateur bidirectionnel BVH
│   ├── bvh_parse.py            # Parseur et exportateur BVH certifié sub-angulaire
│   ├── gltf_mixamo_adapter.py  # Rigs Mixamo (.glb)
│   ├── vrm_adapter.py          # Avatars VRoid (.vrm)
│   └── hybrik_pk_adapter.py    # Inférence vision amont (.pk)
├── examples/                   # Scripts de démonstration prêts à l'emploi
│   ├── minimal_usage.py        # Démonstration minimale pure NumPy
│   └── run_bvh_example.py      # Démonstration BVH autonome
├── tests/                      # Suite de validation automatisée (40 tests PASS)
│   ├── test_pipeline_quality.py   # Invariants mathématiques & robustesse synthétique
│   ├── test_auto_mapper.py        # Identification sémantique & alignement A-T Pose
│   ├── test_temporal_state.py     # Mémoire d'état t-1 -> t & extrapolation so(3)
│   ├── test_ransac_and_deformation.py # RANSAC & Green-Lagrange
│   ├── test_umeyama_and_smoothing.py  # Umeyama & Bézier SQUAD
│   ├── test_mappings.py           # Ponts multi-formats isolés
│   ├── test_universal_markers.py  # Reconstitution géométrique Kabsch
│   └── test_end_to_end_real_assets.py # Pipeline réelle bout-en-bout
├── quickstart.py               # Démonstration instantanée en 1 clic
├── setup.py                    # Définition du package installable mark-target
└── requirements.txt            # Dépendances de production
```

---

## 🧪 Suite de Tests & Qualité de Code

Le projet intègre une suite de **40 tests automatisés** garantissant la non-régression, l'isolation des états et la précision mathématique :

```bash
python -m pytest tests/ -v
```

```text
============================= 40 passed in ~3s =============================
```

### Invariants et tests couverts :
1. **Qualité numérique (`test_pipeline_quality.py`)** :
   - Conservation absolue de l'énergie lors de l'aller-retour hiérarchique ($< 10^{-6}\text{ rad}$).
   - Stabilité stricte sur signal immobile (zéro vibration parasite).
   - Invariance d'échelle Umeyama ($c$ découplé sans cisaillement d'orientation).
   - Repli gracieux sans plantage ni NaN sur marqueurs colinéaires / coplanaires.
   - Réduction prouvée du jerk angulaire via Bézier SQUAD.
2. **Isolation des composants** : Réinitialisation stricte du registre de mapping pour prévenir toute interférence entre tests.
3. **Politique d'assets légers sur GitHub** :
   - Le dépôt GitHub est volontairement exempt de fichiers binaires 3D volumineux (`.glb`, `.vrm`, `.bvh`, `.pk` exclus via `.gitignore`).
   - Le test d'intégration bout-en-bout bascule automatiquement en `SkipTest` dans les environnements CI dénués de modèles binaires locaux.

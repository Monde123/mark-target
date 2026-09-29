# Guide d'utilisation du module `isolate`

Ce document explique comment intégrer, configurer et utiliser la méthode de retargeting par marqueurs 3D dans n'importe quel projet ou pipeline d'animation.

---

## 1. Installation & Dépendances

Le cœur algorithmique du module `isolate` est conçu pour être **ultra-léger** et indépendant de tout framework d'apprentissage profond (comme PyTorch) :

```bash
# Dépendance obligatoire pour le cœur (mathématiques, SVD Kabsch)
pip install numpy

# Dépendances optionnelles selon vos formats de fichiers cibles
pip install pygltflib   # Requis pour importer / exporter des avatars Mixamo (.glb) ou VRM (.vrm)
```

---

## 2. Intégration en Python (API Programmatique)

Vous pouvez utiliser `MarkerRetargetSolver` directement dans votre propre code sans passer par la ligne de commande.

### Exemple minimal autonome

```python
import numpy as np
from isolate.retargeting import MarkerRetargetSolver, global_to_local_hierarchy
from isolate.core import quat_identity

# 1. Définition de la hiérarchie du squelette cible (ex: Épaule -> Bras -> Avant-bras)
parents = {
    "Hips": None,
    "LeftArm": "Hips",
    "LeftForeArm": "LeftArm",
}

# 2. Rotations de repos (T-pose ou bind pose) du squelette cible
rest_rotations = {
    "Hips": quat_identity(),        # [w, x, y, z] = [1.0, 0.0, 0.0, 0.0]
    "LeftArm": quat_identity(),
    "LeftForeArm": quat_identity(),
}

# 3. Instanciation du solveur
solver = MarkerRetargetSolver(
    rest_rotations=rest_rotations,
    parents=parents,
    min_points_kabsch=3,    # Au moins 3 points non-colinéaires par os
    min_rank_ratio=1e-3,    # Détection de dégénérescence géométrique
    max_rms_ratio=0.25,     # Tolérance maximale de déformation résiduelle
)

# 4. Marqueurs 3D en pose de repos (P_rest) pour chaque os : tableau de forme (N, 3)
positions_rest = {
    "LeftArm": np.array([
        [0.2, 1.4, 0.0],
        [0.4, 1.4, 0.0],
        [0.3, 1.5, 0.05],
        [0.3, 1.3, -0.05],
    ]),
    "LeftForeArm": np.array([
        [0.5, 1.4, 0.0],
        [0.7, 1.4, 0.0],
        [0.6, 1.45, 0.04],
        [0.6, 1.35, -0.04],
    ]),
}

# 5. Marqueurs 3D observés à la frame t (P_t) : tableau de forme (N, 3)
positions_t = {
    "LeftArm": positions_rest["LeftArm"] + np.array([0.0, 0.1, -0.05]),
    "LeftForeArm": positions_rest["LeftForeArm"] + np.array([0.0, 0.2, -0.1]),
}

# 6. Résolution de l'orientation globale de chaque os par Kabsch
Q_global_t, diagnostics = solver.solve_frame_multi_kabsch(
    positions_t=positions_t,
    positions_rest=positions_rest,
)

# 7. Conversion des rotations globales vers les rotations locales requises par le rig 3D
Q_local_t = global_to_local_hierarchy(Q_global_t, parents)

print("Rotation locale calculée pour LeftArm :", Q_local_t["LeftArm"])
```

---

## 3. Utilisation en Ligne de Commande (CLI)

Le script `isolate.run_marker_retarget` permet de traiter des animations complètes directement entre différents formats.

### Syntaxe générale

```bash
python -m isolate.run_marker_retarget \
    --source <FICHIER_SOURCE> \
    --source-type <pk|bvh|mixamo|vrm> \
    --target <MODELE_CIBLE> \
    --target-type <mixamo|vrm|bvh> \
    --output <FICHIER_SORTIE> \
    [--root-x-degrees <ANGLE>]
```

### Exemples d'application

1. **Retargeting d'une sortie HybrIK/SMPL-X (`.pk`) vers un avatar Mixamo (`.glb`)** :
   ```bash
   python -m isolate.run_marker_retarget \
       --source output/hybrik_a/res.pk \
       --source-type pk \
       --target clara.glb \
       --target-type mixamo \
       --output animation_mixamo.glb \
       --root-x-degrees -90.0
   ```

2. **Retargeting d'une capture BVH vers un avatar VRM (VRoid)** :
   ```bash
   python -m isolate.run_marker_retarget \
       --source mocap_data.bvh \
       --source-type bvh \
       --target avatar.vrm \
       --target-type vrm \
       --output animation_vrm.vrm
   ```

---

## 4. Guide des Configurations et Hyperparamètres

| Hyperparamètre | Type / Valeur recommandée | Rôle et impact |
|---|---|---|
| `min_points_kabsch` | `int` (défaut : `3`) | Nombre minimal de marqueurs par os pour calculer Kabsch. Avec $\ge 3$ points non colinéaires, les 3 degrés de liberté (roll inclus) sont résolus de façon unique. |
| `min_rank_ratio` | `float` (défaut : `1e-3`) | Rapport entre la plus petite et la plus grande valeur singulière ($S_2 / S_0$). Si ce ratio est trop faible, les points sont quasi colinéaires (ex: os très fins mal échantillonnés), et le solveur applique un repli sécurisé (*fallback*) pour éviter les sauts numériques. |
| `max_rms_ratio` | `float` (défaut : `0.25` ou `None`) | Seuil relatif d'erreur RMS résiduelle après alignement rigide. Si le tissu musculaire ou le maillage subit une déformation plastique excessive, permet de filtrer l'aberration. |
| `weights` | `ndarray (N,)` (optionnel) | Poids attribués à chaque marqueur lors de la résolution SVD. Dans le cas d'un maillage avec skinning, utiliser les poids calculés par `compute_marker_stability_weights` qui favorisent les sommets purs et pénalisent les frontières d'articulation. |
| `root_x_degrees` | `float` (défaut : `0.0`) | Angle de rotation corrective (en degrés) à appliquer autour de l'axe X pour le joint racine. Permet de corriger les décalages de repère caméra (ex: passage repère caméra Y-down vers Y-up). |

---

## 5. Comment nous avons procédé pour les fonctions de `smplx_to_mixamo.py`

Historiquement, le projet utilisait `smplx_to_mixamo.py`, qui concentrait plusieurs responsabilités fortement intriquées :
- Définition en dur des index de joints SMPL-X (`0..54`) et des noms d'os Mixamo (`mixamorig:LeftArm`, etc.).
- Fonctions mathématiques de quaternions mélangées avec des opérations spécifiques glTF.
- Dépendance directe aux fichiers `.glb` de Mixamo.

### Stratégie de découplage adoptée dans `isolate/` :

1. **Extraction de l'algèbre pure dans `isolate/core/geometry.py`** :
   - Les fonctions de quaternions (`quat_mul`, `quat_inv`, `quat_normalize`, `rotate_vector`, `matrix_to_quaternion`) ont été réécrites de façon autonome en pur NumPy.
   - La convention standard retenue est explicitement $[w, x, y, z]$.
   - Aucune dépendance vers `smplx`, `torch` ou `scipy` n'a été conservée dans le cœur.

2. **Découplage de la cinématique dans `isolate/retargeting/kinematics.py`** :
   - La fonction `global_to_local_hierarchy` a été rendue universelle : elle ne dépend plus d'une liste prédéfinie de bones Mixamo, mais d'un dictionnaire abstrait `parents: Dict[str, Optional[str]]`. Elle fonctionne donc aussi bien avec un rig Mixamo, un rig VRM, un squelette BVH ou un squelette personnalisé.

3. **Généralisation de la lecture de rigs dans les Adaptateurs (`isolate/adapters/`)** :
   - La logique d'extraction de bind pose glTF a été déplacée dans `GLTFMixamoAdapter` et `VRMAdapter`.
   - Au lieu d'imposer un avatar spécifique, l'adaptateur lit dynamiquement la structure du fichier `.glb` ou `.vrm` fourni et construit la structure générique `TargetSkeleton`.

4. **Introduction du Registre de Mapping Pivot (`isolate/mappings/`)** :
   - Remplacement des tables codées en dur par `standard_humanoid.py` et `MappingRegistry`.
   - Cela permet d'ajouter de nouvelles correspondances (ex: `BVH -> VRM` ou `Mixamo -> VRM`) sans toucher à une seule ligne du code de calcul géométrique.

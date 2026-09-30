# 🗺️ Feuille de Route Technologique & Algorithmique (ROADMAP) — mark-target v2.0

> **Branche de Développement :** `dev`  
> **Auteur :** Pipeline Signes Team / Core Engineering  
> **Statut :** Spécification & Plan d'Implémentation Actif  
> **Objectif :** Transformer le prototype mathématique en une suite de production temps-réel capable de convertir une **vidéo 2D brute** en un **personnage 3D (.glb) en T-Pose animé**, avec visualisation 3D des marqueurs et stabilité biomécanique absolue.

---

## 1. Réponses Architecturales aux Questions Clés

### Q1 : Est-il possible d'observer/visualiser les marqueurs directement sur les modèles 3D ?
**OUI, et c'est un atout majeur de débogage.**  
Dans l'implémentation de `mark-target`, chaque marqueur correspond à un sommet précis du maillage source ou cible (défini par son indice `vertex_id`). Il est possible de les observer de deux façons :
1. **Mode Debug GLB/VRM :** Injecter des sphères 3D colorées (primitives géométriques sans poids de skinning) aux coordonnées des sommets sélectionnés. Chaque os a sa propre couleur (ex: avant-bras en cyan, pouce en violet).
2. **Visualiseur Web / Viewport temps réel :** Afficher un nuage de points colorés superposé sur la surface semi-transparente du modèle 3D pour inspecter la répartition du Farthest Point Sampling (FPS) et détecter les déformations de peau anormales.

### Q2 : Est-il possible de prendre en entrée une vidéo 2D et de fournir en sortie un avatar `.glb` (fourni en T-Pose) ?
**OUI, c'est précisément le pipeline cible de cette roadmap.**  
La chaîne de traitement complète s'organise ainsi :
```
[ Vidéo 2D brute (MP4/WebM) ]
             ↓
[ Estimateur de Maillage 3D : HybrIK / 4D-Humans / MediaPipe / SMPL-X ]
             ↓
[ Séquence de sommets 3D temporels V(t) ]
             ↓
[ mark-target : Échantillonnage FPS + Marqueurs virtuels ]
             ↓
[ Modèle Cible .glb en T-Pose (Bind Pose) ]
             ↓
[ Solveur Algébrique : Kabsch-Umeyama + RANSAC + Bézier SQUAD ]
             ↓
[ Fichier .glb final animé en coordonnées locales ]
```
Le modèle cible `.glb` fourni en T-Pose fournit la géométrie de repos ($P_{\text{rest}}$). L'animation résolue est ensuite encodée directement sous forme d'une piste d'animation glTF (`AnimationSampler` et `AnimationChannel`) sans altérer le maillage de base.

---

## 2. Sélection & Justification des Algorithmes Algébriques Retenus

Après analyse critique des compromis performance / robustesse, voici le panier d'algorithmes validés pour la branche `dev` :

| Algorithme Sélectionné | Rôle dans le Pipeline | Pourquoi ce choix ? |
| :--- | :--- | :--- |
| **1. Kabsch-Umeyama (1991)** | Solveur d'alignement rigide + facteur d'échelle $c$ | Absorbe les variations de perspective de caméra 2D et les dissemblances de corpulence sans fausser l'orientation. |
| **2. Courbes de Bézier Sphériques (SQUAD sur $\mathbb{S}^3$)** | Continuité temporelle et anti-jitter des quaternions | Garantit une continuité $C^1/C^2$ des rotations, supprime les micro-vibrations et permet d'interpoler les frames intermédiaires (gain x5 en vitesse). |
| **3. RANSAC + IRLS (Perte de Huber)** | Rejet des aberrations (outliers) et occlusions | Évite qu'un marqueur corrompu (ex: main qui passe derrière le torse) ne fasse pivoter tout le membre à 180°. |
| **4. Algèbre de Lie $\mathfrak{so}(3)$ / Logarithme matriciel** | Linéarisation et calcul dans l'espace tangent | Permet de moyenner et dériver les rotations par de simples sommes vectorielles sans singularité de matrice. |
| **5. Tenseur de Green-Lagrange** | Détection d'élasticité non rigide locale | Mesure si la peau s'étire excessivement près d'une articulation et désactive automatiquement les marqueurs instables. |
| **6. Projection sur Cône Convexe (Joint Limits)** | Clamping anatomique des angles articulaires | Empêche mathématiquement toute luxation ou hyperextension (coudes, genoux, doigts). |

---

## 3. Découpage du Pipeline par Sous-Tâches Mathématiques

### Sous-Tâche 1 : Pré-Traitement Temporel des Trajectoires 3D
* **Algorithme :** Algorithme de De Casteljau appliqué aux coordonnées euclidiennes $\mathbb{R}^3$.
* **Entrée :** Trajectoire brute des marqueurs extraits de l'estimateur vidéo.
* **Sortie :** Trajectoires lissées avec élimination du bruit haute fréquence.

### Sous-Tâche 2 : Résolution Robuste d'Orientation
* **Algorithme :** RANSAC sur quadruplets de marqueurs + Kabsch-Umeyama pondéré par M-estimateur de Huber.
* **Entrée :** Nuage de marqueurs de repos $P_{\text{rest}}$ et nuage courant $P(t)$.
* **Sortie :** Matrice de rotation propre $R \in \text{SO}(3)$ et facteur d'échelle $c$.

### Sous-Tâche 3 : Filtrage sur la Variété Sphérique des Quaternions
* **Algorithme :** SQUAD (Spherical and Quadrangle Bézier Splines) sur $\mathbb{S}^3$.
* **Entrée :** Quaternions bruts résolus frame par frame.
* **Sortie :** Courbe continue de rotation avec tangentes harmonieuses.

### Sous-Tâche 4 : Projection Biomécanique & Hiérarchique
* **Algorithme :** Décomposition Swing-Twist et projection sur cônes elliptiques de contraintes.
* **Entrée :** Quaternions locaux $Q_{\text{local}} = Q_{\text{parent}}^{-1} \times Q_{\text{bone}}$.
* **Sortie :** Rotations bornées garantissant la conformité anatomique.

### Sous-Tâche 5 : Injection dans le Conteneur .GLB
* **Technologie :** `pygltflib` avec création des accesseurs de temps et de quaternions.
* **Entrée :** Modèle `.glb` original en T-Pose + Clip d'animation.
* **Sortie :** Nouveau fichier `.glb` intégrant la piste d'animation prête pour le web ou les moteurs de jeu.

---

## 4. Calendrier d'Exécution par Phases

```
Phase 1 (Actuelle) : Spécification, documentation et stabilisation de la branche dev.
Phase 2 : Implémentation du solveur Kabsch-Umeyama et du module de lissage Bézier/De Casteljau.
Phase 3 : Intégration du module RANSAC/Huber et du tenseur de Green-Lagrange.
Phase 4 : Développement du pipeline End-to-End Vidéo 2D -> SMPL-X/HybrIK -> mark-target -> GLB T-Pose.
Phase 5 : Interface de prévisualisation 3D Web (Three.js) avec affichage temps réel des marqueurs.
```

---

## 5. Journal des Modifications (Changelog de la branche dev)

* **v1.0.0-dev (2026-09-30) :**
  - Initialisation de la branche `dev`.
  - Rédaction et intégration de la feuille de route formelle `ROADMAP.md`.
  - Spécification des algorithmes algébriques complémentaires (Bézier SQUAD, Umeyama, RANSAC, Lie $\mathfrak{so}(3)$).
  - Validation architecturale du pipeline Vidéo 2D $\rightarrow$ .GLB T-Pose.

## 6. Architecture Modulaire : Articulation HybrIK-X & mark-target

```
[ Étape 1 : Vision par Ordinateur (GPU / PyTorch) ]
  Vidéo 2D (MP4) 
      └──> HybrIK-X (Estimation de pose & maillage paramétrique SMPL-X)
            └──> Génération de `res.pk` (sommets déformés, quaternions, transl)

[ Étape 2 : Moteur de Retargeting Géométrique (CPU / NumPy) ]
  Fichier `res.pk` + Modèle Cible `.glb` (en T-Pose)
      └──> mark-target (Surface Sampling FPS + Poids LBS)
            └──> Solveur Kabsch-Umeyama (SVD + Roll complet 6-DoF)
            └──> Kinematics (Conversion espace Monde -> Hiérarchie Locale)
            └──> Exporteur pygltflib
                  └──> Fichier final `out_animated.glb`
```

### Pourquoi ce découplage est optimal :
1. **Isolation des dépendances :** HybrIK-X nécessite PyTorch et CUDA. `mark-target` tourne en quelques millisecondes sur un simple CPU avec NumPy.
2. **Agnosticisme de la source :** Tout autre estimateur de maillage peut remplacer HybrIK-X sans modifier une seule ligne du solveur de retargeting.
3. **Fidélité biomécanique :** HybrIK-X résout la vision, `mark-target` garantit qu'aucun os ne vrille sur le rig cible.

## 7. Prise en Charge Bidirectionnelle du Format BVH (.bvh)

Comme implémenté dans `adapters/bvh_adapter.py` via la double héritance `class BVHAdapter(SourceAdapter, TargetAdapter)` :
* **BVH en Source :** Lit la section `HIERARCHY` et `MOTION` d'un fichier MoCap BVH, extrait les trajectoires spatiales 3D des joints pour alimenter le solveur.
* **BVH vers Cible Arbitraire :** Retargeting direct vers `.glb` (Mixamo), `.vrm` (VRoid) ou un autre `.bvh` via le pont `MappingRegistry`.
* **Exemple CLI supporté nativement :**
  ```bash
  python run_marker_retarget.py --source anim.bvh --source-type bvh --target avatar.glb --target-type mixamo --output out_anim.glb
  ```

## 8. Étape 1 Réalisée : Configuration & Stabilisation du Mapping Universel

* **Statut :** Complété & Validé par tests unitaires (`tests/test_mappings.py`).
* **Modifications apportées :**
  1. `mappings/standard_humanoid.py` :
     - Intégration complète des 55 os standards (22 corps + 3 tête/visage/mâchoire/yeux + 30 segments de doigts).
     - Correction des membres inférieurs Mixamo (ajout des jambes `LeftUpLeg`, `LeftLeg`, `LeftFoot`, `LeftToeBase` etc.).
     - Ajout des tables bidirectionnelles `SMPLX_TO_STANDARD` (55 joints HybrIK-X) et `BVH_TO_STANDARD` (conventions CMU/Biovision).
  2. `mappings/registry.py` :
     - Résolution automatique bidirectionnelle multi-formats (`smplx`, `hybrik`, `mixamo`, `vrm`, `bvh`, `standard`).
     - Support des correspondances dynamiques sur mesure (`register()`).
  3. `tests/test_mappings.py` :
     - Suite de tests unitaires couvrant les 55 os, la hiérarchie parentale et les conversions croisées (7/7 tests OK).

## 9. Documentation du Mapping Personnalisé (Custom Rig / Mimo / Blender Rigify)

* **Documentation complète :** Voir `docs/CUSTOM_MAPPING.md`.
* **Fonctionnalités livrées :**
  - Chargement de fichier JSON : `MappingRegistry.register_from_json(path, src, tgt)`
  - Export de template JSON : `MappingRegistry.export_to_json(src, tgt, path)`
  - Argument CLI direct dans `run_marker_retarget.py` : `--custom-mapping custom.json`

## 10. Périmètre Délimité : Moteur de Retargeting Pur

Conformément aux directives d'ingénierie :
* **mark-target ne fait QUE le retargeting géométrique 3D déterministe.**
* L'extraction amont depuis la vidéo 2D (pixels -> 3D) est expressément déléguée à une pipeline indépendante dédiée (ex: **Mimo**, HybrIK-X, ou 4D-Humans).
* `mark-target` prend les sorties 3D de ces outils (`.pk`, `.bvh`, dictionnaires de sommets) et garantit un transfert mathématiquement optimal sans artefacts vers le `.glb` cible.

## 11. Étape 2 Réalisée : Solveur Hybride (Kabsch-Umeyama & Bézier SQUAD)

* **Statut :** Complété & Validé par 18 tests unitaires (`Ran 18 tests in 0.022s OK`).
* **Modifications apportées :**
  1. `core/kabsch.py` :
     - Implémentation de `kabsch_umeyama_rotation()` et `weighted_kabsch_umeyama_rotation()`.
     - Résolution conjointe de la rotation $R \in \text{SO}(3)$, du facteur d'échelle $c = \frac{\operatorname{Tr}(DS)}{\sigma_A^2}$ et de la translation $t$.
     - Absorption naturelle des disparités morphologiques et d'épaisseur corporelle.
  2. `retargeting/smoothing.py` :
     - Algorithme de De Casteljau dans $\mathbb{R}^3$ pour le lissage des trajectoires de position.
     - Splines sphériques Bézier **SQUAD** et interpolation **SLERP** sur la 3-sphère des quaternions $\mathbb{S}^3$.
     - Fonctions exponentielles et logarithmiques de l'algèbre de Lie $\mathfrak{so}(3)$.
     - Élimination prouvée du bruit et du jitter haute fréquence (`test_jitter_reduction`).
  3. `retargeting/solver.py` :
     - Support du flag `use_umeyama=True` dans `solve_frame_multi_kabsch`.
     - Méthode de séquence complète `solve_sequence(...)` avec lissage temporel automatisé.
  4. `run_marker_retarget.py` :
     - Ajout des drapeaux CLI `--use-umeyama` et `--smoothing-factor FLOAT`.
  5. `tests/test_umeyama_and_smoothing.py` :
     - 6 tests unitaires spécifiques validant l'exactitude d'Umeyama et la continuité de Bézier SQUAD.

## 12. Étape 3 Réalisée : Filtre RANSAC Algébrique & Tenseur de Green-Lagrange

* **Statut :** Complété & Validé par 22 tests unitaires (`Ran 22 tests — OK`).
* **Modifications apportées :**
  1. `markers/ransac_filter.py` :
     - Implémentation de `ransac_kabsch_alignment()` : sélectionne les triplets de consensus, évalue les résidus sur tous les marqueurs de l'os et ré-estime la rotation optimale en rejetant les marqueurs aberrants (occlusions, glissements).
     - Implémentation de `compute_green_lagrange_strain()` : calcule le tenseur $E = \frac{1}{2}(F^T F - I_3)$ et l'énergie de déformation non rigide $\|E\|_F$.
  2. `core/kabsch.py` :
     - Condition de rang robuste adaptée pour $N \ge 3$ points.
  3. `retargeting/solver.py` :
     - Intégration de `use_ransac` et `max_strain_threshold` dans `solve_frame_multi_kabsch` et `solve_sequence`.
  4. `run_marker_retarget.py` :
     - Nouveaux drapeaux CLI : `--use-ransac` et `--max-strain FLOAT`.
  5. `tests/test_ransac_and_deformation.py` :
     - 4 tests unitaires validant l'invariance rigide de Green-Lagrange, la détection d'élongation non rigide, le rejet RANSAC d'outliers et la protection du solveur.

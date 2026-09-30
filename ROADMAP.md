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

# Méthodologie et Fondements Théoriques : Retargeting par Marqueurs 3D

Ce document expose les principes scientifiques, mathématiques et algorithmiques de la méthode de retargeting par marqueurs de surface (inspirée des travaux biomécaniques de Marilyn Keller et al., *SMPL2AddBiomechanics*, SIGGRAPH Asia 2023, et des principes de mocap optique Vicon/OptiTrack).

---

## 1. Problématique fondamentale du retargeting

Le retargeting d'animation consiste à transférer le mouvement capturé sur un sujet source (ex: maillage SMPL-X, données MoCap) vers un avatar 3D cible (ex: personnage Mixamo ou avatar VRM).

### Les écueils des approches classiques

1. **Retargeting direct par rotations de joints (Joint-to-joint transfer)** :
   - Suppose que les repères locaux de repos (bind poses) des deux squelettes sont identiques ou parfaitement alignés.
   - En pratique, deux rigs n'ont quasiment jamais les mêmes orientations de repères locaux (axes d'os orientés selon X, Y ou Z, roll intrinsèque différent, décalages anatomiques).
   - Les corrections d'alignement manuel sont fragiles et se désynchronisent facilement le long de la chaîne cinématique.

2. **Cinématique Inverse basée sur les positions (Position IK)** :
   - Ajuste les angles de rotation pour atteindre des cibles de position dans l'espace.
   - Très sensible aux différences de longueurs de membres : un bras cible plus court que la source produit des singularités, des blocages en hyperextension ou des discontinuités temporelles brutales.

3. **Approche par Mono-Marqueur + Visée (Aim vector)** :
   - Un marqueur unique par articulation calcule le vecteur orienté $\mathbf{v} = \mathbf{p}_{\text{enfant}} - \mathbf{p}_{\text{parent}}$.
   - **Limite structurelle majeure** : Un vecteur dans l'espace $\mathbb{R}^3$ ne possède que 2 degrés de liberté (azimut et élévation). Il est mathématiquement impossible de contraindre la rotation axiale propre autour de l'os (le **roll** ou torsion de l'avant-bras / des doigts).

---

## 2. La solution : Marqueurs multiples et algorithme de Kabsch

En plaçant **au moins 3 marqueurs non-colinéaires** solidaires de chaque segment osseux, le segment se comporte comme un corps rigide dans l'espace tridimensionnel, fixant l'intégralité des **3 degrés de liberté** ($SO(3)$).

```
   Nuage de repos P_rest                     Nuage déformé P_t
        (Frame 0)                                (Frame t)
        
         M1 --- M2                                M1' --- M2'
          \     /                                  \     /
           \   /         ===> KABSCH SVD ===>       \   /
             M3                                       M3'
             
    Orientation complète résolue : Roll + Pitch + Yaw (3 DoF)
```

### Formulation Mathématique de l'Algorithme de Kabsch

Soit deux ensembles correspondants de $N$ points 3D pour un os donné :
- $\mathbf{P}^{\text{rest}} = \{\mathbf{p}_1, \dots, \mathbf{p}_N\} \subset \mathbb{R}^3$ (positions en pose de repos)
- $\mathbf{P}^{t} = \{\mathbf{q}_1, \dots, \mathbf{q}_N\} \subset \mathbb{R}^3$ (positions déformées à l'instant $t$)
- $w_i \ge 0$ : poids de confiance associé au marqueur $i$.

#### Étape 1 : Centrage des nuages
On calcule les barycentres respectifs :
$$\mathbf{c}^{\text{rest}} = \frac{\sum_{i=1}^N w_i \mathbf{p}_i}{\sum_{i=1}^N w_i}, \quad \mathbf{c}^{t} = \frac{\sum_{i=1}^N w_i \mathbf{q}_i}{\sum_{i=1}^N w_i}$$

Les coordonnées centrées sont :
$$\tilde{\mathbf{p}}_i = \mathbf{p}_i - \mathbf{c}^{\text{rest}}, \quad \tilde{\mathbf{q}}_i = \mathbf{q}_i - \mathbf{c}^{t}$$

#### Étape 2 : Matrice de covariance croisée
On construit la matrice de dispersion pondérée $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ :
$$\mathbf{H} = \sum_{i=1}^N w_i \tilde{\mathbf{p}}_i \tilde{\mathbf{q}}_i^T$$

#### Étape 3 : Décomposition en valeurs singulières (SVD)
On décompose $\mathbf{H}$ par SVD :
$$\mathbf{H} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

où $\mathbf{U}, \mathbf{V} \in O(3)$ sont des matrices orthogonales et $\mathbf{\Sigma} = \text{diag}(\sigma_1, \sigma_2, \sigma_3)$ contient les valeurs singulières classées $\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge 0$.

#### Étape 4 : Garantie anti-réflexion ($\det(\mathbf{R}) = +1$)
Une décomposition SVD standard peut produire une symétrie axiale (réflexion miroir) si $\det(\mathbf{V} \mathbf{U}^T) = -1$.
Pour garantir que la transformation obtenue appartient bien au groupe spécial orthogonal $SO(3)$ :
$$d = \text{sign}(\det(\mathbf{V} \mathbf{U}^T))$$
$$\mathbf{D} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & d \end{pmatrix}$$

La matrice de rotation optimale au sens des moindres carrés est alors :
$$\mathbf{R} = \mathbf{V} \mathbf{D} \mathbf{U}^T$$

---

## 3. Échantillonnage de surface & Stabilité Numérique

Pour que l'algorithme Kabsch soit stable, les marqueurs ne doivent être ni regroupés en un point, ni alignés sur une droite (colinéaires).

### A. Échantillonnage par Farthest Point Sampling (FPS)
Pour chaque os, le module `isolate.markers.surface_sampling` procède ainsi :
1. Isole les sommets du maillage appartenant au joint (segmentation de sous-maillage par poids de skinning dominants).
2. Sélectionne le premier marqueur comme le sommet le plus proche du centre de rotation articulaire.
3. Choisit itérativement les sommets suivants de manière à **maximiser la distance géodésique/euclidienne** avec les sommets déjà sélectionnés :
   $$i_{k+1} = \arg\max_{j} \left( \min_{m \in \text{sélection}} \|\mathbf{v}_j - \mathbf{v}_m\| \right)$$
4. Calcule l'aire du triangle formé par les 3 premiers points : si $\text{Aire} < \epsilon$, la configuration est rejetée car trop proche d'une ligne droite.

### B. Pondération non-linéaire par la pureté de skinning (LBS)
Les sommets situés aux articulations (ex: pli du coude ou du genou) subissent une déformation d'étirement plastique qui perturbe l'hypothèse de corps rigide de Kabsch.
Le module applique une fonction de pondération :
$$w_i = \max(0, W_{i, \text{propre}})^2 \times \max(0, W_{i, \text{propre}} - W_{i, \text{second}})$$
- Un sommet contrôlé à 98% par l'avant-bras et 2% par le bras aura un poids très élevé.
- Un sommet contrôlé à 50% par l'avant-bras et 50% par le bras aura un poids nul ($w_i = 0$), excluant ainsi les artefacts de pli de peau du calcul d'orientation.

---

## 4. Résolution cinématique : Du Global au Local

La rotation calculée par Kabsch $\mathbf{R}_{\text{move}}$ s'applique dans le repère global du monde :
$$\mathbf{Q}_{\text{global}}^t = \mathbf{Q}_{\text{move}} \otimes \mathbf{Q}_{\text{rest}}$$

Pour qu'un moteur de rendu 3D (Three.js, Unity, Blender, Unreal) anime le squelette sans le disloquer, chaque os doit recevoir une rotation **locale** relative à son parent immédiat :
$$\mathbf{Q}_{\text{local}}^t(\text{os}) = \left(\mathbf{Q}_{\text{global}}^t(\text{parent})\right)^{-1} \otimes \mathbf{Q}_{\text{global}}^t(\text{os})$$

Pour l'os racine (`Hips`), la rotation locale est égale à son orientation globale dans le repère monde.

# 📐 ANALYSE FORENSIC DU CODE : EXTRACTION DE LA POSE DE REPOS & ARCHITECTURE DU RETARGETING

> **Avertissement de méthode :** Ce document est rédigé à partir de l'analyse stricte et exclusive des **lignes de code exécutées** dans le projet `mark_target`, et non à partir des fichiers README ou de documentations théoriques.

---

## 1. La Question Centrale : Comment la Source Animée Obtient-elle sa Pose de Repos ?

Lorsqu'un fichier source (contenant des centaines de frames d'animation) est chargé, comment le moteur extrait-il sa pose de repos neutre sans la confondre avec le premier mouvement de l'animation ?

La réponse dépend du format source et est codée dans les adaptateurs :

---

### A. Cas du format MoCap BVH (`.bvh`)

Le standard Biovision BVH sépare physiquement le fichier en deux blocs :
1. `HIERARCHY` : La définition du squelette et des offsets géométriques relatifs.
2. `MOTION` : La liste chronologique des frames contenant les angles d'Euler et la translation de racine.

#### 🔍 Ce que fait le code dans `adapters/bvh_parse.py` :
* **Preuve de code** ([adapters/bvh_parse.py](file:///D:/dev/mark_target/adapters/bvh_parse.py#L328-L329)) :
```python
rest_pos, _, _ = _forward(joints, _zero_motion(joints, motion.shape[1]))
rest_vectors = {name: rest_pos[name][0] for name in rest_pos}
```

* **Preuve de code** ([adapters/bvh_parse.py](file:///D:/dev/mark_target/adapters/bvh_parse.py#L500-L503)) :
```python
def _zero_motion(joints: Dict[str, BvhJoint], n_channels: int) -> np.ndarray:
    return np.zeros((1, n_channels), dtype=np.float64)
```

#### ⚙️ Mécanisme exact :
1. **La Frame 0 N'EST PAS la pose de repos.** À la frame 0, le personnage est souvent déjà en train de marcher, de parler ou de bouger. Le moteur **n'utilise pas** la frame 0 comme référence.
2. Le moteur génère une frame artificielle appelée `_zero_motion` où **tous les canaux de rotation (Euler) sont rigoureusement mis à zéro (`0.0`)**.
3. À angles nuls, toutes les matrices de rotation locales valent la matrice identité $I_3$.
4. La fonction cinématique directe `_forward()` propage les translations le long de l'arbre en sommant uniquement les vecteurs `OFFSET` déclarés dans le bloc `HIERARCHY` :
   $$\text{Pos}_{\text{rest}}(\text{root}) = \text{OFFSET}(\text{root})$$
   $$\text{Pos}_{\text{rest}}(\text{enfant}) = \text{Pos}_{\text{rest}}(\text{parent}) + \text{OFFSET}(\text{enfant})$$
5. **Résultat :** Le squelette au repos obtenu correspond exactement à la géométrie de fabrication du rig BVH (généralement une T-Pose canonique définie par le studio de MoCap).

---

### B. Cas du format Vision 3D / HybrIK (`.pk`)

Dans les fichiers issus de réseaux de neurones de vision (HybrIK / SMPL-X), il n'y a pas de bloc de hiérarchie BVH standard.

#### 🔍 Ce que fait le code dans `adapters/hybrik_pk_adapter.py` :
* **Cas 1 : Positions articulaires 3D prédites (`pred_xyz_jts`)** ([adapters/hybrik_pk_adapter.py](file:///D:/dev/mark_target/adapters/hybrik_pk_adapter.py#L56-L60)) :
```python
# Frame 0 comme repos de référence
for j_idx in range(min(n_jts, len(smplx_names))):
    bone_name = smplx_names[j_idx]
    rest_markers[bone_name] = raw_jts[0, j_idx].reshape(1, 3)
```
Dans ce cas précis, le fichier ne fournissant pas de bind-pose neutre, le code prend par convention les positions de la **Frame 0**.

* **Cas 2 : Quaternions de joints seuls** ([adapters/hybrik_pk_adapter.py](file:///D:/dev/mark_target/adapters/hybrik_pk_adapter.py#L77-L82)) :
```python
canonical_pos = CanonicalBufferArmature.get_canonical_positions()
rest_markers = {b: np.asarray(p, dtype=np.float64).reshape(1, 3) for b, p in canonical_pos.items()}
```
Le code injecte les positions 3D parfaites de l'armature canonique (T-Pose mathématique pré-calculée sur 55 os).

---

### C. Cas du modèle 3D Cible (Avatar glTF / Mixamo `.glb`)

Pour l'avatar qui doit recevoir l'animation :

#### 🔍 Ce que fait le code dans `adapters/gltf_mixamo_adapter.py` :
* **Preuve de code** ([adapters/gltf_mixamo_adapter.py](file:///D:/dev/mark_target/adapters/gltf_mixamo_adapter.py#L46-L90)) :
Le fichier `.glb` contient la scène 3D statique. Les os du squelette sont des nœuds (`nodes`) possédant chacun :
- Une translation locale au repos (`node.translation`).
- Un quaternion local au repos (`node.rotation` ou $q_{\text{identity}}$ si absent).

Le code calcule la cinématique directe mondiale (`global_pose` et `global_position`) :
```python
def global_position(i: int) -> np.ndarray:
    if i not in parent_of_idx:
        p = node_local_trans[i]
    else:
        p_parent = global_position(parent_of_idx[i])
        q_parent = global_pose(parent_of_idx[i])
        p = p_parent + rotate_vector(q_parent, node_local_trans[i])
    return p
```
Cela fournit à la fois :
1. `skeleton.rest_positions` : Les coordonnées 3D mondes des articulations de l'avatar au repos.
2. `skeleton.rest_rotations` : Les orientations mondes des os dans leur **Bind Pose** (la pose dans laquelle le maillage a été skinné).

---

## 2. Comment le Moteur Aligne-t-il la Source et la Cible ? (A-Pose $\leftrightarrow$ T-Pose)

Une fois que le moteur dispose de la pose de repos source ($\text{Pos}_{\text{rest, src}}$) et de la pose de repos cible ($\text{Pos}_{\text{rest, tgt}}$) :

#### 🔍 Ce que fait le code dans `mappings/auto_mapper.py` :
* **Preuve de code** ([mappings/auto_mapper.py](file:///D:/dev/mark_target/mappings/auto_mapper.py#L225-L245)) :
1. Pour chaque os (par exemple le bras gauche `LeftArm`), il extrait le vecteur directeur reliant l'os à son enfant dans la source :
   $$\vec{v}_{\text{src}} = \text{Enfant}_{\text{src, rest}} - \text{Parent}_{\text{src, rest}}$$
2. Il fait de même pour la cible :
   $$\vec{v}_{\text{tgt}} = \text{Enfant}_{\text{tgt, rest}} - \text{Parent}_{\text{tgt, rest}}$$
3. Si la source est en T-Pose ($\vec{v}_{\text{src}} \approx [1, 0, 0]$ horizontal) et la cible en A-Pose ($\vec{v}_{\text{tgt}} \approx [0.7, -0.7, 0]$ incliné à 45°) :
   Le moteur calcule le quaternion de correction minimale :
   $$q_{\text{align}} = \text{quat\_from\_two\_vectors}(\vec{v}_{\text{src}}, \vec{v}_{\text{tgt}})$$

#### 🔍 Application dans `run_marker_retarget.py` ([L228-L234](file:///D:/dev/mark_target/run_marker_retarget.py#L228-L234)) :
À chaque frame $t$, lorsque la rotation de mouvement $q_{\text{delta}}$ est calculée par le solveur SVD, elle est conjuguée par cette rotation d'alignement :
$$q_{\text{delta, aligned}} = q_{\text{align}} \cdot q_{\text{delta}} \cdot q_{\text{align}}^{-1}$$
C'est cette opération mathématique précise qui empêche l'avatar de se disloquer les bras ou les jambes.

---

## 3. La Vérité sur le Mapping : Est-il "Sans Regarder la Nomenclature" ?

### ❌ NON : Le Mapping Dépend Strictement de Chaînes Textuelles (Regex)

Le code prouve que le mapping n'est **pas** réalisé par des marqueurs ou de la géométrie pure :
* **Preuve de code** ([mappings/auto_mapper.py](file:///D:/dev/mark_target/mappings/auto_mapper.py#L34-L86)) :
```python
LEFT_PATTERNS = [re.compile(r"left", re.I), re.compile(r"(?:^|[_\.\-:])l(?:[_\.\-:]|$)", re.I), ...]
LIMB_PATTERNS = [
    ("Shoulder", ["shoulder", "clavicle", "collar"]),
    ("ForeArm", ["forearm", "lowerarm", "lower_arm", "fore_arm", "ulna", "elbow"]),
    ("Arm", ["upperarm", "upper_arm", "up_arm", "bicep", "shldr", "arm"]),
    ("Hand", ["hand", "wrist", "palm"]),
    ("Leg", ["lowerleg", "lower_leg", "lowleg", "low_leg", "calf", "shin", "knee"]),
    ("UpLeg", ["upleg", "upperleg", "upper_leg", "up_leg", "thigh"]),
    ...
]
```
Si deux modèles utilisent des noms complètement arbitraires (`Bone.001`, `Bone.002`, `Segment_A`), l'Auto-Mapper ne trouvera **aucune correspondance**.

---

## 4. Où Interviennent les Marqueurs et Pourquoi sont-ils Résistants ?

Les marqueurs n'interviennent qu'**après** que l'os a été associé par son nom.

### Le Rôle des Cages Virtuelles Tétraédriques (`markers/virtual_cage.py`)
1. Dans un fichier MoCap BVH, chaque articulation n'est qu'un point $P \in \mathbb{R}^3$.
2. **Le problème mathématique :** 1 point n'a pas d'orientation. Si un os n'a qu'un seul enfant colinéaire, il est impossible de calculer son roulis (torsion axiale) avec un point unique.
3. **La solution codée** ([markers/virtual_cage.py](file:///D:/dev/mark_target/markers/virtual_cage.py#L65-L105)) :
   Le moteur construit pour chaque os un tétraèdre rigide de 4 marqueurs $\{P_0, P_1, P_2, P_3\}$ positionnés autour de l'axe de l'os dans son repère de repos.
   À chaque frame $t$, ces 4 marqueurs sont tournés dans l'espace selon le mouvement de l'os.

### La Résolution Mathématique par Kabsch SVD (`core/kabsch.py`)
Ayant 4 points au repos $P_{\text{rest}}$ et 4 points à la frame $t$ $P_t$ :
1. Calcul du barycentre et centrage des points.
2. Matrice de covariance croisée $H = P_{\text{rest}}^T P_t \in \mathbb{R}^{3 \times 3}$.
3. Décomposition SVD : $H = U \Sigma V^T$.
4. Rotation propre :
   $$R = V \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(V U^T) \end{pmatrix} U^T$$

**Pourquoi est-ce résistant ?**
- **0 itération / 0 risque de divergence** : Contrairement aux solveurs IK numériques (Jacobienne pseudo-inverse, CCD, FABRIK), Kabsch est une **forme fermée exacte**.
- **$\det(R) = +1$ garanti** : Aucun risque d'inversion miroir d'un membre.
- **Rang 3 plein garanti** : Grâce à la forme tétraédrique, la covariance $H$ n'est jamais dégénérée (pas de coplanarité).

---

## 5. Synthèse des Éléments Requis pour un Mapping Géométrique Réel ("Sans les Noms")

Pour s'affranchir complètement de la nomenclature textuelle à l'avenir, le composant à créer devra opérer en amont de l'Auto-Mapper :
1. **Extraction de la topologie de l'arbre (Théorie des graphes)** :
   - Détection de la racine (bassin) = sommet de degré entrant 0 et degré sortant $\ge 3$.
   - Détection des 5 extrémités (tête, 2 mains, 2 pieds) = feuilles de l'arbre.
2. **Plan Sagittal Géométrique ($X=0$)** :
   - Découpage spatial gauche/droite par rapport au plan de symétrie de la pose de repos, sans lire `"left"` ou `"right"`.
3. **Appariement morphologique (ICP squelettique)** :
   - Comparaison des ratios de longueurs d'os par rapport à l'armature canonique.

---
*Document certifié conforme au code source de `mark_target`.*

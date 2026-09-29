# Protocole de Benchmark et Métriques d'Évaluation

Ce document formalise les critères quantitatifs, les métriques objectives et le protocole expérimental permettant d'évaluer la pertinence et la performance de la **méthode par marqueurs (Kabsch multi-points)** face aux autres méthodes de retargeting existantes dans l'état de l'art.

---

## 1. Tableau comparatif des approches de retargeting

| Critère d'évaluation | Méthode Marqueurs Multi-Points (Kabsch) | Transfert Direct (Joint-to-Joint) | IK Positionnelle (Inverse Kinematics) | Transfert de Skinning (ex: MIBURI) |
|---|---|---|---|---|
| **Gestion du Roll (3 DoF)** | **Excellente** (fixé rigoureusement par $\ge 3$ points) | Dépendante de la calibration d'axe | Moyenne (risque de torsion axiale indéterminée) | Sans objet (agit sur les sommets, pas les os) |
| **Indépendance de la bind pose** | **Totale** (mesure le déplacement géométrique réel dans l'espace) | Nulle (exige des axes locaux parfaitement identiques) | Faible (sensible aux différences d'échelle) | Nulle (recalcule des poids pour le maillage complet) |
| **Sensibilité aux proportions corporelles** | **Très faible** (les marqueurs s'adaptent à la surface de chaque os) | Faible | **Très élevée** (hyperextension si membre cible plus court) | Faible |
| **Temps de calcul par frame** | **< 1 ms / frame** (algèbre linéaire SVD analytique) | < 0.1 ms / frame | 5 à 50 ms / frame (optimisation itérative) | Lourd (calcul initial sur des milliers de sommets) |
| **Continuité temporelle** | **Continue** (solution SVD unique sans saut local) | Continue | Risque de sauts locaux / discontinuités | Continue |
| **Compatibilité multi-formats** | **Universelle** (marche sur tout maillage/nuage de points) | Rigs squelettiques uniquement | Rigs squelettiques uniquement | Maillages denses uniquement |

---

## 2. Métriques Quantitatives d'Évaluation

Pour mesurer scientifiquement la qualité du retargeting, 4 métriques sont préconisées :

### A. MPJPE (*Mean Per Joint Position Error*) — en millimètres
Mesure la fidélité de position spatiale entre les articulations de la source et celles du modèle cible ré-échelonné :
$$\text{MPJPE} = \frac{1}{T \times J} \sum_{t=1}^T \sum_{j=1}^J \|\mathbf{p}_{j, t}^{\text{cible}} - s \cdot \mathbf{p}_{j, t}^{\text{source}}\|$$
- $T$ : nombre total de frames.
- $J$ : nombre de joints évalués.
- $s$ : facteur d'échelle global pour compenser la taille des personnages.

### B. Erreur Angulaire Géodésique (sur $SO(3)$) — en degrés
Mesure l'écart d'orientation intrinsèque pour chaque os, indépendamment de sa translation :
$$\Delta \theta(t, j) = \arccos \left( \frac{\text{Trace}(\mathbf{R}_{j, t}^{\text{source} \to \text{cible}} \cdot (\mathbf{R}_{j, t}^{\text{est}})^T) - 1}{2} \right)$$
- Une méthode efficace doit maintenir une erreur médiane $< 5^\circ$ sur les membres majeurs (bras, avant-bras) et $< 8^\circ$ sur les phalanges.

### C. Jerk Résiduel (Fluidité Temporelle) — en $\text{m/s}^3$ ou $\text{rad/s}^3$
Mesure la 3ᵉ dérivée temporelle de la position ou de la rotation (dérivée de l'accélération) :
$$\text{Jerk}(t) = \frac{\mathbf{a}(t + \Delta t) - \mathbf{a}(t)}{\Delta t}$$
- Un Jerk élevé indique des tremblements, du *jittering* ou des micro-sauts numériques. La méthode Kabsch produit un Jerk proche du mouvement source grâce à la régularité de la SVD.

### D. Débit de Traitement (Images par Seconde - FPS)
Mesure la latence d'inférence pour déterminer l'adéquation au temps-réel (MoCap live, réalité virtuelle, jeux vidéo).
- **Temps réel** : $\ge 30 \text{ FPS}$ (idéalement $\ge 60 \text{ FPS}$).
- La méthode Kabsch sur un rig de 55 os s'exécute typiquement entre **300 et 1200 FPS** sur un simple processeur CPU.

---

## 3. Protocole Expérimental Recommandé

Pour comparer le module `isolate` face à une autre méthode :

1. **Jeu de données de test standard** :
   - Sélectionner une séquence de mouvements comportant des torsions d'avant-bras prononcées et des configurations de doigts complexes (ex: vocabulaire LSF / Langue des Signes, ou mouvements martiaux).
2. **Appliquer les deux méthodes sur les mêmes données d'entrée**.
3. **Calculer les 4 métriques ci-dessus** et tracer :
   - Le profil d'erreur angulaire au cours du temps pour les segments critiques (avant-bras, mains).
   - L'histogramme des erreurs RMS résiduelles.
4. **Vérification visuelle qualitative** :
   - Présence ou absence de déformations non-naturelles de la peau au niveau des coudes et des poignets.
   - Respect de l'horizontalité et de la verticalité du buste.

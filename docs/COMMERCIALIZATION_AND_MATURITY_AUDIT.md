# 🚀 Audit de Maturité & Plan de Commercialisation : mark-target

Ce document établit un diagnostic stratégique et technique transparent sur la maturité de **`mark-target`**, son potentiel commercial et les chantiers indispensables pour en faire un produit B2B / B2C viable.

---

## 1. Diagnostic de Maturité Actuelle : Où en sommes-nous ?

### Ce qui est Mature (Niveau Industriel & Mathématique) : 95%
* **Moteur Géométrique :** Algorithmes analytiques en forme fermée (Kabsch-Umeyama SVD $3 \times 3$, absorption d'échelle $c$).
* **Filtres de Robustesse :** RANSAC pour éliminer les occlusions et tenseur de Green-Lagrange pour jauger la déformation élastique de chair.
* **Continuité Temporelle :** Bézier SQUAD sphérique sur $\mathbb{S}^3$ garantissant une continuité $C^2$ anti-tremblement.
* **Standardisation :** Registre universel 55 os avec support personnalisé JSON (`--custom-mapping`).
* **Vitesse :** $< 0.4\text{ ms}$ par image sur un simple CPU, sans dépendance GPU.
* **Stabilité du Code :** 22 tests unitaires passant en $0.04\text{ s}$ sur la branche `dev`.

### Ce qui Manque pour Être Commercialisable : Le "Last-Mile Gap" (50% Produit)
Un studio de jeu, un créateur 3D ou une agence vidéo n'achète pas un script en ligne de commande Python nécessitant d'installer `pip`, `numpy` et de lancer des commandes shell.

---

## 2. Les 4 Bloqueurs Actuels à la Commercialisation

| Bloqueur | Symptôme Client | Impact Commercial |
| :--- | :--- | :--- |
| **1. Pas de flux vidéo direct de bout en bout** | L'utilisateur doit faire tourner Mimo séparément pour générer les poses 3D avant de lancer mark-target. | Friction critique : 90% des clients abandonnent s'il n'y a pas un bouton unique *"Vidéo MP4 $\rightarrow$ Avatar GLB animé"*. |
| **2. Pas d'Addon / Plugin DCC** | Les animateurs travaillent dans Blender, Maya ou Unreal Engine 5. | Rejet par les artistes 3D qui refusent de quitter leur logiciel de travail. |
| **3. Glissement résiduel des pieds (Foot-Skating)** | Si la hauteur des jambes de l'avatar diffère de la vidéo, les pieds patinent au sol lors de la marche. | Refus de validation par les directeurs d'animation exigeant un contact sol parfait. |
| **4. Packaging grand public / Entreprise** | Dépendances Python manuelles, pas d'exécutable `.exe` Windows ou `.dmg` Mac. | Limite l'usage aux seuls ingénieurs Python. |

---

## 3. Feuille de Route pour Commercialiser mark-target (4 Chantiers)

### 🛠️ Chantier 1 : Module de Verrouillage au Sol (Foot-Plant IK)
* **Objectif :** Éliminer 100% du patinage des pieds sur le plan horizontal $XZ$.
* **Mise en œuvre :**
  1. Détecter les frames d'appui (quand la vitesse de la cheville passe sous un seuil $\epsilon$).
  2. Verrouiller la position cartésienne du pied au sol durant la phase d'appui.
  3. Appliquer une passe Inverse Kinematics (2-Bone IK sur hanche-genou-cheville) pour réajuster le genou.

### 🔌 Chantier 2 : L'Addon Blender Officiel (`mark-target-blender`)
* **Pourquoi :** Blender possède plus de 3 millions d'utilisateurs actifs, c'est le canal d'acquisition gratuit numéro 1.
* **Fonctionnalités :**
  - Panneau latéral dans Blender : Sélection de l'armature Mixamo/Rigify.
  - Bouton *"Importer Vidéo ou MoCap"* $\rightarrow$ Application automatique des keyframes en 1 seconde.
  - Vente sur **BlenderMarket** et **Gumroad** (prix recommandé : 39$ à 59$).

### 🌐 Chantier 3 : La Plateforme SaaS Cloud (No-Code B2B / B2C)
* **Architecture :**
  - **Frontend :** WebApp React Three.js (glisser-déposer une vidéo MP4 + avatar 3D GLB).
  - **Backend Worker GPU :** Découpe vidéo $\rightarrow$ Inférence Mimo amont.
  - **Moteur Aval :** Inférence ultra-rapide mark-target sur CPU.
  - **Résultat :** Prévisualisation 3D en direct dans le navigateur et téléchargement du `.GLB` / `.FBX`.
* **Modèle Économique :**
  - **Gratuit :** 3 animations / mois en basse résolution.
  - **Pro (19$ - 49$ / mois) :** Retargeting illimité, 60 FPS, export FBX/GLB propre, support des doigts.
  - **Entreprise API :** Facturation à la minute de vidéo (0.10$ / minute).

### 📦 Chantier 4 : SDK Python & Binaire Windows Autonome
* Publication sur PyPI : `pip install mark-target`.
* Compilation d'un binaire autonome `.exe` (via PyInstaller) : glisser une vidéo sur l'exécutable génère le fichier 3D sans rien installer.

---

## 4. Modèle Commercial & Cibles Prioritaires

1. **Studios de Jeux Indépendants & VR :**
   - *Pain point :* Une combinaison MoCap (Xsens, Rokoko) coûte 3 000$ à 15 000$, un studio Vicon coûte 50 000$.
   - *Proposition de valeur :* Animer des dizaines de PNJs avec un simple smartphone et mark-target.
2. **Créateurs VTuber & Metaverse :**
   - *Pain point :* Transférer des danses TikTok ou chorégraphies réelles sur des avatars VRM/VRChat.
3. **Agences de Publicité & Réseaux Sociaux :**
   - *Pain point :* Créer des doublures 3D virtuelles en quelques minutes à partir de vidéos de danseurs.

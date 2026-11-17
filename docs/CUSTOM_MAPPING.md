# 📖 Guide Complet : Configurer un Mapping Personnalisé dans mark-target

Ce guide détaille comment utiliser **`MappingRegistry`** pour connecter n'importe quel squelette personnalisé (ex: Blender Rigify, Unreal Engine 5 Mannequin, Mimo, ou rig maison) vers votre modèle 3D cible.

---

## 1. Principe de Fonctionnement de `MappingRegistry`

`MappingRegistry` résout les correspondances d'os selon un ordre de priorité strict :
1. **Correspondance directe personnalisée (Priorité 1) :** Si vous avez enregistré une table spécifique pour votre format (ex: `mon_rig->mixamo`), elle est utilisée immédiatement.
2. **Pont Pivot Standard (Priorité 2) :** Si aucun mapping direct n'existe, le registre traduit automatiquement `Source -> Pivot Standard Humanoid (55 os) -> Cible`.

---

## 2. Méthode 1 : Utilisation par Fichier JSON (Recommandé)

Créez un simple fichier JSON `mon_rig_mapping.json` associant chaque nom d'os source au nom d'os cible :

```json
{
  "_description": "Mapping personnalisé pour rig Unreal Engine 5 vers Mixamo",
  "pelvis": "mixamorig:Hips",
  "spine_01": "mixamorig:Spine",
  "spine_02": "mixamorig:Spine1",
  "spine_03": "mixamorig:Spine2",
  "neck_01": "mixamorig:Neck",
  "head": "mixamorig:Head",
  "clavicle_l": "mixamorig:LeftShoulder",
  "upperarm_l": "mixamorig:LeftArm",
  "lowerarm_l": "mixamorig:LeftForeArm",
  "hand_l": "mixamorig:LeftHand",
  "clavicle_r": "mixamorig:RightShoulder",
  "upperarm_r": "mixamorig:RightArm",
  "lowerarm_r": "mixamorig:RightForeArm",
  "hand_r": "mixamorig:RightHand",
  "thigh_l": "mixamorig:LeftUpLeg",
  "calf_l": "mixamorig:LeftLeg",
  "foot_l": "mixamorig:LeftFoot",
  "ball_l": "mixamorig:LeftToeBase",
  "thigh_r": "mixamorig:RightUpLeg",
  "calf_r": "mixamorig:RightLeg",
  "foot_r": "mixamorig:RightFoot",
  "ball_r": "mixamorig:RightToeBase"
}
```

### Exécution directe en ligne de commande (CLI) :
```bash
python run_marker_retarget.py \
  --source animation_source.bvh \
  --source-type bvh \
  --target mon_personnage.glb \
  --target-type mixamo \
  --custom-mapping mon_rig_mapping.json \
  --output out_anime.glb
```

---

## 3. Méthode 2 : Enregistrement par Code Python

Si vous intégrez `mark-target` dans un script ou un pipeline automatisé :

```python
from mappings.registry import MappingRegistry

# Option A : Enregistrement direct d'un dictionnaire Python
custom_map = {
    "root_joint": "mixamorig:Hips",
    "arm_left": "mixamorig:LeftArm",
    "forearm_left": "mixamorig:LeftForeArm",
}
MappingRegistry.register("mon_rig_source", "mixamo", custom_map)

# Option B : Chargement depuis un fichier JSON
MappingRegistry.register_from_json(
    json_path="configs/custom_rig.json",
    source_format="mon_rig_source",
    target_format="mixamo"
)

# Résolution immédiate
resolved = MappingRegistry.get_mapping("mon_rig_source", "mixamo")
print(f"Mapping actif ({len(resolved)} os) :", resolved)
```

---

## 4. Méthode 3 : Exporter un Modèle de Mapping Existant

Pour créer facilement un nouveau mapping sans partir de zéro, vous pouvez exporter un template JSON complet d'un format existant :

```python
from mappings.registry import MappingRegistry

# Exporte les 55 correspondances SMPL-X (HybrIK-X / Mimo) vers Mixamo
MappingRegistry.export_to_json(
    source_format="smplx",
    target_format="mixamo",
    json_path="configs/template_smplx_to_mixamo.json"
)
```
Il ne vous reste plus qu'à modifier les clés de gauche avec vos propres noms d'os !

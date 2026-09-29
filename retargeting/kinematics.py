"""
Module isolate.retargeting.kinematics
=====================================
Gestion de la cinématique directe et inverse pour arbres squelettiques génériques:
- Conversion des orientations globales en rotations relatives locales
- Résolution hiérarchique récursive
- Application de corrections (ex: compensation d'orientation de root)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional
from core.geometry import quat_mul, quat_inv, quat_normalize, quat_identity


def global_to_local_hierarchy(
    Q_global: Dict[str, np.ndarray],
    parents: Dict[str, Optional[str]],
) -> Dict[str, np.ndarray]:
    """
    Convertit un dictionnaire de quaternions globaux en quaternions locaux (relatifs aux parents).
    
    Q_local(bone) = Q_global(parent)^(-1) * Q_global(bone)
    Pour un os racine (parent is None) : Q_local(root) = Q_global(root)
    """
    Q_local: Dict[str, np.ndarray] = {}

    for bone, q_glob in Q_global.items():
        if q_glob is None:
            continue
        parent = parents.get(bone)
        if parent is None or parent not in Q_global or Q_global[parent] is None:
            Q_local[bone] = quat_normalize(q_glob)
        else:
            q_parent_inv = quat_inv(Q_global[parent])
            Q_local[bone] = quat_normalize(quat_mul(q_parent_inv, q_glob))

    return Q_local


def local_to_global_hierarchy(
    Q_local: Dict[str, np.ndarray],
    parents: Dict[str, Optional[str]],
) -> Dict[str, np.ndarray]:
    """
    Reconstitue les rotations globales à partir des rotations locales et de l'arbre généalogique.
    """
    Q_global: Dict[str, np.ndarray] = {}

    def resolve(bone: str) -> np.ndarray:
        if bone in Q_global:
            return Q_global[bone]
        q_loc = Q_local.get(bone, quat_identity())
        parent = parents.get(bone)
        if parent is None or parent not in Q_local:
            q_glob = quat_normalize(q_loc)
        else:
            q_parent_glob = resolve(parent)
            q_glob = quat_normalize(quat_mul(q_parent_glob, q_loc))
        Q_global[bone] = q_glob
        return q_glob

    for bone in Q_local:
        resolve(bone)

    return Q_global


def apply_euler_correction_to_root(
    local_rotations: Dict[str, np.ndarray],
    root_name: str,
    axis: str = "x",
    degrees: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Applique une rotation corrective simple (angle en degrés autour de x, y, ou z) au joint racine.
    """
    if root_name not in local_rotations or abs(degrees) < 1e-6:
        return local_rotations

    rad = np.radians(degrees)
    half = rad * 0.5
    s = np.sin(half)
    c = np.cos(half)

    if axis.lower() == "x":
        q_corr = np.array([c, s, 0.0, 0.0], dtype=np.float64)
    elif axis.lower() == "y":
        q_corr = np.array([c, 0.0, s, 0.0], dtype=np.float64)
    elif axis.lower() == "z":
        q_corr = np.array([c, 0.0, 0.0, s], dtype=np.float64)
    else:
        raise ValueError(f"Axe non reconnu: {axis}")

    res = dict(local_rotations)
    res[root_name] = quat_normalize(quat_mul(q_corr, res[root_name]))
    return res

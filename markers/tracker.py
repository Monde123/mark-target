"""
Module isolate.markers.tracker
==============================
Suivi et reconstruction temporelle des marqueurs:
- Évaluation des positions 3D des marqueurs par frame à partir d'un maillage déformé
- Calcul des poids de confiance / rigidité par marqueur (ex: pureté LBS)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any


def extract_markers_positions(
    deformed_vertices: np.ndarray,
    markers_definition: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Extrait les coordonnées 3D des marqueurs d'une frame.
    
    markers_definition: dict {bone_name: vertex_indices_array}
    deformed_vertices: ndarray (V, 3) des sommets déformés à la frame t
    
    Retourne:
        dict {bone_name: ndarray (N_markers, 3)}
    """
    deformed_vertices = np.asarray(deformed_vertices, dtype=np.float64)
    positions = {}
    for bone_name, v_ids in markers_definition.items():
        v_ids = np.asarray(v_ids, dtype=np.int64)
        positions[bone_name] = deformed_vertices[v_ids]
    return positions


def compute_marker_stability_weights(
    skinning_weights: np.ndarray,
    marker_vertex_ids: np.ndarray,
    target_joint_idx: int,
) -> np.ndarray:
    """
    Calcule un vecteur de poids de confiance pour un ensemble de marqueurs.
    Favorise les sommets très fortement liés à l'os cible et éloignés des zones d'influence secondaires.
    """
    v_ids = np.asarray(marker_vertex_ids, dtype=np.int64)
    w_mat = skinning_weights[v_ids]  # (N_markers, N_joints)

    own = w_mat[:, target_joint_idx]
    sorted_w = np.sort(w_mat, axis=1)
    second = sorted_w[:, -2] if sorted_w.shape[1] >= 2 else np.zeros_like(own)

    # Pondération non-linéaire quadratique : propre^2 * marge_sur_second
    weights = (np.clip(own, 0.0, 1.0) ** 2) * np.clip(own - second, 0.0, 1.0)
    if float(np.sum(weights)) <= 1e-8:
        return np.ones(len(v_ids), dtype=np.float64)

    return weights.astype(np.float64)

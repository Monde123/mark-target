"""
Module isolate.markers.surface_sampling
=======================================
Outils d'échantillonnage de surface génériques et agnostiques :
- Découpage d'un maillage global en sous-maillages (submeshes) par dominance de poids
- Farthest Point Sampling (FPS) pour garantir l'espacement et la non-colinéarité
- Calcul de marqueurs basés sur barycentres de triangles ou sommets discrets
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional


def segment_submesh_by_mask(
    faces: np.ndarray,
    vert_mask: np.ndarray,
    min_vert_in_face: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrait un sous-maillage à partir d'un masque booléen sur les sommets.
    
    Retourne:
        sub_faces: Faces réindexées relatives au sous-maillage (M, 3)
        global_vert_ids: Indices des sommets dans le maillage source original (K,)
    """
    faces = np.asarray(faces, dtype=np.int64)
    vert_mask = np.asarray(vert_mask, dtype=bool)

    face_mask = np.sum(vert_mask[faces.ravel()].reshape(-1, 3), axis=1) >= min_vert_in_face
    global_vert_ids = np.where(vert_mask)[0]

    old_to_new = -np.ones(len(vert_mask), dtype=np.int64)
    old_to_new[global_vert_ids] = np.arange(len(global_vert_ids))

    sub_faces = old_to_new[faces[face_mask]]
    valid_face_rows = np.all(sub_faces >= 0, axis=1)
    sub_faces = sub_faces[valid_face_rows]

    return sub_faces, global_vert_ids


def build_bone_submeshes(
    faces: np.ndarray,
    skinning_weights: np.ndarray,
    joint_indices_map: Dict[str, int],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Découpe le maillage en sous-parties pour chaque segment squelettique selon le joint dominant.
    
    joint_indices_map: dict {nom_os: index_joint_dans_skinning_weights}
    """
    dominant_joint = np.argmax(skinning_weights, axis=1)
    submeshes = {}

    for bone_name, joint_idx in joint_indices_map.items():
        vert_mask = (dominant_joint == joint_idx)
        if vert_mask.sum() < 3:
            continue
        sub_faces, vertex_ids = segment_submesh_by_mask(faces, vert_mask)
        if len(sub_faces) == 0 or len(vertex_ids) < 3:
            continue
        submeshes[bone_name] = {
            "vertex_ids": vertex_ids,
            "faces": sub_faces,
        }

    return submeshes


def farthest_point_sampling(
    points: np.ndarray,
    n_samples: int,
    initial_idx: Optional[int] = None
) -> List[int]:
    """
    Échantillonne n_samples points maximisant la distance mutuelle (FPS).
    Garantit une bonne assise spatiale pour la stabilité de l'algorithme Kabsch.
    """
    points = np.asarray(points, dtype=np.float64)
    n_points = points.shape[0]
    if n_points <= n_samples:
        return list(range(n_points))

    if initial_idx is None:
        initial_idx = 0

    chosen = [int(initial_idx)]
    dists = np.linalg.norm(points - points[chosen[0]][None, :], axis=1)

    for _ in range(n_samples - 1):
        dists[chosen] = -1.0
        next_idx = int(np.argmax(dists))
        chosen.append(next_idx)
        new_dists = np.linalg.norm(points - points[next_idx][None, :], axis=1)
        dists = np.minimum(dists, new_dists)

    return chosen


def sample_multi_markers_for_bone(
    rest_vertices: np.ndarray,
    candidate_vertex_ids: np.ndarray,
    joint_pivot: np.ndarray,
    n_markers: int = 4,
    eps_collinear: float = 1e-6,
) -> Optional[np.ndarray]:
    """
    Sélectionne n_markers sommets non-colinéaires sur un os via FPS,
    à partir du sommet le plus proche du pivot articulaire.
    
    Retourne les indices globaux des sommets choisis, ou None si non viable.
    """
    candidate_ids = np.asarray(candidate_vertex_ids, dtype=np.int64)
    if len(candidate_ids) < n_markers:
        return None

    candidate_pos = rest_vertices[candidate_ids]
    # Démarrage par le sommet le plus proche du joint
    dists_to_pivot = np.linalg.norm(candidate_pos - joint_pivot[None, :], axis=1)
    seed_idx = int(np.argmin(dists_to_pivot))

    chosen_local = farthest_point_sampling(candidate_pos, n_markers, initial_idx=seed_idx)
    chosen_global = candidate_ids[chosen_local]

    # Vérification d'aire minimale (non-colinéarité des 3 premiers marqueurs)
    if n_markers >= 3:
        p0, p1, p2 = rest_vertices[chosen_global[:3]]
        area = np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        if area < eps_collinear:
            return None

    return chosen_global

"""Module markers.virtual_cage
=============================
Générateur de cages tétraédriques rigides pour sources squelettiques filaires (BVH).

Problème mathématique :
Un joint BVH classique n'est qu'un point 3D unique. Un segment osseux (parent -> enfant)
ne définit donc qu'un seul vecteur directeur, ce qui ne procure que 2 degrés de liberté (DoF).
Le 3ème DoF — le roulis de l'os autour de son propre axe (Twist / Roll) — est mathématiquement
indéterminé, provoquant des torsions aberrantes lors du retargeting.

Solution géométrique :
Pour chaque segment osseux, on construit une cage rigide de 4 marqueurs virtuels non coplanaires :
- M0 : origine de l'os (position du joint parent)
- M1 : extrémité de l'os (position du joint enfant)
- M2 : marqueur latéral (fixe le plan de flexion / charnière de l'articulation)
- M3 : marqueur orthogonal (fixe strictement le roulis / twist sans ambiguïté)

Ce tétraèdre rigide solidaire fournit un nuage de rang 3 plein, permettant à Kabsch-Umeyama
de résoudre les 3 DoF complets sans aucune singularité.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.geometry import (
    quat_identity,
    quat_normalize,
    rotate_vector,
    quaternion_to_matrix,
    matrix_to_quaternion,
)


class VirtualBoneCageGenerator:
    """
    Génère des cages tétraédriques rigides à 4 marqueurs pour chaque os d'un squelette filaire.
    """

    def __init__(self, lateral_ratio: float = 0.20, orthogonal_ratio: float = 0.15):
        """
        lateral_ratio : rapport largeur latérale / longueur de l'os (plan de flexion)
        orthogonal_ratio : rapport épaisseur / longueur de l'os (axe de torsion)
        """
        self.lateral_ratio = lateral_ratio
        self.orthogonal_ratio = orthogonal_ratio

    def build_rest_cage_template(
        self,
        p_parent_rest: np.ndarray,
        p_child_rest: np.ndarray,
        q_parent_rest: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Construit le tétraèdre de repos à 4 marqueurs pour un os (forme (4, 3)).
        """
        p0 = np.asarray(p_parent_rest, dtype=np.float64).reshape(3)
        p1 = np.asarray(p_child_rest, dtype=np.float64).reshape(3)

        bone_vec = p1 - p0
        length = float(np.linalg.norm(bone_vec))
        if length < 1e-6:
            # Os de longueur nulle (ex: racines superposées), créer un tétraèdre unitaire centré
            eps = 0.05
            return np.array([
                p0,
                p0 + np.array([eps, 0.0, 0.0]),
                p0 + np.array([0.0, eps, 0.0]),
                p0 + np.array([0.0, 0.0, eps]),
            ], dtype=np.float64)

        u_dir = bone_vec / length
        mid_point = p0 + 0.5 * bone_vec

        # Déterminer deux vecteurs orthogonaux normés (u_lat, u_orth) pour former une base orthonormée
        if q_parent_rest is not None and np.linalg.norm(q_parent_rest) > 1e-6:
            # Utiliser la rotation de repos du joint pour respecter les axes anatomiques locaux
            R_rest = quaternion_to_matrix(quat_normalize(q_parent_rest))
            # Prendre les axes X et Z locaux projetés orthogonalement à u_dir
            candidate_lat = R_rest[:, 0]
            candidate_orth = R_rest[:, 2]

            # Orthogonalisation de Gram-Schmidt contre u_dir
            v_lat = candidate_lat - np.dot(candidate_lat, u_dir) * u_dir
            norm_lat = float(np.linalg.norm(v_lat))
            if norm_lat > 1e-4:
                u_lat = v_lat / norm_lat
            else:
                # Si le candidat était colinéaire, chercher une direction de secours
                fallback = np.array([1.0, 0.0, 0.0]) if abs(u_dir[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
                v_lat = np.cross(u_dir, fallback)
                u_lat = v_lat / np.linalg.norm(v_lat)
        else:
            # Trouver un vecteur orthogonal stable par produit vectoriel
            fallback = np.array([1.0, 0.0, 0.0]) if abs(u_dir[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            v_lat = np.cross(u_dir, fallback)
            u_lat = v_lat / np.linalg.norm(v_lat)

        u_orth = np.cross(u_dir, u_lat)
        u_orth = u_orth / np.linalg.norm(u_orth)

        # Distances caractéristiques
        w_lat = max(length * self.lateral_ratio, 0.02)
        w_orth = max(length * self.orthogonal_ratio, 0.015)

        # Les 4 sommets du tétraèdre rigide :
        m0 = p0
        m1 = p1
        m2 = mid_point + w_lat * u_lat
        m3 = mid_point + w_orth * u_orth

        return np.array([m0, m1, m2, m3], dtype=np.float64)

    def generate_bone_cages_sequence(
        self,
        parents: Dict[str, Optional[str]],
        rest_positions: Dict[str, np.ndarray],
        frames_positions: List[Dict[str, np.ndarray]],
        rest_rotations: Optional[Dict[str, np.ndarray]] = None,
        joint_rotations: Optional[List[Dict[str, np.ndarray]]] = None,
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """
        Génère les cages de repos et la séquence temporelle de cages 4 marqueurs par os.

        Retourne :
            rest_cages : dict {bone_name: ndarray (4, 3)}
            frames_cages : liste de dict {bone_name: ndarray (4, 3)}
        """
        # 1. Identifier l'enfant principal de chaque os
        primary_child: Dict[str, str] = {}
        for bone, parent in parents.items():
            if parent is not None and parent not in primary_child:
                primary_child[parent] = bone

        # 2. Cages de repos
        rest_cages: Dict[str, np.ndarray] = {}
        offsets_local: Dict[str, np.ndarray] = {}  # {bone: 4x3 relatif au joint parent}

        for bone in parents:
            p_parent_rest = rest_positions.get(bone)
            if p_parent_rest is None:
                continue

            child = primary_child.get(bone)
            if child and child in rest_positions:
                p_child_rest = rest_positions[child]
            else:
                # Pour les extrémités (feuilles de l'arbre), projeter dans la direction du parent
                parent_of_bone = parents.get(bone)
                if parent_of_bone and parent_of_bone in rest_positions:
                    p_p = rest_positions[parent_of_bone]
                    dir_vec = p_parent_rest - p_p
                    norm_dir = float(np.linalg.norm(dir_vec))
                    if norm_dir > 1e-6:
                        p_child_rest = p_parent_rest + (dir_vec / norm_dir) * 0.10
                    else:
                        p_child_rest = p_parent_rest + np.array([0.0, 0.10, 0.0])
                else:
                    p_child_rest = p_parent_rest + np.array([0.0, 0.10, 0.0])

            q_rest = rest_rotations.get(bone) if rest_rotations else None
            cage_rest = self.build_rest_cage_template(p_parent_rest, p_child_rest, q_rest)
            rest_cages[bone] = cage_rest

            # Calculer les offsets locaux relatifs au joint parent dans son repère de repos
            p0 = p_parent_rest.reshape(3)
            diffs = cage_rest - p0  # (4, 3)

            if q_rest is not None and np.linalg.norm(q_rest) > 1e-6:
                R_rest = quaternion_to_matrix(quat_normalize(q_rest))
                # diff_local = diffs @ R_rest
                diffs_local = (R_rest.T @ diffs.T).T
            else:
                diffs_local = diffs

            offsets_local[bone] = diffs_local

        # 3. Cages temporelles par frame
        n_frames = len(frames_positions)
        frames_cages: List[Dict[str, np.ndarray]] = []

        for t in range(n_frames):
            frame_pos = frames_positions[t]
            frame_cages_t: Dict[str, np.ndarray] = {}

            # Rotations monde de la frame si disponibles (ex: du clip BVH)
            frame_rots = joint_rotations[t] if (joint_rotations and t < len(joint_rotations)) else None

            for bone, cage_rest in rest_cages.items():
                p_parent_t = frame_pos.get(bone)
                if p_parent_t is None:
                    continue
                p0_t = np.asarray(p_parent_t, dtype=np.float64).reshape(3)

                if frame_rots and bone in frame_rots:
                    # Si la rotation monde de l'os est connue (BVH), faire pivoter le tétraèdre rigide complet
                    q_world_t = quat_normalize(frame_rots[bone])
                    R_world_t = quaternion_to_matrix(q_world_t)
                    diffs_local = offsets_local[bone]
                    # cage_t = p0_t + (R_world_t @ diffs_local.T).T
                    cage_t = p0_t + (R_world_t @ diffs_local.T).T
                    frame_cages_t[bone] = cage_t
                else:
                    # Si seule la position des joints est connue, orienter par rapport à l'enfant
                    child = primary_child.get(bone)
                    if child and child in frame_pos:
                        p_child_t = np.asarray(frame_pos[child], dtype=np.float64).reshape(3)
                        # Reconstruire le tétraèdre sur le vecteur courant
                        cage_t = self.build_rest_cage_template(p0_t, p_child_t)
                        frame_cages_t[bone] = cage_t
                    else:
                        # Fallback rigide : translation pure du tétraèdre de repos
                        delta_t = p0_t - rest_positions[bone].reshape(3)
                        frame_cages_t[bone] = cage_rest + delta_t

            frames_cages.append(frame_cages_t)

        return rest_cages, frames_cages

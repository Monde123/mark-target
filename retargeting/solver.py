"""
Module isolate.retargeting.solver
=================================
Moteur universel de retargeting par marqueurs:
- Mode Multi-Marqueurs : Alignement optimal par Kabsch (résout l'orientation 3D complète, roll inclus)
- Mode Mono-Marqueur : Résolution par visée (aim vectoriel vers l'enfant) avec propagation hiérarchique
- Résilience et gestion intelligente des fallbacks
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, Any

from core.geometry import (
    quat_mul,
    quat_inv,
    quat_normalize,
    quat_identity,
    quat_from_two_vectors,
    rotate_vector,
    matrix_to_quaternion,
)
from core.kabsch import weighted_kabsch_rotation, kabsch_rotation


class MarkerRetargetSolver:
    """
    Solveur générique appliquant les positions de marqueurs 3D
    sur une structure de squelette cible.
    """

    def __init__(
        self,
        rest_rotations: Dict[str, np.ndarray],
        parents: Dict[str, Optional[str]],
        min_points_kabsch: int = 3,
        min_rank_ratio: float = 1e-3,
        max_rms_ratio: Optional[float] = 0.25,
    ):
        """
        rest_rotations: dict {bone_name: quat_rest [w, x, y, z]}
        parents: dict {bone_name: parent_name or None}
        """
        self.rest_rotations = {k: quat_normalize(v) for k, v in rest_rotations.items()}
        self.parents = parents
        self.min_points_kabsch = min_points_kabsch
        self.min_rank_ratio = min_rank_ratio
        self.max_rms_ratio = max_rms_ratio

    def solve_frame_multi_kabsch(
        self,
        positions_t: Dict[str, np.ndarray],
        positions_rest: Dict[str, np.ndarray],
        weights: Optional[Dict[str, np.ndarray]] = None,
        root_rot_override: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Résout l'orientation globale pour chaque os par Kabsch multi-points.
        
        positions_t: {bone_name: (N, 3)}
        positions_rest: {bone_name: (N, 3)}
        weights: {bone_name: (N,)} optionnel
        root_rot_override: {root_bone: q_global} ex: root orientation imposée par caméra
        
        Retourne:
            Q_global_t: {bone_name: quat_global}
            diagnostics: {bone_name: status_info}
        """
        Q_global_t: Dict[str, np.ndarray] = {}
        diagnostics: Dict[str, Any] = {}
        root_override = root_rot_override or {}

        for bone, q_rest in self.rest_rotations.items():
            # Cas 1 : Remplacement direct (ex: racine orientée par capteur/caméra)
            if bone in root_override:
                Q_global_t[bone] = quat_normalize(root_override[bone])
                diagnostics[bone] = {"status": "root_override"}
                continue

            # Cas 2 : Marqueurs absents ou incomplets -> Maintien de la pose de repos
            if bone not in positions_t or bone not in positions_rest:
                Q_global_t[bone] = q_rest.copy()
                diagnostics[bone] = {"status": "fallback_rest_no_markers"}
                continue

            pts_t = positions_t[bone]
            pts_r = positions_rest[bone]
            if len(pts_t) < self.min_points_kabsch or len(pts_r) < self.min_points_kabsch:
                Q_global_t[bone] = q_rest.copy()
                diagnostics[bone] = {"status": "fallback_insufficient_points"}
                continue

            w = weights.get(bone) if weights else None

            # Cas 3 : Résolution par Kabsch
            try:
                if w is not None:
                    R_move, _, _, rms, s_vals = weighted_kabsch_rotation(
                        pts_r,
                        pts_t,
                        weights=w,
                        min_points=self.min_points_kabsch,
                        min_rank_ratio=self.min_rank_ratio,
                        max_rms_ratio=self.max_rms_ratio,
                    )
                else:
                    R_move, _, _, rms, s_vals = kabsch_rotation(
                        pts_r,
                        pts_t,
                        min_rank_ratio=self.min_rank_ratio,
                        max_rms_ratio=self.max_rms_ratio,
                    )

                q_move = matrix_to_quaternion(R_move)
                Q_global_t[bone] = quat_normalize(quat_mul(q_move, q_rest))
                diagnostics[bone] = {
                    "status": "kabsch_success",
                    "rms": rms,
                    "singular_values": s_vals,
                }
            except (ValueError, np.linalg.LinAlgError) as exc:
                # Fallback sécurisé en cas de singularité géométrique
                Q_global_t[bone] = q_rest.copy()
                diagnostics[bone] = {
                    "status": "fallback_error",
                    "error": str(exc),
                }

        return Q_global_t, diagnostics

    def solve_frame_single_aim(
        self,
        positions_t: Dict[str, np.ndarray],
        positions_rest: Dict[str, np.ndarray],
        root_rot_override: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Résout l'orientation globale pour chaque os par Aim (vecteur unitaire vers l'enfant).
        Utile lorsqu'on ne dispose que d'un seul point 3D par os.
        """
        children_of: Dict[str, list[str]] = {}
        for bone, parent in self.parents.items():
            if parent is not None:
                children_of.setdefault(parent, []).append(bone)

        Q_global_t: Dict[str, np.ndarray] = dict(root_rot_override or {})

        def resolve(bone: str) -> np.ndarray:
            if bone in Q_global_t:
                return Q_global_t[bone]
            if bone not in self.rest_rotations:
                return quat_identity()

            q_rest = self.rest_rotations[bone]
            parent = self.parents.get(bone)
            child = next((c for c in children_of.get(bone, []) if c in positions_t), None)

            if parent is None or parent not in self.rest_rotations:
                if child is None or bone not in positions_t or bone not in positions_rest:
                    Q_global_t[bone] = q_rest
                    return q_rest
                dir_rest = positions_rest[child] - positions_rest[bone]
                dir_t = positions_t[child] - positions_t[bone]
                if np.linalg.norm(dir_rest) < 1e-8 or np.linalg.norm(dir_t) < 1e-8:
                    Q_global_t[bone] = q_rest
                    return q_rest
                r_move = quat_from_two_vectors(dir_rest, dir_t)
                Q_global_t[bone] = quat_normalize(quat_mul(r_move, q_rest))
                return Q_global_t[bone]

            q_parent_t = resolve(parent)
            q_parent_rest = self.rest_rotations[parent]

            if child is None or bone not in positions_t or bone not in positions_rest:
                q_local_rest = quat_mul(quat_inv(q_parent_rest), q_rest)
                Q_global_t[bone] = quat_normalize(quat_mul(q_parent_t, q_local_rest))
                return Q_global_t[bone]

            dir_rest_world = positions_rest[child] - positions_rest[bone]
            dir_t_world = positions_t[child] - positions_t[bone]
            if np.linalg.norm(dir_rest_world) < 1e-8 or np.linalg.norm(dir_t_world) < 1e-8:
                q_local_rest = quat_mul(quat_inv(q_parent_rest), q_rest)
                Q_global_t[bone] = quat_normalize(quat_mul(q_parent_t, q_local_rest))
                return Q_global_t[bone]

            dir_rest_local = rotate_vector(quat_inv(q_parent_rest), dir_rest_world)
            dir_t_local = rotate_vector(quat_inv(q_parent_t), dir_t_world)

            r_move_local = quat_from_two_vectors(dir_rest_local, dir_t_local)
            q_local_rest = quat_mul(quat_inv(q_parent_rest), q_rest)
            q_local_t = quat_mul(r_move_local, q_local_rest)
            Q_global_t[bone] = quat_normalize(quat_mul(q_parent_t, q_local_t))
            return Q_global_t[bone]

        for bone in self.rest_rotations:
            resolve(bone)

        return Q_global_t

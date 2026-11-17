"""Module isolate.retargeting.solver
====================================
Solveur d'orientation universel multi-marqueurs basé sur Kabsch / Umeyama, RANSAC et Aim IK.
Supporte :
- La résolution multi-marqueurs par Kabsch rigide ou Umeyama (avec absorption d'échelle)
- Le filtrage robuste RANSAC contre les occlusions et marqueurs aberrants
- Le monitoring de déformation élastique non-rigide par tenseur de Green-Lagrange
- Le mode fallback à 1 marqueur par visée vectorielle (Aim IK)
- Le lissage temporel de séquence via Bézier SQUAD / SLERP
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from core.geometry import (
    quat_normalize,
    quat_identity,
    quat_mul,
    matrix_to_quaternion,
    quat_from_two_vectors,
)
from core.kabsch import (
    kabsch_rotation,
    weighted_kabsch_rotation,
    kabsch_umeyama_rotation,
    weighted_kabsch_umeyama_rotation,
)
from markers.ransac_filter import (
    ransac_kabsch_alignment,
    compute_green_lagrange_strain,
)
from retargeting.smoothing import smooth_quaternion_trajectory


class MarkerRetargetSolver:
    """Solveur d'orientation par frame combinant Kabsch/Umeyama multi-points, RANSAC et Aim IK."""

    def __init__(
        self,
        rest_rotations: Dict[str, np.ndarray],
        parents: Dict[str, Optional[str]],
        children: Optional[Dict[str, List[str]]] = None,
        min_points_kabsch: int = 3,
        min_rank_ratio: float = 1e-3,
        max_rms_ratio: Optional[float] = None,
    ):
        self.rest_rotations = {b: quat_normalize(q) for b, q in rest_rotations.items()}
        self.parents = parents
        self.min_points_kabsch = min_points_kabsch
        self.min_rank_ratio = min_rank_ratio
        self.max_rms_ratio = max_rms_ratio

        if children is None:
            self.children: Dict[str, List[str]] = {b: [] for b in self.rest_rotations}
            for bone, parent in parents.items():
                if parent and parent in self.children:
                    self.children[parent].append(bone)
        else:
            self.children = children

    def solve_frame(
        self,
        frame_markers: Dict[str, np.ndarray],
        rest_markers: Dict[str, np.ndarray],
        mapping: Dict[str, str],
        use_umeyama: bool = False,
        use_ransac: bool = False,
        ransac_threshold: float = 0.04,
        max_strain: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Résout une frame complète en mappant les marqueurs sources vers les os cibles.
        """
        positions_t: Dict[str, List[np.ndarray]] = {b: [] for b in self.rest_rotations}
        positions_rest: Dict[str, List[np.ndarray]] = {b: [] for b in self.rest_rotations}

        for src_marker, tgt_bone in mapping.items():
            if tgt_bone in positions_t and src_marker in frame_markers and src_marker in rest_markers:
                positions_t[tgt_bone].append(frame_markers[src_marker])
                positions_rest[tgt_bone].append(rest_markers[src_marker])

        pts_t = {b: np.array(v) for b, v in positions_t.items() if len(v) > 0}
        pts_r = {b: np.array(v) for b, v in positions_rest.items() if len(v) > 0}

        q_global, _ = self.solve_frame_multi_kabsch(
            positions_t=pts_t,
            positions_rest=pts_r,
            use_umeyama=use_umeyama,
            use_ransac=use_ransac,
            ransac_threshold=ransac_threshold,
            max_strain_threshold=max_strain,
        )
        return q_global

    def solve_frame_multi_kabsch(
        self,
        positions_t: Dict[str, np.ndarray],
        positions_rest: Dict[str, np.ndarray],
        weights: Optional[Dict[str, np.ndarray]] = None,
        root_rot_override: Optional[Dict[str, np.ndarray]] = None,
        use_umeyama: bool = False,
        use_ransac: bool = False,
        ransac_threshold: float = 0.04,
        max_strain_threshold: Optional[float] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Résout l'orientation globale pour chaque os par Kabsch, Umeyama ou RANSAC.
        """
        Q_global_t: Dict[str, np.ndarray] = {}
        diagnostics: Dict[str, Any] = {}
        root_override = root_rot_override or {}

        for bone, q_rest in self.rest_rotations.items():
            if bone in root_override:
                Q_global_t[bone] = quat_normalize(root_override[bone])
                diagnostics[bone] = {"status": "root_override"}
                continue

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

            # Contrôle de déformation non rigide (Green-Lagrange)
            strain_val = 0.0
            if max_strain_threshold is not None:
                _, strain_val = compute_green_lagrange_strain(pts_r, pts_t)
                if strain_val > max_strain_threshold:
                    Q_global_t[bone] = q_rest.copy()
                    diagnostics[bone] = {
                        "status": "fallback_strain_exceeded",
                        "strain": strain_val,
                        "threshold": max_strain_threshold,
                    }
                    continue

            w = weights.get(bone) if weights else None

            try:
                scale_val = 1.0
                s_vals = np.array([1.0, 1.0, 1.0])

                if use_ransac and len(pts_r) >= 4:
                    R_move, inlier_mask, rms, n_inliers = ransac_kabsch_alignment(
                        pts_r, pts_t, inlier_threshold=ransac_threshold
                    )
                    diag_status = "ransac_success"
                elif use_umeyama:
                    if w is not None:
                        R_move, scale_val, _, _, _, rms, s_vals = weighted_kabsch_umeyama_rotation(
                            pts_r, pts_t, weights=w, min_points=self.min_points_kabsch, min_rank_ratio=self.min_rank_ratio
                        )
                    else:
                        R_move, scale_val, _, _, _, rms, s_vals = kabsch_umeyama_rotation(
                            pts_r, pts_t, min_rank_ratio=self.min_rank_ratio, max_rms_ratio=self.max_rms_ratio
                        )
                    diag_status = "umeyama_success"
                else:
                    if w is not None:
                        R_move, _, _, rms, s_vals = weighted_kabsch_rotation(
                            pts_r, pts_t, weights=w, min_points=self.min_points_kabsch,
                            min_rank_ratio=self.min_rank_ratio, max_rms_ratio=self.max_rms_ratio,
                        )
                    else:
                        R_move, _, _, rms, s_vals = kabsch_rotation(
                            pts_r, pts_t, min_rank_ratio=self.min_rank_ratio, max_rms_ratio=self.max_rms_ratio,
                        )
                    diag_status = "kabsch_success"

                q_move = matrix_to_quaternion(R_move)
                Q_global_t[bone] = quat_normalize(quat_mul(q_move, q_rest))
                diagnostics[bone] = {
                    "status": diag_status,
                    "rms": rms,
                    "scale": scale_val,
                    "strain": strain_val,
                    "singular_values": s_vals,
                }
            except ValueError as e:
                Q_global_t[bone] = q_rest.copy()
                diagnostics[bone] = {
                    "status": "fallback_rest_exception",
                    "error": str(e),
                }

        return Q_global_t, diagnostics

    def solve_sequence(
        self,
        sequence_positions_t: List[Dict[str, np.ndarray]],
        positions_rest: Dict[str, np.ndarray],
        weights: Optional[Dict[str, np.ndarray]] = None,
        smoothing_factor: float = 0.0,
        use_umeyama: bool = False,
        use_ransac: bool = False,
        ransac_threshold: float = 0.04,
        max_strain_threshold: Optional[float] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Résout une séquence temporelle complète avec lissage temporel optionnel (Bézier/SLERP).
        """
        n_frames = len(sequence_positions_t)
        if n_frames == 0:
            return []

        # 1. Résolution frame par frame
        raw_frames_rotations: List[Dict[str, np.ndarray]] = []
        for pos_t in sequence_positions_t:
            q_glob, _ = self.solve_frame_multi_kabsch(
                pos_t,
                positions_rest,
                weights=weights,
                use_umeyama=use_umeyama,
                use_ransac=use_ransac,
                ransac_threshold=ransac_threshold,
                max_strain_threshold=max_strain_threshold,
            )
            raw_frames_rotations.append(q_glob)

        if smoothing_factor <= 0.0 or n_frames <= 2:
            return raw_frames_rotations

        # 2. Lissage temporel par os à travers les frames
        bones = list(self.rest_rotations.keys())
        smoothed_series: Dict[str, List[np.ndarray]] = {}

        for bone in bones:
            bone_trajectory = [raw_frames_rotations[f][bone] for f in range(n_frames)]
            smoothed_series[bone] = smooth_quaternion_trajectory(bone_trajectory, smoothing_factor=smoothing_factor)

        # 3. Ré-assemblage par frame
        smoothed_frames: List[Dict[str, np.ndarray]] = []
        for f in range(n_frames):
            frame_dict = {bone: smoothed_series[bone][f] for bone in bones}
            smoothed_frames.append(frame_dict)

        return smoothed_frames

    def solve_frame_single_aim(
        self,
        positions_t: Dict[str, np.ndarray],
        positions_rest: Dict[str, np.ndarray],
        root_rot_override: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Mode dégradé à 1 marqueur par os : Oriente chaque os en visant son os enfant (Aim IK).
        """
        Q_global_t: Dict[str, np.ndarray] = {}
        diagnostics: Dict[str, Any] = {}
        root_override = root_rot_override or {}

        # Traitement racine
        root_candidates = [b for b, p in self.parents.items() if p is None]
        root_bone = root_candidates[0] if root_candidates else "Hips"

        if root_bone in root_override:
            Q_global_t[root_bone] = quat_normalize(root_override[root_bone])
        else:
            Q_global_t[root_bone] = self.rest_rotations.get(root_bone, quat_identity()).copy()

        diagnostics[root_bone] = {"status": "root_assigned"}

        # Propagation hiérarchique
        queue = list(self.children.get(root_bone, []))
        visited = {root_bone}

        while queue:
            bone = queue.pop(0)
            if bone in visited:
                continue
            visited.add(bone)

            parent = self.parents.get(bone)
            q_parent_glob = Q_global_t.get(parent, quat_identity()) if parent else quat_identity()
            q_rest = self.rest_rotations.get(bone, quat_identity())

            children = self.children.get(bone, [])
            if not children or bone not in positions_t or bone not in positions_rest:
                Q_global_t[bone] = q_rest.copy()
                diagnostics[bone] = {"status": "fallback_no_child_or_pos"}
                queue.extend(children)
                continue

            child = children[0]
            if child not in positions_t or child not in positions_rest:
                Q_global_t[bone] = q_rest.copy()
                diagnostics[bone] = {"status": "fallback_child_no_pos"}
                queue.extend(children)
                continue

            dir_rest_world = positions_rest[child] - positions_rest[bone]
            dir_t_world = positions_t[child] - positions_t[bone]

            norm_r = np.linalg.norm(dir_rest_world)
            norm_t = np.linalg.norm(dir_t_world)

            if norm_r < 1e-6 or norm_t < 1e-6:
                Q_global_t[bone] = q_rest.copy()
                diagnostics[bone] = {"status": "degenerate_aim_vector"}
                queue.extend(children)
                continue

            u_rest = dir_rest_world / norm_r
            v_curr = dir_t_world / norm_t

            # Visée vectorielle
            r_move = quat_from_two_vectors(u_rest, v_curr)
            Q_global_t[bone] = quat_normalize(quat_mul(r_move, q_rest))
            diagnostics[bone] = {"status": "aim_success"}

            queue.extend(children)

        return Q_global_t, diagnostics

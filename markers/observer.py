"""Module markers.observer
========================
Moteur d'observation, d'analyse cinématique et de diagnostic 3D des marqueurs.

Fournit :
- Le calcul des vitesses et accélérations scalaires/vectorielles par marqueur et par os
- La vérification de la rigidité inter-marqueurs (détection de distorsions géométriques)
- L'extraction des trièdres de repères locaux (axes RGB : X rouge, Y vert, Z bleu)
- L'exportation d'un payload JSON compact et complet pour le visualiseur WebGL 3D
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import json
import numpy as np


class MarkerObserver:
    """
    Analyseur cinématique et diagnostiqueur des trajectoires de marqueurs 3D.
    """

    def __init__(
        self,
        bones_hierarchy: Dict[str, Optional[str]],
        rest_cages: Dict[str, np.ndarray],
        frames_cages: List[Dict[str, np.ndarray]],
        fps: float = 30.0,
    ):
        """
        Args:
            bones_hierarchy: Arbre hiérarchique {nom_os: nom_parent}
            rest_cages: Dictionnaire {nom_os: np.ndarray (N_markers, 3)} au repos
            frames_cages: Liste temporelle de {nom_os: np.ndarray (N_markers, 3)}
            fps: Fréquence d'échantillonnage temporelle en Hz
        """
        self.bones_hierarchy = bones_hierarchy
        self.rest_cages = rest_cages
        self.frames_cages = frames_cages
        self.fps = fps
        self.dt = 1.0 / max(fps, 1e-3)
        self.n_frames = len(frames_cages)

    def analyze_rigidity(self) -> Dict[str, Dict[str, float]]:
        """
        Vérifie la rigidité des cages tétraédriques au cours du temps.
        Pour une cage rigide parfaite, la distance entre chaque paire de marqueurs
        doit rester strictement constante (déviation type < 1e-4).
        """
        results: Dict[str, Dict[str, float]] = {}

        for bone, rest_pts in self.rest_cages.items():
            k = len(rest_pts)
            if k < 2 or self.n_frames == 0:
                continue

            # Calculer les distances de repos entre toutes les paires (i, j)
            pair_dists_rest: List[float] = []
            pairs = []
            for i in range(k):
                for j in range(i + 1, k):
                    pairs.append((i, j))
                    d = float(np.linalg.norm(rest_pts[i] - rest_pts[j]))
                    pair_dists_rest.append(d)

            pair_dists_rest = np.array(pair_dists_rest, dtype=np.float64)

            # Calculer les distances sur toutes les frames
            max_dev = 0.0
            sum_dev = 0.0
            count = 0

            for frame in self.frames_cages:
                if bone not in frame:
                    continue
                pts_t = frame[bone]
                if len(pts_t) != k:
                    continue

                for idx, (i, j) in enumerate(pairs):
                    d_t = float(np.linalg.norm(pts_t[i] - pts_t[j]))
                    dev = abs(d_t - pair_dists_rest[idx])
                    if dev > max_dev:
                        max_dev = dev
                    sum_dev += dev
                    count += 1

            mean_dev = sum_dev / max(count, 1)
            results[bone] = {
                "max_rigidity_deviation": float(max_dev),
                "mean_rigidity_deviation": float(mean_dev),
                "is_rigid": bool(max_dev < 1e-3),
            }

        return results

    def compute_kinematics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calcule les profils de vitesse et d'accélération pour le joint pivot (marqueur M0).
        """
        kinematics: Dict[str, Dict[str, Any]] = {}

        for bone in self.rest_cages:
            positions: List[np.ndarray] = []
            for frame in self.frames_cages:
                if bone in frame and len(frame[bone]) > 0:
                    positions.append(frame[bone][0])  # M0 = position du joint
                else:
                    positions.append(np.zeros(3))

            pos_arr = np.array(positions, dtype=np.float64)  # (N_frames, 3)

            if len(pos_arr) < 2:
                kinematics[bone] = {
                    "max_speed": 0.0,
                    "mean_speed": 0.0,
                    "max_accel": 0.0,
                }
                continue

            # Vitesse : v(t) = ||p(t) - p(t-1)|| / dt
            diffs = np.diff(pos_arr, axis=0)
            speeds = np.linalg.norm(diffs, axis=1) / self.dt  # (N-1,)

            # Accélération : a(t) = ||v(t) - v(t-1)|| / dt
            if len(diffs) > 1:
                v_diffs = np.diff(diffs / self.dt, axis=0)
                accels = np.linalg.norm(v_diffs, axis=1) / self.dt
                max_accel = float(np.max(accels)) if len(accels) > 0 else 0.0
            else:
                max_accel = 0.0

            kinematics[bone] = {
                "max_speed": float(np.max(speeds)) if len(speeds) > 0 else 0.0,
                "mean_speed": float(np.mean(speeds)) if len(speeds) > 0 else 0.0,
                "max_accel": max_accel,
            }

        return kinematics

    def compute_frames_axes(self) -> List[Dict[str, Dict[str, List[float]]]]:
        """
        Calcule pour chaque os et chaque frame le trièdre orthonormé local (axes RGB) :
        - X (Rouge) : Axe longitudinal de l'os (du joint parent M0 vers l'enfant M1)
        - Y (Vert)  : Axe de flexion latérale (vers le marqueur M2)
        - Z (Bleu)  : Axe orthogonal normalisé (X cross Y, fixant le twist de M3)
        """
        frames_axes: List[Dict[str, Dict[str, List[float]]]] = []

        for frame in self.frames_cages:
            bone_axes: Dict[str, Dict[str, List[float]]] = {}
            for bone, pts in frame.items():
                if len(pts) < 4:
                    continue
                m0 = pts[0]
                m1 = pts[1]
                m2 = pts[2]
                m3 = pts[3]

                # Vecteur directeur de l'os
                v_bone = m1 - m0
                norm_bone = float(np.linalg.norm(v_bone))
                if norm_bone > 1e-6:
                    ux = v_bone / norm_bone
                else:
                    ux = np.array([1.0, 0.0, 0.0])

                # Vecteur latéral
                mid = m0 + 0.5 * v_bone
                v_lat = m2 - mid
                v_lat_proj = v_lat - np.dot(v_lat, ux) * ux
                norm_lat = float(np.linalg.norm(v_lat_proj))
                if norm_lat > 1e-6:
                    uy = v_lat_proj / norm_lat
                else:
                    fallback = np.array([0.0, 1.0, 0.0]) if abs(ux[1]) < 0.8 else np.array([0.0, 0.0, 1.0])
                    uy = np.cross(fallback, ux)
                    uy = uy / np.linalg.norm(uy)

                # Vecteur orthogonal (Z = X x Y)
                uz = np.cross(ux, uy)
                uz = uz / max(float(np.linalg.norm(uz)), 1e-6)

                bone_axes[bone] = {
                    "origin": [float(c) for c in m0],
                    "axis_x": [float(c) for c in ux],
                    "axis_y": [float(c) for c in uy],
                    "axis_z": [float(c) for c in uz],
                }

            frames_axes.append(bone_axes)

        return frames_axes

    def build_diagnostic_payload(
        self,
        source_name: str = "Source BVH",
        target_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Construit l'intégralité du payload JSON pour le visualiseur WebGL 3D.
        """
        rigidity_data = self.analyze_rigidity()
        kinematics_data = self.compute_kinematics()
        frames_axes = self.compute_frames_axes()

        # Conversion des cages de repos
        rest_dict: Dict[str, List[List[float]]] = {}
        for bone, pts in self.rest_cages.items():
            rest_dict[bone] = [[float(coord) for coord in pt] for pt in pts]

        # Conversion des frames
        frames_list: List[Dict[str, List[List[float]]]] = []
        for frame in self.frames_cages:
            frame_dict: Dict[str, List[List[float]]] = {}
            for bone, pts in frame.items():
                frame_dict[bone] = [[float(coord) for coord in pt] for pt in pts]
            frames_list.append(frame_dict)

        # Synthèse des diagnostics par os
        diagnostics: Dict[str, Dict[str, Any]] = {}
        for bone in self.rest_cages:
            rig = rigidity_data.get(bone, {})
            kin = kinematics_data.get(bone, {})
            pts_count = len(self.rest_cages[bone])

            diagnostics[bone] = {
                "markers_count": pts_count,
                "dof_supported": 3 if pts_count >= 3 else (2 if pts_count == 2 else 0),
                "is_rigid": rig.get("is_rigid", True),
                "max_rigidity_deviation": rig.get("max_rigidity_deviation", 0.0),
                "max_speed_m_s": kin.get("max_speed", 0.0),
                "mean_speed_m_s": kin.get("mean_speed", 0.0),
                "max_accel_m_s2": kin.get("max_accel", 0.0),
            }

        payload: Dict[str, Any] = {
            "metadata": {
                "source_name": source_name,
                "target_name": target_name or "Unknown",
                "fps": float(self.fps),
                "frame_count": self.n_frames,
                "bone_count": len(self.rest_cages),
                "marker_cage_type": "rigid_tetrahedron_4pts",
            },
            "bones_hierarchy": self.bones_hierarchy,
            "rest_cages": rest_dict,
            "frames": frames_list,
            "axes": frames_axes,
            "diagnostics": diagnostics,
        }

        return payload

    def export_observation_json(
        self,
        output_path: str,
        source_name: str = "Source BVH",
        target_name: Optional[str] = None,
        indent: Optional[int] = None,
    ) -> str:
        """
        Exporte le payload diagnostic au format JSON.
        """
        payload = self.build_diagnostic_payload(source_name=source_name, target_name=target_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
        return output_path

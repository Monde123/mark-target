"""
Module isolate.adapters.hybrik_pk_adapter
=========================================
Adaptateur spécifique pour les sorties HybrIK / SMPL-X (.pk):
- Lit res.pk (rotations quaternions, translations, fps)
- Utilise les données 3D disponibles (joints/vertices) pour générer des marqueurs cohérents
"""
from __future__ import annotations

import pickle
import numpy as np
from typing import Dict, List, Optional, Any

from adapters.base import SourceAdapter, MarkerFrameSequence


_SMPLX_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]


class HybrIKPKAdapter(SourceAdapter):
    """Adaptateur de lecture pour les fichiers res.pk issus de HybrIK."""

    def __init__(self, smplx_model_path: Optional[str] = None, markers_per_bone: int = 4):
        self.smplx_model_path = smplx_model_path
        self.markers_per_bone = markers_per_bone

    @staticmethod
    def _find_first_array(data: Dict[str, Any], candidate_keys: List[str], ndim: int) -> Optional[np.ndarray]:
        for key in candidate_keys:
            value = data.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == ndim:
                return arr
        return None

    def load(self, source_path: str, **kwargs) -> MarkerFrameSequence:
        """Charge le fichier res.pk."""
        with open(source_path, "rb") as f:
            data = pickle.load(f)

        fps = float(data.get("fps", 30.0))

        quats_arr = np.asarray(data.get("quats", []), dtype=np.float64)
        transl_arr = np.asarray(data.get("transl", []), dtype=np.float64)

        if quats_arr.ndim == 2 and quats_arr.shape[-1] == 4:
            root_rotations = [quats_arr[i] for i in range(quats_arr.shape[0])]
            n_frames_quat = quats_arr.shape[0]
        elif quats_arr.ndim >= 3 and quats_arr.shape[-1] == 4:
            root_rotations = [quats_arr[i, 0] for i in range(quats_arr.shape[0])]
            n_frames_quat = quats_arr.shape[0]
        else:
            root_rotations = None
            n_frames_quat = 0

        if transl_arr.ndim == 2 and transl_arr.shape[-1] >= 3:
            root_translations = transl_arr[:, :3].astype(np.float64)
            n_frames_trans = root_translations.shape[0]
        else:
            root_translations = None
            n_frames_trans = 0

        joints_arr = self._find_first_array(
            data,
            ["joints3d", "joints_3d", "joints", "pred_joints", "pred_xyz_29", "xyz_29", "kp_3d"],
            ndim=3,
        )
        verts_arr = self._find_first_array(
            data,
            ["verts", "vertices", "verts3d", "pred_vertices"],
            ndim=3,
        )

        n_frames_data = [
            int(data.get("n_frames", 0)) if data.get("n_frames") is not None else 0,
            n_frames_quat,
            n_frames_trans,
            int(joints_arr.shape[0]) if joints_arr is not None else 0,
            int(verts_arr.shape[0]) if verts_arr is not None else 0,
        ]
        n_frames = max(n_frames_data)

        if n_frames <= 0:
            raise RuntimeError(
                "HybrIK PK invalide: impossible de déterminer le nombre de frames. "
                "Le fichier doit contenir n_frames, quats, transl, joints3d ou verts."
            )

        frames_markers: List[Dict[str, np.ndarray]] = []
        rest_markers: Dict[str, np.ndarray] = {}

        if joints_arr is not None and joints_arr.shape[-1] >= 3:
            joints_arr = joints_arr[:, :, :3].astype(np.float64)
            n_joints = joints_arr.shape[1]
            names = [
                _SMPLX_JOINT_NAMES[i] if i < len(_SMPLX_JOINT_NAMES) else f"joint_{i}"
                for i in range(n_joints)
            ]

            rest_frame = joints_arr[0]
            for j_idx, name in enumerate(names):
                rest_markers[name] = rest_frame[j_idx].reshape(1, 3)

            for f_idx in range(n_frames):
                src_idx = min(f_idx, joints_arr.shape[0] - 1)
                frame = joints_arr[src_idx]
                frame_markers = {name: frame[j_idx].reshape(1, 3) for j_idx, name in enumerate(names)}
                frames_markers.append(frame_markers)

        elif verts_arr is not None and verts_arr.shape[-1] >= 3:
            verts_arr = verts_arr[:, :, :3].astype(np.float64)
            sample_count = max(1, min(int(self.markers_per_bone), 16))
            v_count = verts_arr.shape[1]
            sample_indices = np.linspace(0, v_count - 1, num=sample_count, dtype=int)
            marker_names = [f"vertex_{i}" for i in sample_indices]

            rest_frame = verts_arr[0]
            for idx, name in zip(sample_indices, marker_names):
                rest_markers[name] = rest_frame[idx].reshape(1, 3)

            for f_idx in range(n_frames):
                src_idx = min(f_idx, verts_arr.shape[0] - 1)
                frame = verts_arr[src_idx]
                frame_markers = {
                    name: frame[idx].reshape(1, 3)
                    for idx, name in zip(sample_indices, marker_names)
                }
                frames_markers.append(frame_markers)

        else:
            if root_translations is not None:
                base = root_translations[0]
                rest_markers = {"pelvis": base.reshape(1, 3)}
                for f_idx in range(n_frames):
                    src_idx = min(f_idx, root_translations.shape[0] - 1)
                    frames_markers.append({"pelvis": root_translations[src_idx].reshape(1, 3)})
            else:
                raise RuntimeError(
                    "Impossible de générer des marqueurs HybrIK: aucune donnée joints/verts/transl exploitable. "
                    "Fournissez un .pk contenant joints3d/verts/transl, ou activez une génération SMPL-X optionnelle "
                    "avec dépendances et modèle disponibles."
                )

        if self.smplx_model_path and joints_arr is None and verts_arr is None:
            raise RuntimeError(
                "Chemin SMPL-X fourni mais génération de marqueurs SMPL-X non disponible avec les données actuelles. "
                "Installez les dépendances optionnelles SMPL-X/PyTorch et fournissez les paramètres nécessaires, "
                "ou utilisez un .pk contenant joints3d/verts."
            )

        return MarkerFrameSequence(
            frames_markers=frames_markers,
            rest_markers=rest_markers,
            root_rotations=root_rotations,
            root_translations=root_translations,
            fps=fps,
        )

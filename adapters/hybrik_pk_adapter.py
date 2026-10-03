"""
Module isolate.adapters.hybrik_pk_adapter
=========================================
Adaptateur spécifique pour les sorties HybrIK / SMPL-X (.pk):
- Lit res.pk (rotations quaternions, translations, fps)
- Utilise un modèle SMPL-X pour poser les sommets et générer les marqueurs 3D
"""
from __future__ import annotations

import pickle
import numpy as np
from typing import Dict, List, Optional, Any

from adapters.base import SourceAdapter, MarkerFrameSequence
class HybrIKPKAdapter(SourceAdapter):
    """Adaptateur de lecture pour les fichiers res.pk issus de HybrIK."""

    def __init__(self, smplx_model_path: Optional[str] = None, markers_per_bone: int = 4):
        self.smplx_model_path = smplx_model_path
        self.markers_per_bone = markers_per_bone

    def load(self, source_path: str, **kwargs) -> MarkerFrameSequence:
        """Charge le fichier res.pk."""
        with open(source_path, "rb") as f:
            data = pickle.load(f)

        n_frames = int(data.get("n_frames", 0))
        if n_frames <= 0:
            raise ValueError("Fichier HybrIK invalide: `n_frames` absent ou non positif.")

        fps = float(data.get("fps", 30.0))
        quats = np.asarray(data.get("quats", []), dtype=np.float64)
        transl = np.asarray(data.get("transl", []), dtype=np.float64)

        marker_frames_data = data.get("markers_3d")
        marker_names = data.get("marker_names")

        if marker_frames_data is None:
            raise ValueError(
                "Le fichier HybrIK ne contient pas de marqueurs 3D (`markers_3d`). "
                "La génération de marqueurs depuis SMPL-X n'est pas implémentée dans cet adaptateur "
                "sans pipeline mesh explicite."
            )

        markers_array = np.asarray(marker_frames_data, dtype=np.float64)
        if markers_array.ndim != 3 or markers_array.shape[2] != 3:
            raise ValueError(
                f"Format `markers_3d` invalide: attendu (n_frames, n_markers, 3), reçu {markers_array.shape}."
            )
        if markers_array.shape[0] != n_frames:
            raise ValueError(
                f"Incohérence HybrIK: n_frames={n_frames} mais markers_3d contient {markers_array.shape[0]} frames."
            )

        n_markers = markers_array.shape[1]
        if marker_names is None:
            marker_names = [f"marker_{i}" for i in range(n_markers)]
        if len(marker_names) != n_markers:
            raise ValueError(
                f"Incohérence HybrIK: marker_names contient {len(marker_names)} entrées pour {n_markers} marqueurs."
            )

        frames_markers: List[Dict[str, np.ndarray]] = []
        for t in range(n_frames):
            frame_markers = {
                str(marker_names[i]): markers_array[t, i].copy()
                for i in range(n_markers)
            }
            frames_markers.append(frame_markers)

        rest_markers_raw = data.get("rest_markers")
        if rest_markers_raw is not None:
            rest_arr = np.asarray(rest_markers_raw, dtype=np.float64)
            if rest_arr.ndim != 2 or rest_arr.shape[1] != 3 or rest_arr.shape[0] != n_markers:
                raise ValueError(
                    f"Format rest_markers invalide: attendu ({n_markers}, 3), reçu {rest_arr.shape}."
                )
            rest_markers = {str(marker_names[i]): rest_arr[i].copy() for i in range(n_markers)}
        else:
            rest_markers = {str(marker_names[i]): markers_array[0, i].copy() for i in range(n_markers)}

        root_rots = None
        if quats.size > 0:
            if quats.ndim != 3 or quats.shape[0] != n_frames or quats.shape[2] != 4:
                raise ValueError(
                    f"Format quats invalide: attendu (n_frames, n_joints, 4), reçu {quats.shape}."
                )
            root_rots = [quats[t, 0].copy() for t in range(n_frames)]

        root_trans = None
        if transl.size > 0:
            if transl.ndim == 1:
                if transl.size == 3:
                    transl = transl.reshape(1, 3)
                elif transl.size % 3 == 0:
                    transl = transl.reshape(-1, 3)
            if transl.ndim != 2 or transl.shape[1] != 3:
                raise ValueError(
                    f"Format transl invalide: attendu (n_frames, 3), reçu {transl.shape}."
                )
            if transl.shape[0] != n_frames:
                raise ValueError(
                    f"Incohérence HybrIK: n_frames={n_frames} mais transl contient {transl.shape[0]} frames."
                )
            root_trans = transl

        return MarkerFrameSequence(
            frames_markers=frames_markers,
            rest_markers=rest_markers,
            root_rotations=root_rots,
            root_translations=root_trans,
            fps=fps,
        )

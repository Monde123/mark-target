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
from markers.surface_sampling import build_bone_submeshes, sample_multi_markers_for_bone
from markers.tracker import extract_markers_positions, compute_marker_stability_weights


class HybrIKPKAdapter(SourceAdapter):
    """Adaptateur de lecture pour les fichiers res.pk issus de HybrIK."""

    def __init__(self, smplx_model_path: Optional[str] = None, markers_per_bone: int = 4):
        self.smplx_model_path = smplx_model_path
        self.markers_per_bone = markers_per_bone

    def load(self, source_path: str, **kwargs) -> MarkerFrameSequence:
        """Charge le fichier res.pk."""
        with open(source_path, "rb") as f:
            data = pickle.load(f)

        n_frames = int(data["n_frames"])
        fps = float(data.get("fps", 30.0))
        quats = np.asarray(data.get("quats", []), dtype=np.float64)
        transl = np.asarray(data.get("transl", []), dtype=np.float64)

        # Si le modèle SMPL-X est fourni, on peut calculer les marqueurs géométriques exacts
        # Sinon, on fournit les trajectoires de translation et quaternions pour traitement direct
        dummy_markers: List[Dict[str, np.ndarray]] = []
        rest_markers: Dict[str, np.ndarray] = {}

        return MarkerFrameSequence(
            frames_markers=dummy_markers,
            rest_markers=rest_markers,
            root_rotations=[quats[t, 0] for t in range(n_frames)] if len(quats) > 0 else None,
            root_translations=transl if len(transl) > 0 else None,
            fps=fps,
        )

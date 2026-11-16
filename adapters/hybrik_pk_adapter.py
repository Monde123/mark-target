"""
Module adapters.hybrik_pk_adapter
=========================================
Adaptateur pour les sorties HybrIK / SMPL-X (.pk):
- Lit les fichiers .pk contenant rotations (quats), translations et/ou coordonnées 3D (pred_xyz_jts)
- Reconstitue les marqueurs 3D ou les cages virtuelles pour le retargeting universel
"""
from __future__ import annotations

import pickle
import numpy as np
from typing import Dict, List, Optional, Any

from adapters.base import SourceAdapter, MarkerFrameSequence
from mappings.standard_humanoid import SMPLX_TO_STANDARD
from mappings.buffer_armature import CanonicalBufferArmature
from markers.virtual_cage import VirtualBoneCageGenerator
from core.geometry import rotate_vector, quaternion_to_matrix


class HybrIKPKAdapter(SourceAdapter):
    """Adaptateur de lecture pour les fichiers res.pk issus de HybrIK / SMPL-X."""

    def __init__(self, smplx_model_path: Optional[str] = None, markers_per_bone: int = 4):
        self.smplx_model_path = smplx_model_path
        self.markers_per_bone = markers_per_bone

    def load(self, source_path: str, **kwargs) -> MarkerFrameSequence:
        """Charge le fichier .pk (HybrIK / SMPL-X)."""
        with open(source_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Le fichier {source_path} ne contient pas un dictionnaire de données valide.")

        n_frames = int(data.get("n_frames", len(data.get("quats", [])) or len(data.get("pred_xyz_jts", [])) or 0))
        if n_frames == 0:
            raise ValueError(f"Le fichier {source_path} ne contient aucune frame d'animation valide.")

        fps = float(data.get("fps", 30.0))
        quats = np.asarray(data.get("quats", []), dtype=np.float64) if "quats" in data else None
        transl = np.asarray(data.get("transl", []), dtype=np.float64) if "transl" in data else None

        # 1. Cas où les positions 3D articulaires sont directement présentes (ex: pred_xyz_jts)
        jts_key = next((k for k in ["pred_xyz_jts", "joints3d", "pred_xyz_jts_29", "joint_positions"] if k in data), None)

        smplx_names = list(SMPLX_TO_STANDARD.keys())

        if jts_key is not None:
            raw_jts = np.asarray(data[jts_key], dtype=np.float64) # (n_frames, n_joints, 3)
            n_jts = raw_jts.shape[1] if raw_jts.ndim >= 3 else 0

            frames_markers: List[Dict[str, np.ndarray]] = []
            rest_markers: Dict[str, np.ndarray] = {}

            # Frame 0 comme repos de référence
            for j_idx in range(min(n_jts, len(smplx_names))):
                bone_name = smplx_names[j_idx]
                rest_markers[bone_name] = raw_jts[0, j_idx].reshape(1, 3)

            for t in range(n_frames):
                frame_dict: Dict[str, np.ndarray] = {}
                for j_idx in range(min(n_jts, len(smplx_names))):
                    bone_name = smplx_names[j_idx]
                    frame_dict[bone_name] = raw_jts[t, j_idx].reshape(1, 3)
                frames_markers.append(frame_dict)

            return MarkerFrameSequence(
                frames_markers=frames_markers,
                rest_markers=rest_markers,
                root_rotations=[quats[t, 0] for t in range(n_frames)] if (quats is not None and len(quats) > 0) else None,
                root_translations=transl if (transl is not None and len(transl) > 0) else None,
                fps=fps,
            )

        # 2. Cas où seules les rotations de joints (quats) sont disponibles
        if quats is not None and len(quats) > 0:
            # Générer des positions à partir de l'armature canonique
            canonical_pos = CanonicalBufferArmature.get_canonical_positions()
            canonical_parents = CanonicalBufferArmature.get_canonical_parents()

            rest_markers = {b: np.asarray(p, dtype=np.float64).reshape(1, 3) for b, p in canonical_pos.items()}

            joint_rots_seq: List[Dict[str, np.ndarray]] = []
            frames_markers = []

            for t in range(n_frames):
                frame_rots: Dict[str, np.ndarray] = {}
                frame_pos: Dict[str, np.ndarray] = {}
                for j_idx in range(min(quats.shape[1], len(smplx_names))):
                    smpl_name = smplx_names[j_idx]
                    std_name = SMPLX_TO_STANDARD[smpl_name]
                    q = quats[t, j_idx]
                    frame_rots[std_name] = q
                    # Position relative approximée
                    p_rest = canonical_pos.get(std_name, np.zeros(3))
                    frame_pos[std_name] = p_rest.reshape(1, 3)

                joint_rots_seq.append(frame_rots)
                frames_markers.append(frame_pos)

            return MarkerFrameSequence(
                frames_markers=frames_markers,
                rest_markers=rest_markers,
                root_rotations=[quats[t, 0] for t in range(n_frames)],
                root_translations=transl if transl is not None else None,
                joint_rotations=joint_rots_seq,
                parents=canonical_parents,
                fps=fps,
            )

        raise ValueError(
            f"Le fichier {source_path} ne contient ni 'pred_xyz_jts' (positions 3D) "
            f"ni 'quats' (rotations articulaires)."
        )

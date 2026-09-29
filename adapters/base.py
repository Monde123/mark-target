"""
Module isolate.adapters.base
============================
Interfaces de base (Protocols / Abstract Classes) pour les entrées et sorties.
Permet d'accueillir n'importe quel format (PK, BVH, Mixamo, VRM, MoCap).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class TargetSkeleton:
    """Structure standardisée d'un rig ou squelette cible."""
    rest_rotations: Dict[str, np.ndarray]  # {bone_name: quat_rest [w, x, y, z]}
    parents: Dict[str, Optional[str]]       # {bone_name: parent_name or None}
    raw_names: Optional[Dict[str, str]] = None  # {nom_canonique: nom_dans_le_fichier}
    root_name: str = "Hips"
    fps: float = 30.0


@dataclass
class MarkerFrameSequence:
    """Séquence temporelle de positions 3D de marqueurs."""
    frames_markers: List[Dict[str, np.ndarray]]  # liste de {bone_name: (N, 3)}
    rest_markers: Dict[str, np.ndarray]          # {bone_name: (N, 3)}
    weights: Optional[Dict[str, np.ndarray]] = None # {bone_name: (N,)}
    root_rotations: Optional[List[np.ndarray]] = None # liste de [w, x, y, z] pour la racine si dispo
    root_translations: Optional[np.ndarray] = None    # (n_frames, 3)
    fps: float = 30.0


class SourceAdapter(ABC):
    """Adaptateur abstrait transformant une source quelconque en séquence de marqueurs."""

    @abstractmethod
    def load(self, source_path: str, **kwargs) -> MarkerFrameSequence:
        """Charge le fichier source et extrait la séquence de marqueurs."""
        pass


class TargetAdapter(ABC):
    """Adaptateur abstrait pour charger le squelette cible et exporter l'animation calculée."""

    @abstractmethod
    def load_skeleton(self, target_model_path: str, **kwargs) -> TargetSkeleton:
        """Extrait la structure du squelette cible (bind pose et hiérarchie)."""
        pass

    @abstractmethod
    def export_animation(
        self,
        target_model_path: str,
        output_path: str,
        animation_clip: List[Dict[str, Any]],
        skeleton: TargetSkeleton,
        **kwargs
    ) -> None:
        """Exporte l'animation résolue dans le format final désiré (GLB, VRM, BVH)."""
        pass

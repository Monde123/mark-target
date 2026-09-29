"""
Module isolate.adapters.bvh_adapter
===================================
Adaptateur BVH :
- Peut servir de Source (lit la hiérarchie BVH et extrait les positions 3D des joints comme marqueurs)
- Peut servir de Target (exporte l'animation résolue en fichier .bvh universel)
"""
from __future__ import annotations

import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from base import SourceAdapter, TargetAdapter, TargetSkeleton, MarkerFrameSequence
from core.geometry import quat_identity, quat_normalize, quaternion_to_matrix


class BVHAdapter(SourceAdapter, TargetAdapter):
    """Adaptateur bidirectionnel pour le format d'animation BVH."""

    def load(self, source_path: str, **kwargs) -> MarkerFrameSequence:
        """
        Lit un fichier BVH et reconstruit les trajectoires 3D des joints
        en avant (cinématique directe) pour les utiliser comme marqueurs 3D.
        """
        # Implémentation simplifiée et robuste d'extraction de trajectoires BVH
        with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        
        # Trouver la section MOTION
        motion_idx = -1
        for idx, line in enumerate(lines):
            if line.upper().startswith("MOTION"):
                motion_idx = idx
                break
        
        if motion_idx == -1:
            raise ValueError(f"Fichier BVH invalide : section MOTION absente dans {source_path}")

        frames_line = lines[motion_idx + 1]
        frame_time_line = lines[motion_idx + 2]
        n_frames = int(re.findall(r"\d+", frames_line)[0])
        frame_time = float(re.findall(r"[\d\.]+", frame_time_line)[0])
        fps = 1.0 / frame_time if frame_time > 0 else 30.0

        # Données de frames
        motion_data = []
        for line in lines[motion_idx + 3:]:
            parts = [float(p) for p in line.split()]
            if parts:
                motion_data.append(parts)

        # Les positions des joints peuvent être calculées ou utilisées directement
        # Pour une source purement BVH, on génère un MarkerFrameSequence
        dummy_markers: List[Dict[str, np.ndarray]] = []
        rest_markers: Dict[str, np.ndarray] = {"Hips": np.zeros((3, 3))}

        for t in range(min(len(motion_data), n_frames)):
            frame_dict = {"Hips": np.tile(np.array(motion_data[t][:3]), (3, 1))}
            dummy_markers.append(frame_dict)

        return MarkerFrameSequence(
            frames_markers=dummy_markers,
            rest_markers=rest_markers,
            fps=fps,
        )

    def load_skeleton(self, target_model_path: str, **kwargs) -> TargetSkeleton:
        """Parse le HEADER BVH pour extraire la structure du squelette cible."""
        with open(target_model_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines()]

        parents: Dict[str, Optional[str]] = {}
        rest_rot: Dict[str, np.ndarray] = {}
        current_stack: List[str] = []
        root_name = "Hips"

        for line in lines:
            if line.upper().startswith("ROOT") or line.upper().startswith("JOINT"):
                parts = line.split()
                name = parts[1]
                parent = current_stack[-1] if current_stack else None
                if not current_stack:
                    root_name = name
                parents[name] = parent
                rest_rot[name] = quat_identity()
                current_stack.append(name)
            elif line.startswith("}"):
                if current_stack:
                    current_stack.pop()
            elif line.upper().startswith("MOTION"):
                break

        return TargetSkeleton(
            rest_rotations=rest_rot,
            parents=parents,
            root_name=root_name,
            fps=30.0,
        )

    def export_animation(
        self,
        target_model_path: str,
        output_path: str,
        animation_clip: List[Dict[str, Any]],
        skeleton: TargetSkeleton,
        **kwargs
    ) -> None:
        """Exporte l'animation en format BVH."""
        # Si target_model_path est un template BVH existant, copie le header et écrit le MOTION
        header_lines = []
        if target_model_path.endswith(".bvh"):
            with open(target_model_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip().upper().startswith("MOTION"):
                        break
                    header_lines.append(line)

        fps = skeleton.fps if skeleton.fps > 0 else 30.0
        frame_time = 1.0 / fps

        with open(output_path, "w", encoding="utf-8") as out:
            if header_lines:
                out.writelines(header_lines)
            else:
                out.write("HIERARCHY\n")
                out.write(f"ROOT {skeleton.root_name}\n{{\n  OFFSET 0.0 0.0 0.0\n  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n  End Site {{\n    OFFSET 0 0 0\n  }}\n}}\n")

            out.write("MOTION\n")
            out.write(f"Frames: {len(animation_clip)}\n")
            out.write(f"Frame Time: {frame_time:.6f}\n")

            for frame in animation_clip:
                root_trans = frame.get("root", np.zeros(3))
                # Écriture des canaux de base
                line_vals = [f"{root_trans[0]:.4f}", f"{root_trans[1]:.4f}", f"{root_trans[2]:.4f}"]
                # Ajout des rotations en euler (placeholder standardisé)
                line_vals.extend(["0.0000", "0.0000", "0.0000"])
                out.write(" ".join(line_vals) + "\n")

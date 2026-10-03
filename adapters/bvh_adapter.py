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

try:
    from adapters.base import SourceAdapter, TargetAdapter, TargetSkeleton, MarkerFrameSequence
except ImportError:
    from base import SourceAdapter, TargetAdapter, TargetSkeleton, MarkerFrameSequence
from core.geometry import (
    quat_identity,
    quat_normalize,
    quat_mul,
    quaternion_to_matrix,
    matrix_to_quaternion,
    rotate_vector,
)


_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _axis_rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    if axis == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "Y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "Z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Axe BVH invalide: {axis}")


def _matrix_to_euler_intrinsic_xyz_orders(R: np.ndarray, order: str) -> Tuple[float, float, float]:
    # Adapté des formules robustes utilisées pour les 6 ordres Tait-Bryan.
    # Retourne (x, y, z) en radians.
    m11, m12, m13 = R[0, 0], R[0, 1], R[0, 2]
    m21, m22, m23 = R[1, 0], R[1, 1], R[1, 2]
    m31, m32, m33 = R[2, 0], R[2, 1], R[2, 2]

    clamp = np.clip
    eps = 1e-7

    if order == "XYZ":
        y = np.arcsin(clamp(m13, -1.0, 1.0))
        if abs(m13) < 1.0 - eps:
            x = np.arctan2(-m23, m33)
            z = np.arctan2(-m12, m11)
        else:
            x = np.arctan2(m32, m22)
            z = 0.0
    elif order == "YXZ":
        x = np.arcsin(-clamp(m23, -1.0, 1.0))
        if abs(m23) < 1.0 - eps:
            y = np.arctan2(m13, m33)
            z = np.arctan2(m21, m22)
        else:
            y = np.arctan2(-m31, m11)
            z = 0.0
    elif order == "ZXY":
        x = np.arcsin(clamp(m32, -1.0, 1.0))
        if abs(m32) < 1.0 - eps:
            y = np.arctan2(-m31, m33)
            z = np.arctan2(-m12, m22)
        else:
            y = 0.0
            z = np.arctan2(m21, m11)
    elif order == "ZYX":
        y = np.arcsin(-clamp(m31, -1.0, 1.0))
        if abs(m31) < 1.0 - eps:
            x = np.arctan2(m32, m33)
            z = np.arctan2(m21, m11)
        else:
            x = 0.0
            z = np.arctan2(-m12, m22)
    elif order == "YZX":
        z = np.arcsin(clamp(m21, -1.0, 1.0))
        if abs(m21) < 1.0 - eps:
            x = np.arctan2(-m23, m22)
            y = np.arctan2(-m31, m11)
        else:
            x = 0.0
            y = np.arctan2(m13, m33)
    elif order == "XZY":
        z = np.arcsin(-clamp(m12, -1.0, 1.0))
        if abs(m12) < 1.0 - eps:
            x = np.arctan2(m32, m22)
            y = np.arctan2(m13, m11)
        else:
            x = np.arctan2(-m23, m33)
            y = 0.0
    else:
        raise ValueError(f"Ordre Euler non supporté: {order}")

    return float(x), float(y), float(z)


def _parse_bvh_file(source_path: str) -> Dict[str, Any]:
    with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    stripped_lines = [line.strip() for line in raw_lines]
    motion_idx = next((i for i, line in enumerate(stripped_lines) if line.upper().startswith("MOTION")), -1)
    if motion_idx < 0:
        raise ValueError(f"Fichier BVH invalide: section MOTION absente dans {source_path}")

    parents: Dict[str, Optional[str]] = {}
    offsets: Dict[str, np.ndarray] = {}
    channels: Dict[str, List[str]] = {}
    joint_order: List[str] = []
    channel_joint_order: List[str] = []
    end_sites: Dict[str, List[np.ndarray]] = {}

    root_name: Optional[str] = None
    stack: List[str] = []
    pending_joint: Optional[str] = None
    pending_end_site_parent: Optional[str] = None
    end_site_stack_depth: List[int] = []

    for line in stripped_lines[:motion_idx]:
        if not line:
            continue

        if line.startswith("ROOT ") or line.startswith("JOINT "):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            joint_name = parts[1].strip()
            parent = stack[-1] if stack else None
            if root_name is None:
                root_name = joint_name
            parents[joint_name] = parent
            offsets[joint_name] = np.zeros(3, dtype=np.float64)
            channels[joint_name] = []
            joint_order.append(joint_name)
            pending_joint = joint_name
            continue

        if line.upper().startswith("END SITE"):
            pending_end_site_parent = stack[-1] if stack else None
            continue

        if line == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
            elif pending_end_site_parent is not None:
                end_site_stack_depth.append(len(stack))
                stack.append("__END_SITE__")
            continue

        if line == "}":
            if stack:
                popped = stack.pop()
                if popped == "__END_SITE__" and end_site_stack_depth:
                    end_site_stack_depth.pop()
                    pending_end_site_parent = None
            continue

        if line.startswith("OFFSET "):
            parts = line.split()
            if len(parts) >= 4:
                vec = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                if stack:
                    current = stack[-1]
                    if current in offsets:
                        offsets[current] = vec
                    elif current == "__END_SITE__" and pending_end_site_parent:
                        end_sites.setdefault(pending_end_site_parent, []).append(vec)
            continue

        if line.startswith("CHANNELS ") and stack:
            current = stack[-1]
            if current in channels:
                parts = line.split()
                declared = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                chs = parts[2:2 + declared]
                channels[current] = chs
                if chs:
                    channel_joint_order.append(current)
            continue

    if root_name is None:
        raise ValueError(f"Fichier BVH invalide: root introuvable dans {source_path}")

    frames_line = stripped_lines[motion_idx + 1] if motion_idx + 1 < len(stripped_lines) else ""
    frame_time_line = stripped_lines[motion_idx + 2] if motion_idx + 2 < len(stripped_lines) else ""
    frames_match = re.findall(r"\d+", frames_line)
    frame_time_match = re.findall(r"[\d\.eE+-]+", frame_time_line)
    n_frames = int(frames_match[0]) if frames_match else 0
    frame_time = float(frame_time_match[-1]) if frame_time_match else (1.0 / 30.0)
    fps = 1.0 / frame_time if frame_time > 0.0 else 30.0

    total_channels = sum(len(channels[j]) for j in channel_joint_order)

    motion_values: List[float] = []
    for line in stripped_lines[motion_idx + 3:]:
        if not line:
            continue
        motion_values.extend(float(v) for v in line.split())

    expected = n_frames * total_channels
    if expected > 0 and len(motion_values) < expected:
        raise ValueError(
            f"Données MOTION insuffisantes dans {source_path}: "
            f"attendu {expected} valeurs, reçu {len(motion_values)}"
        )

    if total_channels == 0:
        motion_array = np.zeros((n_frames, 0), dtype=np.float64)
    else:
        motion_array = np.asarray(motion_values[:expected], dtype=np.float64).reshape(n_frames, total_channels)

    return {
        "root_name": root_name,
        "parents": parents,
        "offsets": offsets,
        "channels": channels,
        "joint_order": joint_order,
        "channel_joint_order": channel_joint_order,
        "end_sites": end_sites,
        "motion": motion_array,
        "n_frames": n_frames,
        "frame_time": frame_time,
        "fps": fps,
        "header_lines": raw_lines[:motion_idx],
    }


def _compute_rest_positions(
    root_name: str,
    joint_order: List[str],
    parents: Dict[str, Optional[str]],
    offsets: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    global_positions: Dict[str, np.ndarray] = {}
    global_rotations: Dict[str, np.ndarray] = {}

    for joint in joint_order:
        parent = parents.get(joint)
        if parent is None:
            global_rotations[joint] = quat_identity()
            global_positions[joint] = offsets.get(joint, np.zeros(3, dtype=np.float64)).copy()
        else:
            parent_q = global_rotations[parent]
            parent_p = global_positions[parent]
            local_t = offsets.get(joint, np.zeros(3, dtype=np.float64))
            global_rotations[joint] = parent_q
            global_positions[joint] = parent_p + rotate_vector(parent_q, local_t)

    if root_name not in global_positions:
        global_positions[root_name] = np.zeros(3, dtype=np.float64)
    return global_positions


def _build_fk_frame(
    frame_values: np.ndarray,
    root_name: str,
    joint_order: List[str],
    channel_joint_order: List[str],
    parents: Dict[str, Optional[str]],
    offsets: Dict[str, np.ndarray],
    channels: Dict[str, List[str]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    local_rotations: Dict[str, np.ndarray] = {j: quat_identity() for j in joint_order}
    local_translations: Dict[str, np.ndarray] = {
        j: offsets.get(j, np.zeros(3, dtype=np.float64)).copy() for j in joint_order
    }

    cursor = 0
    root_translation = np.zeros(3, dtype=np.float64)

    for joint in channel_joint_order:
        chs = channels.get(joint, [])
        local_t = offsets.get(joint, np.zeros(3, dtype=np.float64)).copy()
        rot_m = np.eye(3, dtype=np.float64)

        for ch in chs:
            if cursor >= len(frame_values):
                break
            value = float(frame_values[cursor])
            cursor += 1

            if ch.endswith("position"):
                axis = ch[0].upper()
                local_t[_AXIS_INDEX[axis]] = value
                if joint == root_name:
                    root_translation[_AXIS_INDEX[axis]] = value
            elif ch.endswith("rotation"):
                axis = ch[0].upper()
                rot_m = rot_m @ _axis_rotation_matrix(axis, np.deg2rad(value))

        local_translations[joint] = local_t
        local_rotations[joint] = matrix_to_quaternion(rot_m)

    global_rotations: Dict[str, np.ndarray] = {}
    global_positions: Dict[str, np.ndarray] = {}

    for joint in joint_order:
        parent = parents.get(joint)
        q_local = local_rotations.get(joint, quat_identity())
        t_local = local_translations.get(joint, offsets.get(joint, np.zeros(3, dtype=np.float64)))

        if parent is None:
            global_rotations[joint] = q_local
            global_positions[joint] = t_local
        else:
            q_parent = global_rotations[parent]
            p_parent = global_positions[parent]
            global_rotations[joint] = quat_normalize(quat_mul(q_parent, q_local))
            global_positions[joint] = p_parent + rotate_vector(q_parent, t_local)

    root_global_rotation = global_rotations.get(root_name, quat_identity())
    return global_positions, global_rotations, root_translation, root_global_rotation


def _quaternion_to_channel_degrees(q: np.ndarray, rotation_axes_order: List[str]) -> Dict[str, float]:
    if not rotation_axes_order:
        return {"X": 0.0, "Y": 0.0, "Z": 0.0}

    order = "".join(rotation_axes_order)
    if len(order) != 3 or any(a not in _AXIS_INDEX for a in order):
        return {"X": 0.0, "Y": 0.0, "Z": 0.0}

    R = quaternion_to_matrix(quat_normalize(q))
    x, y, z = _matrix_to_euler_intrinsic_xyz_orders(R, order)
    deg = np.degrees(np.array([x, y, z], dtype=np.float64))
    return {"X": float(deg[0]), "Y": float(deg[1]), "Z": float(deg[2])}


class BVHAdapter(SourceAdapter, TargetAdapter):
    """Adaptateur bidirectionnel pour le format d'animation BVH."""

    def load(self, source_path: str, **kwargs) -> MarkerFrameSequence:
        """
        Lit un fichier BVH et reconstruit les trajectoires 3D des joints
        en avant (cinématique directe) pour les utiliser comme marqueurs 3D.
        """
        parsed = _parse_bvh_file(source_path)
        root_name = parsed["root_name"]
        joint_order = parsed["joint_order"]

        rest_positions = _compute_rest_positions(
            root_name=root_name,
            joint_order=joint_order,
            parents=parsed["parents"],
            offsets=parsed["offsets"],
        )

        frames_markers: List[Dict[str, np.ndarray]] = []
        root_translations: List[np.ndarray] = []
        root_rotations: List[np.ndarray] = []

        for frame_idx in range(parsed["n_frames"]):
            frame_values = parsed["motion"][frame_idx] if frame_idx < len(parsed["motion"]) else np.zeros(0)
            frame_positions, _, root_t, root_q = _build_fk_frame(
                frame_values=frame_values,
                root_name=root_name,
                joint_order=joint_order,
                channel_joint_order=parsed["channel_joint_order"],
                parents=parsed["parents"],
                offsets=parsed["offsets"],
                channels=parsed["channels"],
            )

            frame_markers = {
                joint: np.asarray(frame_positions[joint], dtype=np.float64).reshape(1, 3)
                for joint in joint_order
                if joint in frame_positions
            }
            frames_markers.append(frame_markers)
            root_translations.append(root_t)
            root_rotations.append(root_q)

        rest_markers = {
            joint: np.asarray(rest_positions[joint], dtype=np.float64).reshape(1, 3)
            for joint in joint_order
            if joint in rest_positions
        }

        return MarkerFrameSequence(
            frames_markers=frames_markers,
            rest_markers=rest_markers,
            root_rotations=root_rotations if root_rotations else None,
            root_translations=np.asarray(root_translations, dtype=np.float64) if root_translations else None,
            fps=parsed["fps"],
        )

    def load_skeleton(self, target_model_path: str, **kwargs) -> TargetSkeleton:
        """Parse le HEADER BVH pour extraire la structure du squelette cible."""
        parsed = _parse_bvh_file(target_model_path)

        rest_rot = {joint: quat_identity() for joint in parsed["joint_order"]}
        raw_names = {joint: joint for joint in parsed["joint_order"]}

        return TargetSkeleton(
            rest_rotations=rest_rot,
            parents=parsed["parents"],
            raw_names=raw_names,
            root_name=parsed["root_name"],
            fps=parsed["fps"] if parsed["fps"] > 0.0 else 30.0,
            metadata={
                "bvh_offsets": {k: v.tolist() for k, v in parsed["offsets"].items()},
                "bvh_channels": parsed["channels"],
                "bvh_joint_order": parsed["joint_order"],
                "bvh_channel_joint_order": parsed["channel_joint_order"],
                "bvh_header_lines": parsed["header_lines"],
                "bvh_frame_time": parsed["frame_time"],
            },
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
        parsed = _parse_bvh_file(target_model_path)
        root_name = skeleton.root_name or parsed["root_name"]

        channels = parsed["channels"]
        channel_joint_order = parsed["channel_joint_order"]
        header_lines = parsed["header_lines"]

        fps = skeleton.fps if skeleton.fps > 0 else parsed["fps"]
        if fps <= 0.0:
            fps = 30.0
        frame_time = 1.0 / fps

        with open(output_path, "w", encoding="utf-8") as out:
            if header_lines:
                for line in header_lines:
                    out.write(line if line.endswith("\n") else f"{line}\n")
            else:
                out.write("HIERARCHY\n")
                out.write(
                    f"ROOT {root_name}\n{{\n"
                    "  OFFSET 0.0 0.0 0.0\n"
                    "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
                    "  End Site\n  {\n    OFFSET 0 0 0\n  }\n"
                    "}\n"
                )

            out.write("MOTION\n")
            out.write(f"Frames: {len(animation_clip)}\n")
            out.write(f"Frame Time: {frame_time:.6f}\n")

            for frame in animation_clip:
                local_rotations = frame.get("rotations", {})
                root_trans = np.asarray(frame.get("root", np.zeros(3)), dtype=np.float64)
                if root_trans.shape != (3,):
                    root_trans = np.asarray(root_trans).reshape(-1)[:3]
                    if root_trans.size < 3:
                        root_trans = np.pad(root_trans, (0, 3 - root_trans.size), mode="constant")

                frame_values: List[str] = []
                for joint in channel_joint_order:
                    joint_channels = channels.get(joint, [])
                    rot_axes = [ch[0].upper() for ch in joint_channels if ch.endswith("rotation")]
                    euler_deg = _quaternion_to_channel_degrees(
                        local_rotations.get(joint, quat_identity()),
                        rot_axes,
                    )

                    for ch in joint_channels:
                        axis = ch[0].upper()
                        if ch.endswith("position"):
                            value = float(root_trans[_AXIS_INDEX[axis]]) if joint == root_name else 0.0
                        elif ch.endswith("rotation"):
                            value = float(euler_deg.get(axis, 0.0))
                        else:
                            value = 0.0
                        frame_values.append(f"{value:.6f}")

                out.write(" ".join(frame_values) + "\n")

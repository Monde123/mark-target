"""
Module adapters.bvh_adapter
===================================
Adaptateur BVH :
- Peut servir de Source (lit la hiérarchie BVH et extrait les positions 3D des joints comme marqueurs)
- Peut servir de Target (exporte l'animation résolue en fichier .bvh universel)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Any

from core.geometry import quaternion_to_matrix
from adapters.base import SourceAdapter, TargetAdapter, TargetSkeleton, MarkerFrameSequence
from adapters.bvh_parse import BvhParseError, parse_bvh, _euler_angles, _euler_matrix


def _parse_options(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Ne transmet au parseur que les déclarations de repère, pas les options d'export."""
    keys = (
        "unit",
        "up_axis",
        "handedness",
        "apply_normalization",
        "primary_child",
        "pole",
        "on_duplicate",
    )
    return {key: kwargs[key] for key in keys if key in kwargs}


class BVHAdapter(SourceAdapter, TargetAdapter):
    """Adaptateur bidirectionnel pour le format d'animation BVH."""

    def __init__(self) -> None:
        self.last_clip = None
        self.export_warnings = []

    def load(self, source_path: str, **kwargs) -> MarkerFrameSequence:
        """
        Lit un BVH et place chaque joint en marqueur (position monde, forme (1, 3)).

        Les rotations monde sont dans `joint_rotations` et dans `last_clip`.
        L'export retarget reste hors de cette étape.
        """
        clip = parse_bvh(source_path, **_parse_options(kwargs))
        self.last_clip = clip
        frames_markers: List[Dict[str, np.ndarray]] = []
        joint_rotations: List[Dict[str, np.ndarray]] = []
        for frame_index in range(clip.n_frames):
            frames_markers.append({
                name: positions[frame_index:frame_index + 1]
                for name, positions in clip.world_positions.items()
            })
            joint_rotations.append({
                name: rotations[frame_index]
                for name, rotations in clip.world_rotations.items()
            })
        rest_markers = {
            name: position.reshape(1, 3)
            for name, position in clip.rest_positions.items()
        }
        root = clip.roots[0]
        return MarkerFrameSequence(
            frames_markers=frames_markers,
            rest_markers=rest_markers,
            root_rotations=[clip.world_rotations[root][i] for i in range(clip.n_frames)],
            root_translations=np.array(clip.world_positions[root], dtype=np.float64, copy=True),
            joint_rotations=joint_rotations,
            fps=clip.fps,
        )

    def load_skeleton(self, target_model_path: str, **kwargs) -> TargetSkeleton:
        """Hiérarchie réelle et rotations de repos déduites des OFFSET."""
        clip = parse_bvh(target_model_path, **_parse_options(kwargs))
        self.last_clip = clip
        return TargetSkeleton(
            rest_rotations={name: joint.bone_rest.copy() for name, joint in clip.joints.items()},
            parents={name: joint.parent for name, joint in clip.joints.items()},
            raw_names={name: joint.raw_name for name, joint in clip.joints.items()},
            root_name=clip.roots[0],
            fps=clip.fps,
        )

    def export_animation(
        self,
        target_model_path: str,
        output_path: str,
        animation_clip: List[Dict[str, Any]],
        skeleton: TargetSkeleton,
        **kwargs
    ) -> None:
        """Écrit les canaux du BVH cible à partir des positions et des rotations monde résolues."""
        del skeleton
        self.export_warnings = []
        clip = self.last_clip
        if clip is None:
            raise BvhParseError(
                0,
                "export BVH sans squelette chargé",
                "appeler load_skeleton sur le BVH cible avant d'exporter",
            )
        if not animation_clip:
            raise BvhParseError(
                0,
                "export BVH sans frame",
                "la résolution doit produire au moins une frame",
            )
        header_lines = _hierarchy_header(target_model_path)
        ordered = sorted(clip.joints.values(), key=lambda joint: joint.channel_start)
        still = set((getattr(self, "marker_notes", None) or {}).get("still") or [])
        rows = []
        previous_twist = {}
        previous_swing = {}
        swing_excess = {}
        for frame_index, frame in enumerate(animation_clip):
            rotations = {}
            rows.append(_motion_row(clip, ordered, frame, self.export_warnings, still, rotations))
            statuses = frame.get("marker_status") or {}
            for name, values in rotations.items():
                if statuses.get(name) != "twist_ambiguous":
                    previous_twist.pop(name, None)
                    continue
                if name in previous_twist:
                    _warn_twist_jump(
                        name,
                        clip.joints[name].line,
                        previous_twist[name],
                        values,
                        frame_index,
                        self.export_warnings,
                    )
                previous_twist[name] = list(values)
            _track_swing_excess(frame, frame_index, statuses, previous_swing, swing_excess)
        for name in sorted(swing_excess):
            slot = swing_excess[name]
            self.export_warnings.append(
                f"Ligne {clip.joints[name].line} : {name} : le roulis saute plus que la source. "
                f"Pire pas, frames {slot['frame']}-{slot['frame'] + 1} : "
                f"source {slot['source']:.2f}°, écrit {slot['written']:.2f}°, "
                f"excédent {slot['excess']:.2f}°. "
                f"{slot['count_30']} pas dépassent 30°, dont {slot['count_90']} au-dessus de 90°. "
                f"Un seul enfant ne fixe pas ce roulis. Il n'est pas corrigé."
            )
        expected = clip.n_channels
        if rows and len(rows[0]) != expected:
            raise BvhParseError(
                0,
                f"la ligne de motion a {len(rows[0])} valeurs, le squelette en attend {expected}",
                "l'ordre des canaux doit suivre CHANNELS, parent avant enfants",
            )
        frame_time = kwargs.get("frame_time")
        if frame_time is None:
            frame_time = clip.normalization.frame_time
        else:
            frame_time = float(frame_time)
            if frame_time <= 0.0:
                raise BvhParseError(
                    0,
                    f"Frame Time de la source invalide ({frame_time})",
                    "le temps entre deux frames doit être celui de la source, strictement positif",
                )
        report = format_marker_report(getattr(self, "marker_notes", None), clip)
        if self.export_warnings:
            block = "\n".join(self.export_warnings) + "\n"
            report = (report + block) if report else ("Rapport marqueurs.\n" + block)
        with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
            if report:
                for line in report.splitlines():
                    handle.write(f"# {line}\n")
            handle.writelines(header_lines)
            if header_lines and not header_lines[-1].endswith("\n"):
                handle.write("\n")
            handle.write("MOTION\n")
            handle.write(f"Frames: {len(rows)}\n")
            handle.write(f"Frame Time: {frame_time:.8f}\n")
            for row in rows:
                handle.write(" ".join(f"{value:.8f}" for value in row) + "\n")
        if report:
            with open(output_path + ".report.txt", "w", encoding="utf-8", newline="\n") as handle:
                handle.write(report)


def format_marker_report(notes, clip) -> str:
    """Rapport lisible. Ce texte ne modifie pas les canaux : il décrit l'export."""
    if not notes:
        return ""
    status_of = notes.get("status") or {}
    labels = {
        "full": "Rotation fixée par plusieurs enfants",
        "twist_ambiguous": "Plusieurs roulis expliquent les positions. Le plus petit est écrit. La peau entre ces roulis n'est pas dans les points",
        "swing_only": "Direction seule. Le roulis autour de l'os n'est pas dans les positions. Ce mode a un point par joint : aucun canal ne peut le restituer",
        "non_rigid": "Les distances entre enfants changent. Une rotation ne les explique pas",
        "unobservable_inherits_parent": "Pas d'enfant observé. Rotation locale écrite : identité",
        "unobservable_root": "Racine sans enfant observé",
        "degenerate": "Enfant de longueur nulle. Rotation non mesurée",
    }
    rank = {
        "non_rigid": 5,
        "degenerate": 4,
        "swing_only": 3,
        "twist_ambiguous": 3,
        "full": 2,
        "unobservable_inherits_parent": 1,
        "unobservable_root": 1,
    }
    chosen = {
        bone: max(statuses, key=lambda item: rank.get(item, 0))
        for bone, statuses in status_of.items()
    }
    groups: Dict[str, List[str]] = {}
    for bone, status in sorted(chosen.items()):
        groups.setdefault(status, []).append(bone)
    still = set(notes.get("still") or [])
    undriven = [name for name in clip.joints if name not in chosen and name not in still]
    lines = [
        "Rapport marqueurs.",
        "Les canaux de rotation de la source ne sont pas lus.",
        f"Unité déclarée : {notes.get('unit') or 'non précisée'}.",
        f"Table de noms : {notes.get('mapping_table', notes.get('mapping_count', 0))} entrées, "
        f"{notes.get('mapping_hits', 0)} présentes dans ce fichier. "
        f"Os reliés par nom identique : {notes.get('exact_count', 0)}.",
        f"Os ignorés, absents de la cible : {_join(notes.get('ignored') or [])}.",
        f"Os au repos local (canaux à zéro, ils suivent le parent) : {_join(notes.get('still') or [])}.",
        f"Os source sans cible : {_join(notes.get('dropped') or [])}.",
        f"Os cible sans marqueur : {_join(undriven)}.",
    ]
    if not groups:
        lines.append("Aucune orientation calculée.")
    for status, bones in groups.items():
        lines.append(f"{labels.get(status, status)} ({len(bones)}) : {', '.join(bones)}")
    for bone, status in sorted(chosen.items()):
        if status != "swing_only":
            continue
        joint = clip.joints.get(bone)
        if joint is None:
            continue
        for child_name in joint.children:
            if chosen.get(child_name) != "full":
                continue
            lines.append(
                f"Le roulis absent de {bone} se retrouve dans les canaux locaux de {child_name}. "
                f"L'orientation monde de {child_name} reste celle des marqueurs."
            )
    return "\n".join(lines) + "\n"


def _join(names: List[str]) -> str:
    if not names:
        return "aucun"
    shown = ", ".join(names[:12])
    if len(names) > 12:
        return f"{shown}, … ({len(names)})"
    return shown


def _hierarchy_header(target_model_path: str) -> List[str]:
    if not str(target_model_path).lower().endswith(".bvh"):
        raise BvhParseError(
            0,
            f"cible BVH attendue, reçu « {target_model_path} »",
            "exporter vers un .bvh dont la hiérarchie sert de modèle",
        )
    header: List[str] = []
    with open(target_model_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip().upper().startswith("MOTION"):
                break
            header.append(line)
    if not any(line.strip().upper().startswith("HIERARCHY") or line.strip().upper().startswith("ROOT") for line in header):
        raise BvhParseError(
            0,
            "l'en-tête BVH cible est vide",
            "le fichier cible doit contenir HIERARCHY avant MOTION",
        )
    return header


def _sibling_angles(clip, joint, positions, parent_rotation) -> List[tuple]:
    """Angle, pour chaque autre enfant du même parent, entre l'os écrit et le marqueur."""
    parent = joint.parent
    if parent is None or parent not in positions:
        return []
    parent_pos = np.asarray(positions[parent], dtype=np.float64).reshape(3)
    found = []
    for child_name in clip.joints[parent].children:
        if child_name == joint.name or child_name not in positions:
            continue
        child = clip.joints[child_name]
        length = float(np.linalg.norm(child.offset))
        if length <= 1e-8:
            continue
        delta = np.asarray(positions[child_name], dtype=np.float64).reshape(3) - parent_pos
        local = parent_rotation.T @ delta
        axis = child.offset / length
        perp = float(np.linalg.norm(local - float(np.dot(local, axis)) * axis))
        parallel = float(np.dot(local, axis))
        found.append((float(np.degrees(np.arctan2(perp, parallel))), child_name))
    found.sort()
    return found


def _warn_locked_translation(clip, joint, local_translation, positions, parent_rotation, warnings: List[str]) -> None:
    """Sépare l'écart de longueur, le long de l'os, de l'écart qui quitte l'os."""
    prefix = f"Ligne {joint.line} : {joint.name} :"
    if any(item.startswith(prefix) for item in warnings):
        return
    residual = np.asarray(local_translation, dtype=np.float64).reshape(3) - joint.offset
    for channel in joint.channels:
        if channel.endswith("position"):
            residual[{"X": 0, "Y": 1, "Z": 2}[channel[0]]] = 0.0
    if float(np.linalg.norm(residual)) <= 1e-4:
        return
    length = float(np.linalg.norm(joint.offset))
    parts = []
    if length > 1e-8:
        axis = joint.offset / length
        along = float(np.dot(residual, axis))
        perp = float(np.linalg.norm(residual - along * axis))
        if abs(along) > 1e-4:
            parts.append(f"longueur {along:+.4g} par rapport à l'OFFSET, non écrite")
        if perp > 1e-4:
            marker = np.asarray(local_translation, dtype=np.float64).reshape(3)
            parallel = float(np.dot(marker, axis))
            theta = float(np.degrees(np.arctan2(perp, parallel)))
            siblings = _sibling_angles(clip, joint, positions, parent_rotation)
            if joint.parent is None:
                cause = "Aucune rotation ne pose le marqueur sur cet os"
            elif not siblings:
                cause = f"La rotation de {joint.parent} ne pose pas cet os sur le marqueur"
            elif theta - siblings[0][0] > 0.5:
                cause = (
                    f"La rotation de {joint.parent} vise mieux {siblings[0][1]} "
                    f"({siblings[0][0]:.2f}°) que cet os ({theta:.2f}°)"
                )
            else:
                closest_theta, closest_name = siblings[0]
                farthest_theta, farthest_name = siblings[-1]
                if farthest_theta - theta > 0.5:
                    cause = (
                        f"La rotation de {joint.parent} rate cet os de {theta:.2f}°. "
                        f"Le plus proche est {closest_name} à {closest_theta:.2f}°, "
                        f"le plus loin est {farthest_name} à {farthest_theta:.2f}°"
                    )
                else:
                    cause = (
                        f"La rotation de {joint.parent} rate cet os de {theta:.2f}°. "
                        f"Les autres enfants sont du même ordre "
                        f"(le plus proche : {closest_name}, {closest_theta:.2f}°)"
                    )
            parts.append(f"écart hors de l'os {perp:.4g}, soit {theta:.2f}°. {cause}")
    if not parts:
        locked = []
        for axis_name, index in (("X", 0), ("Y", 1), ("Z", 2)):
            amount = abs(float(residual[index]))
            if amount > 1e-4:
                locked.append(f"{amount:.4g} sur {axis_name}")
        if locked:
            parts.append(", ".join(locked) + " sans canal de position")
    if not parts:
        return
    warnings.append(prefix + " " + ". ".join(parts) + ".")


def _rotation_step_degrees(previous: np.ndarray, current: np.ndarray) -> float:
    mixed = previous.T @ current
    trace = float(mixed[0, 0] + mixed[1, 1] + mixed[2, 2])
    cosine = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _track_swing_excess(frame, frame_index, statuses, previous, excesses) -> None:
    """Compte les pas où un os à un enfant tourne plus que la source."""
    solved = frame.get("world") or {}
    source = frame.get("source_world") or {}
    for name, status in statuses.items():
        if status != "swing_only" or name not in solved or name not in source:
            previous.pop(name, None)
            continue
        solved_matrix = quaternion_to_matrix(np.asarray(solved[name], dtype=np.float64))
        source_matrix = quaternion_to_matrix(np.asarray(source[name], dtype=np.float64))
        if name in previous:
            written_step = _rotation_step_degrees(previous[name][0], solved_matrix)
            source_step = _rotation_step_degrees(previous[name][1], source_matrix)
            extra = written_step - source_step
            if extra > 30.0:
                slot = excesses.setdefault(
                    name, {"count_30": 0, "count_90": 0, "excess": -1.0}
                )
                slot["count_30"] += 1
                if extra > 90.0:
                    slot["count_90"] += 1
                if extra > slot["excess"]:
                    slot["excess"] = extra
                    slot["source"] = source_step
                    slot["written"] = written_step
                    slot["frame"] = frame_index - 1
        previous[name] = (solved_matrix, source_matrix)


def _warn_twist_jump(name, line, previous, current, frame_index, warnings: List[str]) -> None:
    """Chaque changement de branche est signalé, y compris le retour."""
    prefix = f"Ligne {line} : {name} : le roulis saute"
    for old, new in zip(previous, current):
        delta = float(((new - old + 180.0) % 360.0) - 180.0)
        if abs(delta) <= 90.0:
            continue
        warnings.append(
            f"{prefix} de {old:.2f}° à {new:.2f}° entre les frames {frame_index - 1} et {frame_index}. "
            f"Les deux posent le joint. Ce saut n'est pas un déplacement. "
            f"Le plus petit roulis a changé de branche."
        )
        return


def _observed_descendants(clip, joint, positions):
    """Os observés sous un os sans marqueur, et le vecteur d'OFFSET qui y mène."""
    found = []

    def walk(name, chain):
        node = clip.joints[name]
        here = chain + np.asarray(node.offset, dtype=np.float64)
        if name in positions:
            found.append((name, here))
            return
        for child in node.children:
            walk(child, here)

    for child in joint.children:
        walk(child, np.zeros(3, dtype=np.float64))
    return found


def _unobserved_translation(clip, joint, parent_rotation, parent_origin, positions, warnings):
    """Pose au repos. S'il y a un seul os observé en dessous, la translation libre l'y amène."""
    found = _observed_descendants(clip, joint, positions)
    if len(found) != 1:
        if len(found) > 1:
            message = (
                f"Ligne {joint.line} : {joint.name} : {len(found)} os observés en dessous, "
                "sans marqueur ici. La translation n'est pas déduite."
            )
            if message not in warnings:
                warnings.append(message)
        return np.asarray(joint.offset, dtype=np.float64).copy()
    _name, chain = found[0]
    marker = np.asarray(positions[found[0][0]], dtype=np.float64).reshape(3)
    desired = parent_rotation.T @ (marker - parent_origin) - chain
    delta = desired - np.asarray(joint.offset, dtype=np.float64)
    for index, letter in enumerate("XYZ"):
        if f"{letter}position" not in joint.channels:
            delta[index] = 0.0
    return np.asarray(joint.offset, dtype=np.float64) + delta


def _motion_row(clip, ordered, frame: Dict[str, Any], warnings: List[str], still: set, rotations_out=None) -> List[float]:
    positions = frame.get("positions")
    worlds = frame.get("world")
    if not isinstance(positions, dict) or not isinstance(worlds, dict):
        raise BvhParseError(
            0,
            "frame sans positions de marqueurs ou sans rotations monde",
            "le retargeting par marqueurs doit fournir positions et world, pas des rotations nulles",
        )
    written = {None: np.eye(3, dtype=np.float64)}
    origin = {None: np.zeros(3, dtype=np.float64)}
    values: List[float] = []
    for joint in ordered:
        parent_rotation = written[joint.parent]
        parent_origin = origin[joint.parent]
        if joint.name not in positions:
            if joint.name in still:
                local_translation = np.asarray(joint.offset, dtype=np.float64)
            else:
                local_translation = _unobserved_translation(
                    clip, joint, parent_rotation, parent_origin, positions, warnings
                )
            written[joint.name] = parent_rotation
            origin[joint.name] = parent_origin + parent_rotation @ local_translation
            for channel in joint.channels:
                if channel.endswith("position"):
                    axis = {"X": 0, "Y": 1, "Z": 2}[channel[0]]
                    values.append(float(local_translation[axis] - joint.offset[axis]))
                elif channel.endswith("rotation"):
                    values.append(0.0)
                else:
                    raise BvhParseError(
                        joint.line,
                        f"canal inconnu « {channel} » sur « {joint.name} »",
                        "canaux admis : Xposition Yposition Zposition Xrotation Yrotation Zrotation",
                    )
            continue
        observed = np.asarray(positions[joint.name], dtype=np.float64).reshape(3)
        if joint.parent is None:
            delta = observed
        elif joint.parent in positions:
            delta = observed - np.asarray(positions[joint.parent], dtype=np.float64).reshape(3)
        else:
            delta = observed - parent_origin
        local_translation = parent_rotation.T @ delta
        origin[joint.name] = parent_origin + parent_rotation @ local_translation
        _warn_locked_translation(clip, joint, local_translation, positions, parent_rotation, warnings)
        rotation_channels = [channel for channel in joint.channels if channel.endswith("rotation")]
        local_written, rotation_values = _rotation_values(
            joint.name,
            parent_rotation,
            worlds.get(joint.name),
            rotation_channels,
            warnings,
            warn_missing=_child_needs_rotation(clip, joint),
            line=joint.line,
        )
        written[joint.name] = parent_rotation @ local_written
        if rotations_out is not None and rotation_values:
            rotations_out[joint.name] = list(rotation_values)
        rotation_cursor = 0
        for channel in joint.channels:
            if channel.endswith("position"):
                axis = {"X": 0, "Y": 1, "Z": 2}[channel[0]]
                values.append(float(local_translation[axis] - joint.offset[axis]))
            elif channel.endswith("rotation"):
                values.append(rotation_values[rotation_cursor])
                rotation_cursor += 1
            else:
                raise BvhParseError(
                    joint.line,
                    f"canal inconnu « {channel} » sur « {joint.name} »",
                    "canaux admis : Xposition Yposition Zposition Xrotation Yrotation Zrotation",
                )
    return values


def _child_needs_rotation(clip, joint) -> bool:
    """Vrai si un enfant ne peut pas absorber le déplacement par ses seuls canaux de position."""
    for child_name in joint.children:
        child = clip.joints[child_name]
        axes = {channel[0] for channel in child.channels if channel.endswith("position")}
        if not {"X", "Y", "Z"} <= axes:
            return True
    return False


def _axis_advice(residual: np.ndarray, present: str) -> str:
    """Nomme l'axe à ajouter. Si la rotation n'est sur aucun axe du repère, le dit."""
    raw = np.array(
        [
            residual[2, 1] - residual[1, 2],
            residual[0, 2] - residual[2, 0],
            residual[1, 0] - residual[0, 1],
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(raw))
    if norm < 1e-8:
        return "ajouter Xrotation Yrotation Zrotation"
    axis = raw / norm
    peak = float(np.max(np.abs(axis)))
    tied = [letter for letter, value in zip("XYZ", np.abs(axis)) if value >= peak - 1e-8]
    if peak >= 0.99 and len(tied) == 1 and tied[0] not in present:
        return f"ajouter {tied[0]}rotation"
    if peak >= 0.99 and len(tied) == 1:
        return f"le résidu reste autour de {tied[0]}"
    return (
        "aucun axe seul ne suffit (" + " et ".join(tied) + " sont aussi proches). "
        "Ajouter Xrotation Yrotation Zrotation"
    )


def _rotation_values(name, parent_rotation, world_quaternion, rotation_channels, warnings, warn_missing=True, line=0):
    identity = np.eye(3, dtype=np.float64)
    where = f"Ligne {line} : " if line else ""
    if not rotation_channels:
        if warn_missing and world_quaternion is not None:
            world = quaternion_to_matrix(np.asarray(world_quaternion, dtype=np.float64))
            local = parent_rotation.T @ world
            gap = _matrix_degrees(identity, local)
            if gap > 0.05 and not any(item.startswith(f"{where}{name} : aucun canal de rotation") for item in warnings):
                warnings.append(
                    f"{where}{name} : aucun canal de rotation, écart {gap:.3f}° non écrit. "
                    f"Correction proposée : {_axis_advice(local, '')}, ou accepter que les enfants quittent les marqueurs."
                )
        return identity, []
    if world_quaternion is None:
        warnings.append(f"{name} : rotation monde absente, canaux écrits à 0")
        return identity, [0.0] * len(rotation_channels)
    world = quaternion_to_matrix(np.asarray(world_quaternion, dtype=np.float64))
    local = parent_rotation.T @ world
    order = "".join(channel[0] for channel in rotation_channels)
    if len(rotation_channels) == 3 and len(set(order)) == 3:
        degrees = np.degrees(_euler_angles(local, order))
        rebuilt = _euler_matrix(order, degrees)
        return rebuilt, [float(value) for value in degrees]
    if len(rotation_channels) == 1:
        axis = rotation_channels[0][0]
        angle = _single_axis_degrees(local, axis)
        rebuilt = _euler_matrix(axis, np.array([angle]))
        gap = _matrix_degrees(rebuilt, local)
        if gap > 0.05 and not any(item.startswith(f"{where}{name} : un seul canal") for item in warnings):
            warnings.append(
                f"{where}{name} : un seul canal {axis}rotation, écart {gap:.3f}° hors de cet axe. "
                f"Correction proposée : {_axis_advice(rebuilt.T @ local, axis)}, "
                "ou accepter que les enfants quittent les marqueurs."
            )
        return rebuilt, [angle]
    if len(rotation_channels) == 2 and all(channel[0] in "XYZ" for channel in rotation_channels):
        degrees, rebuilt, gap = _two_axis_degrees(local, order)
        if gap > 0.05 and not any(item.startswith(f"{where}{name} : deux canaux") for item in warnings):
            warnings.append(
                f"{where}{name} : deux canaux {order}, écart {gap:.3f}° hors de ces axes. "
                f"Correction proposée : {_axis_advice(rebuilt.T @ local, order)}, "
                "ou accepter que les enfants quittent les marqueurs."
            )
        return rebuilt, [float(value) for value in degrees]
    warnings.append(
        f"{name} : {len(rotation_channels)} canaux de rotation, pas une base complète, écrits à 0"
    )
    return identity, [0.0] * len(rotation_channels)


def _two_axis_degrees(matrix: np.ndarray, order: str):
    """Meilleur couple d'angles dans l'ordre des deux canaux. L'écart restant est en degrés."""
    grid = np.linspace(-180.0, 180.0, 24, endpoint=False)
    best = np.zeros(2, dtype=np.float64)
    best_gap = _matrix_degrees(np.eye(3), matrix)
    for first in grid:
        for second in grid:
            trial = np.array([first, second], dtype=np.float64)
            gap = _matrix_degrees(_euler_matrix(order, trial), matrix)
            if gap < best_gap:
                best_gap = gap
                best = trial
    step = 7.5
    angles = best.copy()
    for _ in range(24):
        improved = False
        for index in (0, 1):
            for sign in (1.0, -1.0):
                trial = angles.copy()
                trial[index] = (trial[index] + sign * step + 180.0) % 360.0 - 180.0
                gap = _matrix_degrees(_euler_matrix(order, trial), matrix)
                if gap + 1e-12 < best_gap:
                    best_gap = gap
                    angles = trial
                    improved = True
        if improved:
            continue
        step *= 0.5
        if step < 1e-4:
            break
    rebuilt = _euler_matrix(order, angles)
    return angles, rebuilt, _matrix_degrees(rebuilt, matrix)


def _single_axis_degrees(matrix: np.ndarray, axis: str) -> float:
    """Angle qui rapproche le plus la rotation de ce seul axe. Pas la lecture d'une case."""
    if axis == "X":
        cosine = matrix[1, 1] + matrix[2, 2]
        sine = matrix[2, 1] - matrix[1, 2]
    elif axis == "Y":
        cosine = matrix[0, 0] + matrix[2, 2]
        sine = matrix[0, 2] - matrix[2, 0]
    elif axis == "Z":
        cosine = matrix[0, 0] + matrix[1, 1]
        sine = matrix[1, 0] - matrix[0, 1]
    else:
        raise BvhParseError(0, f"axe inconnu « {axis} »", "utiliser X, Y ou Z")
    return float(np.degrees(np.arctan2(sine, cosine)))


def _matrix_degrees(left: np.ndarray, right: np.ndarray) -> float:
    mixed = left.T @ right
    cosine = float(np.clip((np.trace(mixed) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))

"""Orientation d'un os à partir des positions de ses enfants.

Un joint BVH n'est qu'un point. Kabsch exige trois points non colinéaires
sur le même os, donc il ne s'applique pas à ce nuage. La rotation d'un os
est celle qui amène les vecteurs repos vers ses enfants sur les vecteurs
courants. Les canaux d'Euler ne sont pas lus.

Un seul enfant sans canal de position fixe la direction. Le roulis autour
de cette direction n'est pas dans cet enfant. Un frère qui a un canal de
position peut le fixer, si un seul angle explique ses axes bloqués.
Deux enfants non colinéaires, ou davantage, fixent la rotation entière.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from core.geometry import matrix_to_quaternion, quat_identity
from core.kabsch import kabsch_rotation


_LENGTH_RATIO = 1e-3
_PLANE = 1e-4
_LOCK = 1e-4


def solve_joint_markers(
    positions_t: Dict[str, np.ndarray],
    positions_rest: Dict[str, np.ndarray],
    parents: Dict[str, Optional[str]],
    translating: Optional[Any] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Retourne les rotations monde [w, x, y, z] déduites des seuls points.

    `translating` nomme les joints qui ont un canal de position. Ce sont des
    déclarations du squelette, pas les valeurs de la frame. S'il reste au moins
    un enfant sans translation, la rotation du parent n'utilise que ceux-là.
    """
    if not positions_t:
        raise ValueError(
            "Aucune position de marqueur. Rien à orienter.\n"
            "Correction proposée : fournir un point 3D par joint, au repos et à la frame."
        )
    current = {name: _as_point(value, name) for name, value in positions_t.items()}
    missing_rest = [name for name in current if name not in positions_rest]
    if missing_rest:
        shown = ", ".join(missing_rest[:8])
        raise ValueError(
            f"Position de repos manquante pour : {shown}.\n"
            "Correction proposée : le repos est la pose aux canaux nuls, pas la frame 0."
        )
    rest = {name: _as_point(positions_rest[name], name) for name in current}
    _axes = _position_axes(translating)
    unknown = [name for name in current if name not in parents]
    if unknown:
        shown = ", ".join(unknown[:8])
        raise ValueError(
            f"Parent non déclaré pour : {shown}.\n"
            "Correction proposée : une racine a le parent None, les autres le nom de leur parent."
        )
    _require_tree(current, parents)

    children: Dict[str, list] = {name: [] for name in current}
    for name, parent in parents.items():
        if name not in current:
            continue
        if parent is not None and parent in current:
            children[parent].append(name)

    quaternions: Dict[str, np.ndarray] = {}
    diagnostics: Dict[str, Any] = {}
    solved: Dict[str, np.ndarray] = {}
    for bone in current:
        records = []
        collapsed = False
        for child in children[bone]:
            rest_vec = rest[child] - rest[bone]
            now_vec = current[child] - current[bone]
            if float(np.linalg.norm(rest_vec)) < 1e-8:
                continue
            if float(np.linalg.norm(now_vec)) < 1e-8:
                collapsed = True
                continue
            records.append((rest_vec, now_vec, _axes.get(child)))
        rigid = [item for item in records if item[2] is None]
        pool = rigid if rigid else records
        if not pool:
            diagnostics[bone] = {
                "status": "degenerate" if collapsed else "pending_unobservable",
                "markers": 0,
                "rms": 0.0,
                "length_changed": collapsed,
            }
            continue
        rest_stack = np.vstack([item[0] for item in pool])
        now_stack = np.vstack([item[1] for item in pool])
        matrix, status, rms, length_changed = _fit_rotation(rest_stack, now_stack)
        translating_records = [item for item in records if item[2] is not None]
        if rigid and translating_records:
            matrix, status, length_changed = _apply_translation_constraint(
                matrix, status, length_changed, rigid, translating_records
            )
        solved[bone] = matrix
        quaternions[bone] = matrix_to_quaternion(matrix)
        diagnostics[bone] = {
            "status": status,
            "markers": int(rest_stack.shape[0]),
            "rms": rms,
            "length_changed": length_changed,
        }

    for bone in current:
        if bone in quaternions:
            continue
        collapsed = diagnostics[bone]["status"] == "degenerate"
        _inherit(bone, parents, quaternions, diagnostics, set(), current)
        if collapsed:
            diagnostics[bone]["status"] = "degenerate"
    return quaternions, diagnostics


def _require_tree(names, parents):
    for name in names:
        seen = set()
        cursor = name
        while cursor is not None:
            if cursor in seen:
                raise ValueError(
                    f"Cycle de parents impliquant « {name} ».\n"
                    "Correction proposée : la hiérarchie doit être un arbre."
                )
            seen.add(cursor)
            cursor = parents.get(cursor)


def _inherit(bone, parents, quaternions, diagnostics, stack, known):
    if bone in quaternions:
        return quaternions[bone]
    if bone in stack:
        raise ValueError(
            f"Cycle de parents impliquant « {bone} ».\n"
            "Correction proposée : la hiérarchie doit être un arbre."
        )
    stack.add(bone)
    parent = parents.get(bone)
    if parent is None or parent not in known:
        quaternions[bone] = quat_identity()
        diagnostics[bone] = {
            "status": "unobservable_root" if parent is None else "unobservable_inherits_parent",
            "markers": 0,
            "rms": 0.0,
            "length_changed": False,
        }
    else:
        quaternions[bone] = _inherit(parent, parents, quaternions, diagnostics, stack, known).copy()
        diagnostics[bone] = {
            "status": "unobservable_inherits_parent",
            "markers": 0,
            "rms": 0.0,
            "length_changed": False,
        }
    return quaternions[bone]


def _as_point(value: np.ndarray, bone: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3,):
        point = array
    elif array.ndim == 2 and array.shape == (1, 3):
        point = array[0]
    else:
        raise ValueError(
            f"Le marqueur de « {bone} » a la forme {tuple(array.shape)}. "
            "Ce solveur attend un seul point par joint, (3,) ou (1, 3).\n"
            "Correction proposée : un nuage de surface (3 points ou plus) reste sur Kabsch."
        )
    if not np.isfinite(point).all():
        raise ValueError(
            f"Position non finie pour « {bone} » : {point.tolist()}.\n"
            "Correction proposée : retirer NaN et Inf avant la résolution."
        )
    return point


def _position_axes(translating: Optional[Any]) -> Dict[str, Set[str]]:
    """Nom → axes de position déclarés. Un ensemble de noms, sans axes, ne contraint pas le roulis."""
    if not translating:
        return {}
    if isinstance(translating, dict):
        axes: Dict[str, Set[str]] = {}
        for name, letters in translating.items():
            found = {str(letter)[:1].upper() for letter in letters}
            found.discard("")
            if found:
                axes[name] = found
        return axes
    return {name: {"X", "Y", "Z"} for name in translating}


def _apply_translation_constraint(matrix, status, length_changed, rigid, translating_records):
    """Un seul enfant rigide laisse le roulis libre. Les frères qui translatent peuvent le fixer."""
    if len(rigid) != 1:
        if _locked_error(matrix, translating_records) > _LOCK:
            return matrix, "non_rigid", True
        return matrix, status, length_changed
    kind, twisted = _solve_twist(matrix, rigid[0][0], translating_records)
    if kind == "unique" and twisted is not None:
        return twisted, "full", length_changed
    if kind == "ambiguous" and twisted is not None:
        return twisted, "twist_ambiguous", length_changed
    if kind == "impossible":
        return matrix, "non_rigid", True
    return matrix, status, length_changed


def _solve_twist(swing, rigid_rest, translating_records):
    axis = rigid_rest / np.linalg.norm(rigid_rest)
    constrained = False
    candidates = [0.0]
    for rest_vec, now_vec, free in translating_records:
        pulled = swing.T @ now_vec
        parallel = float(np.dot(pulled, axis))
        perp = pulled - parallel * axis
        crossed = np.cross(axis, pulled)
        for index, letter in enumerate("XYZ"):
            if letter in free:
                continue
            amplitude_cos = float(perp[index])
            amplitude_sin = float(-crossed[index])
            target = float(rest_vec[index] - parallel * axis[index])
            scale = float(np.hypot(amplitude_cos, amplitude_sin))
            if scale < 1e-8:
                if abs(target) > _LOCK:
                    return "impossible", None
                continue
            if abs(target) > scale + 1e-6:
                return "impossible", None
            constrained = True
            phase = float(np.arctan2(amplitude_sin, amplitude_cos))
            delta = float(np.arccos(np.clip(target / scale, -1.0, 1.0)))
            candidates.append(phase + delta)
            candidates.append(phase - delta)
    if not constrained:
        return "free", None
    good = []
    for theta in candidates:
        wrapped = float((theta + np.pi) % (2.0 * np.pi) - np.pi)
        if _locked_error(swing @ _rodrigues(axis, wrapped), translating_records) <= _LOCK:
            good.append(wrapped)
    if not good:
        return "impossible", None
    clusters: list = []
    for theta in good:
        if any(abs(float(np.degrees(_wrap(theta - other)))) < 0.05 for other in clusters):
            continue
        clusters.append(theta)
    chosen = min(clusters, key=lambda theta: abs(theta))
    matrix = swing @ _rodrigues(axis, chosen)
    if len(clusters) == 1:
        return "unique", matrix
    return "ambiguous", matrix


def _wrap(theta: float) -> float:
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


def _locked_error(matrix, translating_records) -> float:
    worst = 0.0
    for rest_vec, now_vec, free in translating_records:
        delta = matrix.T @ now_vec - rest_vec
        for index, letter in enumerate("XYZ"):
            if letter in free:
                continue
            worst = max(worst, abs(float(delta[index])))
    return worst


def _fit_rotation(
    rest_offsets: np.ndarray,
    current_offsets: np.ndarray,
) -> Tuple[np.ndarray, str, float, bool]:
    rest_lengths = np.linalg.norm(rest_offsets, axis=1)
    current_lengths = np.linalg.norm(current_offsets, axis=1)
    length_ratio = float(np.max(np.abs(current_lengths - rest_lengths) / np.maximum(rest_lengths, 1e-8)))
    length_changed = length_ratio > _LENGTH_RATIO
    count = int(rest_offsets.shape[0])

    if count == 1:
        matrix = _swing(rest_offsets[0], current_offsets[0])
        residual = current_offsets - (matrix @ rest_offsets.T).T
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        return matrix, "swing_only", rms, length_changed

    if count == 2:
        matrix = _two_vectors(rest_offsets, current_offsets)
    else:
        matrix = _kabsch_about_joint(rest_offsets, current_offsets)
    if matrix is None:
        longest = int(np.argmax(rest_lengths))
        matrix = _swing(rest_offsets[longest], current_offsets[longest])
        residual = current_offsets - (matrix @ rest_offsets.T).T
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        scale = float(np.mean(rest_lengths))
        if count >= 2 and scale > 0.0 and rms / scale > 0.02:
            return matrix, "non_rigid", rms, True
        return matrix, "swing_only", rms, length_changed

    residual = current_offsets - (matrix @ rest_offsets.T).T
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    scale = float(np.mean(rest_lengths))
    if length_changed or (scale > 0.0 and rms / scale > 0.02):
        return matrix, "non_rigid", rms, True
    return matrix, "full", rms, False


def _swing(rest_vec: np.ndarray, current_vec: np.ndarray) -> np.ndarray:
    """Rotation minimale qui aligne rest_vec sur current_vec. Le roulis reste nul."""
    source = rest_vec / np.linalg.norm(rest_vec)
    target_norm = float(np.linalg.norm(current_vec))
    if target_norm < 1e-8:
        return np.eye(3)
    target = current_vec / target_norm
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-12 and cosine > 0.0:
        return np.eye(3)
    if sine < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, axis)
        axis = axis / np.linalg.norm(axis)
        return _rodrigues(axis, np.pi)
    axis = cross / sine
    angle = float(np.arctan2(sine, cosine))
    return _rodrigues(axis, angle)


def _two_vectors(rest_offsets: np.ndarray, current_offsets: np.ndarray) -> Optional[np.ndarray]:
    rest_frame = _basis(rest_offsets[0], rest_offsets[1])
    current_frame = _basis(current_offsets[0], current_offsets[1])
    if rest_frame is None or current_frame is None:
        return None
    return current_frame @ rest_frame.T


def _basis(primary: np.ndarray, secondary: np.ndarray) -> Optional[np.ndarray]:
    primary_norm = float(np.linalg.norm(primary))
    secondary_norm = float(np.linalg.norm(secondary))
    if primary_norm < 1e-8 or secondary_norm < 1e-8:
        return None
    normal = np.cross(primary, secondary)
    sine = float(np.linalg.norm(normal)) / (primary_norm * secondary_norm)
    if sine < _PLANE:
        return None
    x_axis = primary / primary_norm
    z_axis = normal / np.linalg.norm(normal)
    y_axis = np.cross(z_axis, x_axis)
    frame = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(frame) < 0.0:
        return None
    return frame


def _kabsch_about_joint(rest_offsets: np.ndarray, current_offsets: np.ndarray) -> Optional[np.ndarray]:
    if _plane_span(rest_offsets) < _PLANE or _plane_span(current_offsets) < _PLANE:
        return None
    try:
        matrix, _, _, _, _ = kabsch_rotation(
            rest_offsets,
            current_offsets,
            min_rank_ratio=0.0,
        )
    except ValueError:
        return None
    if np.linalg.det(matrix) < 0.0:
        return None
    return matrix


def _plane_span(offsets: np.ndarray) -> float:
    """s1/s0 des vecteurs depuis le joint. Un plan (main) vaut ; une ligne ne vaut pas."""
    singular = np.linalg.svd(offsets, compute_uv=False)
    if singular.shape[0] < 2 or float(singular[0]) < 1e-12:
        return 0.0
    return float(singular[1] / singular[0])


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    x_axis, y_axis, z_axis = axis
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    skew = np.array(
        [
            [0.0, -z_axis, y_axis],
            [z_axis, 0.0, -x_axis],
            [-y_axis, x_axis, 0.0],
        ],
        dtype=np.float64,
    )
    outer = np.outer(axis, axis)
    return cosine * np.eye(3) + sine * skew + (1.0 - cosine) * outer

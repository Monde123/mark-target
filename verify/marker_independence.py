"""
Marqueurs joints, sans les canaux d'Euler.

Le solveur ne reçoit que des positions. La référence est la cinématique
de phase_a_oracle (Rx Ry Rz écrits là-bas), pas la cinématique du parseur.
Un os à plusieurs enfants non colinéaires doit retrouver cette rotation.
Un os à un seul enfant ne retrouve que la direction : le roulis absent
des positions est mesuré, il ne fait pas échouer le test.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from adapters.bvh_adapter import BVHAdapter
from adapters.bvh_parse import format_report, parse_bvh
from core.geometry import quaternion_to_matrix
from retargeting.joint_markers import solve_joint_markers
from retargeting.solver import MarkerRetargetSolver

OUT = os.path.join(ROOT, "verify", "out")
PRECISE = os.path.join(ROOT, "a_out_precise.bvh")
CORPUS = [
    (r"D:\dev\Mimo\out\a_pose_hands.bvh", "fourni"),
    (r"D:\dev\Mimo\out\bon_pose_hands.bvh", "fourni"),
    (r"D:\dev\Mimo\out\jumping_jacks.bvh", "fourni"),
    (r"D:\dev\Mimo\out\signes.bvh", "cherché dans Mimo/out"),
    (r"D:\dev\Mimo\out_prefix_body.bvh", "cherché, canaux de position seuls"),
]
CM_TO_MM = 10.0
POS_MM = 0.05
FULL_DEG = 0.05


def _load_oracle():
    path = os.path.join(ROOT, "verify", "phase_a_oracle.py")
    spec = importlib.util.spec_from_file_location("phase_a_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _angle(left: np.ndarray, right: np.ndarray) -> float:
    mixed = left.T @ right
    cosine = float(np.clip((np.trace(mixed) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _ry(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rz(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rx(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _parents_of(names):
    parents = {"Root": None}
    for name in names:
        parents[name] = "Root"
    return parents


def _literals(failures):
    lines = ["Littéraux, calculés à la main (Ry, Rz), pas par le parseur."]
    rest = {"Root": np.zeros(3), "A": np.array([1.0, 0.0, 0.0]), "B": np.array([0.0, 1.0, 0.0]), "C": np.array([0.0, 0.0, 1.0])}
    rotation = _ry(90.0)
    current = {name: rotation @ rest[name] for name in rest}
    quats, info = solve_joint_markers(current, rest, _parents_of(("A", "B", "C")))
    got = quaternion_to_matrix(quats["Root"])
    gap = _angle(got, rotation)
    lines.append(f"  3 enfants, Ry(90) : statut {info['Root']['status']}, écart {gap:.3e}°")
    if info["Root"]["status"] != "full" or gap > 1e-6 or abs(np.linalg.det(got) - 1.0) > 1e-8:
        failures.append("littéral Ry(90)")

    rest_one = {"Root": np.zeros(3), "Child": np.array([1.0, 0.0, 0.0])}
    parents = {"Root": None, "Child": "Root"}
    current_one = {"Root": np.zeros(3), "Child": np.array([0.0, 1.0, 0.0])}
    quats, info = solve_joint_markers(current_one, rest_one, parents)
    gap = _angle(quaternion_to_matrix(quats["Root"]), _rz(90.0))
    lines.append(f"  1 enfant +X → +Y : statut {info['Root']['status']}, écart à Rz(90) {gap:.3e}°")
    if info["Root"]["status"] != "swing_only" or gap > 1e-6:
        failures.append("littéral Rz(90)")

    flat = {
        "Root": np.zeros(3),
        "A": np.array([1.0, 0.0, 0.0]),
        "B": np.array([0.2, 1.0, 0.0]),
        "C": np.array([-0.4, 0.3, 0.0]),
        "D": np.array([0.1, -0.8, 0.0]),
    }
    rotation = _rz(25.0)
    moved = {name: rotation @ flat[name] for name in flat}
    quats, info = solve_joint_markers(moved, flat, _parents_of(("A", "B", "C", "D")))
    gap = _angle(quaternion_to_matrix(quats["Root"]), rotation)
    lines.append(f"  4 enfants coplanaires, Rz(25) : statut {info['Root']['status']}, écart {gap:.3e}°")
    if info["Root"]["status"] != "full" or gap > 1e-6:
        failures.append("plan de la main rejeté")

    twisted = _rz(90.0) @ _rx(40.0)
    current_twist = {"Root": np.zeros(3), "Child": twisted @ np.array([1.0, 0.0, 0.0])}
    quats, info = solve_joint_markers(current_twist, rest_one, parents)
    hidden = _angle(quaternion_to_matrix(quats["Root"]), twisted)
    lines.append(
        f"  même enfant, Rz(90)@Rx(40) : le point ne voit pas Rx(40). "
        f"Écart au roulis {hidden:.3f}° (mesuré, pas un échec). Statut {info['Root']['status']}."
    )
    if hidden < 1.0:
        failures.append("le roulis invisible aurait dû rester visible dans la mesure")
    return lines


def _hostile(failures):
    lines = ["Entrées hostiles."]
    parents = {"Root": None, "Child": "Root"}
    rest = {"Root": np.zeros(3), "Child": np.array([1.0, 0.0, 0.0])}
    cases = [
        ("vide", lambda: solve_joint_markers({}, {}, {})),
        ("NaN", lambda: solve_joint_markers({"Root": np.array([np.nan, 0.0, 0.0])}, {"Root": np.zeros(3)}, {"Root": None})),
        ("nuage 4x3", lambda: solve_joint_markers({"Root": np.zeros((4, 3))}, {"Root": np.zeros(3)}, {"Root": None})),
        ("cycle", lambda: solve_joint_markers(
            {"A": np.zeros(3), "B": np.array([1.0, 0.0, 0.0])},
            {"A": np.zeros(3), "B": np.array([1.0, 0.0, 0.0])},
            {"A": "B", "B": "A"},
        )),
    ]
    for label, call in cases:
        try:
            call()
        except ValueError as error:
            first = str(error).splitlines()[0]
            lines.append(f"  {label} : refusé — {first}")
        else:
            failures.append(f"hostile {label} accepté")
            lines.append(f"  {label} : ACCEPTÉ, c'est un échec")

    current = {"Root": np.zeros(3), "Child": np.zeros(3)}
    _, info = solve_joint_markers(current, rest, parents)
    lines.append(f"  enfant de longueur nulle : {info['Root']['status']}")
    if info["Root"]["status"] != "degenerate":
        failures.append("longueur nulle")

    reflected_rest = {
        "Root": np.zeros(3),
        "A": np.array([1.0, 0.0, 0.0]),
        "B": np.array([0.0, 1.0, 0.0]),
        "C": np.array([0.0, 0.0, 1.0]),
    }
    reflected = {
        "Root": np.zeros(3),
        "A": np.array([1.0, 0.0, 0.0]),
        "B": np.array([0.0, 1.0, 0.0]),
        "C": np.array([0.0, 0.0, -1.0]),
    }
    quats, info = solve_joint_markers(reflected, reflected_rest, _parents_of(("A", "B", "C")))
    det = float(np.linalg.det(quaternion_to_matrix(quats["Root"])))
    lines.append(f"  retournement de C : statut {info['Root']['status']}, det {det:.3f}")
    if info["Root"]["status"] == "full" or det < 0.0:
        failures.append("réflexion traitée comme une rotation")

    doubled = {"Root": np.zeros(3), "Child": np.array([2.0, 0.0, 0.0])}
    _, info = solve_joint_markers(doubled, rest, parents)
    lines.append(
        f"  longueur doublée : statut {info['Root']['status']}, "
        f"length_changed={info['Root']['length_changed']}"
    )
    if not info["Root"]["length_changed"]:
        failures.append("longueur doublée silencieuse")
    return lines


def _walk(roots):
    ordered = []

    def visit(node):
        ordered.append(node)
        for child in node.children:
            visit(child)

    for root in roots:
        visit(root)
    return ordered


def _rotation_channels(node):
    return [channel for channel in node.channels if channel.lower().endswith("rotation")]


def _evaluate_file(path, origin, oracle, failures):
    lines = [f"Fichier ({origin}) : {path}"]
    if not os.path.isfile(path):
        failures.append(f"absent {path}")
        lines.append("  ABSENT")
        return lines
    clip = parse_bvh(path, unit="cm")
    report = format_report(clip).splitlines()
    lines.extend(f"  {line}" for line in report[:16])
    unit_line = next((line for line in report if line.strip().startswith("unité")), "")
    if "cm" not in unit_line:
        failures.append(f"unité non cm {path}")

    text = open(path, "r", encoding="utf-8", errors="replace").read()
    roots, motion, _frame_time = oracle.naive_load(text)
    naive_pos, naive_rot = oracle.independent_fk(roots, motion)
    ordered = _walk(roots)
    names = [node.name for node in ordered]
    if len(names) != len(set(names)):
        lines.append("  Noms en double : le lecteur naïf ne désambiguïse pas. Marqueurs non lancés.")
        failures.append(f"doublons {os.path.basename(path)}")
        return lines

    pos_gap = 0.0
    rot_gap = 0.0
    for name in names:
        if name not in clip.world_positions:
            failures.append(f"nom {name} absent du parseur")
            continue
        pos_gap = max(pos_gap, float(np.max(np.abs(clip.world_positions[name] - naive_pos[name]))))
        for frame in range(motion.shape[0]):
            parsed = quaternion_to_matrix(clip.world_rotations[name][frame])
            rot_gap = max(rot_gap, _angle(parsed, naive_rot[name][frame]))
    lines.append(
        f"  Parseur contre lecteur naïf : {pos_gap * CM_TO_MM:.6f} mm, {rot_gap:.6e}°"
    )
    if pos_gap * CM_TO_MM > POS_MM or rot_gap > FULL_DEG:
        failures.append(f"parseur ≠ naïf {os.path.basename(path)}")

    rest_pos, _rest_rot = oracle.independent_fk(roots, np.zeros_like(motion))
    parents = {node.name: None if node.parent is None else node.parent.name for node in ordered}
    full_channels = {
        node.name
        for node in ordered
        if len(_rotation_channels(node)) == 3 and len({channel[0].upper() for channel in _rotation_channels(node)}) == 3
    }
    pose_deg = 0.0
    pose_bone = names[0] if names else ""
    full_gap = 0.0
    full_bone = ""
    full_count = 0
    swing_twist = 0.0
    swing_bone = ""
    counts = {}
    n_frames = motion.shape[0]
    solved_world = []
    for frame in range(n_frames):
        current = {name: naive_pos[name][frame] for name in names}
        rest = {name: rest_pos[name][0] for name in names}
        translating = {
            node.name: {channel[0].upper() for channel in node.channels if channel.endswith("position")}
            for node in ordered
            if any(channel.endswith("position") for channel in node.channels)
        }
        quats, info = solve_joint_markers(current, rest, parents, translating)
        solved_world.append(quats)
        if frame == 0:
            for info_bone in info.values():
                counts[info_bone["status"]] = counts.get(info_bone["status"], 0) + 1
        for name in names:
            oracle_r = naive_rot[name][frame]
            away = _angle(oracle_r, np.eye(3))
            if away > pose_deg:
                pose_deg = away
                pose_bone = name
            status = info[name]["status"]
            if status != "full" or info[name]["length_changed"] or name not in full_channels:
                if status == "swing_only" and name in full_channels and not info[name]["length_changed"]:
                    twist = _angle(quaternion_to_matrix(quats[name]), oracle_r)
                    if twist > swing_twist:
                        swing_twist = twist
                        swing_bone = name
                continue
            full_count += 1
            gap = _angle(quaternion_to_matrix(quats[name]), oracle_r)
            if gap > full_gap:
                full_gap = gap
                full_bone = f"{name} frame {frame}"
    lines.append(f"  Statuts frame 0 : {counts}")
    lines.append(f"  Pose max hors repos (lecteur naïf) : {pose_deg:.3f}° sur {pose_bone}")
    lines.append(
        f"  Rotation retrouvée, os à nuage rigide et 3 canaux : "
        f"max {full_gap:.6e}° ({full_bone or 'aucun'}), comparaisons {full_count}"
    )
    lines.append(
        f"  Roulis non contenu dans un seul enfant : max {swing_twist:.3f}° sur {swing_bone or 'aucun'}"
    )
    if full_count == 0:
        lines.append("  Aucun os à nuage rigide : la rotation des canaux n'est pas testée sur ce fichier.")
    elif full_gap > FULL_DEG:
        failures.append(f"rotation marqueurs {os.path.basename(path)}")

    adapter = BVHAdapter()
    adapter.load_skeleton(path, unit="cm")
    animation = []
    for frame in range(n_frames):
        animation.append({
            "positions": {name: naive_pos[name][frame] for name in names},
            "world": solved_world[frame],
            "root": naive_pos[ordered[0].name][frame],
        })
    destination = os.path.join(OUT, f"markers_{os.path.basename(path)}")
    adapter.export_animation(path, destination, animation, adapter.load_skeleton(path, unit="cm"))
    written = open(destination, "r", encoding="utf-8", errors="replace").read()
    out_roots, out_motion, _out_time = oracle.naive_load(written)
    out_pos, _out_rot = oracle.independent_fk(out_roots, out_motion)
    written_gap = 0.0
    written_bone = names[0] if names else ""
    for name in names:
        if name not in out_pos:
            failures.append(f"export sans {name}")
            continue
        gap = float(np.max(np.abs(out_pos[name] - naive_pos[name])))
        if gap > written_gap:
            written_gap = gap
            written_bone = name
    lines.append(
        f"  BVH réécrit, relu par le lecteur naïf : {written_gap * CM_TO_MM:.6f} mm sur {written_bone}"
    )
    lines.append(f"  Avertissements d'export : {len(adapter.export_warnings)}")
    for warning in adapter.export_warnings[:6]:
        lines.append(f"    - {warning}")
    if written_gap * CM_TO_MM > POS_MM:
        failures.append(f"positions réécrites {os.path.basename(path)}")
    return lines


def _old_path(path, oracle, failures):
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    roots, motion, _frame_time = oracle.naive_load(text)
    ordered = _walk(roots)
    names = [node.name for node in ordered]
    parents = {node.name: None if node.parent is None else node.parent.name for node in ordered}
    solver = MarkerRetargetSolver(
        rest_rotations={name: np.array([1.0, 0.0, 0.0, 0.0]) for name in names},
        parents=parents,
    )
    frame = min(motion.shape[0] - 1, max(motion.shape[0] // 2, 0))
    positions, rotations = oracle.independent_fk(roots, motion)
    cloud = {name: positions[name][frame].reshape(1, 3) for name in names}
    rest = {name: positions[name][0].reshape(1, 3) for name in names}
    _quats, info = solver.solve_frame_multi_kabsch(cloud, rest)
    statuses = {}
    for item in info.values():
        statuses[item["status"]] = statuses.get(item["status"], 0) + 1
    pose = 0.0
    for name in names:
        pose = max(pose, _angle(rotations[name][frame], np.eye(3)))
    line = (
        f"Ancien Kabsch, 1 point par os, {os.path.basename(path)} frame {frame} : {statuses}. "
        f"La pose réelle s'écarte de {pose:.3f}° de l'identité."
    )
    if statuses.get("fallback_insufficient_points", 0) != len(names):
        failures.append("ancien Kabsch n'a pas refusé le point unique")
    if pose < 1.0:
        line += " Cette frame est presque au repos : l'écart ancien/nouveau y est faible."
    return line


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(OUT, exist_ok=True)
    failures = []
    oracle = _load_oracle()
    blocks = [
        "Marqueurs joints. Unité déclarée : cm. Conversion non appliquée.",
        "Le solveur ne lit pas les canaux. La référence est le lecteur naïf.",
        "",
    ]
    try:
        blocks.extend(_literals(failures))
        blocks.append("")
        blocks.extend(_hostile(failures))
        blocks.append("")
        blocks.append(_old_path(CORPUS[0][0], oracle, failures))
        blocks.append("")
        for path, origin in CORPUS:
            blocks.extend(_evaluate_file(path, origin, oracle, failures))
            blocks.append("")
        blocks.extend(_evaluate_file(PRECISE, "a_out_precise, unité confirmée cm", oracle, failures))
    except Exception:
        failures.append("exception")
        blocks.append(traceback.format_exc())
    blocks.append(f"Échecs : {len(failures)}")
    for item in failures:
        blocks.append(f"  - {item}")
    text = "\n".join(blocks) + "\n"
    destination = os.path.join(OUT, "marker_independence.txt")
    with open(destination, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()

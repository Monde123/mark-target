"""
Oracle de la partie A.

La cinématique de ce fichier est écrite ici, avec des matrices Rx Ry Rz
explicites. Elle n'appelle pas la cinématique du parseur. Les positions
attendues des cas algébriques sont des littéraux, pas des valeurs
renvoyées par le parseur.

Ce script ne remplace pas le corpus réel de 5 BVH : il fabrique des
fichiers irréguliers pour se contrôler, et il lit a_out_precise.bvh s'il
est présent.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from adapters.bvh_adapter import BVHAdapter
from adapters.bvh_parse import (
    BvhParseError,
    check_euler_roundtrip,
    format_report,
    parse_bvh,
    parse_bvh_text,
    write_bvh,
)

OUT = os.path.join(ROOT, "verify", "out")
REAL = os.path.join(ROOT, "a_out_precise.bvh")


class Node:
    def __init__(self, name, offset, channels, children=None, end=None):
        self.name = name
        self.offset = np.array(offset, dtype=np.float64)
        self.channels = list(channels)
        self.children = list(children or [])
        self.end = None if end is None else np.array(end, dtype=np.float64)
        self.parent = None
        for child in self.children:
            child.parent = self


def _rx(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _ry(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rz(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


_AXIS = {"X": _rx, "Y": _ry, "Z": _rz}


def _walk(roots):
    ordered = []

    def visit(node):
        ordered.append(node)
        for child in node.children:
            visit(child)

    for root in roots:
        visit(root)
    return ordered


def independent_fk(roots, motion):
    """Positions et matrices monde. `motion` est (frames, canaux) dans l'ordre DFS."""
    ordered = _walk(roots)
    cursor = 0
    spans = []
    for node in ordered:
        spans.append((node, cursor))
        cursor += len(node.channels)
    if motion.shape[1] != cursor:
        raise RuntimeError(f"harnais : {motion.shape[1]} colonnes, {cursor} canaux")
    n_frames = motion.shape[0]
    positions = {node.name: np.zeros((n_frames, 3)) for node in ordered}
    rotations = {node.name: np.zeros((n_frames, 3, 3)) for node in ordered}
    for frame in range(n_frames):
        row = motion[frame]
        world = {}
        for node, start in spans:
            local_r = np.eye(3)
            local_t = node.offset.copy()
            for index, channel in enumerate(node.channels):
                value = float(row[start + index])
                if channel.endswith("position"):
                    local_t["XYZ".index(channel[0])] += value
                elif channel.endswith("rotation"):
                    local_r = _AXIS[channel[0]](value) @ local_r
                else:
                    raise RuntimeError(channel)
            if node.parent is None:
                world_t = local_t
                world_r = local_r
            else:
                parent_t, parent_r = world[node.parent.name]
                world_t = parent_t + parent_r @ local_t
                world_r = parent_r @ local_r
            world[node.name] = (world_t, world_r)
            positions[node.name][frame] = world_t
            rotations[node.name][frame] = world_r
    return positions, rotations


def _quat_to_matrix(q):
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_degrees(a, b):
    mixed = a.T @ b
    cosine = (np.trace(mixed) - 1.0) / 2.0
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _max_errors(clip, positions, rotations):
    pos_err = 0.0
    rot_err = 0.0
    for name, expected in positions.items():
        got = clip.world_positions[name]
        pos_err = max(pos_err, float(np.max(np.abs(got - expected))))
        for frame in range(expected.shape[0]):
            got_m = _quat_to_matrix(clip.world_rotations[name][frame])
            rot_err = max(rot_err, _rotation_degrees(rotations[name][frame], got_m))
    return pos_err, rot_err


def _fmt_vec(values):
    return " ".join(f"{float(v):.8f}" for v in values)


def emit_bvh(roots, motion, frame_time, style="canonical"):
    lines = []
    if style != "no_header":
        lines.append("HIERARCHY")
    if style == "comment":
        lines.append("# commentaire à ignorer")
    written = []

    def rec(node, depth, is_root):
        kind = "ROOT" if is_root else "JOINT"
        if style == "lower":
            kind = kind.lower()
            channels = [ch[0].lower() + ch[1:] for ch in node.channels]
        else:
            channels = list(node.channels)
        written.extend(channels)
        channel_text = " ".join(channels)
        if style == "jammed":
            lines.append(
                f"{kind} {node.name}{{OFFSET {_fmt_vec(node.offset)} "
                f"CHANNELS {len(channels)} {channel_text}".rstrip()
            )
            for child in node.children:
                rec(child, depth + 1, False)
            if node.end is not None:
                lines.append(f"End Site{{OFFSET {_fmt_vec(node.end)}}}")
            lines.append("}")
            return
        pad = "\t" * depth
        lines.append(f"{pad}{kind} {node.name}")
        lines.append(f"{pad}{{")
        offset_word = "offset" if style == "lower" else "OFFSET"
        channel_word = "channels" if style == "lower" else "CHANNELS"
        lines.append(f"{pad}\t{offset_word} {_fmt_vec(node.offset)}")
        lines.append(f"{pad}\t{channel_word} {len(channels)} {channel_text}".rstrip())
        for child in node.children:
            rec(child, depth + 1, False)
        if node.end is not None:
            lines.append(f"{pad}\tEnd Site")
            lines.append(f"{pad}\t{{")
            lines.append(f"{pad}\t\tOFFSET {_fmt_vec(node.end)}")
            lines.append(f"{pad}\t}}")
        lines.append(f"{pad}}}")

    for root in roots:
        rec(root, 0, True)
    expected = []
    for node in _walk(roots):
        expected.extend(node.channels if style != "lower" else [c[0].lower() + c[1:] for c in node.channels])
    if [item.lower() for item in written] != [item.lower() for item in expected]:
        raise RuntimeError("harnais : l'ordre d'écriture des canaux ne suit pas le DFS")
    lines.append("MOTION")
    if style == "swap_header":
        lines.append(f"Frame Time: {frame_time:.8f}")
        lines.append(f"Frames: {motion.shape[0]}")
    else:
        lines.append(f"Frames: {motion.shape[0]}")
        lines.append(f"Frame Time: {frame_time:.8f}")
    for row in motion:
        lines.append(" ".join(f"{float(v):.8f}" for v in row))
    return "\n".join(lines) + "\n"


def _split_naive_clauses(line):
    """Une accolade collée au nom, « Arm{ », ouvre le joint. OFFSET ne coupe le nom que devant trois nombres."""
    spaced = []
    for char in line:
        if char in "{}":
            spaced.append(" ")
            spaced.append(char)
            spaced.append(" ")
        else:
            spaced.append(char)
    tokens = "".join(spaced).split()
    groups = []
    current = []

    def flush():
        if current:
            groups.append(" ".join(current))
            current.clear()

    def number(token):
        try:
            float(token)
        except ValueError:
            return False
        return True

    for index, token in enumerate(tokens):
        upper = token.upper().rstrip(":")
        starts = token in "{}" or upper in {"JOINT", "ROOT", "END", "ENDSITE", "MOTION", "HIERARCHY"}
        if upper == "OFFSET":
            nxt = tokens[index + 1 : index + 4]
            starts = len(nxt) == 3 and all(number(item) for item in nxt)
        elif upper == "CHANNELS":
            nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
            starts = bool(nxt) and nxt.lstrip("+-").isdigit()
        if starts:
            flush()
            if token in "{}":
                groups.append(token)
                continue
        current.append(token)
    flush()
    return groups


def naive_load(text):
    """Lecteur canonique, volontairement ligne à ligne, distinct du parseur."""
    raw_lines = []
    for line in text.splitlines():
        raw = line.split("#", 1)[0].strip()
        if not raw:
            continue
        raw_lines.extend(_split_naive_clauses(raw))
    cursor = {"i": 0, "lines": raw_lines}

    def current():
        i = cursor["i"]
        if i >= len(cursor["lines"]):
            raise RuntimeError("lecteur naïf : fin de fichier")
        return cursor["lines"][i]

    def take():
        line = current()
        cursor["i"] += 1
        return line

    if current().upper() == "HIERARCHY":
        take()
    roots = []

    def parse_joint(is_root):
        line = take()
        parts = line.split()
        head = parts[0].upper()
        if is_root and head != "ROOT":
            raise RuntimeError(parts[0])
        if not is_root and head != "JOINT":
            raise RuntimeError(parts[0])
        if parts[-1] == "{":
            name = " ".join(parts[1:-1])
            opened = True
        else:
            name = " ".join(parts[1:])
            opened = False
        if not name:
            raise RuntimeError("nom manquant")
        if not opened and take() != "{":
            raise RuntimeError("accolade")
        offset = None
        channels = []
        children = []
        end = None
        while current() != "}":
            line = current()
            upper = line.upper()
            if upper.startswith("OFFSET"):
                nums = line.split()[1:]
                offset = np.array([float(v) for v in nums], dtype=np.float64)
                take()
            elif upper.startswith("CHANNELS"):
                bits = line.split()
                channels = bits[2:]
                take()
            elif upper.startswith("JOINT"):
                children.append(parse_joint(False))
            elif upper.startswith("END"):
                take()
                if take() != "{":
                    raise RuntimeError("end")
                end_line = take()
                nums = end_line.split()[1:]
                end = np.array([float(v) for v in nums], dtype=np.float64)
                if take() != "}":
                    raise RuntimeError("fin end site")
            else:
                raise RuntimeError(f"ligne naïve inattendue : {line}")
        take()
        node = Node(name, offset, channels, children, end)
        for child in node.children:
            child.parent = node
        return node

    while cursor["i"] < len(raw_lines) and current().split()[0].upper() == "ROOT":
        roots.append(parse_joint(True))
    if take().upper() != "MOTION":
        raise RuntimeError("MOTION")
    frames = None
    frame_time = None
    while frames is None or frame_time is None:
        line = take()
        upper = line.upper()
        if upper.startswith("FRAMES"):
            frames = int(line.split(":")[-1].strip())
        elif upper.startswith("FRAME TIME"):
            frame_time = float(line.split(":")[-1].strip())
        else:
            raise RuntimeError(line)
    rows = []
    for _ in range(frames):
        if cursor["i"] >= len(raw_lines):
            raise RuntimeError("motion tronqué")
        rows.append([float(v) for v in take().split()])
    motion = np.zeros((frames, 0)) if not rows else np.asarray(rows, dtype=np.float64)
    if frames == 0:
        motion = np.zeros((0, 0))
    return roots, motion, frame_time


def _compare_clip(clip, positions, rotations, label, failures, unit_to_mm):
    pos_err, rot_err = _max_errors(clip, positions, rotations)
    pos_mm = pos_err * unit_to_mm
    ok = pos_err < 1e-5 and rot_err < 1e-4
    if not ok:
        failures.append(label)
    return (
        f"{label}\n"
        f"  écart position max : {pos_err:.3e} unités fichier  ({pos_mm:.6f} mm)\n"
        f"  écart rotation max : {rot_err:.3e} degrés\n"
        f"  seuil : 1e-5 unité et 1e-4 degré → {'OK' if ok else 'ÉCHEC'}"
    )


def _algebra_child():
    return Node("Child", (0, 1, 0), ["Xrotation", "Yrotation", "Zrotation"], end=(0, 1, 0))


def _run_literals(failures):
    blocks = []
    specs = [
        (
            "XYZ 90/90/0",
            ["Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation"],
            [0, 0, 0, 90, 90, 0, 0, 0, 0],
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        ),
        (
            "YXZ 90/90/0",
            ["Xposition", "Yposition", "Zposition", "Yrotation", "Xrotation", "Zrotation"],
            [0, 0, 0, 90, 90, 0, 0, 0, 0],
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 2.0]),
        ),
        (
            "ZXY 90/0/0",
            ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"],
            [0, 0, 0, 90, 0, 0, 0, 0, 0],
            np.array([-1.0, 0.0, 0.0]),
            np.array([-2.0, 0.0, 0.0]),
        ),
    ]
    for title, root_channels, row, child_at, end_at in specs:
        child = _algebra_child()
        root = Node("Hips", (0, 0, 0), root_channels, [child])
        text = emit_bvh([root], np.array([row], dtype=np.float64), 0.04)
        clip = parse_bvh_text(text, source_name=title)
        got = clip.world_positions["Child"][0]
        end = clip.end_positions["Child"][0]
        pos = float(np.max(np.abs(got - child_at)))
        end_err = float(np.max(np.abs(end - end_at)))
        ok = pos < 1e-8 and end_err < 1e-8
        if not ok:
            failures.append(title)
        blocks.append(
            f"{title} : enfant attendu {child_at.tolist()} obtenu {got.tolist()} "
            f"(écart {pos:.3e}) ; End Site attendu {end_at.tolist()} écart {end_err:.3e} "
            f"→ {'OK' if ok else 'ÉCHEC'}"
        )

    child = Node("Child", (0, 1, 0), ["Yposition"])
    root = Node(
        "Hips",
        (0, 0, 0),
        ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"],
        [child],
    )
    motion = np.array([[0, 0, 0, 90, 0, 0, 2]], dtype=np.float64)
    clip = parse_bvh_text(emit_bvh([root], motion, 0.05), source_name="position hors racine")
    got = clip.world_positions["Child"][0]
    expected = np.array([-3.0, 0.0, 0.0])
    err = float(np.max(np.abs(got - expected)))
    ok = err < 1e-8
    if not ok:
        failures.append("position hors racine")
    blocks.append(
        f"Position hors racine : Rz(90) sur (0, 3, 0) attendu [-3, 0, 0] "
        f"obtenu {got.tolist()} écart {err:.3e} → {'OK' if ok else 'ÉCHEC'}"
    )

    two = emit_bvh(
        [Node("A", (1, 0, 0), []), Node("B", (0, 2, 0), [])],
        np.zeros((1, 0)),
        0.05,
    )
    clip = parse_bvh_text(two, source_name="deux racines")
    err_a = float(np.max(np.abs(clip.world_positions["A"][0] - np.array([1.0, 0.0, 0.0]))))
    err_b = float(np.max(np.abs(clip.world_positions["B"][0] - np.array([0.0, 2.0, 0.0]))))
    ok = err_a < 1e-8 and err_b < 1e-8 and len(clip.roots) == 2
    if not ok:
        failures.append("deux racines")
    blocks.append(
        f"Deux ROOT : A écart {err_a:.3e}, B écart {err_b:.3e}, racines {clip.roots} "
        f"→ {'OK' if ok else 'ÉCHEC'}"
    )

    arm = Node("Arm", (0, 5, 0), [])
    hand = Node("Hand", (0, 3, 0), [], end=(0, 1, 0))
    arm.children = [hand]
    hand.parent = arm
    hips = Node("Hips", (0, 0, 0), ["Xposition", "Yposition", "Zposition"], [arm])
    text = emit_bvh([hips], np.array([[1.0, 2.0, 3.0]]), 0.1, style="canonical")
    text = text.replace("CHANNELS 0\n", "", 1)
    clip = parse_bvh_text(text, source_name="sans rotation")
    expected = {
        "Hips": np.array([1.0, 2.0, 3.0]),
        "Arm": np.array([1.0, 7.0, 3.0]),
        "Hand": np.array([1.0, 10.0, 3.0]),
    }
    worst = 0.0
    for name, target in expected.items():
        worst = max(worst, float(np.max(np.abs(clip.world_positions[name][0] - target))))
    warned = any("sans CHANNELS" in warning for warning in clip.warnings)
    ok = worst < 1e-8 and warned and "Hand" in clip.joints and clip.joints["Hand"].channels == []
    if not ok:
        failures.append("sans rotation")
    blocks.append(
        f"Joint sans CHANNELS + CHANNELS 0 : écart {worst:.3e}, avertissement={warned} "
        f"→ {'OK' if ok else 'ÉCHEC'}"
    )
    along_x = Node("Bone", (10, 0, 0), [], end=(1, 0, 0))
    root = Node("Root", (0, 0, 0), [], [along_x])
    clip = parse_bvh_text(emit_bvh([root], np.zeros((1, 0)), 0.04), source_name="repos")
    frame = clip.joints["Root"].bone_frame
    y_error = float(np.linalg.norm(frame[:, 1] - np.array([1.0, 0.0, 0.0])))
    # Pôle +Z rejeté sur le plan orthogonal à +X, puis X = Y × Z.
    z_error = float(np.linalg.norm(frame[:, 2] - np.array([0.0, 0.0, 1.0])))
    x_expected = np.cross(frame[:, 1], frame[:, 2])
    x_error = float(np.linalg.norm(frame[:, 0] - x_expected))
    ok = y_error < 1e-8 and z_error < 1e-8 and x_error < 1e-8 and abs(np.linalg.det(frame) - 1.0) < 1e-8
    if not ok:
        failures.append("repos depuis OFFSET")
    blocks.append(
        f"Repos depuis OFFSET (enfant unique +X, pôle +Z) : "
        f"écart Y {y_error:.3e}, Z {z_error:.3e}, X {x_error:.3e} "
        f"→ {'OK' if ok else 'ÉCHEC'}"
    )
    return "\n".join(blocks)


def _synthetic_cases():
    zxy = ["Zrotation", "Xrotation", "Yrotation"]
    xyz = ["Xrotation", "Yrotation", "Zrotation"]
    zyx = ["Zrotation", "Yrotation", "Xrotation"]
    cases = []

    hips = Node("mixamorig:Hips", (0, 90, 0), ["Xposition", "Yposition", "Zposition", *zxy])
    spine = Node("mixamorig:Spine", (0, 30, 0), zxy)
    chest = Node("mixamorig:Spine1", (0, 25, 0), zxy)
    head = Node("mixamorig:Head", (0, 20, 0), zxy, end=(0, 18, 0))
    twist = Node("mixamorig:LeftArmTwist", (8, 0, 0), zxy)
    arm = Node("mixamorig:LeftArm", (15, 5, 0), zxy, [twist])
    chest.children = [head, arm]
    for child in chest.children:
        child.parent = chest
    spine.children = [chest]
    chest.parent = spine
    hips.children = [spine]
    spine.parent = hips
    motion = np.zeros((3, 6 + 3 * 5))
    motion[1, 6:9] = (20, -10, 5)
    motion[2, 0:3] = (4, 0, -2)
    motion[2, 15:18] = (0, 40, 0)
    cases.append(("1 mixamo ZXY, sans jambes ni doigts, commentaire", [hips], motion, 0.04, "comment", 10.0, None))

    pelvis = Node("Pelvis", (0, 1.0, 0), ["Xposition", "Yposition", "Zposition", *xyz])
    chest = Node("Chest", (0, 0.35, 0), xyz)
    l_arm = Node("UpperArm_L", (0.2, 0.1, 0), xyz, [Node("Hand_L", (0.55, 0, 0), xyz, end=(0.1, 0, 0))])
    r_arm = Node("UpperArm_R", (-0.2, 0.1, 0), xyz, [Node("Hand_R", (-0.5, 0, 0), xyz, end=(-0.1, 0, 0))])
    chest.children = [l_arm, r_arm]
    for child in chest.children:
        child.parent = chest
        for grand in child.children:
            grand.parent = child
    pelvis.children = [chest]
    chest.parent = pelvis
    motion = np.zeros((2, 6 + 3 * 5))
    motion[1, 6:9] = (0, 15, -20)
    cases.append(("2 mètres XYZ, accolades collées, bras seuls", [pelvis], motion, 0.02, "jammed", 1000.0, None))

    bip = Node(
        "Bip01_Pelvis",
        (0, 0, 40),
        ["Xposition", "Yposition", "Zposition", *zyx],
    )
    spine = Node("Bip01_Spine", (0, 0, 18), zyx)
    roll = Node("Bip01_SpineRoll", (0, 0, 4), zyx)
    head = Node("Bip01_Head", (0, 0, 12), zyx, end=(0, 0, 8))
    spine.children = [roll, head]
    for child in spine.children:
        child.parent = spine
    bip.children = [spine]
    spine.parent = bip
    motion = np.zeros((2, 6 + 3 * 3))
    motion[1, 3:6] = (0, 25, 10)
    cases.append((
        "3 pouces ZYX Z-up, Bip01, os roll",
        [bip],
        motion,
        0.00833333,
        "canonical",
        25.4,
        {"unit": "inch", "up_axis": "Z", "apply_normalization": True},
    ))

    marker = Node("Marker", (0, 1, 0), ["Yposition"])
    body = Node("Body", (0, 0, 0), ["Xposition", "Yposition", "Zposition"], [marker])
    camera = Node("Camera", (0, 10, -2), ["Xrotation"])
    motion = np.zeros((1, 5))
    motion[0, 0:3] = (1.0, 2.0, 3.0)
    motion[0, 3] = 4.0
    motion[0, 4] = 25.0
    cases.append((
        "4 deux racines, position sur un non-root, une seule rotation",
        [body, camera],
        motion,
        1.0 / 60.0,
        "lower",
        1.0,
        None,
    ))

    origin = Node("Character1_Root", (0, 0, 0), ["Xposition", "Yposition", "Zposition", "Xrotation"])
    short = Node("bras_court", (0, 12, 0), ["Zrotation"])
    tip = Node("bout", (0, 4, 0), [], end=(0, 2, 0))
    short.children = [tip]
    tip.parent = short
    long_1 = Node("os_gauche_haut", (-3, 8, 0), ["Zrotation", "Xrotation"])
    long_2 = Node("os_gauche_2", (0, 6, 0), ["Yrotation"])
    long_3 = Node("machin_42", (0, 5, 0), ["Xrotation"])
    long_4 = Node("libre_fin", (0, 3, 1), [], end=(0, 1, 0))
    long_3.children = [long_4]
    long_4.parent = long_3
    long_2.children = [long_3]
    long_3.parent = long_2
    long_1.children = [long_2]
    long_2.parent = long_1
    origin.children = [short, long_1]
    for child in origin.children:
        child.parent = origin
    motion = np.zeros((4, 3 + 1 + 1 + 2 + 1 + 1))
    motion[1, 3] = 15
    motion[2, 4] = -30
    motion[3, 5:7] = (10, -20)
    cases.append((
        "5 préfixe Character1, noms libres, profondeurs 2 et 5, canaux partiels",
        [origin],
        motion,
        0.05,
        "swap_header",
        10.0,
        None,
    ))
    return cases


def _basis_z_up():
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def _run_synthetics(failures):
    blocks = []
    for title, roots, motion, frame_time, style, unit_to_mm, options in _synthetic_cases():
        text = emit_bvh(roots, motion, frame_time, style=style)
        positions, rotations = independent_fk(roots, motion)
        clip = parse_bvh_text(text, source_name=title)
        shown = clip
        if options is None:
            blocks.append(_compare_clip(clip, positions, rotations, title, failures, unit_to_mm))
        else:
            raw_pos, raw_rot = _max_errors(clip, positions, rotations)
            converted_pos = {}
            converted_rot = {}
            basis = _basis_z_up()
            scale = 0.0254
            for name in positions:
                converted_pos[name] = (scale * (basis @ positions[name].T)).T
                converted_rot[name] = np.zeros_like(rotations[name])
                for frame in range(rotations[name].shape[0]):
                    converted_rot[name][frame] = basis @ rotations[name][frame] @ basis.T
            clip_n = parse_bvh_text(text, source_name=title, **options)
            shown = clip_n
            pos_err, rot_err = _max_errors(clip_n, converted_pos, converted_rot)
            ok = raw_pos < 1e-5 and raw_rot < 1e-4 and pos_err < 1e-5 and rot_err < 1e-4
            if not ok:
                failures.append(title)
            blocks.append(
                f"{title}\n"
                f"  avant conversion : {raw_pos:.3e} unité, {raw_rot:.3e} deg\n"
                f"  après inch + Z-up → Y-up mètres : {pos_err:.3e} m ({pos_err * 1000:.6f} mm), "
                f"{rot_err:.3e} deg\n"
                f"  fps : {clip_n.fps:.6f} (Frame Time {frame_time})\n"
                f"  → {'OK' if ok else 'ÉCHEC'}"
            )
        written = os.path.join(OUT, "roundtrip.bvh")
        write_bvh(clip if options is None else parse_bvh_text(text, source_name=title), written)
        naive_roots, naive_motion, naive_time = naive_load(open(written, encoding="utf-8").read())
        naive_pos, naive_rot = independent_fk(naive_roots, naive_motion)
        round_pos = 0.0
        round_rot = 0.0
        for name in positions:
            round_pos = max(round_pos, float(np.max(np.abs(naive_pos[name] - positions[name]))))
            for frame in range(positions[name].shape[0]):
                round_rot = max(round_rot, _rotation_degrees(naive_rot[name][frame], rotations[name][frame]))
        time_err = abs(naive_time - frame_time)
        ok = round_pos < 1e-5 and round_rot < 1e-4 and time_err < 1e-8
        if not ok:
            failures.append(title + " aller-retour")
        blocks.append(
            f"  aller-retour écriture puis lecteur naïf : {round_pos:.3e} unité "
            f"({round_pos * unit_to_mm:.6f} mm), {round_rot:.3e} deg, "
            f"Δt={time_err:.3e} → {'OK' if ok else 'ÉCHEC'}"
        )
        orders = sorted({
            "".join(ch[0] for ch in joint.channels if ch.endswith("rotation")) or "—"
            for joint in shown.joints.values()
        })
        blocks.append(
            f"  parse : {len(shown.joints)} joints, racines {shown.roots}, "
            f"ordres {orders}, fps {shown.fps:.6f}, avertissements {len(shown.warnings)}"
        )
        if shown.warnings:
            blocks.extend(f"    - {warning}" for warning in shown.warnings)
    return "\n".join(blocks)


def _run_handedness(failures):
    root = Node("Hips", (1, 2, 3), ["Xposition", "Yposition", "Zposition"])
    motion = np.array([[4.0, 5.0, 6.0]])
    text = emit_bvh([root], motion, 0.04)
    clip = parse_bvh_text(
        text,
        source_name="miroir",
        unit="m",
        up_axis="Y",
        handedness="left",
        apply_normalization=True,
    )
    got = clip.world_positions["Hips"][0]
    expected = np.array([-5.0, 7.0, 9.0])
    err = float(np.max(np.abs(got - expected)))
    ok = err < 1e-8
    if not ok:
        failures.append("chiralité gauche")
    return (
        f"Main gauche déclarée, Y-up, mètres : offset (1,2,3) + position (4,5,6) "
        f"attendu {expected.tolist()} obtenu {got.tolist()} écart {err:.3e} → {'OK' if ok else 'ÉCHEC'}"
    )


def _expect_error(text, source, line, fragments):
    try:
        parse_bvh_text(text, source_name=source)
    except BvhParseError as exc:
        message = str(exc)
        missing = [fragment for fragment in fragments if fragment not in message]
        if line > 0 and f"Ligne {line}" not in message:
            missing.append(f"Ligne {line}")
        if "Correction proposée" not in message:
            missing.append("Correction proposée")
        return message, missing
    return "aucune exception", ["exception absente"]


def _run_failures():
    samples = [
        (
            "canal inconnu",
            """HIERARCHY
ROOT Hips
{
OFFSET 0 0 0
CHANNELS 1 Wrotation
}
MOTION
Frames: 1
Frame Time: 0.04
0
""",
            5,
            ["canal inconnu", "Wrotation"],
        ),
        (
            "compte de frame",
            """HIERARCHY
ROOT Hips
{
OFFSET 0 0 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
}
MOTION
Frames: 1
Frame Time: 0.04
1 2 3
""",
            10,
            ["3 valeurs", "6 attendues"],
        ),
        (
            "motion tronqué",
            """HIERARCHY
ROOT Hips
{
OFFSET 0 0 0
CHANNELS 3 Xposition Yposition Zposition
}
MOTION
Frames: 4
Frame Time: 0.02
0 0 0
""",
            10,
            ["tronqué", "1 frame"],
        ),
        (
            "hiérarchie tronquée",
            """HIERARCHY
ROOT Hips
{
OFFSET 0 0 0
CHANNELS 3 Xposition Yposition Zposition
JOINT Arm
{
OFFSET 0 1 0
""",
            6,
            ["tronqué", "Arm"],
        ),
        (
            "Frame Time absent",
            """HIERARCHY
ROOT Hips
{
OFFSET 0 0 0
CHANNELS 3 Xposition Yposition Zposition
}
MOTION
Frames: 1
0 1 2
""",
            9,
            ["données de frame", "0"],
        ),
        (
            "Frame Time nul",
            """HIERARCHY
ROOT Hips
{
OFFSET 0 0 0
CHANNELS 0
}
MOTION
Frames: 0
Frame Time: 0
""",
            9,
            ["Frame Time invalide"],
        ),
        (
            "fichier vide",
            "",
            1,
            ["fichier vide"],
        ),
        (
            "virgule décimale",
            """HIERARCHY
ROOT Hips
{
OFFSET 0 1,5 0
CHANNELS 0
}
MOTION
Frames: 0
Frame Time: 0.04
""",
            4,
            ["1,5", "point"],
        ),
    ]
    blocks = []
    failed = []
    for title, text, line, fragments in samples:
        message, missing = _expect_error(text, title, line, fragments)
        status = "OK" if not missing else "ÉCHEC " + ", ".join(missing)
        if missing:
            failed.append(title)
        blocks.append(f"--- {title} [{status}]\n{message}")
    try:
        parse_bvh(os.path.join(ROOT, "ce_fichier_n_existe_pas.bvh"))
        blocks.append("--- fichier absent [ÉCHEC exception absente]")
        failed.append("fichier absent")
    except BvhParseError as exc:
        message = str(exc)
        ok = "introuvable" in message and "Correction proposée" in message
        if not ok:
            failed.append("fichier absent")
        blocks.append(f"--- fichier absent [{'OK' if ok else 'ÉCHEC'}]\n{message}")
    try:
        parse_bvh_text("HIERARCHY\n", source_name="x", apply_normalization=True)
    except BvhParseError as exc:
        message = str(exc)
        ok = "unité" in message and "Correction proposée" in message
        if not ok:
            failed.append("normalisation sans unité")
        blocks.append(f"--- normalisation sans unité [{'OK' if ok else 'ÉCHEC'}]\n{message}")
    return "\n\n".join(blocks), failed


def _control_arm():
    elbow = Node("Elbow", (10, 0, 0), ["Zrotation", "Xrotation", "Yrotation"])
    wrist = Node("Wrist", (8, 0, 0), ["Zrotation", "Xrotation", "Yrotation"], end=(3, 0, 0))
    elbow.children = [wrist]
    wrist.parent = elbow
    shoulder = Node("Shoulder", (0, 0, 0), ["Zrotation", "Xrotation", "Yrotation"], [elbow])
    motion = np.zeros((3, 9))
    motion[1, 0] = 90
    motion[2, 0] = 90
    motion[2, 4] = 80
    text = emit_bvh([shoulder], motion, 0.04)
    positions, rotations = independent_fk([shoulder], motion)
    clip = parse_bvh_text(text, source_name="contrôle bras")
    return clip, positions, rotations


def _draw_text(image, text, x, y, color, scale=2):
    for char in text:
        glyph = _FONT.get(char, _FONT[" "])
        for row in range(7):
            for col in range(5):
                if glyph[row * 5 + col] == "1":
                    y0 = y + row * scale
                    x0 = x + col * scale
                    image[y0:y0 + scale, x0:x0 + scale] = color
        x += 6 * scale


def _draw_line(image, x0, y0, x1, y1, color, thickness=1):
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    h, w = image.shape[:2]
    while True:
        for oy in range(-thickness, thickness + 1):
            for ox in range(-thickness, thickness + 1):
                xx, yy = x0 + ox, y0 + oy
                if 0 <= xx < w and 0 <= yy < h:
                    image[yy, xx] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _project(points, origin_x, origin_y, width, height, fit=None):
    def screen(cloud):
        return np.stack([cloud[:, 0] + 0.35 * cloud[:, 2], cloud[:, 1] + 0.15 * cloud[:, 2]], axis=1)

    basis = screen(points if fit is None else fit)
    query = screen(points)
    min_x, max_x = float(basis[:, 0].min()), float(basis[:, 0].max())
    min_y, max_y = float(basis[:, 1].min()), float(basis[:, 1].max())
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    scale = 0.72 * min(width, height) / span
    cx = origin_x + width * 0.5
    cy = origin_y + height * 0.58
    mid_x = (min_x + max_x) * 0.5
    mid_y = (min_y + max_y) * 0.5
    px = cx + (query[:, 0] - mid_x) * scale
    py = cy - (query[:, 1] - mid_y) * scale
    return np.stack([px, py], axis=1)


def _paint_skeleton(image, clip, frame, ox, oy, w, h, color, thickness, cross=None):
    names = list(clip.joints)
    pts = np.array([clip.world_positions[name][frame] for name in names])
    projected = _project(pts, ox, oy, w, h)
    index = {name: i for i, name in enumerate(names)}
    for name, joint in clip.joints.items():
        if joint.parent is not None and joint.parent in index:
            a = projected[index[joint.parent]]
            b = projected[index[name]]
            _draw_line(image, a[0], a[1], b[0], b[1], color, thickness)
        if name in clip.end_positions:
            end_pts = np.vstack([pts, clip.end_positions[name][frame]])
            end_proj = _project(end_pts, ox, oy, w, h)
            a = end_proj[index[name]]
            b = end_proj[-1]
            _draw_line(image, a[0], a[1], b[0], b[1], color, thickness)
    if cross is not None:
        joint_name, direction = cross
        if joint_name in index:
            start = clip.world_positions[joint_name][frame]
            end = start + direction
            both = np.vstack([pts, start, end])
            proj = _project(both, ox, oy, w, h)
            _draw_line(image, proj[-2, 0], proj[-2, 1], proj[-1, 0], proj[-1, 1], color, thickness + 1)


def _write_png(path, image):
    import struct
    import zlib

    height, width = image.shape[:2]
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as handle:
        handle.write(png)


def _render(control, real_clip, real_note):
    panel_w, panel_h = 380, 340
    image = np.full((panel_h * 2, panel_w * 3, 3), 244, dtype=np.uint8)
    clip, positions, rotations = control
    titles = ["REPOS", "BRAS LEVE", "TORSION"]
    for frame, title in enumerate(titles):
        ox = frame * panel_w
        _paint_from_oracle(image, clip, positions, rotations, frame, ox, 0, panel_w, panel_h)
        _draw_text(image, title, ox + 12, 12, (20, 20, 20), 2)
    if real_clip is not None and real_clip.n_frames > 0:
        frames = [0, real_clip.n_frames // 2, real_clip.n_frames - 1]
        labels = [f"REEL {frames[0]}", f"REEL {frames[1]}", f"REEL {frames[2]}"]
        for index, frame in enumerate(frames):
            ox = index * panel_w
            oy = panel_h
            _paint_skeleton(image, real_clip, frame, ox, oy, panel_w, panel_h, (25, 70, 140), 2)
            _draw_text(image, labels[index], ox + 12, oy + 12, (20, 20, 20), 2)
    else:
        _draw_text(image, "REEL " + real_note[:18], 12, panel_h + 12, (140, 30, 30), 2)
    path = os.path.join(OUT, "phase_a_control.png")
    _write_png(path, image)
    return path


def _paint_from_oracle(image, clip, positions, rotations, frame, ox, oy, w, h):
    names = list(positions)
    pts = np.array([positions[name][frame] for name in names])
    parser_pts = np.array([clip.world_positions[name][frame] for name in names])
    fit = np.vstack([pts, parser_pts])
    projected = _project(pts, ox, oy, w, h, fit=fit)
    index = {name: i for i, name in enumerate(names)}
    for name in names:
        parent = None
        for joint_name, joint in clip.joints.items():
            if joint_name == name:
                parent = joint.parent
        if parent is not None and parent in index:
            a = projected[index[parent]]
            b = projected[index[name]]
            _draw_line(image, a[0], a[1], b[0], b[1], (30, 70, 160), 3)
    parser_proj = _project(parser_pts, ox, oy, w, h, fit=fit)
    for name in names:
        joint = clip.joints[name]
        if joint.parent is not None and joint.parent in index:
            a = parser_proj[index[joint.parent]]
            b = parser_proj[index[name]]
            _draw_line(image, a[0], a[1], b[0], b[1], (220, 130, 20), 1)
    if "Elbow" in rotations:
        start = positions["Elbow"][frame]
        direction = rotations["Elbow"][frame] @ np.array([0.0, 6.0, 0.0])
        both = np.vstack([fit, start, start + direction])
        proj = _project(both, ox, oy, w, h, fit=fit)
        _draw_line(image, proj[-2, 0], proj[-2, 1], proj[-1, 0], proj[-1, 1], (0, 130, 150), 3)
        q = clip.world_rotations["Elbow"][frame]
        parser_dir = _quat_to_matrix(q) @ np.array([0.0, 6.0, 0.0])
        both_p = np.vstack([
            fit,
            clip.world_positions["Elbow"][frame],
            clip.world_positions["Elbow"][frame] + parser_dir,
        ])
        proj_p = _project(both_p, ox, oy, w, h, fit=fit)
        _draw_line(image, proj_p[-2, 0], proj_p[-2, 1], proj_p[-1, 0], proj_p[-1, 1], (190, 50, 40), 1)


def _matrix_angle_deg(left, right):
    """Angle stable. arccos(trace) explose quand la trace vaut 3 à 1 ulp près."""
    rel = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    cosine = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    skew = np.array(
        [
            rel[2, 1] - rel[1, 2],
            rel[0, 2] - rel[2, 0],
            rel[1, 0] - rel[0, 1],
        ],
        dtype=np.float64,
    )
    sine = float(np.clip(0.5 * np.linalg.norm(skew), 0.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))


def _local_from_row(channels, row, start):
    """Matrice locale, mêmes Rx Ry Rz que la FK de cet oracle. Pas la FK du parseur."""
    local = np.eye(3, dtype=np.float64)
    for index, channel in enumerate(channels):
        if channel.endswith("rotation"):
            local = _AXIS[channel[0]](float(row[start + index])) @ local
    return local


def rotation_roundtrip_deg(clip):
    """R = (R @ F) @ F.T doit rendre la rotation des canaux. F est le repère d'os."""
    worst = 0.0
    worst_name = ""
    for joint in clip.joints.values():
        frame = np.asarray(joint.bone_frame, dtype=np.float64)
        for frame_index in range(clip.motion.shape[0]):
            local = _local_from_row(joint.channels, clip.motion[frame_index], joint.channel_start)
            recovered = (local @ frame) @ frame.T
            angle = _matrix_angle_deg(local, recovered)
            if angle > worst:
                worst = angle
                worst_name = joint.name
    return worst, worst_name


def _frame_geometry(clip):
    """Y suit la direction du cahier des charges. Z est le pôle rejeté. X = Y × Z."""
    worst_dir = 0.0
    worst_roll = 0.0
    worst_name = ""
    notes = []
    pole = np.asarray(clip.normalization.pole, dtype=np.float64)
    pole = pole / np.linalg.norm(pole)
    for joint in clip.joints.values():
        frame = np.asarray(joint.bone_frame, dtype=np.float64)
        if abs(float(np.linalg.det(frame)) - 1.0) > 1e-5:
            notes.append(f"det {joint.name} = {np.linalg.det(frame):.3e}")
        if np.max(np.abs(frame.T @ frame - np.eye(3))) > 1e-5:
            notes.append(f"non orthogonal {joint.name}")
        expected = _spec_direction(joint, clip.joints)
        y = frame[:, 1]
        if expected is None:
            if np.linalg.norm(frame - np.eye(3)) > 1e-5:
                notes.append(f"{joint.name} devrait être l'identité")
            continue
        err = float(np.linalg.norm(y - expected))
        if err > worst_dir:
            worst_dir = err
            worst_name = joint.name
        proj = pole - y * float(np.dot(pole, y))
        proj_norm = float(np.linalg.norm(proj))
        if proj_norm > 1e-6:
            proj = proj / proj_norm
            worst_roll = max(worst_roll, abs(1.0 - abs(float(np.dot(frame[:, 2], proj)))))
        if np.linalg.norm(frame[:, 0] - np.cross(y, frame[:, 2])) > 1e-5:
            notes.append(f"X ≠ Y×Z sur {joint.name}")
    return worst_dir, worst_roll, worst_name, notes


def _spec_direction(joint, joints):
    children = [joints[name] for name in joint.children]
    if len(children) == 1:
        vector = np.asarray(children[0].offset, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        return None if norm < 1e-6 else vector / norm
    if len(children) > 1:
        mean = np.mean([np.asarray(child.offset, dtype=np.float64) for child in children], axis=0)
        norm = float(np.linalg.norm(mean))
        return None if norm < 1e-6 else mean / norm
    if joint.end_site is not None:
        vector = np.asarray(joint.end_site, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm >= 1e-6:
            return vector / norm
    if joint.parent is not None:
        return np.array(joints[joint.parent].bone_frame[:, 1], dtype=np.float64, copy=True)
    return None


def _motion_row(width, angles):
    row = np.zeros(width, dtype=np.float64)
    row[: len(angles)] = angles
    return row


def _run_decisions(failures):
    blocks = []
    roundtrip_worst = 0.0

    fork = """HIERARCHY
ROOT Root
{
OFFSET 0 0 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
JOINT Long
{
OFFSET 3 0 0
CHANNELS 3 Zrotation Xrotation Yrotation
End Site
{
OFFSET 0.2 0 0
}
}
JOINT Short
{
OFFSET 0 1 0
CHANNELS 3 Zrotation Xrotation Yrotation
End Site
{
OFFSET 0 0.2 0
}
}
}
MOTION
Frames: 2
Frame Time: 0.04
0 0 0 30 20 10 5 -8 3 1 2 3
1 0 -1 -12 40 7 0 0 0 9 -4 2
"""
    clip = parse_bvh_text(fork, source_name="barycentre")
    # Moyenne de (3,0,0) et (0,1,0) = (1.5, 0.5, 0). Norme sqrt(2.5), à la main.
    expected = np.array([0.9486832980505138, 0.31622776601683794, 0.0])
    y = clip.joints["Root"].bone_frame[:, 1]
    bary_err = float(np.linalg.norm(y - expected))
    longest_dot = abs(float(np.dot(y, np.array([1.0, 0.0, 0.0]))))
    rt, rt_name = rotation_roundtrip_deg(clip)
    roundtrip_worst = max(roundtrip_worst, rt)
    ok = bary_err < 1e-8 and longest_dot < 0.96 and rt < 1e-6
    if not ok:
        failures.append("barycentre contre plus long enfant")
    blocks.append(
        f"Plusieurs enfants, sans chaîne primaire : écart au barycentre {bary_err:.3e}, "
        f"|Y·(+X)| {longest_dot:.3f} (le plus long enfant serait 1), "
        f"aller-retour rotations {rt:.3e} deg sur {rt_name} → {'OK' if ok else 'ÉCHEC'}"
    )

    forced = parse_bvh_text(fork, source_name="primaire", primary_child={"Root": "Short"})
    y_short = forced.joints["Root"].bone_frame[:, 1]
    short_err = float(np.linalg.norm(y_short - np.array([0.0, 1.0, 0.0])))
    rt, rt_name = rotation_roundtrip_deg(forced)
    roundtrip_worst = max(roundtrip_worst, rt)
    ok = short_err < 1e-8 and rt < 1e-6
    if not ok:
        failures.append("chaîne primaire")
    blocks.append(
        f"Chaîne primaire Root→Short : écart à +Y {short_err:.3e}, "
        f"aller-retour {rt:.3e} deg → {'OK' if ok else 'ÉCHEC'}"
    )

    inherit = """HIERARCHY
ROOT Root
{
OFFSET 0 0 0
CHANNELS 3 Zrotation Xrotation Yrotation
JOINT Bone
{
OFFSET 2 0 0
CHANNELS 3 Zrotation Xrotation Yrotation
End Site
{
OFFSET 0 0 0
}
}
}
MOTION
Frames: 2
Frame Time: 0.04
30 20 10 0 0 0
-15 40 5 0 0 0
"""
    clip = parse_bvh_text(inherit, source_name="end site nul")
    y_tip = clip.joints["Bone"].bone_frame[:, 1]
    inherit_err = float(np.linalg.norm(y_tip - np.array([1.0, 0.0, 0.0])))
    warned = any("End Site" in warning and "parent" in warning for warning in clip.warnings)
    rt, rt_name = rotation_roundtrip_deg(clip)
    roundtrip_worst = max(roundtrip_worst, rt)
    ok = inherit_err < 1e-8 and warned and rt < 1e-6
    if not ok:
        failures.append("End Site nul")
    blocks.append(
        f"End Site nul : Bone hérite +X du parent, écart {inherit_err:.3e}, "
        f"avertissement={warned}, aller-retour {rt:.3e} deg → {'OK' if ok else 'ÉCHEC'}"
    )

    along_y = """HIERARCHY
ROOT Root
{
OFFSET 0 0 0
CHANNELS 3 Zrotation Xrotation Yrotation
JOINT Bone
{
OFFSET 0 2 0
CHANNELS 3 Zrotation Xrotation Yrotation
End Site
{
OFFSET 0 1 0
}
}
}
MOTION
Frames: 1
Frame Time: 0.04
30 20 10 0 0 0
"""
    pole_z = parse_bvh_text(along_y, source_name="pole Z", pole=(0.0, 0.0, 1.0))
    pole_x = parse_bvh_text(along_y, source_name="pole X", pole=(1.0, 0.0, 0.0))
    # Direction +Y, pôle +X : Z=(1,0,0), X=Y×Z=(0,0,-1). Littéral, pas la fonction du parseur.
    expected_x = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    frame_err = float(np.linalg.norm(pole_x.joints["Root"].bone_frame - expected_x))
    differ = float(np.linalg.norm(pole_z.joints["Root"].bone_frame - pole_x.joints["Root"].bone_frame))
    rt_z, _ = rotation_roundtrip_deg(pole_z)
    rt_x, _ = rotation_roundtrip_deg(pole_x)
    roundtrip_worst = max(roundtrip_worst, rt_z, rt_x)
    ok = frame_err < 1e-8 and differ > 0.5 and rt_z < 1e-6 and rt_x < 1e-6
    if not ok:
        failures.append("pôle")
    blocks.append(
        f"Deux pôles sur un os +Y : écart au repère littéral pôle +X {frame_err:.3e}, "
        f"écart entre pôles {differ:.3e}, aller-retour {max(rt_z, rt_x):.3e} deg "
        f"→ {'OK' if ok else 'ÉCHEC'}"
    )

    cancel = """HIERARCHY
ROOT Root
{
OFFSET 0 0 0
CHANNELS 3 Zrotation Xrotation Yrotation
JOINT East
{
OFFSET 2 0 0
CHANNELS 3 Zrotation Xrotation Yrotation
End Site
{
OFFSET 0.1 0 0
}
}
JOINT West
{
OFFSET -2 0 0
CHANNELS 3 Zrotation Xrotation Yrotation
End Site
{
OFFSET -0.1 0 0
}
}
}
MOTION
Frames: 1
Frame Time: 0.04
0 0 0 0 0 0 0 0 0
"""
    clip = parse_bvh_text(cancel, source_name="annulation")
    identity_err = float(np.linalg.norm(clip.joints["Root"].bone_frame - np.eye(3)))
    not_longest = abs(float(clip.joints["Root"].bone_frame[0, 1])) < 1e-8
    warned = any("barycentre" in warning and "plus long" in warning for warning in clip.warnings)
    ok = identity_err < 1e-8 and not_longest and warned
    if not ok:
        failures.append("barycentre nul")
    blocks.append(
        f"Enfants opposés : identité {identity_err:.3e}, pas le plus long={not_longest}, "
        f"avertissement={warned} → {'OK' if ok else 'ÉCHEC'}"
    )

    unit_clip = parse_bvh_text(fork, source_name="unite vide")
    unit_ok = (
        unit_clip.normalization.unit == "non déclarée"
        and not np.isfinite(unit_clip.normalization.unit_to_meters)
        and any("Aucune conversion" in warning for warning in unit_clip.warnings)
        and unit_clip.normalization.unit not in ("m", "cm", "inch")
    )
    if not unit_ok:
        failures.append("unité non déclarée")
    blocks.append(
        f"Unité laissée vide : « {unit_clip.normalization.unit} », "
        f"facteur fini={np.isfinite(unit_clip.normalization.unit_to_meters)} "
        f"→ {'OK' if unit_ok else 'ÉCHEC'}"
    )

    right_body = _chirality_bvh(mirror=False)
    left_body = _chirality_bvh(mirror=True)
    plain = _chirality_bvh(shoulders_only=True)
    right = parse_bvh_text(right_body, source_name="chiralité droite")
    mirrored = parse_bvh_text(left_body, source_name="chiralité miroir")
    unclear = parse_bvh_text(plain, source_name="chiralité sans avant")
    right_ok = (
        "compatible" in right.normalization.chirality_measurement
        and right.normalization.handedness == "right"
        and not any("contredit" in warning for warning in right.warnings)
    )
    mirror_ok = (
        mirrored.normalization.handedness == "right"
        and any("contredit" in warning and "Correction proposée" in warning for warning in mirrored.warnings)
        and "contradiction" in mirrored.normalization.chirality_measurement
    )
    unclear_ok = (
        "non concluante" in unclear.normalization.chirality_measurement
        and unclear.normalization.handedness == "right"
        and not any("contredit" in warning for warning in unclear.warnings)
    )
    if not (right_ok and mirror_ok and unclear_ok):
        failures.append("mesure de chiralité")
    blocks.append(
        f"Chiralité mesurée, hypothèse droite conservée :\n"
        f"  squelette cohérent : {right.normalization.chirality_measurement} → {'OK' if right_ok else 'ÉCHEC'}\n"
        f"  miroir : {mirrored.normalization.chirality_measurement} → {'OK' if mirror_ok else 'ÉCHEC'}\n"
        f"  sans axe avant : {unclear.normalization.chirality_measurement} → {'OK' if unclear_ok else 'ÉCHEC'}"
    )

    dup = _duplicate_bvh()
    clip = parse_bvh_text(dup, source_name="doublons")
    expected_names = {"Hips", "Left", "Right", "Hips/Left/Hand", "Hips/Right/Hand"}
    names = set(clip.joints)
    hash_names = [name for name in names if "#2" in name or "\0" in name]
    path_warn = any("chemin" in warning and "Correction proposée" in warning for warning in clip.warnings)
    rt, _ = rotation_roundtrip_deg(clip)
    roundtrip_worst = max(roundtrip_worst, rt)
    ok = names == expected_names and not hash_names and path_warn
    if not ok:
        failures.append("doublons par chemin")
    blocks.append(
        f"Doublons : identités {sorted(names)}, #2={hash_names or 'aucun'}, "
        f"avertissement de chemin={path_warn} → {'OK' if ok else 'ÉCHEC'}"
    )

    blocks.append("Entrées hostiles :")
    blocks.append(_hostile(failures, "chaîne primaire absente", lambda: parse_bvh_text(
        fork, source_name="primaire absente", primary_child={"Root": "Nope"}
    )))
    blocks.append(_hostile(failures, "frères homonymes", lambda: parse_bvh_text(
        _sibling_bvh(), source_name="frères"
    )))
    blocks.append(_hostile(failures, "on_duplicate=error", lambda: parse_bvh_text(
        dup, source_name="refus doublon", on_duplicate="error"
    )))
    blocks.append(_hostile(failures, "pôle nul", lambda: parse_bvh_text(
        along_y, source_name="pole nul", pole=(0.0, 0.0, 0.0)
    )))
    blocks.append(_hostile(failures, "on_duplicate inconnu", lambda: parse_bvh_text(
        fork, source_name="politique", on_duplicate="rename"
    )))
    french = parse_bvh_text(_chirality_bvh(french=True), source_name="noms français")
    blocks.append(
        "  noms Gauche/Droite (non reconnus) : "
        + french.normalization.chirality_measurement
    )
    parallel = parse_bvh_text(inherit, source_name="pôle parallèle", pole=(1.0, 0.0, 0.0))
    parallel_warn = any("parallèle" in warning for warning in parallel.warnings)
    rt, _ = rotation_roundtrip_deg(parallel)
    roundtrip_worst = max(roundtrip_worst, rt)
    det = float(np.linalg.det(parallel.joints["Bone"].bone_frame))
    ok = parallel_warn and abs(det - 1.0) < 1e-6 and rt < 1e-6
    if not ok:
        failures.append("pôle parallèle")
    blocks.append(
        f"  pôle parallèle à l'os : det {det:.6f}, avertissement={parallel_warn}, "
        f"aller-retour {rt:.3e} deg → {'OK' if ok else 'ÉCHEC'}"
    )
    blocks.append(f"Écart angulaire max de l'aller-retour (ces cas) : {roundtrip_worst:.3e} degrés.")
    return "\n".join(blocks), roundtrip_worst


def _hostile(failures, label, call):
    try:
        call()
    except BvhParseError as exc:
        text = str(exc).replace("\n", " | ")
        ok = "Correction proposée" in str(exc) and (exc.line > 0 or "on_duplicate" in label or "pôle" in label)
        if not ok:
            failures.append(label)
        return f"  {label} : REFUS — {text} → {'OK' if ok else 'ÉCHEC'}"
    failures.append(label)
    return f"  {label} : accepté alors qu'un refus était attendu → ÉCHEC"


def _chirality_bvh(mirror=False, shoulders_only=False, french=False):
    if french:
        left_hip, right_hip = "Gauche", "Droite"
        left_shoulder, right_shoulder = "EpauleG", "EpauleD"
    else:
        left_hip, right_hip = "LeftHip", "RightHip"
        left_shoulder, right_shoulder = "LeftShoulder", "RightShoulder"
    sign = -1.0 if mirror else 1.0
    lx, rx = sign * 1.0, -sign * 1.0
    lines = [
        "HIERARCHY",
        "ROOT Hips",
        "{",
        "OFFSET 0 0 0",
        "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
    ]
    if not shoulders_only and not french:
        lines += _joint_block(left_hip, (lx, 0, 0), [_joint_leaf("LeftKnee", (0, -1, 0.4), (0, -0.4, 0.2))])
        lines += _joint_block(right_hip, (rx, 0, 0), [_joint_leaf("RightKnee", (0, -1, 0.4), (0, -0.4, 0.2))])
    chest_children = [
        _joint_leaf(left_shoulder, (lx, 0, 0), (sign * 0.4, 0, 0)),
        _joint_leaf(right_shoulder, (rx, 0, 0), (-sign * 0.4, 0, 0)),
        _joint_leaf("Head", (0, 0.5, 0), (0, 0.2, 0)),
    ]
    lines += _joint_block("Chest", (0, 1, 0), chest_children)
    lines += ["}", "MOTION", "Frames: 1", "Frame Time: 0.05"]
    count = 6 + 3 * (0 if shoulders_only or french else 4) + 3 * 4
    lines.append(" ".join(["0"] * count))
    return "\n".join(lines) + "\n"


def _joint_leaf(name, offset, end):
    return "\n".join([
        f"JOINT {name}",
        "{",
        f"OFFSET {offset[0]} {offset[1]} {offset[2]}",
        "CHANNELS 3 Zrotation Xrotation Yrotation",
        "End Site",
        "{",
        f"OFFSET {end[0]} {end[1]} {end[2]}",
        "}",
        "}",
    ])


def _joint_block(name, offset, children):
    body = [
        f"JOINT {name}",
        "{",
        f"OFFSET {offset[0]} {offset[1]} {offset[2]}",
        "CHANNELS 3 Zrotation Xrotation Yrotation",
    ]
    for child in children:
        body.extend(line for line in child.splitlines())
    body.append("}")
    return body


def _duplicate_bvh():
    return """HIERARCHY
ROOT Hips
{
OFFSET 0 0 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
JOINT Left
{
OFFSET 1 0 0
CHANNELS 3 Zrotation Xrotation Yrotation
JOINT Hand
{
OFFSET 0 1 0
CHANNELS 3 Zrotation Xrotation Yrotation
End Site
{
OFFSET 0 0.2 0
}
}
}
JOINT Right
{
OFFSET -1 0 0
CHANNELS 3 Zrotation Xrotation Yrotation
JOINT Hand
{
OFFSET 0 1 0
CHANNELS 3 Zrotation Xrotation Yrotation
End Site
{
OFFSET 0 0.2 0
}
}
}
}
MOTION
Frames: 1
Frame Time: 0.05
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
"""


def _sibling_bvh():
    return """HIERARCHY
ROOT Hips
{
OFFSET 0 0 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
JOINT Hand
{
OFFSET 1 0 0
CHANNELS 3 Zrotation Xrotation Yrotation
End Site
{
OFFSET 0.2 0 0
}
}
JOINT Hand
{
OFFSET -1 0 0
CHANNELS 3 Zrotation Xrotation Yrotation
End Site
{
OFFSET -0.2 0 0
}
}
}
MOTION
Frames: 1
Frame Time: 0.05
0 0 0 0 0 0 0 0 0 0 0 0
"""


def _run_real(failures):
    if not os.path.isfile(REAL):
        return "a_out_precise.bvh absent. Aucun corpus réel fourni dans le dépôt.", None, "ABSENT"
    try:
        clip = parse_bvh(REAL)
    except BvhParseError as exc:
        return (
            "a_out_precise.bvh est refusé par le parseur.\n" + str(exc),
            None,
            "REFUSE",
        )
    try:
        roots, motion, frame_time = naive_load(open(REAL, encoding="utf-8").read())
        positions, rotations = independent_fk(roots, motion)
        pos_err, rot_err = _max_errors(clip, positions, rotations)
        time_err = abs(frame_time - clip.normalization.frame_time)
        rt_deg, rt_name = rotation_roundtrip_deg(clip)
        dir_err, roll_err, dir_name, geom_notes = _frame_geometry(clip)
        unit_name = clip.normalization.unit
        proportion = any(
            "unité non déclarée" in warning and "Aucune conversion" in warning
            for warning in clip.warnings
        )
        hash_names = [joint.name for joint in clip.joints.values() if "#2" in joint.name or "\0" in joint.name]
        renamed = [f"{joint.raw_name} → {joint.name}" for joint in clip.joints.values() if joint.name != joint.raw_name]
        unit_ok = unit_name == "non déclarée" and proportion and unit_name not in ("m", "cm", "inch", "unknown")
        geom_ok = dir_err < 1e-5 and roll_err < 1e-5 and not geom_notes and rt_deg < 1e-6
        ok = pos_err < 1e-4 and rot_err < 1e-3 and time_err < 1e-8 and unit_ok and geom_ok and not hash_names
        if not ok:
            failures.append("a_out_precise")
        naive_block = (
            f"Lecteur naïf indépendant sur le fichier réel :\n"
            f"  écart position max {pos_err:.3e} unités fichier\n"
            f"  écart rotation max {rot_err:.3e} degrés\n"
            f"  Δ Frame Time {time_err:.3e}\n"
            f"  aller-retour des rotations via le repère d'os : {rt_deg:.3e} deg (pire joint {rt_name})\n"
            f"  direction d'os max {dir_err:.3e} ({dir_name}), roulis pôle {roll_err:.3e}\n"
            f"  unité « {unit_name} » (non devinée), avertissement de proportion={proportion}\n"
            f"  hypothèse chiralité : {clip.normalization.handedness} — {clip.normalization.handedness_source}\n"
            f"  mesure : {clip.normalization.chirality_measurement}\n"
            f"  doublons renommés : {', '.join(renamed) if renamed else 'aucun'} ; suffixe #2 : {hash_names or 'aucun'}\n"
            f"  notes de géométrie : {geom_notes or 'aucune'}\n"
            f"  → {'OK' if ok else 'ÉCHEC'}\n"
            "  L'unité reste non déclarée : l'écart n'est pas converti en mm."
        )
    except Exception as exc:
        naive_block = "Lecteur naïf en échec sur le fichier réel :\n" + "".join(
            traceback.format_exception_only(type(exc), exc)
        )
        failures.append("lecteur naïf fichier réel")
    adapter = BVHAdapter()
    sequence = adapter.load(REAL)
    skeleton = adapter.load_skeleton(REAL)
    adapter_block = (
        f"Adaptateur : {len(sequence.frames_markers)} frames, "
        f"{len(sequence.frames_markers[0])} marqueurs, fps {sequence.fps:.6f}, "
        f"repos squelette {len(skeleton.rest_rotations)} os, fps squelette {skeleton.fps:.6f}."
    )
    return "\n".join([naive_block, adapter_block, "", format_report(clip)]), clip, "OK"


def _run_adapter_control(failures):
    clip, positions, _rotations = _control_arm()
    folder = tempfile.mkdtemp(prefix="bvh_a_")
    path = os.path.join(folder, "arm.bvh")
    write_bvh(clip, path)
    adapter = BVHAdapter()
    sequence = adapter.load(path)
    skeleton = adapter.load_skeleton(path)
    marker = sequence.frames_markers[1]["Wrist"]
    expected = positions["Wrist"][1]
    err = float(np.max(np.abs(marker.reshape(3) - expected)))
    fps_ok = abs(sequence.fps - 25.0) < 1e-9 and abs(skeleton.fps - 25.0) < 1e-9
    rest = skeleton.rest_rotations["Shoulder"]
    rest_not_identity = abs(float(rest[0])) < 0.99
    ok = err < 1e-5 and fps_ok and rest_not_identity and sequence.joint_rotations is not None
    if not ok:
        failures.append("adaptateur")
    return (
        f"Adaptateur sur le bras de contrôle : écart poignet frame bras levé {err:.3e}, "
        f"fps {sequence.fps:.6f} (pas 30), repos épaule non identité={rest_not_identity} "
        f"→ {'OK' if ok else 'ÉCHEC'}"
    )


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    os.makedirs(OUT, exist_ok=True)
    failures = []
    sections = []
    euler_err = check_euler_roundtrip()
    euler_ok = euler_err < 1e-8
    if not euler_ok:
        failures.append("reconstruction Euler")
    sections.append(
        "1. Reconstruction Euler du parseur (contrôle interne, pas l'oracle de positions)\n"
        f"   écart matrice max {euler_err:.3e} → {'OK' if euler_ok else 'ÉCHEC'}"
    )
    sections.append("2. Littéraux algébriques (positions écrites à la main)\n" + _run_literals(failures))
    sections.append("3. Cinq BVH synthétiques irréguliers, puis aller-retour\n" + _run_synthetics(failures))
    decision_text, decision_rt = _run_decisions(failures)
    sections.append(
        "3b. Repère d'os, unité non devinée, chiralité mesurée, doublons\n"
        + decision_text
        + f"\nÉcart angulaire max retenu pour ces preuves : {decision_rt:.3e} degrés."
    )
    sections.append("4. Chiralité déclarée\n" + _run_handedness(failures))
    failure_text, failure_names = _run_failures()
    failures.extend(failure_names)
    sections.append("5. Entrées invalides — message complet\n" + failure_text)
    sections.append("6. Adaptateur\n" + _run_adapter_control(failures))
    real_text, real_clip, real_note = _run_real(failures)
    sections.append("7. Fichier présent dans le dépôt\n" + real_text)
    image = _render(_control_arm(), real_clip, real_note)
    sections.append(
        "8. Rendu\n"
        f"   {image}\n"
        "   Rangée haute : bras synthétique, oracle en bleu, parseur en orange, "
        "barre locale +Y du coude (torsion). Repos, bras levé (Z=90), torsion coude (X=80).\n"
        "   Rangée basse : a_out_precise.bvh si le parseur l'accepte. "
        "Il n'y a pas de pose cible : l'export vers un autre squelette est la partie C."
    )
    sections.append(
        "9. Ce qui n'est pas vérifié\n"
        "   - Unité de a_out_precise.bvh : toujours non déclarée. Aucune valeur n'a été choisie à sa place.\n"
        "   - Noms de côté autres que left/right ou le suffixe _L/_R (le cas Gauche/Droite est seulement montré).\n"
        "   - Chaîne primaire désignée sur un parent lui-même en double, quand deux parents portent le même nom brut.\n"
        "   - Pôle exprimé dans un autre repère que le repère joint du parent.\n"
        "   - Barycentre rendu nul par un OFFSET plus petit que 1e-6 alors qu'un os réel est là.\n"
        "   - Aller-retour du pipeline de retarget : partie C, pas faite.\n"
        "   - Image d'une pose cible : pas d'export d'animation retargetée."
    )
    if real_clip is not None:
        real_rt, real_rt_name = rotation_roundtrip_deg(real_clip)
        real_unit = real_clip.normalization.unit
        real_chiral = real_clip.normalization.chirality_measurement
        real_dups = [
            joint.name for joint in real_clip.joints.values() if joint.name != joint.raw_name
        ]
    else:
        real_rt, real_rt_name = float("nan"), ""
        real_unit = "fichier absent"
        real_chiral = "non mesurée"
        real_dups = []
    header = (
        "RAPPORT PARTIE A — parseur BVH\n"
        f"Échecs : {len(failures)}\n"
        + ("\n".join(f"  - {name}" for name in failures) if failures else "  aucun")
        + "\n\n"
        "Décisions reprises\n"
        "1. Le repère d'os n'est plus le plus long enfant. +Y va vers l'enfant unique, "
        "vers la chaîne primaire si elle est nommée, sinon vers le barycentre. "
        "Un End Site nul hérite de la direction du parent. Le pôle fixe le roulis. "
        "Ce repère reste distinct du repère joint BVH ; son inverse est sa transposée. "
        f"Aller-retour des rotations, indépendant de la FK du parseur : "
        f"{decision_rt:.3e} deg sur les cas construits, "
        f"{real_rt:.3e} deg sur a_out_precise.bvh (pire joint {real_rt_name or '—'}).\n"
        f"2. Unité de a_out_precise.bvh : « {real_unit} ». "
        "Elle n'est pas devinée. Seul un avertissement de proportion humaine est ajouté. "
        "Décision restante : quelle unité déclarer.\n"
        f"3. Hypothèse droitière explicite, non remplacée par la mesure. Mesure du fichier : {real_chiral}\n"
        "4. Un nom unique reste le nom brut. Un doublon prend le chemin hiérarchique, "
        "ou le parseur refuse. Pas de suffixe #2. "
        f"Sur a_out_precise.bvh : {', '.join(real_dups) if real_dups else 'aucun doublon'}.\n"
    )
    report = header + "\n\n".join(sections) + "\n"
    report_path = os.path.join(OUT, "phase_a_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    sys.stdout.write(report)
    sys.stdout.write(f"\nRapport écrit : {report_path}\n")
    return 1 if failures else 0


_FONT = {
    " ": "00000000000000000000000000000000000",
    "A": "01110100011000111111100011000110001",
    "B": "11110100011000111110100011000111110",
    "C": "01111100001000010000100001000001111",
    "E": "11111100001000011110100001000011111",
    "F": "11111100001000011110100001000010000",
    "H": "10001100011111110001100011000110001",
    "I": "01110001000010000100001000010001110",
    "L": "10000100001000010000100001000011111",
    "M": "10001110111010110001100011000110001",
    "N": "10001110011010110011100011000110001",
    "O": "01110100011000110001100011000101110",
    "P": "11110100011000111110100001000010000",
    "R": "11110100011000111110101001001010001",
    "S": "01111100001000001110000010000111110",
    "T": "11111001000010000100001000010000100",
    "U": "10001100011000110001100011000101110",
    "V": "10001100011000110001100010101000100",
    "0": "01110100011001110101110011000101110",
    "1": "00100011000010000100001000010001110",
    "2": "01110100010000100010001000100011111",
    "3": "01110100010000100110000011000101110",
    "4": "00100001100101010010111110001000010",
    "5": "11111100001000011110000010000111110",
    "6": "01110100001000011110100011000101110",
    "7": "11111000010001000100010000100001000",
    "8": "01110100011000101110100011000101110",
    "9": "01110100011000101111000010000101110",
}


if __name__ == "__main__":
    sys.exit(main())

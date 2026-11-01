"""
Lecture et réécriture fidèle d'un BVH.

Convention d'Euler, par joint : les canaux de rotation sont appliqués dans
l'ordre où ils sont écrits. Xrotation puis Yrotation puis Zrotation donne
R = Rz @ Ry @ Rx (vecteurs colonne). Ce n'est pas un ordre global au fichier.

Les canaux de position sont lus par nom d'axe et ajoutés à l'OFFSET, dans le
repère du parent. La position d'un joint ne dépend pas de sa propre rotation.

Le repère d'os est distinct du repère joint BVH. Au repos, le repère joint
reste l'identité (canaux de rotation nuls). Le repère d'os a +Y vers
l'enfant unique, vers la chaîne primaire désignée, ou vers le barycentre
des enfants. Un End Site d'OFFSET nul hérite de la direction du parent.
Le roulis est fixé par un pôle secondaire. La matrice est inversible
(inverse = transposée). Elle ne remplace pas les rotations des canaux.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.geometry import (
    matrix_to_quaternion,
    quat_identity,
    quaternion_to_matrix,
)


_VALID_CHANNELS = {
    "Xposition",
    "Yposition",
    "Zposition",
    "Xrotation",
    "Yrotation",
    "Zrotation",
}
_CHANNEL_CANON = {name.lower(): name for name in _VALID_CHANNELS}
_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
_UNIT_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "cm": 0.01,
    "inch": 0.0254,
    "in": 0.0254,
    "inches": 0.0254,
}
_UNIT_ALIAS = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "cm": "cm",
    "inch": "inch",
    "in": "inch",
    "inches": "inch",
}
_UP_HINTS = {
    "head": 4.0,
    "neck": 3.0,
    "spine": 3.0,
    "chest": 2.0,
    "abdomen": 2.0,
    "hips": 2.0,
    "pelvis": 2.0,
    "shoulder": 0.5,
}
_PREFIX_RULES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^mixamorig:", re.IGNORECASE), "mixamorig:"),
    (re.compile(r"^Bip\d+[_\s]", re.IGNORECASE), "BipNN_"),
    (re.compile(r"^Character\d+_", re.IGNORECASE), "CharacterN_"),
    (re.compile(r"^Armature\|", re.IGNORECASE), "Armature|"),
    (re.compile(r"^[A-Za-z_][\w]*:"), "namespace:"),
)


class BvhParseError(ValueError):
    """Erreur de lecture BVH : ligne, cause, correction proposée."""

    def __init__(self, line: int, cause: str, correction: str):
        self.line = int(line)
        self.cause = cause
        self.correction = correction
        if self.line > 0:
            head = f"Ligne {self.line} : {cause}"
        else:
            head = cause
        super().__init__(f"{head}\nCorrection proposée : {correction}")


@dataclass
class BvhJoint:
    name: str
    raw_name: str
    prefix: str
    stripped_name: str
    parent: Optional[str]
    children: List[str]
    offset: np.ndarray
    end_site: Optional[np.ndarray]
    channels: List[str]
    channel_start: int
    line: int
    depth: int
    is_root: bool
    bone_rest: np.ndarray = field(default_factory=quat_identity)
    bone_frame: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    bone_direction_source: str = ""
    end_sites: List[np.ndarray] = field(default_factory=list)


@dataclass
class BvhNormalization:
    """Décisions de repère. Rien n'est converti sans déclaration explicite."""

    unit: str
    unit_to_meters: float
    unit_source: str
    up_axis: str
    up_axis_source: str
    handedness: str
    handedness_source: str
    frame_time: float
    fps: float
    fps_source: str
    applied: bool
    output_unit: str
    output_up_axis: str
    output_handedness: str
    scale_applied: float
    basis_applied: np.ndarray
    chain_length_file_units: float
    notes: List[str]
    chirality_measurement: str = ""
    pole: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    duplicate_policy: str = "path"


@dataclass
class BvhClip:
    joints: Dict[str, BvhJoint]
    roots: List[str]
    motion: np.ndarray
    normalization: BvhNormalization
    source_name: str
    warnings: List[str]
    world_positions: Dict[str, np.ndarray]
    world_rotations: Dict[str, np.ndarray]
    local_rotations: Dict[str, np.ndarray]
    rest_positions: Dict[str, np.ndarray]
    end_positions: Dict[str, np.ndarray]

    @property
    def n_frames(self) -> int:
        return int(self.motion.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.motion.shape[1]) if self.motion.ndim == 2 else 0

    @property
    def fps(self) -> float:
        return self.normalization.fps


def split_joint_name(raw_name: str) -> Tuple[str, str]:
    """Retourne (préfixe reconnu, nom sans préfixe). Le nom d'origine est conservé ailleurs."""
    for pattern, label in _PREFIX_RULES:
        match = pattern.match(raw_name)
        if not match:
            continue
        prefix = raw_name[: match.end()]
        stripped = raw_name[match.end() :]
        if stripped:
            return prefix if label != "namespace:" else prefix, stripped
    return "", raw_name


def parse_bvh(
    path: str,
    *,
    unit: Optional[str] = None,
    up_axis: Optional[str] = None,
    handedness: Optional[str] = None,
    apply_normalization: bool = False,
    primary_child: Optional[Dict[str, str]] = None,
    pole: Optional[Sequence[float]] = None,
    on_duplicate: str = "path",
) -> BvhClip:
    """Lit un fichier BVH. `apply_normalization` exige `unit` déclaré."""
    if not os.path.isfile(path):
        raise BvhParseError(
            0,
            f"fichier introuvable : {path}",
            "vérifier le chemin du BVH",
        )
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    return parse_bvh_text(
        text,
        source_name=path,
        unit=unit,
        up_axis=up_axis,
        handedness=handedness,
        apply_normalization=apply_normalization,
        primary_child=primary_child,
        pole=pole,
        on_duplicate=on_duplicate,
    )


def parse_bvh_text(
    text: str,
    *,
    source_name: str = "<mémoire>",
    unit: Optional[str] = None,
    up_axis: Optional[str] = None,
    handedness: Optional[str] = None,
    apply_normalization: bool = False,
    primary_child: Optional[Dict[str, str]] = None,
    pole: Optional[Sequence[float]] = None,
    on_duplicate: str = "path",
) -> BvhClip:
    declared_unit = _canonicalize_unit(unit) if unit is not None else None
    declared_up = _canonicalize_up(up_axis) if up_axis is not None else None
    declared_hand = _canonicalize_hand(handedness) if handedness is not None else None
    if apply_normalization and declared_unit is None:
        raise BvhParseError(
            0,
            "apply_normalization est vrai mais aucune unité n'est déclarée",
            "passer unit='m', unit='cm' ou unit='inch'",
        )

    tokens, comment_lines = _tokenize(text)
    if not tokens:
        raise BvhParseError(
            1,
            "fichier vide",
            "fournir un BVH avec HIERARCHY, au moins un ROOT, et MOTION",
        )

    parser = _Parser(tokens, comment_lines)
    joints, roots, motion, frame_time, warnings = parser.parse()
    joints, roots = _disambiguate_names(joints, roots, warnings, on_duplicate)
    chain = _longest_chain(joints, roots)
    up_guess, up_source = _infer_up_axis(joints)

    notes: List[str] = []
    if declared_unit is not None:
        unit_name = declared_unit
        unit_to_m = _UNIT_TO_METERS[unit_name]
        unit_src = f"déclaré : {unit_name}"
    else:
        unit_name = "non déclarée"
        unit_to_m = float("nan")
        unit_src = "non déclarée — laissée vide, non devinée, aucune conversion"
        warnings.append(_proportion_warning(chain, roots))

    if declared_up is not None:
        up_name = declared_up
        up_src = f"déclaré : {up_name}-up"
        if up_guess not in ("unknown", declared_up):
            notes.append(
                f"axe haut déclaré {declared_up}, différent de l'inférence ({up_source})"
            )
    else:
        up_name = up_guess
        up_src = up_source

    if declared_hand is not None:
        hand_name = declared_hand
        hand_src = (
            f"hypothèse explicite déclarée : {declared_hand}. "
            "La mesure épaules/hanches ne la remplace pas."
        )
    else:
        hand_name = "right"
        hand_src = (
            "hypothèse explicite : droitier. "
            "Un BVH Biovision est en général droitier. "
            "La mesure épaules/hanches ne remplace pas cette hypothèse."
        )

    scale = 1.0
    basis = np.eye(3)
    output_unit = "fichier"
    output_up = "fichier"
    output_hand = "fichier"
    applied = False

    if apply_normalization:
        applied = True
        scale = unit_to_m
        output_unit = "m"
        if declared_up is None:
            output_up = "fichier"
            notes.append("axe haut non déclaré : axes du fichier conservés")
            up_for_basis = "Y"
            convert_axes = False
        else:
            output_up = "Y"
            up_for_basis = declared_up
            convert_axes = True
        if declared_hand is None:
            output_hand = "right"
            hand_for_basis = "right"
            notes.append(
                "chiralité non déclarée : l'hypothèse droitière est conservée, "
                "la mesure du fichier ne la change pas"
            )
        else:
            output_hand = "right"
            hand_for_basis = declared_hand
        if convert_axes or hand_for_basis == "left":
            basis = _change_of_basis(
                up_for_basis if convert_axes else "Y",
                hand_for_basis,
            )
        _apply_basis(joints, motion, scale, basis, convert_rotations=not np.allclose(basis, np.eye(3)))

    pole_vector = _canonicalize_pole(pole)
    _assign_bone_frames(joints, warnings, primary_child, pole_vector)
    rest_pos, _, _ = _forward(joints, _zero_motion(joints, motion.shape[1]))
    rest_vectors = {name: rest_pos[name][0] for name in rest_pos}
    chirality_text, chirality_warning = _measure_chirality(joints, rest_vectors, hand_name)
    if chirality_warning:
        warnings.append(chirality_warning)
    world_pos, world_rot, local_rot = _forward(joints, motion)
    end_pos = _end_site_positions(joints, world_pos, world_rot)

    normalization = BvhNormalization(
        unit=unit_name,
        unit_to_meters=float(unit_to_m),
        unit_source=unit_src,
        up_axis=up_name,
        up_axis_source=up_src,
        handedness=hand_name,
        handedness_source=hand_src,
        frame_time=float(frame_time),
        fps=float(1.0 / frame_time),
        fps_source="Frame Time du fichier, sans valeur par défaut",
        applied=applied,
        output_unit=output_unit,
        output_up_axis=output_up,
        output_handedness=output_hand,
        scale_applied=float(scale if applied else 1.0),
        basis_applied=np.array(basis, dtype=np.float64, copy=True),
        chain_length_file_units=float(chain),
        notes=notes,
        chirality_measurement=chirality_text,
        pole=(float(pole_vector[0]), float(pole_vector[1]), float(pole_vector[2])),
        duplicate_policy=_duplicate_policy(on_duplicate),
    )
    return BvhClip(
        joints=joints,
        roots=roots,
        motion=motion,
        normalization=normalization,
        source_name=source_name,
        warnings=warnings,
        world_positions=world_pos,
        world_rotations=world_rot,
        local_rotations=local_rot,
        rest_positions=rest_vectors,
        end_positions=end_pos,
    )


def write_bvh(clip: BvhClip, path: str) -> None:
    """Réécrit le clip tel qu'il est en mémoire (inverse du parseur, pas l'export retarget)."""
    lines: List[str] = ["HIERARCHY"]

    def emit(name: str, indent: int) -> None:
        joint = clip.joints[name]
        pad = "\t" * indent
        kind = "ROOT" if joint.is_root else "JOINT"
        lines.append(f"{pad}{kind} {joint.name}")
        lines.append(f"{pad}{{")
        offset = joint.offset
        lines.append(
            f"{pad}\tOFFSET {offset[0]:.8f} {offset[1]:.8f} {offset[2]:.8f}"
        )
        channel_text = " ".join(joint.channels)
        lines.append(f"{pad}\tCHANNELS {len(joint.channels)} {channel_text}".rstrip())
        for child in joint.children:
            emit(child, indent + 1)
        sites = joint.end_sites or ([joint.end_site] if joint.end_site is not None else [])
        for end in sites:
            lines.append(f"{pad}\tEnd Site")
            lines.append(f"{pad}\t{{")
            lines.append(f"{pad}\t\tOFFSET {end[0]:.8f} {end[1]:.8f} {end[2]:.8f}")
            lines.append(f"{pad}\t}}")
        lines.append(f"{pad}}}")

    for root in clip.roots:
        emit(root, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {clip.n_frames}")
    lines.append(f"Frame Time: {clip.normalization.frame_time:.8f}")
    for row in clip.motion:
        lines.append(" ".join(f"{value:.8f}" for value in row))
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def format_report(clip: BvhClip) -> str:
    """Rapport lisible : rien d'important n'est omis."""
    joints = list(clip.joints.values())
    depths = [joint.depth for joint in joints] or [0]
    orders = sorted({_rotation_order(joint.channels) for joint in joints})
    prefixes: Dict[str, int] = {}
    for joint in joints:
        if joint.prefix:
            prefixes[joint.prefix] = prefixes.get(joint.prefix, 0) + 1
    pos_off_root = [
        joint.name
        for joint in joints
        if not joint.is_root and any(ch.endswith("position") for ch in joint.channels)
    ]
    no_rotation = [
        joint.name
        for joint in joints
        if not any(ch.endswith("rotation") for ch in joint.channels)
    ]
    renamed = [joint for joint in joints if joint.name != joint.raw_name]
    norm = clip.normalization
    quat_flips = _quaternion_sign_flips(clip.world_rotations)

    rows = [
        f"Fichier : {clip.source_name}",
        f"Joints : {len(joints)}    Racines : {len(clip.roots)} ({', '.join(clip.roots)})",
        f"Profondeur max : {max(depths)}    Frames : {clip.n_frames}    Canaux : {clip.n_channels}",
        f"Frame Time : {norm.frame_time:.8f} s    fps : {norm.fps:.6f}    ({norm.fps_source})",
        f"Ordres de rotation présents : {', '.join(orders) if orders else 'aucun'}",
        f"End Sites : {sum(joint.end_site is not None for joint in joints)}",
        f"Joints sans rotation : {', '.join(no_rotation) if no_rotation else 'aucun'}",
        f"Positions hors racine : {', '.join(pos_off_root) if pos_off_root else 'aucune'}",
        f"Préfixes retirés (nom de fichier conservé) : {_format_prefixes(prefixes)}",
        f"Noms désambiguïsés : {_format_renames(renamed)}",
        f"Longueur de chaîne (unités du fichier, avant conversion) : {norm.chain_length_file_units:.6f}",
        "Normalisation :",
        f"  unité : {norm.unit} — {norm.unit_source}",
        _unit_factor_line(norm),
        f"  axe haut : {norm.up_axis} — {norm.up_axis_source}",
        f"  chiralité : {norm.handedness} — {norm.handedness_source}",
        f"  mesure de chiralité : {norm.chirality_measurement}",
        f"  appliquée : {'oui' if norm.applied else 'non'}",
        f"  sortie : unité={norm.output_unit}, haut={norm.output_up_axis}, main={norm.output_handedness}",
        f"  échelle appliquée : {norm.scale_applied:.8f}",
        (
            "Repère d'os : +Y = direction (un enfant ; chaîne primaire si elle est nommée ; "
            "sinon barycentre). End Site nul : direction du parent. "
            f"Pôle secondaire {tuple(round(float(v), 6) for v in norm.pole)}. "
            "Distinct du repère joint BVH. Inverse = transposée."
        ),
        f"Doublons : politique « {norm.duplicate_policy} » (chemin hiérarchique, ou refus). Pas de suffixe #2.",
        f"Inversions de signe quaternion (double couverture, pas une erreur de pose) : {quat_flips}",
    ]
    if norm.notes:
        rows.append("Notes de normalisation :")
        rows.extend(f"  - {note}" for note in norm.notes)
    if clip.warnings:
        rows.append("Avertissements :")
        rows.extend(f"  - {warning}" for warning in clip.warnings)
    else:
        rows.append("Avertissements : aucun")
    rows.append("Hiérarchie :")
    for root in clip.roots:
        rows.extend(_tree_lines(clip, root))
    return "\n".join(rows) + "\n"


def check_euler_roundtrip(samples: int = 40, seed: int = 7) -> float:
    """Écart max de reconstruction matrice → Euler → matrice, en norme de Frobenius."""
    rng = np.random.default_rng(seed)
    orders = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")
    worst = 0.0
    for order in orders:
        for _ in range(samples):
            angles = rng.uniform(-120.0, 120.0, size=3)
            matrix = _euler_matrix(order, angles)
            recovered = np.degrees(_euler_angles(matrix, order))
            rebuilt = _euler_matrix(order, recovered)
            worst = max(worst, float(np.linalg.norm(matrix - rebuilt)))
        for mid in (90.0, -90.0):
            angles = np.array([rng.uniform(-120, 120), mid, rng.uniform(-120, 120)])
            matrix = _euler_matrix(order, angles)
            recovered = np.degrees(_euler_angles(matrix, order))
            rebuilt = _euler_matrix(order, recovered)
            worst = max(worst, float(np.linalg.norm(matrix - rebuilt)))
    return worst


def _tokenize(text: str) -> Tuple[List[Tuple[int, str]], int]:
    if text.startswith("\ufeff"):
        text = text[1:]
    tokens: List[Tuple[int, str]] = []
    comment_lines = 0
    for line_no, raw in enumerate(text.splitlines(), 1):
        if "#" in raw:
            comment_lines += 1
            raw = raw.split("#", 1)[0]
        i = 0
        length = len(raw)
        while i < length:
            char = raw[i]
            if char.isspace():
                i += 1
                continue
            if char in "{}":
                tokens.append((line_no, char))
                i += 1
                continue
            j = i + 1
            while j < length and (not raw[j].isspace()) and raw[j] not in "{}":
                j += 1
            tokens.append((line_no, raw[i:j]))
            i = j
    return tokens, comment_lines


class _Parser:
    def __init__(self, tokens: Sequence[Tuple[int, str]], comment_lines: int):
        self.tokens = list(tokens)
        self.n = len(self.tokens)
        self.i = 0
        self.comment_lines = comment_lines
        self.warnings: List[str] = []
        self.seen_names: Dict[str, int] = {}
        self.issued_names: set = set()
        self.channel_cursor = 0

    def parse(self) -> Tuple[Dict[str, BvhJoint], List[str], np.ndarray, float, List[str]]:
        if self.comment_lines:
            self.warnings.append(
                f"{self.comment_lines} ligne(s) contiennent un commentaire #, ignoré jusqu'à la fin de la ligne"
            )
        self._skip_hierarchy_keyword()
        joints: Dict[str, BvhJoint] = {}
        roots: List[str] = []
        if self._peek_keyword() != "ROOT":
            line = self._line()
            raise BvhParseError(
                line,
                "aucun ROOT au premier niveau de la hiérarchie",
                "ajouter une ligne ROOT <nom> suivie d'un bloc { OFFSET ... }",
            )
        while self._peek_keyword() == "ROOT":
            self._pop()
            joint = self._parse_joint(joints, parent=None, is_root=True, depth=0)
            roots.append(joint.name)
        if self._peek_keyword() != "MOTION":
            line = self._line()
            raise BvhParseError(
                line,
                "fichier tronqué : section MOTION absente ou accolade non fermée",
                "fermer chaque bloc { } puis ajouter MOTION, Frames et Frame Time",
            )
        self._pop()
        frame_count, frame_time = self._parse_motion_header()
        motion = self._parse_motion_values(frame_count)
        return joints, roots, motion, frame_time, self.warnings

    def _parse_joint(
        self,
        joints: Dict[str, BvhJoint],
        parent: Optional[str],
        is_root: bool,
        depth: int,
    ) -> BvhJoint:
        name_line, raw_name = self._read_joint_name()
        unique = self._unique_name(raw_name, name_line)
        brace_line, brace = self._pop()
        if brace != "{":
            raise BvhParseError(
                brace_line,
                f"accolade ouvrante attendue après le joint « {raw_name} », reçu « {brace} »",
                "placer { juste après le nom du joint",
            )
        offset: Optional[np.ndarray] = None
        channels: Optional[List[str]] = None
        end_site: Optional[np.ndarray] = None
        end_extra_line = name_line
        prefix, stripped = split_joint_name(raw_name)
        joint = BvhJoint(
            name=unique,
            raw_name=raw_name,
            prefix=prefix,
            stripped_name=stripped,
            parent=parent,
            children=[],
            offset=np.zeros(3, dtype=np.float64),
            end_site=None,
            channels=[],
            channel_start=self.channel_cursor,
            line=name_line,
            depth=depth,
            is_root=is_root,
        )
        joints[unique] = joint

        while True:
            peeked = self._peek()
            if peeked is None:
                raise BvhParseError(
                    name_line,
                    f"fichier tronqué : le bloc du joint « {raw_name} » n'est pas fermé",
                    f"ajouter }} pour fermer « {raw_name} » (ouvert ligne {name_line})",
                )
            line, token = peeked
            keyword = _keyword(token)
            if token == "}":
                self._pop()
                break
            if keyword == "OFFSET":
                if offset is not None:
                    raise BvhParseError(
                        line,
                        f"OFFSET en double sur « {raw_name} »",
                        "ne garder qu'un seul OFFSET par joint",
                    )
                self._pop()
                offset = self._read_vec3(raw_name, "OFFSET")
                joint.offset = offset
            elif keyword == "CHANNELS":
                if channels is not None:
                    raise BvhParseError(
                        line,
                        f"CHANNELS en double sur « {raw_name} »",
                        "ne garder qu'une seule ligne CHANNELS par joint",
                    )
                self._pop()
                channels = self._read_channels(raw_name)
                joint.channels = channels
                joint.channel_start = self.channel_cursor
                self.channel_cursor += len(channels)
                self._warn_if_interleaved(raw_name, channels, line)
            elif keyword == "JOINT":
                self._pop()
                child = self._parse_joint(joints, parent=unique, is_root=False, depth=depth + 1)
                joint.children.append(child.name)
            elif keyword == "ROOT":
                raise BvhParseError(
                    line,
                    f"ROOT imbriqué dans « {raw_name} »",
                    "fermer le ROOT courant et placer le ROOT suivant au premier niveau, après son bloc",
                )
            elif keyword in ("END", "ENDSITE"):
                site = self._read_end_site(raw_name, combined=(keyword == "ENDSITE"))
                if not joint.end_sites:
                    end_extra_line = line
                elif len(joint.end_sites) == 1:
                    end_extra_line = line
                joint.end_sites.append(site)
                joint.end_site = site
                end_site = site
            else:
                raise BvhParseError(
                    line,
                    f"mot inattendu « {token} » dans le joint « {raw_name} »",
                    "attendu : OFFSET, CHANNELS, JOINT, End Site, ou }",
                )

        if len(joint.end_sites) > 1:
            joint.end_site = np.mean(np.stack(joint.end_sites, axis=0), axis=0)
            self.warnings.append(
                f"Ligne {end_extra_line} : {len(joint.end_sites)} End Sites sur « {raw_name} ». "
                "Cause : un joint n'a normalement qu'un segment terminal. "
                "Correction proposée : la direction de repos est le barycentre de ces sites. "
                "Ils sont conservés, le fichier n'est pas refusé."
            )
        if offset is None:
            raise BvhParseError(
                name_line,
                f"OFFSET manquant sur « {raw_name} »",
                "ajouter OFFSET x y z dans le bloc du joint",
            )
        if channels is None:
            self.warnings.append(
                f"Ligne {name_line} : joint « {raw_name} » sans CHANNELS, traité comme 0 canal"
            )
            joint.channels = []
            joint.channel_start = self.channel_cursor
        return joint

    def _read_joint_name(self) -> Tuple[int, str]:
        """Le nom va jusqu'à l'accolade. « End Site » n'est le bloc terminal qu'une fois le joint ouvert."""
        name_line, first = self._pop()
        if first in "{}":
            raise BvhParseError(
                name_line,
                "nom de joint manquant",
                "écrire JOINT <nom> ou ROOT <nom> avant l'accolade",
            )
        parts = [first]
        while True:
            peeked = self._peek()
            if peeked is None:
                break
            _line, token = peeked
            if token in "{}" or self._name_token_stops(token):
                break
            self._pop()
            parts.append(token)
        return name_line, " ".join(parts)

    def _name_token_stops(self, token: str) -> bool:
        """Un mot réservé suivi de { fait partie du nom. OFFSET ne coupe que s'il a trois nombres."""
        key = _keyword(token)
        if key not in _NAME_STOP:
            return False
        if self.i + 1 >= self.n:
            return True
        if self.tokens[self.i + 1][1] == "{":
            return False
        if key == "OFFSET":
            return self._three_numbers_ahead()
        if key == "CHANNELS":
            nxt = self.tokens[self.i + 1][1]
            return bool(re.fullmatch(r"[+-]?\d+", nxt))
        return True

    def _three_numbers_ahead(self) -> bool:
        if self.i + 3 >= self.n:
            return False
        return all(_looks_number(self.tokens[self.i + 1 + step][1]) for step in range(3))

    def _read_channels(self, joint_name: str) -> List[str]:
        count_line, count_tok = self._pop()
        if not re.fullmatch(r"[+-]?\d+", count_tok):
            raise BvhParseError(
                count_line,
                f"nombre de canaux illisible « {count_tok} » sur « {joint_name} »",
                "écrire CHANNELS <entier> puis la liste des canaux",
            )
        count = int(count_tok)
        if count < 0:
            raise BvhParseError(
                count_line,
                f"nombre de canaux négatif sur « {joint_name} »",
                "utiliser 0 si le joint n'a ni position ni rotation",
            )
        names: List[str] = []
        seen = set()
        for _ in range(count):
            line, token = self._pop()
            canon = _CHANNEL_CANON.get(token.lower())
            if canon is None:
                raise BvhParseError(
                    line,
                    f"canal inconnu « {token} » sur « {joint_name} »",
                    "canaux admis : Xposition Yposition Zposition Xrotation Yrotation Zrotation",
                )
            if canon in seen:
                raise BvhParseError(
                    line,
                    f"canal « {canon} » répété sur « {joint_name} »",
                    "chaque axe de position ou de rotation ne doit apparaître qu'une fois",
                )
            seen.add(canon)
            names.append(canon)
        return names

    def _read_end_site(self, joint_name: str, combined: bool = False) -> np.ndarray:
        end_line, _ = self._pop()
        if not combined:
            peeked = self._peek()
            if peeked is not None and _keyword(peeked[1]) == "SITE":
                self._pop()
        brace = self._peek()
        if brace is None or brace[1] != "{":
            line = end_line if brace is None else brace[0]
            raise BvhParseError(
                line,
                f"End Site de « {joint_name} » sans bloc",
                "écrire End Site { OFFSET x y z }",
            )
        self._pop()
        inner = self._peek()
        if inner is None or _keyword(inner[1]) != "OFFSET":
            line = end_line if inner is None else inner[0]
            raise BvhParseError(
                line,
                f"End Site de « {joint_name} » sans OFFSET",
                "écrire OFFSET x y z dans le End Site",
            )
        self._pop()
        offset = self._read_vec3(joint_name, "End Site")
        close = self._peek()
        if close is None or close[1] != "}":
            line = end_line if close is None else close[0]
            raise BvhParseError(
                line,
                f"End Site de « {joint_name} » non fermé",
                "ajouter } après l'OFFSET du End Site",
            )
        self._pop()
        return offset

    def _read_vec3(self, joint_name: str, what: str) -> np.ndarray:
        values = [self._read_float(f"{what} de « {joint_name} »") for _ in range(3)]
        return np.array(values, dtype=np.float64)

    def _read_float(self, what: str) -> float:
        line, token = self._pop()
        try:
            value = float(token)
        except ValueError:
            hint = "utiliser un point décimal, pas une virgule" if "," in token else (
                f"écrire un nombre pour {what}"
            )
            raise BvhParseError(line, f"nombre attendu pour {what}, reçu « {token} »", hint) from None
        if not math.isfinite(value):
            raise BvhParseError(
                line,
                f"valeur non finie pour {what}",
                "remplacer NaN ou Inf par un nombre fini",
            )
        return value

    def _parse_motion_header(self) -> Tuple[int, float]:
        frames: Optional[int] = None
        frame_time: Optional[float] = None
        seen_order: List[str] = []
        while frames is None or frame_time is None:
            peeked = self._peek()
            if peeked is None:
                raise BvhParseError(
                    self._line(),
                    "fichier tronqué : Frames ou Frame Time manquant",
                    "ajouter les deux lignes « Frames: N » et « Frame Time: S » avant les données",
                )
            line, token = peeked
            keyword = _keyword(token)
            if _looks_number(token):
                raise BvhParseError(
                    line,
                    f"données de frame « {token} » avant la fin de l'en-tête MOTION",
                    "placer Frames et Frame Time avant la première ligne de valeurs",
                )
            if keyword == "FRAMES":
                self._pop()
                frames = self._read_frame_count()
                seen_order.append("Frames")
            elif keyword == "FRAME":
                self._pop()
                nxt = self._peek()
                if nxt is None or _keyword(nxt[1]) != "TIME":
                    raise BvhParseError(
                        line,
                        "Frame sans Time",
                        "écrire « Frame Time: <secondes> », avec un temps strictement positif",
                    )
                self._pop()
                frame_time = self._read_frame_time()
                seen_order.append("Frame Time")
            else:
                raise BvhParseError(
                    line,
                    f"en-tête MOTION inattendu « {token} »",
                    "écrire « Frames: N » puis « Frame Time: S »",
                )
        if seen_order != ["Frames", "Frame Time"]:
            self.warnings.append(
                "Frames et Frame Time ne sont pas dans l'ordre habituel ; les deux ont été lus"
            )
        assert frames is not None and frame_time is not None
        return frames, frame_time

    def _read_frame_count(self) -> int:
        self._skip_colon()
        line, token = self._pop()
        if not re.fullmatch(r"[+]?\d+", token):
            raise BvhParseError(
                line,
                f"nombre de frames illisible « {token} »",
                "écrire un entier positif ou nul, par exemple Frames: 120",
            )
        return int(token)

    def _read_frame_time(self) -> float:
        self._skip_colon()
        value = self._read_float("Frame Time")
        if value <= 0.0:
            raise BvhParseError(
                self._previous_line(),
                f"Frame Time invalide ({value})",
                "écrire une durée positive en secondes, par exemple 0.033333. Aucun fps par défaut n'est substitué",
            )
        return value

    def _parse_motion_values(self, frame_count: int) -> np.ndarray:
        n_channels = self.channel_cursor
        if frame_count == 0:
            if self.i < self.n:
                line = self.tokens[self.i][0]
                raise BvhParseError(
                    line,
                    "des valeurs sont présentes alors que Frames vaut 0",
                    "mettre Frames au nombre réel de lignes de données, ou supprimer ces lignes",
                )
            return np.zeros((0, n_channels), dtype=np.float64)
        if n_channels == 0:
            if self.i < self.n:
                line = self.tokens[self.i][0]
                raise BvhParseError(
                    line,
                    "des valeurs de mouvement sont présentes alors qu'aucun joint n'a de canal",
                    "retirer la section numérique ou ajouter des CHANNELS aux joints",
                )
            return np.zeros((frame_count, 0), dtype=np.float64)

        groups: List[Tuple[int, List[str]]] = []
        while self.i < self.n:
            line = self.tokens[self.i][0]
            bucket: List[str] = []
            while self.i < self.n and self.tokens[self.i][0] == line:
                bucket.append(self.tokens[self.i][1])
                self.i += 1
            groups.append((line, bucket))

        if len(groups) < frame_count:
            last = groups[-1][0] if groups else self._line()
            raise BvhParseError(
                last,
                f"fichier tronqué : {len(groups)} frame(s) lue(s), {frame_count} déclarée(s)",
                f"compléter jusqu'à {frame_count} lignes, ou corriger Frames",
            )
        if len(groups) > frame_count:
            extra_line = groups[frame_count][0]
            raise BvhParseError(
                extra_line,
                f"{len(groups)} lignes de mouvement pour {frame_count} frame(s) déclarée(s)",
                "aligner Frames sur le nombre de lignes, une frame par ligne",
            )

        rows: List[List[float]] = []
        for line, bucket in groups:
            if len(bucket) != n_channels:
                raise BvhParseError(
                    line,
                    f"la frame contient {len(bucket)} valeurs, {n_channels} attendues (somme des CHANNELS)",
                    f"mettre exactement {n_channels} nombres sur cette ligne, dans l'ordre des canaux du fichier",
                )
            values: List[float] = []
            for token in bucket:
                try:
                    value = float(token)
                except ValueError:
                    hint = "utiliser un point décimal" if "," in token else "remplacer ce jeton par un nombre"
                    raise BvhParseError(
                        line,
                        f"valeur de frame illisible « {token} »",
                        hint,
                    ) from None
                if not math.isfinite(value):
                    raise BvhParseError(
                        line,
                        "valeur de frame non finie",
                        "remplacer NaN ou Inf par un nombre fini",
                    )
                values.append(value)
            rows.append(values)
        return np.asarray(rows, dtype=np.float64)

    def _unique_name(self, raw_name: str, line: int) -> str:
        """Clé provisoire. L'identité publique est décidée ensuite : chemin, ou refus."""
        count = self.seen_names.get(raw_name, 0) + 1
        self.seen_names[raw_name] = count
        if count == 1 and raw_name not in self.issued_names:
            self.issued_names.add(raw_name)
            return raw_name
        index = count
        while True:
            unique = f"{raw_name}\0{index}"
            if unique not in self.issued_names:
                break
            index += 1
        self.issued_names.add(unique)
        return unique

    def _warn_if_interleaved(self, joint_name: str, channels: Sequence[str], line: int) -> None:
        kinds = ["p" if ch.endswith("position") else "r" for ch in channels]
        if "p" in kinds and "r" in kinds and kinds != sorted(kinds):
            self.warnings.append(
                f"Ligne {line} : positions et rotations entrelacées sur « {joint_name} ». "
                "Les positions sont sommées par axe, les rotations composées dans leur ordre d'apparition"
            )

    def _skip_hierarchy_keyword(self) -> None:
        keyword = self._peek_keyword()
        if keyword == "HIERARCHY":
            self._pop()
            return
        if keyword == "ROOT":
            line = self._line()
            self.warnings.append(
                f"Ligne {line} : mot-clé HIERARCHY absent, lecture à partir du premier ROOT"
            )
            return
        raise BvhParseError(
            self._line(),
            "début de fichier illisible : HIERARCHY ou ROOT attendu",
            "commencer le fichier par HIERARCHY puis ROOT <nom>",
        )

    def _skip_colon(self) -> None:
        peeked = self._peek()
        if peeked is not None and peeked[1] == ":":
            self._pop()

    def _peek(self) -> Optional[Tuple[int, str]]:
        if self.i >= self.n:
            return None
        return self.tokens[self.i]

    def _peek_keyword(self) -> Optional[str]:
        peeked = self._peek()
        if peeked is None:
            return None
        return _keyword(peeked[1])

    def _pop(self) -> Tuple[int, str]:
        peeked = self._peek()
        if peeked is None:
            raise BvhParseError(
                self._line(),
                "fichier tronqué : jeton attendu en fin de fichier",
                "compléter la hiérarchie ou la section MOTION",
            )
        self.i += 1
        return peeked

    def _line(self) -> int:
        if not self.tokens:
            return 1
        index = min(self.i, self.n - 1)
        return self.tokens[index][0]

    def _previous_line(self) -> int:
        if self.i <= 0:
            return 1
        return self.tokens[self.i - 1][0]


_NAME_STOP = {
    "OFFSET",
    "CHANNELS",
    "JOINT",
    "ROOT",
    "MOTION",
    "HIERARCHY",
    "FRAMES",
    "FRAME",
}


def _keyword(token: str) -> str:
    return token.upper().rstrip(":")


def _looks_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _canonicalize_unit(unit: str) -> str:
    key = unit.strip().lower()
    if key not in _UNIT_ALIAS:
        raise BvhParseError(
            0,
            f"unité déclarée inconnue « {unit} »",
            "utiliser m, cm ou inch",
        )
    return _UNIT_ALIAS[key]


def _canonicalize_up(up_axis: str) -> str:
    key = up_axis.strip().upper().replace("-UP", "")
    if key not in ("Y", "Z"):
        raise BvhParseError(
            0,
            f"axe haut déclaré inconnu « {up_axis} »",
            "utiliser Y ou Z",
        )
    return key


def _canonicalize_hand(handedness: str) -> str:
    key = handedness.strip().lower()
    aliases = {
        "right": "right",
        "r": "right",
        "droite": "right",
        "left": "left",
        "l": "left",
        "gauche": "left",
    }
    if key not in aliases:
        raise BvhParseError(
            0,
            f"chiralité déclarée inconnue « {handedness} »",
            "utiliser right ou left",
        )
    return aliases[key]


def _infer_up_axis(joints: Dict[str, BvhJoint]) -> Tuple[str, str]:
    weights = np.zeros(3, dtype=np.float64)
    for joint in joints.values():
        weight = 0.0
        lowered = joint.stripped_name.lower()
        for hint, hint_weight in _UP_HINTS.items():
            if hint in lowered:
                weight = max(weight, hint_weight)
        if weight > 0.0:
            weights += weight * np.abs(joint.offset)
    total = float(weights.sum())
    if total <= 1e-8:
        return "unknown", "aucun nom d'axe haut reconnaissable. Déclarer up_axis=Y ou Z"
    index = int(np.argmax(weights))
    share = float(weights[index] / total)
    axis = "XYZ"[index]
    if share < 0.6 or axis == "X":
        return (
            "unknown",
            f"votes d'axe insuffisants ou latéraux (part {share:.2f} sur {axis}). Déclarer up_axis",
        )
    return axis, f"inféré {axis}-up à partir des OFFSET nommés (part {share:.2f}). Non appliqué tant que non déclaré"


def _longest_chain(joints: Dict[str, BvhJoint], roots: Sequence[str]) -> float:
    def walk(name: str) -> float:
        joint = joints[name]
        lengths = [np.linalg.norm(joints[child].offset) + walk(child) for child in joint.children]
        if joint.end_site is not None:
            lengths.append(float(np.linalg.norm(joint.end_site)))
        return max(lengths) if lengths else 0.0

    if not roots:
        return 0.0
    return float(max(walk(root) for root in roots))


def _change_of_basis(up_axis: str, handedness: str) -> np.ndarray:
    """Matrice qui envoie le repère fichier vers Y-up droitier."""
    if up_axis == "Y":
        basis = np.eye(3, dtype=np.float64)
    elif up_axis == "Z":
        angle = math.radians(-90.0)
        cosine, sine = math.cos(angle), math.sin(angle)
        basis = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cosine, -sine],
                [0.0, sine, cosine],
            ],
            dtype=np.float64,
        )
    else:
        raise BvhParseError(0, f"axe haut non géré « {up_axis} »", "utiliser Y ou Z")
    if handedness == "left":
        mirror = np.diag([-1.0, 1.0, 1.0])
        basis = mirror @ basis
    return basis


def _apply_basis(
    joints: Dict[str, BvhJoint],
    motion: np.ndarray,
    scale: float,
    basis: np.ndarray,
    *,
    convert_rotations: bool,
) -> None:
    for joint in joints.values():
        joint.offset = scale * (basis @ joint.offset)
        if joint.end_sites:
            joint.end_sites = [scale * (basis @ site) for site in joint.end_sites]
            stacked = np.stack(joint.end_sites, axis=0)
            joint.end_site = stacked[0] if len(joint.end_sites) == 1 else np.mean(stacked, axis=0)
        elif joint.end_site is not None:
            joint.end_site = scale * (basis @ joint.end_site)
        _scale_position_channels(joint, motion, scale, basis)
        if convert_rotations:
            _convert_rotation_channels(joint, motion, basis)


def _scale_position_channels(
    joint: BvhJoint,
    motion: np.ndarray,
    scale: float,
    basis: np.ndarray,
) -> None:
    columns = [
        (ch[0], joint.channel_start + index)
        for index, ch in enumerate(joint.channels)
        if ch.endswith("position")
    ]
    if not columns or motion.shape[0] == 0:
        return
    local = np.zeros((motion.shape[0], 3), dtype=np.float64)
    for axis, column in columns:
        local[:, _AXIS_INDEX[axis]] = motion[:, column]
    converted = scale * (basis @ local.T).T
    for axis, column in columns:
        motion[:, column] = converted[:, _AXIS_INDEX[axis]]


def _convert_rotation_channels(joint: BvhJoint, motion: np.ndarray, basis: np.ndarray) -> None:
    rot_channels = [ch for ch in joint.channels if ch.endswith("rotation")]
    if not rot_channels or motion.shape[0] == 0:
        return
    order = "".join(ch[0] for ch in rot_channels)
    if len(order) != 3 or len(set(order)) != 3:
        nonzero = _rotation_nonzero(joint, motion)
        if nonzero:
            raise BvhParseError(
                joint.line,
                f"« {joint.name} » n'a pas 3 axes de rotation ({order or 'aucun'}), "
                "impossible de réécrire son orientation après changement de repère",
                "déclarer l'axe haut du fichier sans conversion, ou donner au joint les canaux Xrotation Yrotation Zrotation",
            )
        return
    for frame in range(motion.shape[0]):
        angles = _channel_angles(joint, motion[frame])
        matrix = _euler_matrix(order, angles)
        converted = basis @ matrix @ basis.T
        rewritten = np.degrees(_euler_angles(converted, order))
        _write_angles(joint, motion[frame], rewritten)


def _rotation_nonzero(joint: BvhJoint, motion: np.ndarray) -> bool:
    for index, channel in enumerate(joint.channels):
        if channel.endswith("rotation") and np.any(np.abs(motion[:, joint.channel_start + index]) > 1e-8):
            return True
    return False


def _channel_angles(joint: BvhJoint, row: np.ndarray) -> np.ndarray:
    values = [
        row[joint.channel_start + index]
        for index, channel in enumerate(joint.channels)
        if channel.endswith("rotation")
    ]
    return np.asarray(values, dtype=np.float64)


def _write_angles(joint: BvhJoint, row: np.ndarray, angles: np.ndarray) -> None:
    cursor = 0
    for index, channel in enumerate(joint.channels):
        if channel.endswith("rotation"):
            row[joint.channel_start + index] = angles[cursor]
            cursor += 1


_DIR_EPS = 1e-6


def _unit_factor_line(norm: BvhNormalization) -> str:
    if norm.unit == "non déclarée" or not math.isfinite(norm.unit_to_meters):
        return "  facteur vers mètres : non applicable (unité non déclarée, aucune conversion)"
    return f"  facteur vers mètres : {norm.unit_to_meters:.8f} (déclaré, non déduit de la chaîne)"


def _proportion_warning(chain_length: float, roots: Sequence[str]) -> str:
    root = ", ".join(roots) if roots else "(aucune racine)"
    return (
        f"Ligne 0 : unité non déclarée. "
        f"Cause : la chaîne la plus longue mesure {chain_length:.4f} unités fichier "
        f"(racine {root}). Une stature adulte vaut environ 1,5–2 m, 150–200 cm ou 60–80 in, "
        f"et cette chaîne n'est pas forcément une stature. "
        f"Correction proposée : déclarer unit='m', unit='cm' ou unit='inch'. "
        f"Aucune conversion n'a été faite."
    )


def _duplicate_policy(on_duplicate: str) -> str:
    policy = (on_duplicate or "path").strip().lower()
    if policy not in ("path", "error"):
        raise BvhParseError(
            0,
            f"on_duplicate inconnu « {on_duplicate} »",
            "utiliser 'path' (le chemin hiérarchique est l'identité) ou 'error' (refus)",
        )
    return policy


def _disambiguate_names(
    joints: Dict[str, BvhJoint],
    roots: List[str],
    warnings: List[str],
    on_duplicate: str,
) -> Tuple[Dict[str, BvhJoint], List[str]]:
    """Les noms uniques restent bruts. Un doublon prend le chemin, ou le parseur refuse."""
    policy = _duplicate_policy(on_duplicate)
    ordered = list(joints.values())
    by_raw: Dict[str, List[BvhJoint]] = {}
    for joint in ordered:
        by_raw.setdefault(joint.raw_name, []).append(joint)
    duplicated = {raw for raw, group in by_raw.items() if len(group) > 1}
    if duplicated and policy == "error":
        raw = sorted(duplicated)[0]
        group = by_raw[raw]
        raise BvhParseError(
            group[1].line,
            f"nom de joint « {raw} » déjà utilisé ligne {group[0].line}",
            "renommer ce joint dans le BVH, ou relancer avec on_duplicate='path' "
            "pour que le chemin hiérarchique soit l'identité",
        )

    by_provisional = {joint.name: joint for joint in ordered}

    def raw_path(joint: BvhJoint) -> str:
        parts: List[str] = []
        current: Optional[BvhJoint] = joint
        guard = 0
        while current is not None:
            parts.append(current.raw_name)
            if current.parent is None:
                break
            parent = by_provisional.get(current.parent)
            if parent is None:
                raise BvhParseError(
                    joint.line,
                    f"parent introuvable pour « {joint.raw_name} »",
                    "la hiérarchie est incohérente, relire le BVH",
                )
            current = parent
            guard += 1
            if guard > len(ordered) + 1:
                raise BvhParseError(
                    joint.line,
                    "cycle dans les parents",
                    "corriger la hiérarchie du BVH",
                )
        parts.reverse()
        return "/".join(parts)

    renamed: Dict[str, str] = {}
    for joint in ordered:
        if joint.raw_name in duplicated:
            renamed[joint.name] = raw_path(joint)
        else:
            renamed[joint.name] = joint.raw_name

    seen: Dict[str, BvhJoint] = {}
    for joint in ordered:
        path = renamed[joint.name]
        other = seen.get(path)
        if other is not None:
            raise BvhParseError(
                joint.line,
                f"deux joints partagent le chemin hiérarchique « {path} » (l'autre est ligne {other.line})",
                "renommer un des joints dans le fichier : le chemin ne les distingue pas, "
                "et un suffixe #2 n'est pas une identité",
            )
        seen[path] = joint

    for joint in ordered:
        path = renamed[joint.name]
        if path == joint.raw_name:
            continue
        others = [str(other.line) for other in by_raw[joint.raw_name] if other is not joint]
        warnings.append(
            f"Ligne {joint.line} : nom « {joint.raw_name} » en double (lignes {', '.join(others)}). "
            f"Cause : ce nom n'identifie pas un seul joint. "
            f"Correction proposée : l'identité retenue est le chemin « {path} ». "
            f"Pas de renommage silencieux en {joint.raw_name}#2."
        )

    for joint in ordered:
        joint.name = renamed[joint.name]
    for joint in ordered:
        if joint.parent is not None:
            joint.parent = renamed[joint.parent]
        joint.children = [renamed[child] for child in joint.children]
    new_joints = {joint.name: joint for joint in ordered}
    new_roots = [renamed[root] for root in roots]
    return new_joints, new_roots


def _canonicalize_pole(pole: Optional[Sequence[float]]) -> np.ndarray:
    if pole is None:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    arr = np.asarray(list(pole), dtype=np.float64).reshape(-1)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or float(np.linalg.norm(arr)) < _DIR_EPS:
        raise BvhParseError(
            0,
            "pôle secondaire nul, incomplet ou non fini",
            "passer pole=(x, y, z) non nul. Le défaut est (0, 0, 1), dans le repère du joint parent",
        )
    return arr


def _as_unit(vector: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < _DIR_EPS:
        return None
    return arr / norm


def _assign_bone_frames(
    joints: Dict[str, BvhJoint],
    warnings: List[str],
    primary_child: Optional[Dict[str, str]],
    pole: np.ndarray,
) -> None:
    if primary_child is not None and not isinstance(primary_child, dict):
        raise BvhParseError(
            0,
            "primary_child doit être un dictionnaire parent → enfant",
            "exemple : primary_child={'Chest': 'Spine'}",
        )
    if primary_child:
        known = set()
        for joint in joints.values():
            known.add(joint.name)
            known.add(joint.raw_name)
        for key in primary_child:
            if key not in known:
                raise BvhParseError(
                    0,
                    f"chaîne primaire : aucun joint « {key} »",
                    "utiliser le nom du parent, ou son chemin hiérarchique s'il est en double",
                )
    for joint in joints.values():
        direction, source = _bone_direction(joint, joints, primary_child, warnings)
        joint.bone_direction_source = source
        if direction is None:
            joint.bone_frame = np.eye(3, dtype=np.float64)
            joint.bone_rest = quat_identity()
            continue
        joint.bone_frame = _frame_from_direction(direction, pole, warnings, joint)
        joint.bone_rest = matrix_to_quaternion(joint.bone_frame)


def _primary_token(joint: BvhJoint, primary_child: Optional[Dict[str, str]]) -> Optional[str]:
    if not primary_child:
        return None
    if joint.name in primary_child:
        return str(primary_child[joint.name])
    if joint.raw_name in primary_child and joint.raw_name != joint.name:
        return str(primary_child[joint.raw_name])
    if joint.raw_name in primary_child:
        return str(primary_child[joint.raw_name])
    return None


def _bone_direction(
    joint: BvhJoint,
    joints: Dict[str, BvhJoint],
    primary_child: Optional[Dict[str, str]],
    warnings: List[str],
) -> Tuple[Optional[np.ndarray], str]:
    children = [joints[name] for name in joint.children]
    if len(children) == 1:
        direction = _as_unit(children[0].offset)
        if direction is not None:
            return direction, f"enfant unique « {children[0].name} »"
        return _inherit_or_identity(
            joint,
            joints,
            warnings,
            f"l'unique enfant « {children[0].name} » a un OFFSET nul",
        )
    if len(children) > 1:
        chosen = _primary_token(joint, primary_child)
        if chosen is not None:
            hits = [child for child in children if chosen in (child.name, child.raw_name)]
            listed = ", ".join(child.name for child in children)
            if len(hits) != 1:
                cause = (
                    f"chaîne primaire « {chosen} » ambiguë parmi les enfants de « {joint.name} »"
                    if len(hits) > 1
                    else f"chaîne primaire « {chosen} » introuvable parmi les enfants de « {joint.name} »"
                )
                raise BvhParseError(
                    joint.line,
                    cause,
                    f"indiquer un enfant direct parmi : {listed}",
                )
            direction = _as_unit(hits[0].offset)
            if direction is None:
                warnings.append(
                    f"Ligne {joint.line} : la chaîne primaire « {hits[0].name} » a un OFFSET nul. "
                    f"Cause : cet enfant ne donne pas de direction. "
                    f"Correction proposée : corriger son OFFSET, ou désigner un autre enfant. "
                    f"Repère d'os = identité. Le plus long enfant n'est pas choisi à sa place."
                )
                return None, "chaîne primaire d'OFFSET nul, identité"
            return direction, f"chaîne primaire « {hits[0].name} »"
        mean = np.mean([child.offset for child in children], axis=0)
        direction = _as_unit(mean)
        if direction is None:
            warnings.append(
                f"Ligne {joint.line} : barycentre des enfants de « {joint.name} » nul. "
                f"Cause : les OFFSET des enfants s'annulent et aucune chaîne primaire n'est désignée. "
                f"Correction proposée : passer primary_child avec l'enfant qui porte la chaîne. "
                f"Le plus long enfant n'est pas utilisé. Repère d'os = identité."
            )
            return None, "barycentre nul, identité"
        return direction, "barycentre des enfants"
    if len(joint.end_sites) > 1:
        direction = _as_unit(np.mean(np.stack(joint.end_sites, axis=0), axis=0))
        if direction is not None:
            return direction, f"barycentre de {len(joint.end_sites)} End Sites"
        return _inherit_or_identity(
            joint,
            joints,
            warnings,
            f"barycentre des End Sites de « {joint.name} » nul",
        )
    if joint.end_site is not None:
        direction = _as_unit(joint.end_site)
        if direction is not None:
            return direction, "End Site"
        return _inherit_or_identity(
            joint,
            joints,
            warnings,
            f"End Site de « {joint.name} » nul (OFFSET de norme < {_DIR_EPS:g})",
        )
    return _inherit_or_identity(
        joint,
        joints,
        warnings,
        f"« {joint.name} » n'a ni enfant ni End Site",
    )


def _inherit_or_identity(
    joint: BvhJoint,
    joints: Dict[str, BvhJoint],
    warnings: List[str],
    cause: str,
) -> Tuple[Optional[np.ndarray], str]:
    if joint.parent is None:
        warnings.append(
            f"Ligne {joint.line} : {cause}. "
            f"Cause : aucune direction d'os, et pas de parent dont hériter. "
            f"Correction proposée : ajouter un enfant ou un End Site avec un OFFSET non nul. "
            f"Repère d'os = identité."
        )
        return None, "aucune direction, identité"
    parent = joints[joint.parent]
    axis = np.array(parent.bone_frame[:, 1], dtype=np.float64, copy=True)
    if float(np.linalg.norm(axis)) < _DIR_EPS:
        warnings.append(
            f"Ligne {joint.line} : {cause}. "
            f"Cause : le parent « {parent.name} » n'a pas de direction à transmettre. "
            f"Correction proposée : donner un OFFSET non nul à cet os. Repère d'os = identité."
        )
        return None, "parent sans direction, identité"
    warnings.append(
        f"Ligne {joint.line} : {cause}. "
        f"Cause : la direction propre est inutilisable. "
        f"Correction proposée : la direction retenue est celle du parent « {parent.name} ». "
        f"Donner un OFFSET non nul pour une direction propre à cet os."
    )
    return axis, f"direction du parent « {parent.name} »"


def _frame_from_direction(
    direction: np.ndarray,
    pole: np.ndarray,
    warnings: List[str],
    joint: BvhJoint,
) -> np.ndarray:
    """Colonnes = axes de l'os dans le repère joint du parent. +Y = direction."""
    y = _as_unit(direction)
    if y is None:
        raise BvhParseError(
            joint.line,
            f"direction d'os nulle sur « {joint.name} »",
            "cette direction aurait dû être refusée avant la construction du repère",
        )
    p = _as_unit(pole)
    if p is None:
        raise BvhParseError(
            0,
            "pôle secondaire nul",
            "passer pole=(x, y, z) non nul",
        )
    z_raw = p - y * float(np.dot(p, y))
    if float(np.linalg.norm(z_raw)) < 1e-8:
        warnings.append(
            f"Ligne {joint.line} : le pôle est parallèle à l'os « {joint.name} ». "
            f"Cause : le pôle ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) ne fixe pas le roulis. "
            f"Correction proposée : choisir un pôle non colinéaire à la direction. "
            f"Un axe de secours orthogonal a été utilisé pour garder un repère inversible."
        )
        eye = np.eye(3, dtype=np.float64)
        axis = min((eye[index] for index in range(3)), key=lambda vector: abs(float(np.dot(vector, y))))
        z_raw = axis - y * float(np.dot(axis, y))
    z = z_raw / float(np.linalg.norm(z_raw))
    x = np.cross(y, z)
    x = x / float(np.linalg.norm(x))
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def _side_of(name: str) -> Optional[str]:
    token = name.strip().lower()
    if token.startswith("left") or re.search(r"(^|[_\-:\s])left([_\-:\s]|$)", token):
        return "left"
    if token.startswith("right") or re.search(r"(^|[_\-:\s])right([_\-:\s]|$)", token):
        return "right"
    if re.search(r"(^|[_\-.])l$", token) or re.match(r"l[_\-.]", token):
        return "left"
    if re.search(r"(^|[_\-.])r$", token) or re.match(r"r[_\-.]", token):
        return "right"
    return None


def _has_part(token: str, part: str) -> bool:
    return re.search(rf"(^|[^a-z]){re.escape(part)}([^a-z]|$)", token) is not None


def _part_token(name: str) -> str:
    token = name.strip().lower()
    token = re.sub(r"left|right", " ", token)
    token = re.sub(r"(^|[_\-.])[lr](?=[_\-.]|$)", " ", token)
    return token


def _measure_chirality(
    joints: Dict[str, BvhJoint],
    rest: Dict[str, np.ndarray],
    assumption: str,
) -> Tuple[str, Optional[str]]:
    """Mesure le fichier. Ne change jamais l'hypothèse de chiralité."""
    classified: List[Tuple[BvhJoint, Optional[str], str]] = []
    for joint in joints.values():
        classified.append((joint, _side_of(joint.stripped_name), _part_token(joint.stripped_name)))

    def find(part: str, side: Optional[str]) -> List[BvhJoint]:
        return [
            joint
            for joint, joint_side, token in classified
            if joint_side == side and _has_part(token, part)
        ]

    lateral_part = None
    left = right = None
    for part in ("hip", "upleg", "thigh", "shoulder", "clavicle", "collar"):
        left_hits = find(part, "left")
        right_hits = find(part, "right")
        if len(left_hits) == 1 and len(right_hits) == 1:
            lateral_part = part
            left, right = left_hits[0], right_hits[0]
            break
    if left is None or right is None or lateral_part is None:
        return (
            "mesure non concluante : aucun couple gauche/droite unique "
            "(hanche, épaule ou clavicule). L'hypothèse explicite n'est pas modifiée.",
            None,
        )
    lateral = np.asarray(rest[left.name], dtype=np.float64) - np.asarray(rest[right.name], dtype=np.float64)
    if float(np.linalg.norm(lateral)) < _DIR_EPS:
        return (
            "mesure non concluante : les repères gauche et droite sont confondus. "
            "L'hypothèse explicite n'est pas modifiée.",
            None,
        )
    origin = 0.5 * (
        np.asarray(rest[left.name], dtype=np.float64) + np.asarray(rest[right.name], dtype=np.float64)
    )
    up, up_src = _chirality_up(find, rest, origin)
    if up is None:
        return (
            f"mesure non concluante : couple {lateral_part} trouvé, axe haut absent. "
            "L'hypothèse explicite n'est pas modifiée.",
            None,
        )
    forward, forward_src = _chirality_forward(classified, rest, origin)
    if forward is None:
        return (
            f"mesure non concluante : couple {lateral_part} et axe haut ({up_src}) trouvés, "
            "mais pas d'axe avant (genou, pied, orteil, nez ou mâchoire) hors de ce plan. "
            "L'hypothèse explicite n'est pas modifiée. Aucune conversion.",
            None,
        )
    score = _signed_triple(lateral, up, forward)
    if abs(score) < 0.2:
        return (
            f"mesure non concluante : produit mixte {score:.3f} "
            f"({lateral_part}, haut {up_src}, avant {forward_src}), repère presque plat. "
            "L'hypothèse explicite n'est pas modifiée.",
            None,
        )
    agrees = (assumption == "right" and score > 0.0) or (assumption == "left" and score < 0.0)
    if agrees:
        return (
            f"mesure {score:.3f}, compatible avec l'hypothèse {assumption} "
            f"({lateral_part}, haut {up_src}, avant {forward_src}). "
            "L'hypothèse reste celle qui a été déclarée ou assumée, elle n'est pas déduite de la mesure.",
            None,
        )
    opposite = "left" if assumption == "right" else "right"
    label = "droitière" if assumption == "right" else "gauche"
    warning = (
        f"Ligne {left.line} : la mesure de chiralité contredit l'hypothèse {label}. "
        f"Cause : le produit mixte (gauche − droite, haut, avant) vaut {score:.3f} "
        f"sur {lateral_part} / {up_src} / {forward_src}. "
        f"Correction proposée : déclarer handedness='{opposite}' si ce signe est voulu. "
        f"L'hypothèse reste « {assumption} ». Aucune conversion n'a été faite."
    )
    text = (
        f"mesure {score:.3f}, en contradiction avec l'hypothèse {assumption} "
        f"({lateral_part}, haut {up_src}, avant {forward_src}). "
        "L'hypothèse est conservée telle quelle."
    )
    return text, warning


def _chirality_up(
    find,
    rest: Dict[str, np.ndarray],
    origin: np.ndarray,
) -> Tuple[Optional[np.ndarray], str]:
    hip_l, hip_r = find("hip", "left"), find("hip", "right")
    sho_l, sho_r = find("shoulder", "left"), find("shoulder", "right")
    if len(hip_l) == 1 and len(hip_r) == 1 and len(sho_l) == 1 and len(sho_r) == 1:
        hip_mid = 0.5 * (rest[hip_l[0].name] + rest[hip_r[0].name])
        sho_mid = 0.5 * (rest[sho_l[0].name] + rest[sho_r[0].name])
        up = np.asarray(sho_mid, dtype=np.float64) - np.asarray(hip_mid, dtype=np.float64)
        if float(np.linalg.norm(up)) >= _DIR_EPS:
            return up, "épaules moins hanches"
    for part, label in (("head", "tête"), ("neck", "cou"), ("chest", "poitrine"), ("spine", "colonne")):
        hits = find(part, None) + find(part, "left") + find(part, "right")
        seen = set()
        for joint in hits:
            if joint.name in seen:
                continue
            seen.add(joint.name)
            up = np.asarray(rest[joint.name], dtype=np.float64) - origin
            if float(np.linalg.norm(up)) >= _DIR_EPS:
                return up, label
    return None, ""


def _chirality_forward(
    classified: Sequence[Tuple[BvhJoint, Optional[str], str]],
    rest: Dict[str, np.ndarray],
    origin: np.ndarray,
) -> Tuple[Optional[np.ndarray], str]:
    for part, label in (
        ("toe", "orteil"),
        ("foot", "pied"),
        ("ankle", "cheville"),
        ("knee", "genou"),
        ("nose", "nez"),
        ("jaw", "mâchoire"),
    ):
        points = [
            np.asarray(rest[joint.name], dtype=np.float64)
            for joint, _side, token in classified
            if _has_part(token, part)
        ]
        if not points:
            continue
        mean = np.mean(np.stack(points, axis=0), axis=0)
        forward = mean - origin
        if float(np.linalg.norm(forward)) >= _DIR_EPS:
            return forward, label
    return None, ""


def _signed_triple(lateral: np.ndarray, up: np.ndarray, forward: np.ndarray) -> float:
    vectors = []
    for vector in (lateral, up, forward):
        unit = _as_unit(vector)
        if unit is None:
            return 0.0
        vectors.append(unit)
    return float(np.dot(np.cross(vectors[0], vectors[1]), vectors[2]))


def _zero_motion(joints: Dict[str, BvhJoint], n_channels: int) -> np.ndarray:
    return np.zeros((1, n_channels), dtype=np.float64)


def _forward(
    joints: Dict[str, BvhJoint],
    motion: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    n_frames = motion.shape[0]
    positions: Dict[str, np.ndarray] = {}
    rotations: Dict[str, np.ndarray] = {}
    locals_q: Dict[str, np.ndarray] = {}
    matrices: Dict[str, np.ndarray] = {}

    for name, joint in joints.items():
        local = _local_matrices(joint, motion)
        translation = np.repeat(joint.offset.reshape(1, 3), n_frames, axis=0)
        for index, channel in enumerate(joint.channels):
            if channel.endswith("position") and n_frames:
                translation[:, _AXIS_INDEX[channel[0]]] += motion[:, joint.channel_start + index]
        parent = joint.parent
        if parent is None:
            world_m = local
            world_t = translation
        else:
            parent_m = matrices[parent]
            world_t = positions[parent] + np.einsum("fij,fj->fi", parent_m, translation)
            world_m = np.einsum("fij,fjk->fik", parent_m, local)
        matrices[name] = world_m
        positions[name] = world_t
        rotations[name] = _matrices_to_quaternions(world_m)
        locals_q[name] = _matrices_to_quaternions(local)
    return positions, rotations, locals_q


def _local_matrices(joint: BvhJoint, motion: np.ndarray) -> np.ndarray:
    n_frames = motion.shape[0]
    matrix = np.zeros((n_frames, 3, 3), dtype=np.float64)
    if n_frames == 0:
        return matrix
    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = 1.0
    matrix[:, 2, 2] = 1.0
    for index, channel in enumerate(joint.channels):
        if not channel.endswith("rotation"):
            continue
        angles = np.deg2rad(motion[:, joint.channel_start + index])
        matrix = np.einsum("fij,fjk->fik", _axis_matrices(channel[0], angles), matrix)
    return matrix


def _axis_matrices(axis: str, angles: np.ndarray) -> np.ndarray:
    cosine = np.cos(angles)
    sine = np.sin(angles)
    count = angles.shape[0]
    out = np.zeros((count, 3, 3), dtype=np.float64)
    if axis == "X":
        out[:, 0, 0] = 1.0
        out[:, 1, 1] = cosine
        out[:, 1, 2] = -sine
        out[:, 2, 1] = sine
        out[:, 2, 2] = cosine
    elif axis == "Y":
        out[:, 0, 0] = cosine
        out[:, 0, 2] = sine
        out[:, 1, 1] = 1.0
        out[:, 2, 0] = -sine
        out[:, 2, 2] = cosine
    elif axis == "Z":
        out[:, 0, 0] = cosine
        out[:, 0, 1] = -sine
        out[:, 1, 0] = sine
        out[:, 1, 1] = cosine
        out[:, 2, 2] = 1.0
    else:
        raise BvhParseError(0, f"axe de rotation invalide « {axis} »", "utiliser X, Y ou Z")
    return out


def _euler_matrix(order: str, degrees: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float64)
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    for axis, angle in zip(order, radians):
        matrix = _axis_matrices(axis, np.array([angle]))[0] @ matrix
    return matrix


def _euler_angles(matrix: np.ndarray, order: str) -> np.ndarray:
    """Inverse de `_euler_matrix` pour un ordre de 3 axes distincts. Radians."""
    if order == "XYZ":
        mid = math.asin(_clamp(-matrix[2, 0]))
        if abs(math.cos(mid)) > 1e-8:
            first = math.atan2(matrix[2, 1], matrix[2, 2])
            third = math.atan2(matrix[1, 0], matrix[0, 0])
        elif mid > 0.0:
            first = math.atan2(matrix[0, 1], matrix[0, 2])
            third = 0.0
        else:
            first = math.atan2(-matrix[0, 1], -matrix[0, 2])
            third = 0.0
    elif order == "XZY":
        mid = math.asin(_clamp(matrix[1, 0]))
        if abs(math.cos(mid)) > 1e-8:
            first = math.atan2(-matrix[1, 2], matrix[1, 1])
            third = math.atan2(-matrix[2, 0], matrix[0, 0])
        elif mid > 0.0:
            first = math.atan2(matrix[0, 2], -matrix[0, 1])
            third = 0.0
        else:
            first = math.atan2(-matrix[0, 2], matrix[0, 1])
            third = 0.0
    elif order == "YXZ":
        mid = math.asin(_clamp(matrix[2, 1]))
        if abs(math.cos(mid)) > 1e-8:
            first = math.atan2(-matrix[2, 0], matrix[2, 2])
            third = math.atan2(-matrix[0, 1], matrix[1, 1])
        else:
            first = math.atan2(matrix[0, 2], matrix[0, 0])
            third = 0.0
    elif order == "YZX":
        mid = math.asin(_clamp(-matrix[0, 1]))
        if abs(math.cos(mid)) > 1e-8:
            first = math.atan2(matrix[0, 2], matrix[0, 0])
            third = math.atan2(matrix[2, 1], matrix[1, 1])
        elif mid > 0.0:
            first = math.atan2(matrix[1, 2], matrix[1, 0])
            third = 0.0
        else:
            first = math.atan2(-matrix[1, 2], -matrix[1, 0])
            third = 0.0
    elif order == "ZXY":
        mid = math.asin(_clamp(-matrix[1, 2]))
        if abs(math.cos(mid)) > 1e-8:
            first = math.atan2(matrix[1, 0], matrix[1, 1])
            third = math.atan2(matrix[0, 2], matrix[2, 2])
        else:
            first = math.atan2(-matrix[0, 1], matrix[0, 0])
            third = 0.0
    elif order == "ZYX":
        mid = math.asin(_clamp(matrix[0, 2]))
        if abs(math.cos(mid)) > 1e-8:
            first = math.atan2(-matrix[0, 1], matrix[0, 0])
            third = math.atan2(-matrix[1, 2], matrix[2, 2])
        else:
            first = math.atan2(matrix[1, 0], matrix[1, 1])
            third = 0.0
    else:
        raise BvhParseError(
            0,
            f"ordre d'Euler incomplet « {order} »",
            "un changement de repère demande les trois canaux Xrotation Yrotation Zrotation",
        )
    return np.array([first, mid, third], dtype=np.float64)


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    count = matrices.shape[0]
    out = np.zeros((count, 4), dtype=np.float64)
    for index in range(count):
        out[index] = matrix_to_quaternion(matrices[index])
    return out


def _end_site_positions(
    joints: Dict[str, BvhJoint],
    positions: Dict[str, np.ndarray],
    rotations: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    ends: Dict[str, np.ndarray] = {}
    for joint in joints.values():
        if joint.end_site is None:
            continue
        worlds = positions[joint.name]
        quats = rotations[joint.name]
        tip = np.zeros_like(worlds)
        for frame in range(worlds.shape[0]):
            tip[frame] = worlds[frame] + quaternion_to_matrix(quats[frame]) @ joint.end_site
        ends[joint.name] = tip
    return ends


def _rotation_order(channels: Sequence[str]) -> str:
    letters = [channel[0] for channel in channels if channel.endswith("rotation")]
    return "".join(letters) if letters else "—"


def _quaternion_sign_flips(rotations: Dict[str, np.ndarray]) -> int:
    flips = 0
    for series in rotations.values():
        if series.shape[0] < 2:
            continue
        dots = np.sum(series[1:] * series[:-1], axis=1)
        flips += int(np.count_nonzero(dots < 0.0))
    return flips


def _format_prefixes(prefixes: Dict[str, int]) -> str:
    if not prefixes:
        return "aucun"
    return ", ".join(f"{prefix} ×{count}" for prefix, count in sorted(prefixes.items()))


def _format_renames(joints: Iterable[BvhJoint]) -> str:
    items = list(joints)
    if not items:
        return "aucun"
    return ", ".join(f"{joint.raw_name} → {joint.name}" for joint in items)


def _tree_lines(clip: BvhClip, name: str) -> List[str]:
    joint = clip.joints[name]
    pad = "  " * (joint.depth + 1)
    channels = " ".join(joint.channels) if joint.channels else "(aucun canal)"
    prefix = f" préfixe={joint.prefix!r} nu={joint.stripped_name}" if joint.prefix else ""
    renamed = f" brut={joint.raw_name}" if joint.name != joint.raw_name else ""
    end = " +EndSite" if joint.end_site is not None else ""
    bone = f" os={joint.bone_direction_source}" if joint.bone_direction_source else ""
    lines = [f"{pad}{joint.name}  [{channels}]{prefix}{renamed}{end}{bone}  ligne {joint.line}"]
    for child in joint.children:
        lines.extend(_tree_lines(clip, child))
    return lines


def main() -> None:
    import argparse
    import sys

    argument_parser = argparse.ArgumentParser(description="Lit un BVH et affiche le rapport de parse.")
    argument_parser.add_argument("path")
    argument_parser.add_argument("--unit", default=None)
    argument_parser.add_argument("--up-axis", default=None)
    argument_parser.add_argument("--handedness", default=None)
    argument_parser.add_argument("--apply-normalization", action="store_true")
    args = argument_parser.parse_args()
    try:
        clip = parse_bvh(
            args.path,
            unit=args.unit,
            up_axis=args.up_axis,
            handedness=args.handedness,
            apply_normalization=args.apply_normalization,
        )
    except BvhParseError as exc:
        sys.stderr.write(str(exc) + "\n")
        sys.exit(1)
    sys.stdout.write(format_report(clip))


if __name__ == "__main__":
    main()

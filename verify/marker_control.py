"""Trois frames : pose source (bleu) et BVH réécrit par les marqueurs (orange)."""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parents(ordered):
    return {node.name: None if node.parent is None else node.parent.name for node in ordered}


def _walk(roots):
    ordered = []

    def visit(node):
        ordered.append(node)
        for child in node.children:
            visit(child)

    for root in roots:
        visit(root)
    return ordered


def _cloud(positions, frame):
    return np.array([positions[name][frame] for name in positions])


def _paint(image, positions, parents, frame, ox, oy, w, h, color, thickness, fit, project, line):
    names = list(positions)
    pts = _cloud(positions, frame)
    projected = project(pts, ox, oy, w, h, fit=fit)
    index = {name: i for i, name in enumerate(names)}
    for name, parent in parents.items():
        if parent is None or name not in index or parent not in index:
            continue
        a = projected[index[parent]]
        b = projected[index[name]]
        line(image, a[0], a[1], b[0], b[1], color, thickness)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    oracle = _load("phase_a_oracle", os.path.join(ROOT, "verify", "phase_a_oracle.py"))
    panel_w, panel_h = 420, 360
    rows = [
        (
            r"D:\dev\Mimo\out\jumping_jacks.bvh",
            os.path.join(ROOT, "verify", "out", "markers_jumping_jacks.bvh"),
            "CORPS",
        ),
        (
            r"D:\dev\Mimo\out\a_pose_hands.bvh",
            os.path.join(ROOT, "verify", "out", "markers_a_pose_hands.bvh"),
            "MAIN",
        ),
        (
            r"D:\dev\Mimo\out_prefix_body.bvh",
            os.path.join(ROOT, "verify", "out", "markers_out_prefix_body.bvh"),
            "POS",
        ),
        (
            r"D:\dev\Mimo\out\a_pose_hands.bvh",
            os.path.join(ROOT, "verify", "out", "retarget_a_on_bon.bvh"),
            "ECART",
        ),
    ]
    image = np.full((panel_h * len(rows), panel_w * 3, 3), 244, dtype=np.uint8)
    gaps = []
    for row_index, (source, rewritten, label) in enumerate(rows):
        roots, motion, _time = oracle.naive_load(open(source, encoding="utf-8", errors="replace").read())
        out_roots, out_motion, _out_time = oracle.naive_load(
            open(rewritten, encoding="utf-8", errors="replace").read()
        )
        positions, _rot = oracle.independent_fk(roots, motion)
        rewritten_pos, _rewritten_rot = oracle.independent_fk(out_roots, out_motion)
        parents = _parents(_walk(roots))
        rewritten_parents = _parents(_walk(out_roots))
        shared = [name for name in parents if name in rewritten_parents]
        parent_gap = [
            name for name in shared if parents[name] != rewritten_parents[name]
        ]
        print(f"{label} : os communs {len(shared)}, parents différents {len(parent_gap)}")
        frames = [0, motion.shape[0] // 2, motion.shape[0] - 1]
        fit = np.vstack(
            [_cloud(positions, frame) for frame in frames]
            + [_cloud(rewritten_pos, frame) for frame in frames]
        )
        gap = 0.0
        gap_bone = next(iter(positions))
        for name in positions:
            if name not in rewritten_pos:
                raise RuntimeError(f"{label} : « {name} » absent du BVH réécrit")
            bone_gap = float(np.max(np.abs(positions[name] - rewritten_pos[name])))
            if bone_gap > gap:
                gap = bone_gap
                gap_bone = name
        gaps.append((label, gap * 10.0, gap_bone))
        oy = row_index * panel_h
        for index, frame in enumerate(frames):
            ox = index * panel_w
            _paint(
                image, positions, parents, frame, ox, oy, panel_w, panel_h,
                (25, 70, 140), 3, fit, oracle._project, oracle._draw_line,
            )
            _paint(
                image, rewritten_pos, rewritten_parents, frame, ox, oy, panel_w, panel_h,
                (210, 90, 20), 1, fit, oracle._project, oracle._draw_line,
            )
            oracle._draw_text(image, f"{label} {frame}", ox + 12, oy + 12, (20, 20, 20), 2)
    path = os.path.join(ROOT, "verify", "out", "marker_control.png")
    oracle._write_png(path, image)
    for label, millimetres, bone in gaps:
        print(f"{label} : écart positions source/réécrit {millimetres:.6f} mm sur {bone}")
    print(path)


if __name__ == "__main__":
    main()

"""Module isolate.markers
======================
Sous-package de placement, sélection, filtrage RANSAC et suivi des marqueurs 3D.
"""

from markers.surface_sampling import (
    segment_submesh_by_mask,
    build_bone_submeshes,
    farthest_point_sampling,
    sample_multi_markers_for_bone,
)
from markers.tracker import (
    extract_markers_positions,
    compute_marker_stability_weights,
)
from markers.ransac_filter import (
    ransac_kabsch_alignment,
    compute_green_lagrange_strain,
)

__all__ = [
    "segment_submesh_by_mask",
    "build_bone_submeshes",
    "farthest_point_sampling",
    "sample_multi_markers_for_bone",
    "extract_markers_positions",
    "compute_marker_stability_weights",
    "ransac_kabsch_alignment",
    "compute_green_lagrange_strain",
]

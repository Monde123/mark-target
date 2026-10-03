"""
Module isolate
==============
Package autonome de Retargeting par Marqueurs Multi-Points (Kabsch) et Mono-Marqueurs (Aim).
Indépendant du modèle source et du modèle final.
"""

from isolate.core import (
    quat_normalize,
    quat_identity,
    quat_inv,
    quat_mul,
    rotate_vector,
    matrix_to_quaternion,
    quaternion_to_matrix,
    kabsch_rotation,
    weighted_kabsch_rotation,
)
from isolate.markers import (
    build_bone_submeshes,
    sample_multi_markers_for_bone,
    extract_markers_positions,
    compute_marker_stability_weights,
)
from isolate.retargeting import (
    MarkerRetargetSolver,
    global_to_local_hierarchy,
    local_to_global_hierarchy,
    apply_euler_correction_to_root,
    apply_axis_correction_to_quaternion,
)
from isolate.mappings import MappingRegistry
from isolate.adapters import (
    SourceAdapter,
    TargetAdapter,
    TargetSkeleton,
    MarkerFrameSequence,
)

__version__ = "1.0.0"

__all__ = [
    "quat_normalize",
    "quat_identity",
    "quat_inv",
    "quat_mul",
    "rotate_vector",
    "matrix_to_quaternion",
    "quaternion_to_matrix",
    "kabsch_rotation",
    "weighted_kabsch_rotation",
    "build_bone_submeshes",
    "sample_multi_markers_for_bone",
    "extract_markers_positions",
    "compute_marker_stability_weights",
    "MarkerRetargetSolver",
    "global_to_local_hierarchy",
    "local_to_global_hierarchy",
    "apply_euler_correction_to_root",
    "apply_axis_correction_to_quaternion",
    "MappingRegistry",
    "SourceAdapter",
    "TargetAdapter",
    "TargetSkeleton",
    "MarkerFrameSequence",
]

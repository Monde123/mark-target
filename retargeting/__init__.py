"""
Module retargeting
==========================
Export du moteur de résolution, de l'état temporel et cinématique.
"""
from retargeting.solver import MarkerRetargetSolver
from retargeting.temporal_state import (
    TemporalRetargetState,
    align_quaternion_hemisphere,
    quaternion_log_map,
    quaternion_exp_map,
)
from retargeting.kinematics import (
    global_to_local_hierarchy,
    local_to_global_hierarchy,
    apply_euler_correction_to_root,
)

__all__ = [
    "MarkerRetargetSolver",
    "TemporalRetargetState",
    "align_quaternion_hemisphere",
    "quaternion_log_map",
    "quaternion_exp_map",
    "global_to_local_hierarchy",
    "local_to_global_hierarchy",
    "apply_euler_correction_to_root",
]

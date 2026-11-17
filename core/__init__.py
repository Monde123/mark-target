"""Module isolate.core
===================
Export des briques de géométrie et Kabsch / Umeyama.
"""

try:
    from isolate.core.geometry import (
        quat_normalize,
        quat_identity,
        quat_inv,
        quat_mul,
        rotate_vector,
        quat_from_two_vectors,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
    from isolate.core.kabsch import (
        kabsch_rotation,
        weighted_kabsch_rotation,
        kabsch_umeyama_rotation,
        weighted_kabsch_umeyama_rotation,
    )
except ImportError:
    from core.geometry import (
        quat_normalize,
        quat_identity,
        quat_inv,
        quat_mul,
        rotate_vector,
        quat_from_two_vectors,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
    from core.kabsch import (
        kabsch_rotation,
        weighted_kabsch_rotation,
        kabsch_umeyama_rotation,
        weighted_kabsch_umeyama_rotation,
    )

__all__ = [
    "quat_normalize",
    "quat_identity",
    "quat_inv",
    "quat_mul",
    "rotate_vector",
    "quat_from_two_vectors",
    "matrix_to_quaternion",
    "quaternion_to_matrix",
    "kabsch_rotation",
    "weighted_kabsch_rotation",
    "kabsch_umeyama_rotation",
    "weighted_kabsch_umeyama_rotation",
]

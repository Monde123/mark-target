"""Module isolate.adapters
=======================
Exports des adaptateurs de formats d'entrée et de sortie.
Supporte le chargement paresseux (lazy-loading) pour fonctionner même sans pygltflib.
"""

from adapters.base import (
    SourceAdapter,
    TargetAdapter,
    TargetSkeleton,
    MarkerFrameSequence,
)
from adapters.bvh_adapter import BVHAdapter

try:
    from adapters.gltf_mixamo_adapter import GLTFMixamoAdapter
except ImportError:
    GLTFMixamoAdapter = None

try:
    from adapters.vrm_adapter import VRMAdapter
except ImportError:
    VRMAdapter = None

try:
    from adapters.hybrik_pk_adapter import HybrIKPKAdapter
except ImportError:
    HybrIKPKAdapter = None

__all__ = [
    "SourceAdapter",
    "TargetAdapter",
    "TargetSkeleton",
    "MarkerFrameSequence",
    "BVHAdapter",
    "GLTFMixamoAdapter",
    "VRMAdapter",
    "HybrIKPKAdapter",
]

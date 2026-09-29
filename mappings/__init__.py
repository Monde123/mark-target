"""
Module isolate.mappings
=======================
Exports des structures et registres de mappings.
"""
from mappings.standard_humanoid import (
    STANDARD_HUMANOID_BONES,
    STANDARD_HUMANOID_PARENTS,
    MIXAMO_TO_STANDARD,
    STANDARD_TO_MIXAMO,
    VRM_TO_STANDARD,
    STANDARD_TO_VRM,
)
from mappings.registry import MappingRegistry

__all__ = [
    "STANDARD_HUMANOID_BONES",
    "STANDARD_HUMANOID_PARENTS",
    "MIXAMO_TO_STANDARD",
    "STANDARD_TO_MIXAMO",
    "VRM_TO_STANDARD",
    "STANDARD_TO_VRM",
    "MappingRegistry",
]

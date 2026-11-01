"""
Module mappings
=======================
Exports des structures et registres de mappings, de l'ossature tampon et de l'auto-mapper.
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
from mappings.buffer_armature import CanonicalBufferArmature
from mappings.auto_mapper import AutoBoneIdentifier, BufferRigMapper

__all__ = [
    "STANDARD_HUMANOID_BONES",
    "STANDARD_HUMANOID_PARENTS",
    "MIXAMO_TO_STANDARD",
    "STANDARD_TO_MIXAMO",
    "VRM_TO_STANDARD",
    "STANDARD_TO_VRM",
    "MappingRegistry",
    "CanonicalBufferArmature",
    "AutoBoneIdentifier",
    "BufferRigMapper",
]

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

import numpy as np
from pygltflib import (
    ACCESSOR_TYPE_SCALAR,
    ACCESSOR_TYPE_VEC4,
    ARRAY_BUFFER,
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Buffer,
    BufferView,
    FLOAT,
    GLTF2,
)

from core.geometry import quat_identity, quat_normalize


def _align4(n: int) -> int:
    return (n + 3) & ~3


def export_rotation_animation_to_gltf(
    gltf: GLTF2,
    output_path: str,
    animation_clip: List[Dict[str, Any]],
    skeleton_fps: float,
    bone_names: List[str],
    node_lookup: Callable[[str], int],
    animation_name: str = "marker_animation",
) -> None:
    if not animation_clip:
        if output_path.lower().endswith(".glb") or output_path.lower().endswith(".vrm"):
            gltf.save_binary(output_path)
        else:
            gltf.save(output_path)
        return

    fps = skeleton_fps if skeleton_fps > 0 else 30.0
    n_frames = len(animation_clip)

    valid_bones = [b for b in bone_names if node_lookup(b) is not None]
    if not valid_bones:
        raise ValueError("Aucun os cible valide trouvé pour construire une animation glTF.")

    times = np.arange(n_frames, dtype=np.float32) / float(fps)
    time_bytes = times.tobytes()
    bone_track_bytes: Dict[str, bytes] = {}

    for bone in valid_bones:
        track = np.zeros((n_frames, 4), dtype=np.float32)
        for i, frame in enumerate(animation_clip):
            q = frame.get("rotations", {}).get(bone, quat_identity())
            q = quat_normalize(np.asarray(q, dtype=np.float64))
            track[i] = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)  # glTF: [x, y, z, w]
        bone_track_bytes[bone] = track.tobytes()

    if gltf.buffers is None:
        gltf.buffers = []
    if gltf.bufferViews is None:
        gltf.bufferViews = []
    if gltf.accessors is None:
        gltf.accessors = []
    if gltf.animations is None:
        gltf.animations = []

    if not gltf.buffers:
        gltf.buffers.append(Buffer(byteLength=0))

    existing_blob = gltf.binary_blob() or b""
    base_offset = _align4(len(existing_blob))
    prefix_padding = b"\x00" * (base_offset - len(existing_blob))
    anim_blob = bytearray()

    time_offset = base_offset
    anim_blob.extend(time_bytes)
    time_padding = _align4(len(anim_blob)) - len(anim_blob)
    if time_padding:
        anim_blob.extend(b"\x00" * time_padding)

    track_offsets: Dict[str, int] = {}
    for bone in valid_bones:
        track_offsets[bone] = base_offset + len(anim_blob)
        anim_blob.extend(bone_track_bytes[bone])
        pad = _align4(len(anim_blob)) - len(anim_blob)
        if pad:
            anim_blob.extend(b"\x00" * pad)

    new_blob = existing_blob + prefix_padding + bytes(anim_blob)
    gltf.set_binary_blob(new_blob)
    gltf.buffers[0].byteLength = len(new_blob)
    gltf.buffers[0].uri = None

    time_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        BufferView(
            buffer=0,
            byteOffset=time_offset,
            byteLength=len(time_bytes),
            target=ARRAY_BUFFER,
            name="marker_animation_time",
        )
    )
    time_accessor_idx = len(gltf.accessors)
    gltf.accessors.append(
        Accessor(
            bufferView=time_bv_idx,
            byteOffset=0,
            componentType=FLOAT,
            count=n_frames,
            type=ACCESSOR_TYPE_SCALAR,
            min=[float(times[0])],
            max=[float(times[-1])],
            name="marker_animation_time_accessor",
        )
    )

    samplers: List[AnimationSampler] = []
    channels: List[AnimationChannel] = []

    for bone in valid_bones:
        node_idx = node_lookup(bone)
        if node_idx is None:
            continue
        track_bytes = bone_track_bytes[bone]
        track_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            BufferView(
                buffer=0,
                byteOffset=track_offsets[bone],
                byteLength=len(track_bytes),
                target=ARRAY_BUFFER,
                name=f"marker_animation_{bone}_rot",
            )
        )
        track_accessor_idx = len(gltf.accessors)
        gltf.accessors.append(
            Accessor(
                bufferView=track_bv_idx,
                byteOffset=0,
                componentType=FLOAT,
                count=n_frames,
                type=ACCESSOR_TYPE_VEC4,
                name=f"marker_animation_{bone}_rot_accessor",
            )
        )
        sampler_idx = len(samplers)
        samplers.append(
            AnimationSampler(
                input=time_accessor_idx,
                output=track_accessor_idx,
                interpolation="LINEAR",
            )
        )
        channels.append(
            AnimationChannel(
                sampler=sampler_idx,
                target=AnimationChannelTarget(node=node_idx, path="rotation"),
            )
        )

    gltf.animations.append(Animation(name=animation_name, samplers=samplers, channels=channels))

    if output_path.lower().endswith(".glb") or output_path.lower().endswith(".vrm"):
        gltf.save_binary(output_path)
    elif output_path.lower().endswith(".gltf"):
        gltf.save(output_path)
    else:
        if os.path.splitext(output_path)[1]:
            gltf.save(output_path)
        else:
            gltf.save_binary(output_path + ".glb")

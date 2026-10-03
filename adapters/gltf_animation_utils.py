from __future__ import annotations

import base64
import numpy as np
from typing import Dict, List, Any

from pygltflib import (
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Accessor,
    Buffer,
    BufferView,
    FLOAT,
    LINEAR,
    SCALAR,
    VEC3,
    VEC4,
)

from adapters.base import TargetSkeleton
from core.geometry import quat_identity, quat_normalize


def _align4_length(length: int) -> int:
    return (length + 3) & ~3


def _append_blob_with_padding(storage: bytearray, blob: bytes) -> int:
    start = _align4_length(len(storage))
    if start > len(storage):
        storage.extend(b"\x00" * (start - len(storage)))
    storage.extend(blob)
    end = _align4_length(len(storage))
    if end > len(storage):
        storage.extend(b"\x00" * (end - len(storage)))
    return start


def _to_data_uri(blob: bytes) -> str:
    encoded = base64.b64encode(blob).decode("ascii")
    return f"data:application/octet-stream;base64,{encoded}"


def export_quaternion_animation_to_gltf(
    gltf,
    output_path: str,
    animation_clip: List[Dict[str, Any]],
    skeleton: TargetSkeleton,
    node_map: Dict[str, int],
    animation_name: str = "marker_animation",
) -> None:
    n_frames = len(animation_clip)
    fps = skeleton.fps if skeleton.fps > 0 else 30.0
    if n_frames <= 0:
        gltf.save(output_path)
        return

    if gltf.animations is None:
        gltf.animations = []
    gltf.animations = [anim for anim in gltf.animations if getattr(anim, "name", None) != animation_name]

    if gltf.buffers is None:
        gltf.buffers = []
    if gltf.bufferViews is None:
        gltf.bufferViews = []
    if gltf.accessors is None:
        gltf.accessors = []

    animation_binary = bytearray()

    times = np.arange(n_frames, dtype=np.float32) / float(fps)
    time_blob = times.tobytes()
    time_offset = _append_blob_with_padding(animation_binary, time_blob)

    new_buffer_index = len(gltf.buffers)
    gltf.buffers.append(Buffer(byteLength=0, uri=None))

    time_view_index = len(gltf.bufferViews)
    gltf.bufferViews.append(
        BufferView(
            buffer=new_buffer_index,
            byteOffset=time_offset,
            byteLength=len(time_blob),
        )
    )

    time_accessor_index = len(gltf.accessors)
    gltf.accessors.append(
        Accessor(
            bufferView=time_view_index,
            byteOffset=0,
            componentType=FLOAT,
            count=n_frames,
            type=SCALAR,
            min=[float(times.min())],
            max=[float(times.max())],
        )
    )

    samplers: List[AnimationSampler] = []
    channels: List[AnimationChannel] = []

    for bone_name, node_idx in sorted(node_map.items(), key=lambda kv: kv[0]):
        rotations_xyzw = np.zeros((n_frames, 4), dtype=np.float32)
        for f_idx, frame in enumerate(animation_clip):
            q = frame.get("rotations", {}).get(bone_name, quat_identity())
            q = quat_normalize(q)
            rotations_xyzw[f_idx] = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)

        rot_blob = rotations_xyzw.tobytes()
        rot_offset = _append_blob_with_padding(animation_binary, rot_blob)

        rot_view_index = len(gltf.bufferViews)
        gltf.bufferViews.append(
            BufferView(
                buffer=new_buffer_index,
                byteOffset=rot_offset,
                byteLength=len(rot_blob),
            )
        )

        rot_accessor_index = len(gltf.accessors)
        gltf.accessors.append(
            Accessor(
                bufferView=rot_view_index,
                byteOffset=0,
                componentType=FLOAT,
                count=n_frames,
                type=VEC4,
            )
        )

        sampler_index = len(samplers)
        samplers.append(
            AnimationSampler(
                input=time_accessor_index,
                output=rot_accessor_index,
                interpolation=LINEAR,
            )
        )
        channels.append(
            AnimationChannel(
                sampler=sampler_index,
                target=AnimationChannelTarget(node=node_idx, path="rotation"),
            )
        )

    root_translations = [frame.get("root") for frame in animation_clip]
    has_root_translation_track = any(v is not None for v in root_translations)
    root_node = node_map.get(skeleton.root_name)
    if has_root_translation_track and root_node is not None:
        translations = np.zeros((n_frames, 3), dtype=np.float32)
        last = np.zeros(3, dtype=np.float32)
        for f_idx, value in enumerate(root_translations):
            if value is None:
                translations[f_idx] = last
                continue
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size >= 3:
                last = arr[:3]
            else:
                pad = np.zeros(3, dtype=np.float32)
                pad[:arr.size] = arr
                last = pad
            translations[f_idx] = last

        tr_blob = translations.tobytes()
        tr_offset = _append_blob_with_padding(animation_binary, tr_blob)

        tr_view_index = len(gltf.bufferViews)
        gltf.bufferViews.append(
            BufferView(
                buffer=new_buffer_index,
                byteOffset=tr_offset,
                byteLength=len(tr_blob),
            )
        )

        tr_accessor_index = len(gltf.accessors)
        gltf.accessors.append(
            Accessor(
                bufferView=tr_view_index,
                byteOffset=0,
                componentType=FLOAT,
                count=n_frames,
                type=VEC3,
            )
        )

        sampler_index = len(samplers)
        samplers.append(
            AnimationSampler(
                input=time_accessor_index,
                output=tr_accessor_index,
                interpolation=LINEAR,
            )
        )
        channels.append(
            AnimationChannel(
                sampler=sampler_index,
                target=AnimationChannelTarget(node=root_node, path="translation"),
            )
        )

    binary_blob = bytes(animation_binary)
    gltf.buffers[new_buffer_index].byteLength = len(binary_blob)
    gltf.buffers[new_buffer_index].uri = _to_data_uri(binary_blob)

    gltf.animations.append(Animation(name=animation_name, samplers=samplers, channels=channels))
    gltf.save(output_path)

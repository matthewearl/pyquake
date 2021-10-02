__all__ = (
    'add_model',
)


import dataclasses
import bpy
import numpy as np
from typing import List, Tuple

from . import md3


def _create_shape_key(surf_obj, surf, frame_num):
    shape_key = surf_obj.shape_key_add(name=f'frame_{frame_num}')
    for vert_idx, shape_key_vert in enumerate(shape_key.data):
        shape_key_vert.co = surf.verts[frame_num, vert_idx]
    return shape_key


def add_model(m: md3.MD3, anim_info: md3.AnimationInfo,
              times: np.ndarray, anim_idxs: np.ndarray,
              obj_name: str, fps: float):
    obj = bpy.data.objects.new(obj_name, None)
    surf_objs = []
    bpy.context.scene.collection.objects.link(obj)

    shape_keys = []
    for surf in m.surfaces:
        surf_obj_name = f'{obj_name}_{surf.name}'
        mesh = bpy.data.meshes.new(surf_obj_name)
        mesh.from_pydata([list(v) for v in surf.verts[0]],
                         [],
                         [list(t) for t in surf.tris])

        surf_obj = bpy.data.objects.new(surf_obj_name, mesh)
        surf_objs.append(surf_obj)
        bpy.context.scene.collection.objects.link(surf_obj)

        surf_obj.parent = obj

        for frame_num in range(surf.num_frames):
            shape_keys.append(_create_shape_key(surf_obj, surf, frame_num))

    # De-duplicate frames
    change_idxs = np.where(np.diff(anim_idxs) != 0)[0]
    change_idxs = np.concatenate([[0], change_idxs + 1, [len(anim_idxs) - 1]])
    start_times = times[change_idxs[:-1]]
    end_times = times[change_idxs[1:]]
    changed_anim_idxs = anim_idxs[change_idxs[:-1]]

    prev_anim_frame = None
    prev_blender_frame = None
    for anim_idx, start_time, end_time in zip(changed_anim_idxs, start_times, end_times):
        anim_idx &= ~128    # This bit is flipped to force an anim restart, but it should otherwise be ignored.

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        anim = anim_info.anims[anim_idx]
        num_anim_frames = int(anim.fps * (end_time - start_time))
        anim_frames = np.arange(num_anim_frames)
        if anim.looping_frames:
            anim_frames[anim.num_frames:] = (
                ((anim_frames[anim.num_frames:] - anim.num_frames) % anim.looping_frames)
                + anim.num_frames - anim.looping_frames
            )
        else:
            anim_frames = np.minimum(anim_frames, anim.num_frames - 1)

        anim_frames += anim.first_frame
        anim_frame_times = start_time + np.arange(num_anim_frames) / anim.fps

        for anim_frame, anim_frame_time in zip(anim_frames, anim_frame_times):
            blender_frame = int(fps * anim_frame_time)

            # if prev frame is same as previous frame then this no longer works.
            if prev_blender_frame is not None and prev_anim_frame != anim_frame:
                shape_keys[anim_frame].value = 0
                shape_keys[anim_frame].keyframe_insert('value', frame=prev_blender_frame)

                shape_keys[prev_anim_frame].value = 0
                shape_keys[prev_anim_frame].keyframe_insert('value', frame=blender_frame)

            shape_keys[anim_frame].value = 1
            shape_keys[anim_frame].keyframe_insert('value', frame=blender_frame)

            prev_anim_frame = anim_frame
            prev_blender_frame = blender_frame

    for surf_obj in surf_objs:
        for c in surf_obj.data.shape_keys.animation_data.action.fcurves:
            for kfp in c.keyframe_points:
                kfp.interpolation = 'LINEAR'

    return obj


def add_player(lower_md3: md3.MD3, upper_md3: md3.MD3, head_md3: md3.MD3,
               pmove_frames: md3.PmoveFrames, anim_info: md3.AnimationInfo,
               fps: float,
               obj_name: str):

    root_obj = bpy.data.objects.new(obj_name, None)
    bpy.context.scene.collection.objects.link(root_obj)

    lower_obj = add_model(lower_md3, anim_info, pmove_frames.times, pmove_frames.leg_anim_idxs, 'lower', fps)

    for time, origin, angles in zip(pmove_frames.times,
                                    pmove_frames.origins,
                                    pmove_frames.angles):
        blender_frame = int(fps * time)
        lower_obj.location = origin
        lower_obj.keyframe_insert('location', frame=blender_frame)

        # TODO:  Take rotation logic from CG_PlayerAngles
        lower_obj.rotation_euler = (0., 0., angles[1])
        lower_obj.keyframe_insert('rotation_euler', frame=blender_frame)

    lower_obj.parent = root_obj

    return root_obj


__all__ = (
    'add_model',
)


import dataclasses
from typing import List, Tuple, Optional

import bpy
import mathutils
import numpy as np

from . import md3


def _create_shape_key(surf_obj, surf, frame_num):
    shape_key = surf_obj.shape_key_add(name=f'frame_{frame_num}')
    for vert_idx, shape_key_vert in enumerate(shape_key.data):
        shape_key_vert.co = surf.verts[frame_num, vert_idx]
    return shape_key


def _quat_from_axis(axis, prev_quat):
    q = mathutils.Quaternion([1, 0, 0], 0)
    q.rotate(mathutils.Matrix(axis))

    # q and -q correspond with the same rotation, but flipping direction suddenly will lead to incorrect interpolation,
    # since blender interpolates on the quaternion components.  Fix this here.
    if prev_quat is not None:
        if q.dot(prev_quat) < 0:
            q = -q

    return q


def add_model(m: md3.MD3, anim_info: md3.AnimationInfo,
              times: np.ndarray, anim_idxs: np.ndarray,
              obj_name: str, origin_tag_name: Optional[str], fps: float):
    obj = bpy.data.objects.new(obj_name, None)
    bpy.context.scene.collection.objects.link(obj)

    if origin_tag_name is not None:
        # Set up an origin object, which applies the inverse transformation of a named tag.
        origin_obj = bpy.data.objects.new(f'{obj_name}_origin', None)
        bpy.context.scene.collection.objects.link(origin_obj)
        origin_obj.parent = obj

        origin_tag = m.tags[origin_tag_name][0]
        origin_obj.rotation_quaternion = _quat_from_axis(origin_tag['axis'].T, None)
        origin_obj.location = -origin_tag['axis'].T @ origin_tag['origin']
        origin_obj.rotation_mode = 'QUATERNION'
    else:
        origin_obj = obj

    # Create an object for each tag.
    tag_objs = {}
    for tag_name, tags in m.tags.items():
        tag_obj = bpy.data.objects.new(f'{obj_name}_{tag_name}', None)
        bpy.context.scene.collection.objects.link(tag_obj)
        tag_obj.rotation_quaternion = _quat_from_axis(tags[0]['axis'], None)
        tag_obj.location = tags[0]['origin']
        tag_obj.rotation_mode = 'QUATERNION'
        tag_obj.parent = origin_obj
        tag_objs[tag_name] = tag_obj

    # Create shape keys.
    # TODO: Currently shape_keys is a flat list, but should be 2D with surf idx and frame as indices, and shapes should
    # be set for each surface.  This is most likely broken for animated multi-surface models.
    shape_keys = []
    surf_objs = []
    for surf in m.surfaces:
        surf_obj_name = f'{obj_name}_{surf.name}'
        mesh = bpy.data.meshes.new(surf_obj_name)
        mesh.from_pydata([list(v) for v in surf.verts[0]],
                         [],
                         [list(t) for t in surf.tris])

        surf_obj = bpy.data.objects.new(surf_obj_name, mesh)
        surf_objs.append(surf_obj)
        bpy.context.scene.collection.objects.link(surf_obj)

        surf_obj.parent = origin_obj

        for frame_num in range(surf.num_frames):
            shape_keys.append(_create_shape_key(surf_obj, surf, frame_num))

    # De-duplicate frames
    change_idxs = np.where(np.diff(anim_idxs) != 0)[0]
    change_idxs = np.concatenate([[0], change_idxs + 1, [len(anim_idxs) - 1]])
    start_times = times[change_idxs[:-1]]
    end_times = times[change_idxs[1:]]
    changed_anim_idxs = anim_idxs[change_idxs[:-1]]

    # Each loop is a period where the animation index remains the same
    prev_anim_frame = None
    prev_blender_frame = None
    for anim_idx, start_time, end_time in zip(changed_anim_idxs, start_times, end_times):
        # This bit is flipped to force an anim restart, but it should otherwise be ignored.
        anim_idx &= ~128

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Work out which frames should be played during this period.
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

        # Insert keyframes according to the animations.
        for anim_frame, anim_frame_time in zip(anim_frames, anim_frame_times):
            blender_frame = int(fps * anim_frame_time)

            # Animate shape keys
            if prev_blender_frame is not None and prev_anim_frame != anim_frame:
                shape_keys[anim_frame].value = 0
                shape_keys[anim_frame].keyframe_insert('value', frame=prev_blender_frame)
                shape_keys[prev_anim_frame].value = 0
                shape_keys[prev_anim_frame].keyframe_insert('value', frame=blender_frame)
            shape_keys[anim_frame].value = 1
            shape_keys[anim_frame].keyframe_insert('value', frame=blender_frame)
            prev_anim_frame = anim_frame
            prev_blender_frame = blender_frame

            # Animate tags
            for tag_name, tag_obj in tag_objs.items():
                tag = m.tags[tag_name][anim_frame]

                tag_obj.rotation_quaternion = _quat_from_axis(tag['axis'],
                                                              tag_obj.rotation_quaternion)
                tag_obj.keyframe_insert('rotation_quaternion', frame=blender_frame)
                tag_obj.location = tag['origin']
                tag_obj.keyframe_insert('location', frame=blender_frame)

            # Animate origin obj, if applicable
            if origin_tag_name is not None:
                origin_tag = m.tags[origin_tag_name][anim_frame]
                origin_obj.rotation_quaternion = _quat_from_axis(origin_tag['axis'].T,
                                                                 origin_obj.rotation_quaternion)
                origin_obj.location = -origin_tag['axis'].T @ origin_tag['origin']

    # Make shape keys linearly interpolated.
    for surf_obj in surf_objs:
        for c in surf_obj.data.shape_keys.animation_data.action.fcurves:
            for kfp in c.keyframe_points:
                kfp.interpolation = 'LINEAR'

    return obj, tag_objs


def add_player(lower_md3: md3.MD3, upper_md3: md3.MD3, head_md3: md3.MD3,
               pmove_frames: md3.PmoveFrames, anim_info: md3.AnimationInfo,
               fps: float,
               obj_name: str):

    root_obj = bpy.data.objects.new(obj_name, None)
    bpy.context.scene.collection.objects.link(root_obj)

    lower_obj, lower_tag_objs = add_model(lower_md3, anim_info,
                                          pmove_frames.times, pmove_frames.leg_anim_idxs,
                                          'lower', None, fps)
    upper_obj, upper_tag_objs = add_model(upper_md3, anim_info,
                                          pmove_frames.times, pmove_frames.torso_anim_idxs,
                                          'upper', 'tag_torso', fps)
    upper_obj.parent = lower_tag_objs['tag_torso']

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


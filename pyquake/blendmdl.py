# Copyright (c) 2020 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

from dataclasses import dataclass
from typing import Dict, Set, List, Optional

import bmesh
import bpy
import bpy_types
import numpy as np

from . import mdl, blendmat


BLENDMATS: Dict[str, blendmat.BlendMat] = {}


@dataclass
class BlendMdl:
    am: "AliasMdl"
    obj: bpy_types.Object
    sub_objs: List[bpy_types.Object]
    sample_as_light_mats: Set[blendmat.BlendMat]

    _initial_pose_num: int
    _group_frame_times: Optional[List[float]]
    _shape_keys: List[List[bpy.types.ShapeKey]]
    _current_pose_num: Optional[int] = None
    _last_time: Optional[float] = None

    def _update_pose(self, time: float, pose_num: int, fps: float):
        if self._current_pose_num is None or self._current_pose_num != pose_num:
            for sub_obj, shape_keys in zip(self.sub_objs, self._shape_keys):
                blender_frame = int(round(fps * time))
                if self._current_pose_num is not None:
                    shape_keys[self._current_pose_num].value = 0
                    shape_keys[self._current_pose_num].keyframe_insert('value', frame=blender_frame)
                    last_blender_frame = int(round(fps * self._last_time))
                    shape_keys[pose_num].value = 0
                    shape_keys[pose_num].keyframe_insert('value', frame=last_blender_frame)

                shape_keys[pose_num].value = 1
                shape_keys[pose_num].keyframe_insert('value', frame=blender_frame)

            self._current_pose_num = pose_num
            self._last_time = time

    def add_pose_keyframe(self, pose_num: int, time: float, fps: float):
        if self._group_frame_times is not None:
            if pose_num != self._initial_pose_num:
                raise Exception("Changing pose of a model whose initial pose is a group frame "
                                "is unsupported")
        else:
            self._update_pose(time, pose_num, fps)

    def set_invisible_to_camera(self):
        for sub_obj in self.sub_objs:
            sub_obj.cycles_visibility.camera = False

    def done(self, final_time: float, fps: float):
        if self._group_frame_times is not None:
            loop_time = 0
            while loop_time < final_time:
                for pose_num, offset in enumerate([0] + self._group_frame_times[:-1]):
                    self._update_pose(loop_time + offset, pose_num, fps)
                loop_time += self._group_frame_times[-1]

        for sub_obj in self.sub_objs:
            for c in sub_obj.data.shape_keys.animation_data.action.fcurves:
                for kfp in c.keyframe_points:
                    kfp.interpolation = 'LINEAR'


def _set_uvs(mesh, am, tri_set):
    mesh.uv_layers.new()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_layer = bm.loops.layers.uv[0]

    for bm_face, tri_idx in zip(bm.faces, tri_set):
        tcs = am.get_tri_tcs(tri_idx)

        for bm_loop, (s, t) in zip(bm_face.loops, tcs):
            bm_loop[uv_layer].uv = s / am.header['skin_width'], t / am.header['skin_height']

    bm.to_mesh(mesh)


def _simplify_pydata(verts, tris):
    vert_map = [] 
    new_tris = []
    for tri in tris:
        new_tri = []
        for vert_idx in tri:
            if vert_idx not in vert_map:
                vert_map.append(vert_idx)
            new_tri.append(vert_map.index(vert_idx))
        new_tris.append(new_tri)

    return ([verts[old_vert_idx] for old_vert_idx in vert_map], [], new_tris), vert_map


def _get_tri_set_fullbright_frac(am, tri_set, skin_idx):
    skin_area = 0
    fullbright_area = 0
    for tri_idx in tri_set:
        mask, skin = am.get_tri_skin(tri_idx, skin_idx)
        skin_area += np.sum(mask)
        fullbright_area += np.sum(mask * (skin >= 224))

    return fullbright_area / skin_area


def _create_shape_key(obj, simple_frame, vert_map):
    shape_key = obj.shape_key_add(name=simple_frame.name)
    for old_vert_idx, shape_key_vert in zip(vert_map, shape_key.data):
        shape_key_vert.co = simple_frame.frame_verts[old_vert_idx]
    return shape_key


def add_model(am, pal, mdl_name, obj_name, skin_num, mdl_cfg, initial_pose_num):
    pal = np.concatenate([pal, np.ones(256)[:, None]], axis=1)

    # If the initial pose is a group frame, just load frames from that group.
    if am.frames[initial_pose_num].frame_type == mdl.FrameType.GROUP:
        group_frame = am.frames[initial_pose_num]
        group_times = list(group_frame.times)
    else:
        group_frame = None
        group_times = None
        for frame in am.frames:
            if frame.frame_type != mdl.FrameType.SINGLE:
                raise Exception(f"Frame type {frame.frame_type} not supported for non-static models")

    # Set up things specific to each tri-set
    sample_as_light_mats: Set[blendmat.BlendMat] = set()
    obj = bpy.data.objects.new(obj_name, None)
    sub_objs = []
    shape_keys = []
    bpy.context.scene.collection.objects.link(obj)
    for tri_set_idx, tri_set in enumerate(am.disjoint_tri_sets):
        # Create the mesh and object
        subobj_name = f"{obj_name}_triset{tri_set_idx}"
        mesh = bpy.data.meshes.new(subobj_name)
        if am.frames[0].frame_type == mdl.FrameType.SINGLE:
            initial_verts = am.frames[0].frame.frame_verts
        else:
            initial_verts = am.frames[0].frames[0].frame_verts
        pydata, vert_map = _simplify_pydata([list(v) for v in initial_verts],
                                            [list(am.tris[t]) for t in tri_set])
        mesh.from_pydata(*pydata)
        subobj = bpy.data.objects.new(subobj_name, mesh)
        subobj.parent = obj
        sub_objs.append(subobj)
        bpy.context.scene.collection.objects.link(subobj)

        # Create shape keys, used for animation.
        if group_frame is None:
            shape_keys.append([
                _create_shape_key(subobj, frame.frame, vert_map) for frame in am.frames
            ])
        else:
            shape_keys.append([
                _create_shape_key(subobj, simple_frame, vert_map)
                for simple_frame in group_frame.frames
            ])

        # Set up material
        sample_as_light = mdl_cfg['sample_as_light']
        mat_name = f"{mdl_name}_skin{skin_num}"

        if sample_as_light:
            mat_name = f"{mat_name}_{obj_name}_triset{tri_set_idx}_fullbright"

        if mat_name not in BLENDMATS:
            array_im, fullbright_array_im, _ = blendmat.array_ims_from_indices(
                pal,
                am.skins[skin_num],
                force_fullbright=mdl_cfg['force_fullbright']
            )
            im = blendmat.im_from_array(mat_name, array_im)
            if fullbright_array_im is not None:
                fullbright_im = blendmat.im_from_array(f"{mat_name}_fullbright", fullbright_array_im)
                strength = mdl_cfg['strength']

                bm = blendmat.setup_fullbright_material(
                    blendmat.BlendMatImages.from_single_pair(im, fullbright_im),
                    mat_name,
                    strength,
                    warp=False
                )
            else:
                bm = blendmat.setup_diffuse_material(
                    blendmat.BlendMatImages.from_single_diffuse(im),
                    mat_name,
                    warp=False
                )
            bm.mat.cycles.sample_as_light = sample_as_light

            if sample_as_light:
                sample_as_light_mats.add(bm)

            BLENDMATS[mat_name] = bm
        bm = BLENDMATS[mat_name]

        # Apply the material
        mesh.materials.append(bm.mat)
        _set_uvs(mesh, am, tri_set)

    return BlendMdl(am, obj, sub_objs, sample_as_light_mats,
                    initial_pose_num, group_times, shape_keys)


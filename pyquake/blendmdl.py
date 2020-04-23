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

import io
from dataclasses import dataclass
from typing import Dict, Any

import bmesh
import bpy
import bpy_types
import numpy as np

from . import pak, mdl, blendmat


def _create_block(obj, simple_frame, vert_map):
    block = obj.shape_key_add(name=simple_frame.name)
    for old_vert_idx, block_vert in zip(vert_map, block.data):
        block_vert.co = simple_frame.frame_verts[old_vert_idx]
    return block


@dataclass
class BlendMdl:
    am: "AliasMdl"
    blocks: Dict
    obj: bpy_types.Object


def _animate(am, blocks, obj, frames, fps=30):
    prev_block = None
    prev_time = None
    for time, frame_num in frames:
        block = blocks[frame_num]

        block.value = 1.0
        block.keyframe_insert('value', frame=int(fps * time))
        if prev_block:
            block.value = 0.0
            block.keyframe_insert('value', frame=int(fps * prev_time))
            prev_block.value = 0.0
            prev_block.keyframe_insert('value', frame=int(fps * time))

        prev_block = block
        prev_time = time

    for c in obj.data.animation_data.action.fcurves:
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


def load_model(pak_root, mdl_name, obj_name, frames, skin_idx=0, fps=30):
    fs = pak.Filesystem(pak_root)
    am = mdl.AliasModel(fs.open(f"progs/{mdl_name}.mdl"))
    pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 255
    add_model(am, pal, mdl_name, obj_name, frames, skin_idx, fps)


def add_model(am, pal, mdl_name, obj_name, frames, skin_idx=0, fps=30):
    frames = list(frames)

    pal = np.concatenate([pal, np.ones(256)[:, None]], axis=1)

    obj = bpy.data.objects.new(obj_name, None)
    bpy.context.scene.collection.objects.link(obj)
    for tri_set_idx, tri_set in enumerate(am.disjoint_tri_sets):
        # Create the mesh and object
        subobj_name = f"{obj_name}_triset{tri_set_idx}"
        mesh = bpy.data.meshes.new(subobj_name)
        pydata, vert_map = _simplify_pydata([list(v) for v in am.frames[0].frame.frame_verts],
                                            [list(am.tris[t]) for t in tri_set])
        mesh.from_pydata(*pydata)
        subobj = bpy.data.objects.new(subobj_name, mesh)
        subobj.parent = obj
        bpy.context.scene.collection.objects.link(subobj)

        # Create shape key blocks, used for animation.
        blocks = {}
        for frame_num, frame in enumerate(am.frames):
            if frame.frame_type != mdl.FrameType.SINGLE:
                raise Exception(f"Frame type {frame.frame_type} not supported")
            simple_frame = frame.frame
            blocks[frame_num] = _create_block(subobj, simple_frame, vert_map)
        _animate(am, blocks, subobj, frames, fps)

        # Set up material
        fullbright_frac = _get_tri_set_fullbright_frac(am, tri_set, skin_idx)
        sample_as_light = fullbright_frac > 0.8
        mat_name = f"{mdl_name}_skin{skin_idx}"
        if sample_as_light:
            mat_name = f"{mat_name}_fullbright"
        if mat_name not in bpy.data.materials:
            mat, nodes, links = blendmat.new_mat(mat_name)
            array_im, fullbright_array_im, _ = blendmat.array_ims_from_indices(pal, am.skins[skin_idx])
            im = blendmat.im_from_array(mat_name, array_im)
            if fullbright_array_im is not None:
                fullbright_im = blendmat.im_from_array(f"{mat_name}_fullbright", fullbright_array_im)
                strength = 10_000. if sample_as_light else 1.0
                blendmat.setup_fullbright_material(nodes, links, im, fullbright_im, strength)
            else:
                blendmat.setup_diffuse_material(nodes, links, im)
            mat.cycles.sample_as_light = sample_as_light
        mat = bpy.data.materials[mat_name]

        # Apply the material
        mesh.materials.append(mat)
        _set_uvs(mesh, am, tri_set)

    return BlendMdl(am, blocks, obj)


import io
from dataclasses import dataclass
from typing import Dict, Any

import bmesh
import bpy
import bpy_types
import numpy as np

from . import pak, mdl, blendmat


def _create_block(obj, simple_frame):
    block = obj.shape_key_add(name=simple_frame.name)
    for vert, co in zip(block.data, simple_frame.frame_verts):
        vert.co = co
    return block


@dataclass
class BlendMdl:
    am: "AliasMdl"
    blocks: Dict
    obj: bpy_types.Object

    def animate(self, frames, fps=30):
        prev_block = None
        prev_time = None
        for time, frame_num in frames:
            simple_frame = self.am.frames[frame_num].frame

            block = self.blocks[simple_frame.name]

            block.value = 1.0
            block.keyframe_insert('value', frame=int(fps * time))
            if prev_block:
                block.value = 0.0
                block.keyframe_insert('value', frame=int(fps * prev_time))
                prev_block.value = 0.0
                prev_block.keyframe_insert('value', frame=int(fps * time))

            prev_block = block
            prev_time = time

        for c in self.obj.data.animation_data.action.fcurves:
            for kfp in c.keyframe_points:
                kfp.interpolation = 'LINEAR'


def _set_uvs(mesh, am):
    mesh.uv_layers.new()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_layer = bm.loops.layers.uv[0]

    for tri_idx, bm_face in enumerate(bm.faces):
        tcs = am.get_tri_tcs(tri_idx)

        for bm_loop, (s, t) in zip(bm_face.loops, tcs):
            bm_loop[uv_layer].uv = s / am.header['skin_width'], t / am.header['skin_height']
            
    bm.to_mesh(mesh)


def load_mdl(pak_root, mdl_name, obj_name, skin=0, fps=30):
    # Load the alias model
    fs = pak.Filesystem(pak_root)
    fname = f'progs/{mdl_name}.mdl'
    am = mdl.AliasModel(io.BytesIO(fs[fname]))

    # Create the mesh and object
    mesh = bpy.data.meshes.new(obj_name)
    mesh.from_pydata([list(v) for v in am.frames[0].frame.frame_verts], [], [list(t) for t in am.tris])
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    # Create shape key blocks, used for animation.
    blocks = {}
    for frame in am.frames:
        if frame.frame_type != mdl.FrameType.SINGLE:
            raise Exception(f"Frame type {frame.frame_type} not supported")
        simple_frame = frame.frame
        if simple_frame.name in blocks:
            raise Exception("Duplicate frame name")
        blocks[simple_frame.name] = _create_block(obj, simple_frame)

    # Set up material
    mat_name = f"{mdl_name}_{skin}"
    if mat_name not in bpy.data.materials:
        pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 255
        pal = np.concatenate([pal, np.ones(256)[:, None]], axis=1)

        mat, nodes, links = blendmat.new_mat(mat_name)
        array_im, fullbright_array_im = blendmat.array_ims_from_indices(mat_name, pal, am.skins[skin])
        im = blendmat.im_from_array(mat_name, array_im)
        if fullbright_array_im is not None:
            fullbright_im = blendmat.im_from_array(f"{mat_name}_fullbright", fullbright_array_im)
            blendmat.setup_fullbright_material(nodes, links, im, fullbright_im, 1.0)
        else:
            blendmat.setup_diffuse_material(nodes, links, im)
    mat = bpy.data.materials[mat_name]

    # Apply the material
    mesh.materials.append(mat)
    _set_uvs(mesh, am)

    return BlendMdl(am, blocks, obj)



import io
from dataclasses import dataclass
from typing import Dict, Any

import bpy
import bpy_types

from . import pak, mdl


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


def load_mdl(pak_root, mdl_name, fps=30):
    fs = pak.Filesystem(pak_root)
    fname = f'progs/{mdl_name}.mdl'
    am = mdl.AliasModel(io.BytesIO(fs[fname]))

    mesh = bpy.data.meshes.new(mdl_name)
    mesh.from_pydata([list(v) for v in am.frames[0].frame.frame_verts], [], [list(t) for t in am.tris])

    obj = bpy.data.objects.new(mdl_name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    blocks = {}
    for frame in am.frames:
        if frame.frame_type != mdl.FrameType.SINGLE:
            raise Exception(f"Frame type {frame.frame_type} not supported")
        simple_frame = frame.frame
        if simple_frame.name in blocks:
            raise Exception("Duplicate frame name")
        blocks[simple_frame.name] = _create_block(obj, simple_frame)

    return BlendMdl(am, blocks, obj)



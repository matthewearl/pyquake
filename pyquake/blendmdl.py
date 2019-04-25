import io

import bpy

from . import pak
from .mdl import AliasModel


def load_mdl(pak_root, mdl_name):
    fs = pak.Filesystem(pak_root)
    fname = f'progs/{mdl_name}.mdl'
    mdl = AliasModel(io.BytesIO(fs[fname]))

    mesh = bpy.data.meshes.new(mdl_name)
    mesh.from_pydata([list(v) for v in mdl.frames[0].frame.frame_verts], [], [list(t) for t in mdl.tris])

    obj = bpy.data.objects.new(mdl_name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    return obj


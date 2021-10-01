__all__ = (
    'add_model',
)


import bpy

from . import md3


def _create_shape_key(surf_obj, surf, frame_num):
    shape_key = surf_obj.shape_key_add(name=f'frame_{frame_num}')
    for vert_idx, shape_key_vert in enumerate(shape_key.data):
        shape_key_vert.co = surf.verts[frame_num, vert_idx]
    return shape_key


def add_model(m: md3.MD3, obj_name: str):
    obj = bpy.data.objects.new(obj_name, None)
    bpy.context.scene.collection.objects.link(obj)

    for surf in m.surfaces:
        subobj_name = f'{obj_name}_{surf.name}'
        mesh = bpy.data.meshes.new(subobj_name)
        mesh.from_pydata([list(v) for v in surf.verts[0]],
                         [],
                         [list(t) for t in surf.tris])

        surf_obj = bpy.data.objects.new(subobj_name, mesh)
        bpy.context.scene.collection.objects.link(surf_obj)

        surf_obj.parent = obj

        for frame_num in range(surf.num_frames):
            _create_shape_key(surf_obj, surf, frame_num)

    return obj

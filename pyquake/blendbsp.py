import io

import bpy

from . import bsp
from . import pak


def _get_face_vert_indices(bsp_file, face_idx):
    face = bsp_file.faces[face_idx]
    for edge_id in bsp_file.edge_list[face.edge_list_idx:face.edge_list_idx + face.num_edges]:
        if edge_id < 0:
            v = bsp_file.edges[-edge_id][1]
        else:
            v = bsp_file.edges[edge_id][0]
        yield v


def _add_mesh_obj(mesh, obj_name):
    scn = bpy.context.scene

    for o in scn.objects:
        o.select = False

    mesh.update()
    mesh.validate()

    nobj = bpy.data.objects.new(obj_name, mesh)
    scn.objects.link(nobj)
    nobj.select = True

    if scn.objects.active is None or scn.objects.active.mode == 'OBJECT':
        scn.objects.active = nobj


def load_bsp(pak_root, map_name):
    fs = pak.Filesystem(pak_root)
    fname = f'maps/{map_name}.bsp'
    bsp_file = bsp.BspFile(io.BytesIO(fs[fname]))

    mesh = bpy.data.meshes.new(map_name)

    model_faces = {i for m in bsp_file.models[1:]
                     for i in range(m.first_face_idx, m.first_face_idx + m.num_faces)}

    mesh.from_pydata(bsp_file.vertices, [],
                     [list(_get_face_vert_indices(bsp_file, face_idx))
                        for face_idx in range(len(bsp_file.faces))
                        if face_idx not in model_faces])

    _add_mesh_obj(mesh, map_name)

    return bsp_file

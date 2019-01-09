import io

import bpy
import numpy as np

from .bsp import Bsp
from . import pak


def _get_face_vert_indices(bsp, face_idx):
    face = bsp.faces[face_idx]
    for edge_id in bsp.edge_list[face.edge_list_idx:face.edge_list_idx + face.num_edges]:
        if edge_id < 0:
            v = bsp.edges[-edge_id][1]
        else:
            v = bsp.edges[edge_id][0]
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


def _load_material(pal, texture):
    # Read the image from the BSP texture
    array_im = pal[np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))]
    array_im = np.flip(array_im, axis=0)

    # Create the image object in blender
    im = bpy.data.images.new(texture.name, width=texture.width, height=texture.height)
    im.pixels = np.ravel(array_im)
    im.pack(as_png=True)

    # Create the material in blender
    mat = bpy.data.materials.new(texture.name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    diffuse_node = nodes['Diffuse BSDF']
    texture_node = nodes.new('ShaderNodeTexImage')

    links.new(diffuse_node.inputs['Color'], texture_node.outputs['Color'])


def _set_uvs(bsp):
    tex_coords = np.array(list(bsp.iter_face_tex_coords(face_idx)))


def _load_object(bsp, map_name):
    mesh = bpy.data.meshes.new(map_name)

    model_faces = {i for m in bsp.models[1:]
                     for i in range(m.first_face_idx, m.first_face_idx + m.num_faces)}

    mesh.from_pydata(bsp.vertices, [],
                     [list(bsp.iter_face_vert_indices(face_idx))
                        for face_idx in range(len(bsp.faces))
                        if face_idx not in model_faces])

    _add_mesh_obj(mesh, map_name)


def load_bsp(pak_root, map_name):
    fs = pak.Filesystem(pak_root)
    fname = f'maps/{map_name}.bsp'
    bsp = Bsp(io.BytesIO(fs[fname]))


    pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 256
    pal = np.concatenate([pal, np.ones(256)[:, None]], axis=1)
    
    for texture in bsp.textures:
        _load_material(pal, texture)

    return bsp

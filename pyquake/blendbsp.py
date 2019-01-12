import io
import logging

import bpy
import bmesh
import numpy as np

from .bsp import Bsp
from . import pak

_EXTRA_BRIGHT_TEXTURES = [
    'tlight02',
    'tlight07',
    'tlight11',
    'tlight01',
]


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

    nobj = bpy.data.objects.new(obj_name, mesh)
    scn.objects.link(nobj)
    nobj.select = True

    if scn.objects.active is None or scn.objects.active.mode == 'OBJECT':
        scn.objects.active = nobj


def _load_material(pal, texture):
    # Read the image from the BSP texture
    im_indices = np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))
    fullbright = np.flip((im_indices >= 224), axis=0)

    do_fullbright = np.any(fullbright)

    array_im = pal[np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))]
    array_im = np.flip(array_im, axis=0)

    array_im = 255 * (array_im / 255.) ** 0.8

    # Create the image object in blender
    im = bpy.data.images.new(texture.name, width=texture.width, height=texture.height)
    im.pixels = np.ravel(array_im)
    im.pack(as_png=True)

    if do_fullbright:
        # Create the glow image object in blender
        glow_im = bpy.data.images.new(f'{texture.name}_glow', width=texture.width, height=texture.height)
        glow_im.pixels = np.ravel(array_im * fullbright[..., None])
        glow_im.pack(as_png=True)

    # Create the material in blender
    mat = bpy.data.materials.new(texture.name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    while nodes:
        nodes.remove(nodes[0])
    while links:
        links.remove(links[0])
    
    texture_node = nodes.new('ShaderNodeTexImage')
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes.new('ShaderNodeOutputMaterial')

    texture_node.image = im
    links.new(diffuse_node.inputs['Color'], texture_node.outputs['Color'])

    if do_fullbright:
        add_node = nodes.new('ShaderNodeAddShader')
        glow_texture_node = nodes.new('ShaderNodeTexImage')
        emission_node = nodes.new('ShaderNodeEmission')

        glow_texture_node.image = glow_im
        links.new(emission_node.inputs['Color'], glow_texture_node.outputs['Color'])
        links.new(add_node.inputs[0], diffuse_node.outputs['BSDF'])
        links.new(add_node.inputs[1], emission_node.outputs['Emission'])
        links.new(output_node.inputs['Surface'], add_node.outputs['Shader'])

        if texture.name in _EXTRA_BRIGHT_TEXTURES:
            emission_node.inputs['Strength'].default_value = 100.
    else:
        links.new(output_node.inputs['Surface'], diffuse_node.outputs['BSDF'])


def _set_uvs(bsp, mesh, face_indices):
    mesh.uv_textures.new()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_layer = bm.loops.layers.uv[0]

    assert len(bm.faces) == len(face_indices)
    for bm_face, bsp_face_idx in zip(bm.faces, face_indices):
        bsp_face = bsp.faces[bsp_face_idx]
        assert bsp_face.num_edges == len(bm_face.loops)

        texture = bsp.textures[bsp.texinfo[bsp_face.texinfo_id].texture_id]
        
        for bm_loop, (s, t) in zip(bm_face.loops, bsp.iter_face_tex_coords(bsp_face_idx)):
            bm_loop[uv_layer].uv = s / texture.width, -t / texture.height

    bm.to_mesh(mesh)


def _apply_materials(bsp, mesh, face_indices):
    for texture in bsp.textures:
        mesh.materials.append(bpy.data.materials[texture.name])

    for mesh_poly, bsp_face_idx in zip(mesh.polygons, face_indices):
        bsp_face = bsp.faces[bsp_face_idx]
        mesh_poly.material_index = bsp.texinfo[bsp_face.texinfo_id].texture_id


def _load_object(bsp, map_name, do_materials):
    mesh = bpy.data.meshes.new(map_name)

    model_faces = {i for m in bsp.models[1:]
                     for i in range(m.first_face_idx, m.first_face_idx + m.num_faces)}

    sky_tex_id = next(iter(i for i, t in enumerate(bsp.textures) if t.name.startswith('sky')))

    face_indices = [face_idx for face_idx, face in enumerate(bsp.faces)
                        if bsp.texinfo[face.texinfo_id].texture_id != sky_tex_id
                        if face_idx not in model_faces]
    faces = [list(bsp.iter_face_vert_indices(face_idx)) for face_idx in face_indices]

    mesh.from_pydata(bsp.vertices, [], faces)

    if do_materials:
        _set_uvs(bsp, mesh, face_indices)
        _apply_materials(bsp, mesh, face_indices)

    mesh.validate()

    obj = bpy.data.objects.new(map_name, mesh)
    bpy.context.scene.objects.link(obj)


def load_bsp(pak_root, map_name, do_materials=True):
    fs = pak.Filesystem(pak_root)
    fname = f'maps/{map_name}.bsp'
    bsp = Bsp(io.BytesIO(fs[fname]))
    pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 256
    pal = np.concatenate([pal, np.ones(256)[:, None]], axis=1)

    if do_materials:
        for texture in bsp.textures:
            _load_material(pal, texture)

    m = _load_object(bsp, map_name, do_materials)
    
    return bsp

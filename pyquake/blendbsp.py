import io
import logging

import bpy
import bmesh
import numpy as np

from .bsp import Bsp, get_tex_coords
from . import pak


_EXTRA_BRIGHT_TEXTURES = [
    'tlight02',
    'tlight07',
    'tlight11',
    'tlight01',
    '*slime0',
]


_LIGHT_TINT = {
    'tlight11': [1., 0.85, 0.7, 1.],
    'tlight01': [1., 0.80, 1., 1.],
    'tlight07': [1., 1., 3.33, 1.],
}


_EMISSION_COLORS= {
    '*slime0': (0., 1., 0.),
}


_ALL_FULLBRIGHT_IN_OVERLAY = True
_FULLBRIGHT_OBJECT_OVERLAY = False

_USE_LUXCORE = False


def _texture_to_array(pal, texture):
    im_indices = np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))
    fullbright = (im_indices >= 224)

    if not np.any(fullbright):
        fullbright = None

    array_im = pal[np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))]
    array_im = array_im ** 0.8

    return array_im, fullbright


def _setup_diffuse_material(nodes, links, im):
    texture_node = nodes.new('ShaderNodeTexImage')
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes.new('ShaderNodeOutputMaterial')

    texture_node.image = im
    texture_node.interpolation = 'Closest'
    links.new(diffuse_node.inputs['Color'], texture_node.outputs['Color'])
    links.new(output_node.inputs['Surface'], diffuse_node.outputs['BSDF'])


def _setup_transparent_fullbright(nodes, links, im, glow_im, extra_bright=False):
    texture_node = nodes.new('ShaderNodeTexImage')
    emission_node = nodes.new('ShaderNodeEmission')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    glow_texture_node = nodes.new('ShaderNodeTexImage')
    mix_node = nodes.new('ShaderNodeMixShader')
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')

    texture_node.image = im
    texture_node.interpolation = 'Closest'
    glow_texture_node.image = glow_im
    glow_texture_node.interpolation = 'Closest'

    if extra_bright:
        emission_node.inputs['Strength'].default_value = 100.

    links.new(emission_node.inputs['Color'], texture_node.outputs['Color'])
    links.new(mix_node.inputs[0], glow_texture_node.outputs['Color'])
    links.new(mix_node.inputs[1], transparent_node.outputs['BSDF'])
    links.new(mix_node.inputs[2], emission_node.outputs['Emission'])
    links.new(output_node.inputs['Surface'], mix_node.outputs['Shader'])


def _setup_transparent_diffuse(nodes, links, im, glow_im):
    texture_node = nodes.new('ShaderNodeTexImage')
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    glow_texture_node = nodes.new('ShaderNodeTexImage')
    mix_node = nodes.new('ShaderNodeMixShader')
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')

    texture_node.image = im
    texture_node.interpolation = 'Closest'
    glow_texture_node.image = glow_im
    glow_texture_node.interpolation = 'Closest'

    links.new(diffuse_node.inputs['Color'], texture_node.outputs['Color'])
    links.new(mix_node.inputs[0], glow_texture_node.outputs['Color'])
    links.new(mix_node.inputs[1], diffuse_node.outputs['BSDF'])
    links.new(mix_node.inputs[2], transparent_node.outputs['BSDF'])
    links.new(output_node.inputs['Surface'], mix_node.outputs['Shader'])


def _setup_fullbright_material(nodes, links, im, glow_im, extra_bright=False):
    texture_node = nodes.new('ShaderNodeTexImage')
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    add_node = nodes.new('ShaderNodeAddShader')
    glow_texture_node = nodes.new('ShaderNodeTexImage')
    emission_node = nodes.new('ShaderNodeEmission')

    texture_node.image = im
    texture_node.interpolation = 'Closest'
    glow_texture_node.image = glow_im
    glow_texture_node.interpolation = 'Closest'

    if extra_bright:
        emission_node.inputs['Strength'].default_value = 100.

    links.new(diffuse_node.inputs['Color'], texture_node.outputs['Color'])
    links.new(emission_node.inputs['Color'], glow_texture_node.outputs['Color'])
    links.new(add_node.inputs[0], diffuse_node.outputs['BSDF'])
    links.new(add_node.inputs[1], emission_node.outputs['Emission'])
    links.new(output_node.inputs['Surface'], add_node.outputs['Shader'])


def _blender_im_from_array(name, array_im):
    im = bpy.data.images.new(name, width=array_im.shape[1], height=array_im.shape[0])
    im.pixels = np.ravel(array_im)
    #im.pack(as_png=True)
    return im


def _blender_new_mat(name):
    mat = bpy.data.materials.new(name)

    if _USE_LUXCORE:
        mat.luxcore.node_tree = bpy.data.node_groups.new(name=name, type="luxcore_material_nodes")
        nodes = mat.luxcore.node_tree.nodes
        links = mat.luxcore.node_tree.links
    else:
        mat.use_nodes = True

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        while nodes:
            nodes.remove(nodes[0])
        while links:
            links.remove(links[0])

    return mat, nodes, links


def _setup_luxcore_material(nodes, links, im, glow_im, extra_bright=False, emission_color=None):
    matte_node = nodes.new('LuxCoreNodeMatGlossy2')
    output_node = nodes.new('LuxCoreNodeMatOutput')
    #diffuse_node = nodes.new('LuxCoreNodeTexImagemap')
    matte_node.inputs['Roughness'].default_value = 0.3

    if glow_im is not None or emission_color is not None:
        emission_node = nodes.new('LuxCoreNodeMatEmission')
        if extra_bright:
            emission_node.gain = 100

    if glow_im is not None:
        glow_node = nodes.new('LuxCoreNodeTexImagemap')
    elif emission_color is not None:
        emission_node.inputs['Color'].default_value = emission_color

    #diffuse_node.image = im
    if glow_im is not None:
        glow_node.image = glow_im

    links.new(output_node.inputs['Material'], matte_node.outputs['Material'])
    #links.new(matte_node.inputs['Diffuse Color'], diffuse_node.outputs['Color'])
    if glow_im is not None or emission_color is not None:
        links.new(matte_node.inputs['Emission'], emission_node.outputs['Emission'])
    if glow_im is not None:
        links.new(emission_node.inputs['Color'], glow_node.outputs['Color'])
    

def _load_images(pal, bsp):
    ims = []
    fullbright_ims = []
    for texture_id, texture in enumerate(bsp.textures):
        array_im, fullbright = _texture_to_array(pal, texture)
        im = _blender_im_from_array(texture.name, array_im)
        if fullbright is not None:
            glow_im = array_im * fullbright[..., None]
            if texture.name in _LIGHT_TINT:
                glow_im *= np.array(_LIGHT_TINT[texture.name])
            glow_im = np.clip(glow_im, 0., 1.)
            fullbright_im = _blender_im_from_array('{}_fullbright'.format(texture.name), glow_im)
        else:
            fullbright_im = None

        ims.append(im)
        fullbright_ims.append(fullbright_im)

    return ims, fullbright_ims


def _load_material(texture_id, texture, ims, fullbright_ims):
    im = ims[texture_id]
    fullbright_im = fullbright_ims[texture_id]

    mat, nodes, links = _blender_new_mat('{}_main'.format(texture.name))

    extra_bright = texture.name in _EXTRA_BRIGHT_TEXTURES

    if _USE_LUXCORE:
        emission_color = _EMISSION_COLORS.get(texture.name)
        _setup_luxcore_material(nodes, links, im, fullbright_im, extra_bright, emission_color)
    elif fullbright_im is not None:
        if _FULLBRIGHT_OBJECT_OVERLAY and (_ALL_FULLBRIGHT_IN_OVERLAY or extra_bright):
            _setup_diffuse_material(nodes, links, im)
        else:
            _setup_fullbright_material(nodes, links, im, fullbright_im, extra_bright)
    else:
        _setup_diffuse_material(nodes, links, im)


def _load_fullbright_obj_material(texture_id, texture, ims, fullbright_ims):
    im = ims[texture_id]
    fullbright_im = fullbright_ims[texture_id]
    if fullbright_im is not None:
        mat, nodes, links = _blender_new_mat('{}_fullbright'.format(texture.name))
        _setup_transparent_fullbright(nodes, links, im, fullbright_im, texture.name in _EXTRA_BRIGHT_TEXTURES)


def _set_uvs(mesh, textures, texinfos, vertices, faces):
    mesh.uv_textures.new()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_layer = bm.loops.layers.uv[0]

    assert len(bm.faces) == len(faces)
    assert len(bm.faces) == len(texinfos)
    assert len(bm.faces) == len(textures)
    for bm_face, face, texinfo, texture in zip(bm.faces, faces, texinfos, textures):
        assert len(face) == len(bm_face.loops)
        for bm_loop, vert_idx in zip(bm_face.loops, face):
            vert = vertices[vert_idx]
            s, t = get_tex_coords(texinfo, vert)
            bm_loop[uv_layer].uv = s / texture.width, t / texture.height

    bm.to_mesh(mesh)


def _apply_materials(bsp, mesh, face_indices, mat_suffix):
    tex_id_to_mat_idx = {}
    mat_idx = 0
    for texture_id, texture in enumerate(bsp.textures):
        mat_name = '{}_{}'.format(texture.name, mat_suffix)
        if mat_name in bpy.data.materials:
            mesh.materials.append(bpy.data.materials[mat_name])
            tex_id_to_mat_idx[texture_id] = mat_idx
            mat_idx += 1

    for mesh_poly, bsp_face_idx in zip(mesh.polygons, face_indices):
        bsp_face = bsp.faces[bsp_face_idx]
        mesh_poly.material_index = tex_id_to_mat_idx[bsp.texinfo[bsp_face.texinfo_id].texture_id]


def _get_visible_face_indices(bsp):
    # Exclude faces from all but the first model (hides doors, buttons, etc).
    model_faces = {i for m in bsp.models[1:]
                     for i in range(m.first_face_idx, m.first_face_idx + m.num_faces)}
    model_faces = {}

    # Don't render the sky (we just use the world surface shader / sun light for that)
    banned_tex_ids = {i for i, t in enumerate(bsp.textures)
                        if t.name.startswith('sky') or t.name == 'trigger'}
    return [face_idx for face_idx, face in enumerate(bsp.faces)
                        if bsp.texinfo[face.texinfo_id].texture_id not in banned_tex_ids
                        if face_idx not in model_faces]


def _get_bbox(a):
    out = []
    for axis in range(2):
        b = np.where(np.any(a, axis=axis))
        out.append([np.min(b), np.max(b)])
    return np.array(out).T


def _get_face_normal(vert_indices, vertices):
    first_edge = None
    best_normal = None

    prev_vert = vertices[vert_indices[-1]]
    for vert_index in vert_indices:
        vert = vertices[vert_index]

        edge = np.array(vert) - np.array(prev_vert)
        edge /= np.linalg.norm(edge)
        if first_edge is None:
            first_edge = edge
        else:
            normal = np.cross(edge, first_edge)
            if best_normal is None or np.linalg.norm(best_normal) < np.linalg.norm(normal):
                best_normal = normal
        prev_vert = vert

    return normal / np.linalg.norm(normal)


def _offset_face(vert_indices, vertices, distance):
    new_vertices, new_vert_indices = [], []
    normal = _get_face_normal(vert_indices, vertices)

    for vert_index in vert_indices:
        vert = vertices[vert_index]

        new_vert = vert + distance * normal
        new_vert_indices.append(len(vertices) + len(new_vertices))
        new_vertices.append(new_vert)

    return new_vert_indices, vertices + new_vertices


def _truncate_face(vert_indices, vertices, normal, plane_dist):
    if len(vert_indices) == 0:
        return [], vertices

    new_vertices, new_vert_indices = [], []
    prev_vert = vertices[vert_indices[-1]]
    for vert_index in vert_indices:
        vert = vertices[vert_index]
        dist = np.dot(vert, normal) - plane_dist
        prev_dist = np.dot(prev_vert, normal) - plane_dist

        if (prev_dist >= 0) != (dist >= 0):
            alpha = -dist / (prev_dist - dist)
            new_vert = tuple(alpha * np.array(prev_vert) + (1 - alpha) * np.array(vert))
            new_vert_indices.append(len(vertices) + len(new_vertices))
            new_vertices.append(new_vert)

        if dist >= 0:
            new_vert_indices.append(vert_index)

        prev_vert = vert

    return new_vert_indices, vertices + new_vertices


def _load_fullbright_object(bsp, map_name, pal, do_materials):
    # Calculate bounding boxes for regions of full brightness.
    bbox_ids, bboxes = [], []
    for texture in bsp.textures:
        _, fullbright_array_im = _texture_to_array(pal, texture)
        if fullbright_array_im is not None:
            bbox_ids.append(len(bboxes))
            bboxes.append(_get_bbox(fullbright_array_im))
        else:
            bbox_ids.append(None)

    # For each fullbright face in the original BSP, create a set of new faces, one for each wrap of the texture image.
    # The new faces bounds the fullbright texels for that particular wrap of the texture
    new_faces, new_face_indices = [], []
    vertices = bsp.vertices
    for face_idx in _get_visible_face_indices(bsp):
        texinfo = bsp.texinfo[bsp.faces[face_idx].texinfo_id]
        texture_id = texinfo.texture_id
        texture = bsp.textures[texture_id]
        bbox_id = bbox_ids[texture_id]
        if bbox_id is None or (not _ALL_FULLBRIGHT_IN_OVERLAY and texture.name not in _EXTRA_BRIGHT_TEXTURES):
            continue

        bbox = bboxes[bbox_id]

        tex_size = np.array([texture.width, texture.height])
        vert_indices = list(bsp.iter_face_vert_indices(face_idx))
        tex_coords = np.array(list(bsp.iter_face_tex_coords(face_idx)))

        face_bbox = np.stack([np.min(tex_coords, axis=0), np.max(tex_coords, axis=0)])
        
        # Iterate over each potential wraps of the texture.  Number of wraps is determined using bounding boxes in
        # texture space.
        start_indices = np.ceil((face_bbox[0] - bbox[1]) / tex_size).astype(np.int)
        end_indices = np.ceil((face_bbox[1] - bbox[0]) / tex_size).astype(np.int)
        for t_offset in range(start_indices[1], end_indices[1]):
            for s_offset in range(start_indices[0], end_indices[0]):
                new_vert_indices = vert_indices
                planes = [(np.array(texinfo.vec_s), s_offset * tex_size[0] + bbox[0, 0] - texinfo.dist_s),
                          (-np.array(texinfo.vec_s), -(s_offset * tex_size[0] + bbox[1, 0] - texinfo.dist_s)),
                          (np.array(texinfo.vec_t), t_offset * tex_size[1] + bbox[0, 1] - texinfo.dist_t),
                          (-np.array(texinfo.vec_t), -(t_offset * tex_size[1] + bbox[1, 1] - texinfo.dist_t))]
                for n, d in planes:
                    new_vert_indices, vertices = _truncate_face(new_vert_indices, vertices, n, d)
                new_vert_indices, vertices = _offset_face(new_vert_indices, vertices, -0.01)
                if new_vert_indices:
                    new_faces.append(new_vert_indices)
                    new_face_indices.append(face_idx)


    # Actually make the mash and add it to the scene
    mesh = bpy.data.meshes.new(map_name)
    mesh.from_pydata(vertices, [], new_faces)

    obj = bpy.data.objects.new('{}_fullbright'.format(map_name), mesh)
    bpy.context.scene.objects.link(obj)

    if do_materials:
        texinfos = [bsp.texinfo[bsp.faces[face_idx].texinfo_id] for face_idx in new_face_indices]
        textures = [bsp.textures[texinfo.texture_id] for texinfo in texinfos]
        _set_uvs(mesh, textures, texinfos, vertices, new_faces)
        _apply_materials(bsp, mesh, new_face_indices, 'fullbright')


def _load_object(bsp, map_name, do_materials):
    face_indices = _get_visible_face_indices(bsp)
    faces = [list(bsp.iter_face_vert_indices(face_idx)) for face_idx in face_indices]

    mesh = bpy.data.meshes.new(map_name)
    mesh.from_pydata(bsp.vertices, [], faces)

    if do_materials:
        texinfos = [bsp.texinfo[bsp.faces[face_idx].texinfo_id] for face_idx in face_indices]
        textures = [bsp.textures[texinfo.texture_id] for texinfo in texinfos]
        _set_uvs(mesh, textures, texinfos, bsp.vertices, faces)
        _apply_materials(bsp, mesh, face_indices, 'main')

    mesh.validate()

    obj = bpy.data.objects.new(map_name, mesh)
    bpy.context.scene.objects.link(obj)


def load_bsp(pak_root, map_name, do_materials=True):
    print("hello")

    fs = pak.Filesystem(pak_root)
    fname = 'maps/{}.bsp'.format(map_name)
    bsp = Bsp(io.BytesIO(fs[fname]))
    pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 255
    pal = np.concatenate([pal, np.ones(256)[:, None]], axis=1)

    if do_materials:
        ims, fullbright_ims = _load_images(pal, bsp)

    if do_materials:
        for texture_id, texture in enumerate(bsp.textures):
            _load_material(texture_id, texture, ims, fullbright_ims)

    _load_object(bsp, map_name, do_materials)

    if _FULLBRIGHT_OBJECT_OVERLAY:
        if do_materials:
            for texture_id, texture in enumerate(bsp.textures):
                _load_fullbright_obj_material(texture_id, texture, ims, fullbright_ims)

        _load_fullbright_object(bsp, map_name, pal, do_materials)

    return bsp

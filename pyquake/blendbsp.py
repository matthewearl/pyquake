import io
import logging

import bpy
import bmesh
import numpy as np

from .bsp import Bsp
from . import pak

_EXTRA_BRIGHT_TEXTURES = [
    'tlight02',
    #'tlight07',
    #'tlight11',
    #'tlight01',
]


def _texture_to_array(pal, texture):
    im_indices = np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))
    fullbright = (im_indices >= 224)

    if not np.any(fullbright):
        fullbright = None

    array_im = pal[np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))]
    array_im = 255 * (array_im / 255.) ** 0.8

    return array_im, fullbright


def _load_material(pal, texture):
    # Read the image from the BSP texture
    array_im, fullbright = _texture_to_array(pal, texture)

    # Create the image object in blender
    im = bpy.data.images.new(texture.name, width=texture.width, height=texture.height)
    im.pixels = np.ravel(array_im)
    im.pack(as_png=True)

    if fullbright is not None:
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
    texture_node.interpolation = 'Closest'
    links.new(diffuse_node.inputs['Color'], texture_node.outputs['Color'])

    if fullbright is not None:
        add_node = nodes.new('ShaderNodeAddShader')
        glow_texture_node = nodes.new('ShaderNodeTexImage')
        emission_node = nodes.new('ShaderNodeEmission')

        glow_texture_node.image = glow_im
        glow_texture_node.interpolation = 'Closest'
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
            bm_loop[uv_layer].uv = s / texture.width, t / texture.height

    bm.to_mesh(mesh)


def _apply_materials(bsp, mesh, face_indices):
    for texture in bsp.textures:
        mesh.materials.append(bpy.data.materials[texture.name])

    for mesh_poly, bsp_face_idx in zip(mesh.polygons, face_indices):
        bsp_face = bsp.faces[bsp_face_idx]
        mesh_poly.material_index = bsp.texinfo[bsp_face.texinfo_id].texture_id


def _get_visible_face_indices(bsp):
    # Exclude faces from all but the first model (hides doors, buttons, etc).
    model_faces = {i for m in bsp.models[1:]
                     for i in range(m.first_face_idx, m.first_face_idx + m.num_faces)}

    # Don't render the sky (we just use the world surface shader / sun light for that)
    sky_tex_id = next(iter(i for i, t in enumerate(bsp.textures) if t.name.startswith('sky')))
    return [face_idx for face_idx, face in enumerate(bsp.faces)
                        if bsp.texinfo[face.texinfo_id].texture_id != sky_tex_id
                        if face_idx not in model_faces]


def _get_bbox(a):
    out = []
    for axis in range(2):
        b = np.where(np.any(a, axis=axis))
        out.append([np.min(b), np.max(b)])
    return np.array(out).T


def _truncate_face(vert_indices, vertices, normal, plane_dist):
    if len(vert_indices) == 0:
        return [], vertices

    new_vertices, new_vert_indices = [], []
    prev_vert = vertices[vert_indices[-1]]
    for vert_index in vert_indices:
        vert = vertices[vert_index]
        dist = np.dot(vert, normal) - plane_dist
        prev_dist = np.dot(prev_vert, normal) - plane_dist

        print('prev vert', prev_vert, 'vert', vert, 'dist', dist, 'prev_dist', prev_dist)

        if (prev_dist >= 0) != (dist >= 0):
            alpha = -dist / (prev_dist - dist)
            new_vert = tuple(alpha * np.array(prev_vert) + (1 - alpha) * np.array(vert))
            new_vert_indices.append(len(vertices) + len(new_vertices))
            new_vertices.append(new_vert)

        if dist >= 0:
            new_vert_indices.append(vert_index)

        prev_vert = vert

    return new_vert_indices, vertices + new_vertices


def _load_fullbright_object(bsp, map_name, pal):

    # Calculate texture colours, fullbright boolean values, bboxes on the fullbrights, and a map from texture ids to
    # indices into the arrays just created.
    ims, fullbrights, bboxes, fullbright_ids = [], [], [], []
    for texture in bsp.textures:
        im, fullbright = _texture_to_array(pal, texture)
        if fullbright is not None:
            fullbright_ids.append(len(fullbrights))
            fullbrights.append(fullbright)
            bboxes.append(_get_bbox(fullbright))
            ims.append(im)
        else:
            fullbright_ids.append(None)

    # For each fullbright face in the original BSP, create a set of new faces, one for each wrap of the texture image.
    # The new faces bounds the fullbright texels for that particular wrap of the texture
    new_faces = []
    vertices = bsp.vertices
    for face_idx in _get_visible_face_indices(bsp):
        texinfo = bsp.texinfo[bsp.faces[face_idx].texinfo_id]
        texture_id = texinfo.texture_id
        texture = bsp.textures[texture_id]
        fullbright_id = fullbright_ids[texture_id]
        if fullbright_id is None or texture.name not in _EXTRA_BRIGHT_TEXTURES:
            continue

        #im = ims[fullbright_id]
        #fullbright = fullbrights[fullbright_id]
        bbox = bboxes[fullbright_id]

        tex_size = np.array([texture.width, texture.height])
        vert_indices = list(bsp.iter_face_vert_indices(face_idx))
        tex_coords = np.array(list(bsp.iter_face_tex_coords(face_idx)))

        face_bbox = np.stack([np.min(tex_coords, axis=0), np.max(tex_coords, axis=0)])
        print('texinfo', texinfo)
        print('face_bbox', face_bbox)
        print('bbox', bbox)
        print('tex_size', tex_size)
        
        # Iterate over each potential wraps of the texture.  Number of wraps is determined using bounding boxes in
        # texture space.
        start_indices = np.ceil((face_bbox[0] - bbox[1]) / tex_size).astype(np.int)
        end_indices = np.ceil((face_bbox[1] - bbox[0]) / tex_size).astype(np.int)
        print(start_indices, end_indices)
        for t_offset in range(start_indices[1], end_indices[1]):
            for s_offset in range(start_indices[0], end_indices[0]):
                new_vert_indices = vert_indices
                planes = [(np.array(texinfo.vec_s), s_offset * tex_size[0] + bbox[0, 0] - texinfo.dist_s),
                          (-np.array(texinfo.vec_s), -(s_offset * tex_size[0] + bbox[1, 0] - texinfo.dist_s)),
                          (np.array(texinfo.vec_t), t_offset * tex_size[1] + bbox[0, 1] - texinfo.dist_t),
                          (-np.array(texinfo.vec_t), -(t_offset * tex_size[1] + bbox[1, 1] - texinfo.dist_t))]
                print('Vertices before truncating\n', np.array([vertices[i] for i in vert_indices]))
                for n, d in planes:
                    print(f'Truncating {n} {d}')
                    new_vert_indices, vertices = _truncate_face(new_vert_indices, vertices, n, d)
                    print('Vertices after truncating\n', np.array([vertices[i] for i in new_vert_indices]))
                print('--')

                print(f'Created a face with {len(new_vert_indices)} faces. vert count = {len(vertices)}')
                if new_vert_indices:
                    new_faces.append(new_vert_indices)

        break

    # Actually make the mash and add it to the scene
    mesh = bpy.data.meshes.new(map_name)
    mesh.from_pydata(vertices, [], new_faces)

    obj = bpy.data.objects.new(f'{map_name}_fullbright', mesh)
    bpy.context.scene.objects.link(obj)


def _load_object(bsp, map_name, do_materials):
    face_indices = _get_visible_face_indices(bsp)
    faces = [list(bsp.iter_face_vert_indices(face_idx)) for face_idx in face_indices]

    mesh = bpy.data.meshes.new(map_name)
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

    _load_object(bsp, map_name, do_materials)
    _load_fullbright_object(bsp, map_name, pal)
    
    return bsp

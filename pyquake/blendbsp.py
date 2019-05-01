import io
import logging
from typing import NamedTuple, Optional, Any, Dict

import bpy
import bmesh
import numpy as np

from .bsp import Bsp, Face
from . import pak, blendmat


_ALL_FULLBRIGHT_IN_OVERLAY = True
_FULLBRIGHT_OBJECT_OVERLAY = True


def _texture_to_arrays(pal, texture, light_tint=(1, 1, 1, 1)):
    im_indices = np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))
    return blendmat.array_ims_from_indices(pal, im_indices, light_tint=light_tint, gamma=0.8)


def _load_images(pal, bsp, map_cfg):
    ims = []
    fullbright_ims = []
    for texture_id, texture in enumerate(bsp.textures):
        tex_cfg = _get_texture_config(texture, map_cfg)
        array_im, fullbright_array_im, _ = _texture_to_arrays(pal, texture, tex_cfg['tint'])
        im = blendmat.im_from_array(texture.name, array_im)

        if fullbright_array_im is not None:
            fullbright_im = blendmat.im_from_array(f"{texture.name}_fullbright", fullbright_array_im)
        else:
            fullbright_im = None

        ims.append(im)
        fullbright_ims.append(fullbright_im)

    return ims, fullbright_ims


def _get_texture_config(texture, map_config):
    cfg = map_config['textures']['__default__']
    cfg.update(map_config['textures'].get(texture.name, {}))
    return cfg


def _load_material(texture_id, texture, ims, fullbright_ims, map_cfg):
    im = ims[texture_id]
    fullbright_im = fullbright_ims[texture_id]

    mat, nodes, links = blendmat.new_mat('{}_main'.format(texture.name))

    tex_cfg = _get_texture_config(texture, map_cfg)

    if fullbright_im is not None:
        if map_cfg['fullbright_object_overlay'] and tex_cfg['overlay']:
            blendmat.setup_diffuse_material(nodes, links, im)
            mat.cycles.sample_as_light = False
        else:
            blendmat.setup_fullbright_material(nodes, links, im, fullbright_im, tex_cfg['strength'])
            mat.cycles.sample_as_light = tex_cfg['sample_as_light']
    else:
        blendmat.setup_diffuse_material(nodes, links, im)
        mat.cycles.sample_as_light = False


def _load_fullbright_obj_material(texture_id, texture, ims, fullbright_ims, map_cfg):
    im = ims[texture_id]
    fullbright_im = fullbright_ims[texture_id]
    if fullbright_im is not None:
        mat, nodes, links = blendmat.new_mat('{}_fullbright'.format(texture.name))
        tex_cfg = _get_texture_config(texture, map_cfg)
        blendmat.setup_transparent_fullbright_material(nodes, links, im, fullbright_im, tex_cfg['strength'])
        mat.cycles.sample_as_light = tex_cfg['sample_as_light']


def _set_uvs(mesh, texinfos, faces):
    mesh.uv_layers.new()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_layer = bm.loops.layers.uv[0]

    assert len(bm.faces) == len(faces)
    assert len(bm.faces) == len(texinfos)
    for bm_face, face, texinfo in zip(bm.faces, faces, texinfos):
        assert len(face) == len(bm_face.loops)
        for bm_loop, vert in zip(bm_face.loops, face):
            s, t = texinfo.vert_to_tex_coords(vert)
            bm_loop[uv_layer].uv = s / texinfo.texture.width, t / texinfo.texture.height

    bm.to_mesh(mesh)


def _apply_materials(model, mesh, bsp_faces, mat_suffix):
    tex_to_mat_idx = {}
    mat_idx = 0
    for texture in {f.tex_info.texture for f in model.faces}:
        mat_name = '{}_{}'.format(texture.name, mat_suffix)
        if mat_name in bpy.data.materials:
            mesh.materials.append(bpy.data.materials[mat_name])
            tex_to_mat_idx[texture] = mat_idx
            mat_idx += 1

    for mesh_poly, bsp_face in zip(mesh.polygons, bsp_faces):
        mesh_poly.material_index = tex_to_mat_idx[bsp_face.tex_info.texture]


def _get_visible_faces(model):
    return [(face_id, face)
            for face_id, face in zip(range(model.first_face_idx, model.first_face_idx + model.num_faces), model.faces)
            if face.tex_info.texture.name != 'trigger'
            if not face.tex_info.texture.name.startswith('sky')]


def _get_bbox(a):
    out = []
    for axis in range(2):
        b = np.where(np.any(a, axis=axis))
        out.append([np.min(b), np.max(b)])
    return np.array(out).T


def _get_face_normal(vertices):
    first_edge = None
    best_normal = None

    prev_vert = vertices[-1]
    for vert in vertices:
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


def _offset_face(vertices, distance):
    new_vertices = []
    normal = _get_face_normal(vertices)

    for vert in vertices:
        new_vertices.append(tuple(vert + distance * normal))

    return new_vertices


def _truncate_face(vertices, normal, plane_dist):
    if len(vertices) == 0:
        return vertices

    new_vertices = []
    prev_vert = vertices[-1]
    for vert in vertices:
        dist = np.dot(vert, normal) - plane_dist
        prev_dist = np.dot(prev_vert, normal) - plane_dist

        if (prev_dist >= 0) != (dist >= 0):
            alpha = -dist / (prev_dist - dist)
            new_vert = tuple(alpha * np.array(prev_vert) + (1 - alpha) * np.array(vert))
            new_vertices.append(new_vert)

        if dist >= 0:
            new_vertices.append(vert)

        prev_vert = vert

    return new_vertices


def _pydata_from_faces(tuple_faces):
    d = {}
    int_faces = []
    for tuple_face in tuple_faces:
        int_face = []
        for vert in tuple_face:
            if id(vert) not in d:
                d[id(vert)] = (len(d), vert)
            int_face.append(d[id(vert)][0])
        int_faces.append(int_face)

    verts = [None] * len(d)
    for i, vert in d.values():
        verts[i] = vert
    assert None not in verts

    return verts, [], int_faces


def _load_fullbright_objects(model, map_name, pal, do_materials, map_cfg):
    # Calculate bounding boxes for regions of full brightness.
    bboxes = {}
    for texture in {f.tex_info.texture for f in model.faces}:
        _, _, fullbright_array = _texture_to_arrays(pal, texture)
        if np.any(fullbright_array):
            bboxes[texture] = _get_bbox(fullbright_array)

    fullbright_objects = {}

    # For each fullbright face in the original BSP, create a set of new faces, one for each wrap of the texture image.
    # The new faces bounds the fullbright texels for that particular wrap of the texture
    for face_id, face in _get_visible_faces(model):
        new_faces, new_bsp_faces = [], []

        texinfo = face.tex_info
        texture = texinfo.texture
        bbox = bboxes.get(texture)
        tex_cfg = _get_texture_config(texture, map_cfg)
        if bbox is None or not tex_cfg['overlay']:
            continue

        tex_size = np.array([texture.width, texture.height])
        face_verts = list(face.vertices)
        tex_coords = np.array(list(face.tex_coords))
        face_bbox = np.stack([np.min(tex_coords, axis=0), np.max(tex_coords, axis=0)])
        
        # Iterate over each potential wraps of the texture.  Number of wraps is determined using bounding boxes in
        # texture space.
        start_indices = np.ceil((face_bbox[0] - bbox[1]) / tex_size).astype(np.int)
        end_indices = np.ceil((face_bbox[1] - bbox[0]) / tex_size).astype(np.int)
        for t_offset in range(start_indices[1], end_indices[1]):
            for s_offset in range(start_indices[0], end_indices[0]):
                new_face = face_verts
                planes = [(np.array(texinfo.vec_s), s_offset * tex_size[0] + bbox[0, 0] - texinfo.dist_s),
                          (-np.array(texinfo.vec_s), -(s_offset * tex_size[0] + bbox[1, 0] - texinfo.dist_s)),
                          (np.array(texinfo.vec_t), t_offset * tex_size[1] + bbox[0, 1] - texinfo.dist_t),
                          (-np.array(texinfo.vec_t), -(t_offset * tex_size[1] + bbox[1, 1] - texinfo.dist_t))]
                for n, d in planes:
                    new_face = _truncate_face(new_face, n, d)

                if new_face:
                    new_face = _offset_face(new_face, -0.01)
                    new_faces.append(new_face)
                    new_bsp_faces.append(face)

        # Actually make the mesh and add it to the scene
        mesh = bpy.data.meshes.new(map_name)
        mesh.from_pydata(*_pydata_from_faces(new_faces))

        obj = bpy.data.objects.new(f'{map_name}_fullbright_{face_id}', mesh)
        bpy.context.scene.collection.objects.link(obj)

        fullbright_objects[face] = obj

        if do_materials:
            texinfos = [face.tex_info for face in new_bsp_faces]
            _set_uvs(mesh, texinfos, new_faces)
            _apply_materials(model, mesh, new_bsp_faces, 'fullbright')

    return fullbright_objects


def _load_object(model_id, model, map_name, do_materials):
    bsp_faces = [face for _, face in _get_visible_faces(model)]
    faces = [list(bsp_face.vertices) for bsp_face in bsp_faces]

    name = f"{map_name}_{model_id}"

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(*_pydata_from_faces(faces))

    if do_materials:
        texinfos = [bsp_face.tex_info for bsp_face in bsp_faces]
        _set_uvs(mesh, texinfos, faces)
        _apply_materials(model, mesh, bsp_faces, 'main')

    mesh.validate()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    return obj


class BlendBsp(NamedTuple):
    bsp: Bsp
    fullbright_objects: Optional[Dict[Face, Any]]

    def hide_invisible_fullbright_objects(self, pos):
        leaf = self.bsp.models[0].get_leaf(pos)
        visible_leaves = {next_leaf for leaf in leaf.visible_leaves for next_leaf in leaf.visible_leaves}
        visible_faces = {f for l in visible_leaves for f in l.faces}

        for face in self.bsp.models[0].faces:
            if face in self.fullbright_objects:
                self.fullbright_objects[face].hide_render = face not in visible_faces


def load_bsp(pak_root, map_name, config):
    fs = pak.Filesystem(pak_root)
    fname = 'maps/{}.bsp'.format(map_name)
    bsp = Bsp(io.BytesIO(fs[fname]))
    pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 255
    pal = np.concatenate([pal, np.ones(256)[:, None]], axis=1)

    map_cfg = config['maps'][fname]

    if map_cfg['do_materials']:
        ims, fullbright_ims = _load_images(pal, bsp, map_cfg)

        for texture_id, texture in enumerate(bsp.textures):
            _load_material(texture_id, texture, ims, fullbright_ims, map_cfg)
    
        if map_cfg['fullbright_object_overlay']:
            for texture_id, texture in enumerate(bsp.textures):
                _load_fullbright_obj_material(texture_id, texture, ims, fullbright_ims, map_cfg)

    map_obj = bpy.data.objects.new(map_name, None)
    bpy.context.scene.collection.objects.link(map_obj)

    fullbright_objects = {}
    for model_id, model in enumerate(bsp.models):
        model_obj = _load_object(model_id, model, map_name, map_cfg['do_materials'])
        model_obj.parent = map_obj

        if map_cfg['fullbright_object_overlay']:
            model_fullbright_objects = _load_fullbright_objects(model, map_name, pal, map_cfg['do_materials'], map_cfg)
        else:
            model_fullbright_objects = {}

        for obj in model_fullbright_objects.values():
            obj.parent = model_obj

        fullbright_objects.update(model_fullbright_objects)

    return BlendBsp(bsp, fullbright_objects)

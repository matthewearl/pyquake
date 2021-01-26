# Copyright (c) 2020 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


import collections
import functools
import logging
import operator
from typing import NamedTuple, Optional, Any, Dict, Set

import bpy
import bpy_types
import bmesh
import numpy as np

from .bsp import Bsp, Face, Leaf, Model, Texture
from . import pak, blendmat


def _texture_to_arrays(pal, texture, light_tint=(1, 1, 1, 1)):
    im_indices = np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))
    return blendmat.array_ims_from_indices(pal, im_indices, light_tint=light_tint, gamma=0.8)


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


def _get_texture_config(texture, map_cfg):
    cfg = dict(map_cfg['textures']['__default__'])
    cfg.update(map_cfg['textures'].get(texture.name, {}))
    return cfg


def _get_anim_textures(texture: Texture, texture_dict: Dict[str, Texture]) -> blendmat.BlendMatImages:
    if texture.name.startswith('+0'):
        print([k for k in texture_dict.keys() if k.startswith('+')])
        main_textures = []
        for i in range(10):
            tex_name = f'+{i}{texture.name[2:]}'
            print(tex_name)
            if tex_name in texture_dict:
                main_textures.append(texture_dict[tex_name])
            else:
                break

        alt_textures = []
        for i in range(10):
            tex_name_lower = f'+{chr(ord("a") + i)}texture.name[2:]'
            tex_name_upper = f'+{chr(ord("A") + i)}texture.name[2:]'
            if tex_name_lower in texture_dict:
                main_textures.append(texture_dict[tex_name_lower])
            elif tex_name_upper in texture_dict:
                main_textures.append(texture_dict[tex_name_upper])
            else:
                break
    else:
        main_textures, alt_textures = [texture], []

    return main_textures, alt_textures


class _MaterialApplier:
    def __init__(self, pal, texture_dict, map_cfg):
        self._pal = pal
        self._map_cfg = map_cfg
        self.sample_as_light_info = collections.defaultdict(lambda: collections.defaultdict(dict))
        self._all_textures: Dict[str, Texture] = texture_dict

    def _load_image(self, texture):
        tex_cfg = _get_texture_config(texture, self._map_cfg)
        array_im, fullbright_array_im, _ = _texture_to_arrays(self._pal, texture, tex_cfg['tint'])
        im = blendmat.im_from_array(texture.name, array_im)
        if fullbright_array_im is not None:
            fullbright_im = blendmat.im_from_array(f"{texture.name}_fullbright", fullbright_array_im)
        else:
            fullbright_im = None
        return blendmat.BlendMatImagePair(im, fullbright_im)

    @functools.lru_cache(None)
    def _load_anim_images(self, texture: Texture) -> blendmat.BlendMatImages:
        main_textures, alt_textures = _get_anim_textures(texture, self._all_textures)
        print(texture.name, [t.name for t in main_textures], [t.name for t in alt_textures])
        return blendmat.BlendMatImages(
            frames=[self._load_image(tex) for tex in main_textures],
            alt_frames=[self._load_image(tex) for tex in alt_textures]
        )

    def _get_sample_as_light(self, texture, mat_type):
        if not self._load_anim_images(texture).any_fullbright:
            return False
        tex_cfg = _get_texture_config(texture, self._map_cfg)
        if not tex_cfg['sample_as_light']:
            return False
        overlay_enabled = self._map_cfg['fullbright_object_overlay']
        if overlay_enabled and mat_type == 'main':
            return False
        return True

    def _load_material(self, texture, leaf):
        images = self._load_anim_images(texture)

        if leaf is None:
            mat_name = f"{texture.name}_main"
        else:
            mat_name = f"{texture.name}_leaf_{leaf.id_}_main"
        tex_cfg = _get_texture_config(texture, self._map_cfg)

        if images.any_fullbright and (
                not self._map_cfg['fullbright_object_overlay'] or not tex_cfg['overlay']):
            bmat = blendmat.setup_fullbright_material(images, mat_name, tex_cfg['strength'])
        else:
            bmat = blendmat.setup_diffuse_material(images, mat_name)

        bmat.mat.cycles.sample_as_light = self._get_sample_as_light(texture, "main")

        return bmat

    def _load_fullbright_obj_material(self, texture, leaf):
        images = self._load_anim_images(texture)

        assert images.any_fullbright, "Should only be called with fullbright textures"

        if leaf is None:
            mat_name = f"{texture.name}_fullbright"
        else:
            mat_name = f"{texture.name}_leaf_{leaf.id_}_fullbright"

        tex_cfg = _get_texture_config(texture, self._map_cfg)

        bmat = blendmat.setup_transparent_fullbright_material(images, mat_name, tex_cfg['strength'])
        bmat.mat.cycles.sample_as_light = self._get_sample_as_light(texture, "fullbright")

        return bmat

    @functools.lru_cache(None)
    def _get_material(self, mat_type, texture, leaf):
        if not self._map_cfg['fullbright_object_overlay']:
            assert mat_type == "main"
        if mat_type == "main":
            bmat = self._load_material(texture, leaf)
        elif mat_type == "fullbright":
            bmat = self._load_fullbright_obj_material(texture, leaf)
        else:
            raise ValueError(f"Invalid mat_type {mat_type}")
        return bmat

    def apply(self, model, mesh, bsp_faces, mat_type):
        slots = []
        mat_to_slot_idx = {}

        for mesh_poly, bsp_face in zip(mesh.polygons, bsp_faces):
            texture = bsp_face.tex_info.texture
            tex_cfg = _get_texture_config(texture, self._map_cfg)
            if self._get_sample_as_light(texture, mat_type):
                # Use a different material for each leaf
                bmat = self._get_material(mat_type, texture, bsp_face.leaf)
                self.sample_as_light_info[model][bsp_face.leaf][bmat] = tex_cfg
            else:
                # Single material
                bmat = self._get_material(mat_type, texture, None)

            if bmat not in mat_to_slot_idx:
                mesh.materials.append(bmat.mat)
                mat_to_slot_idx[bmat] = len(mat_to_slot_idx)

            mesh_poly.material_index = mat_to_slot_idx[bmat]


def _get_visible_faces(model):
    return [(face_id, face)
            for face_id, face in zip(range(model.first_face_idx, model.first_face_idx + model.num_faces), model.faces)
            if face.tex_info.texture.name != 'trigger'
            if not face.tex_info.texture.name.startswith('sky')]


def _get_bbox(a):
    out = []
    for axis in range(2):
        b = np.where(np.any(a, axis=axis))
        out.append([np.min(b), np.max(b) + 1])
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

        if (prev_dist > 0) != (dist > 0):
            alpha = -dist / (prev_dist - dist)
            new_vert = tuple(alpha * np.array(prev_vert) + (1 - alpha) * np.array(vert))
            new_vertices.append(new_vert)

        if dist > 0:
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


def _get_union_fullbright_array(pal, texture, texture_dict):
    return functools.reduce(operator.or_,
                            (_texture_to_arrays(pal, tex)[2]
                             for tex_list in _get_anim_textures(texture, texture_dict)
                             for tex in tex_list))


def _load_fullbright_objects(model, map_name, pal, texture_dict, mat_applier, map_cfg, obj_name_prefix):
    # Calculate bounding boxes for regions of full brightness.
    bboxes = {}
    for texture in {f.tex_info.texture for f in model.faces}:
        fullbright_array = _get_union_fullbright_array(pal, texture, texture_dict)
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

        if new_faces:
            # Actually make the mesh and add it to the scene
            obj_name = f'{obj_name_prefix}{map_name}_fullbright_{face_id}'
            mesh = bpy.data.meshes.new(obj_name)
            mesh.from_pydata(*_pydata_from_faces(new_faces))

            obj = bpy.data.objects.new(obj_name, mesh)
            bpy.context.scene.collection.objects.link(obj)

            fullbright_objects[face] = obj

            if mat_applier is not None:
                texinfos = [face.tex_info for face in new_bsp_faces]
                _set_uvs(mesh, texinfos, new_faces)
                mat_applier.apply(model, mesh, new_bsp_faces, 'fullbright')

    return fullbright_objects


def _load_object(model_id, model, map_name, mat_applier, obj_name_prefix):
    bsp_faces = [face for _, face in _get_visible_faces(model)]
    faces = [list(bsp_face.vertices) for bsp_face in bsp_faces]

    name = f"{obj_name_prefix}{map_name}_{model_id}"

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(*_pydata_from_faces(faces))

    if mat_applier is not None:
        texinfos = [bsp_face.tex_info for bsp_face in bsp_faces]
        _set_uvs(mesh, texinfos, faces)
        mat_applier.apply(model, mesh, bsp_faces, 'main')

    mesh.validate()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    return obj


class BlendBsp(NamedTuple):
    bsp: Bsp
    map_obj: bpy_types.Object
    model_objs: Dict[int, bpy_types.Object]
    fullbright_objects: Optional[Dict[Face, bpy_types.Object]]
    sample_as_light_info: Optional[Dict[Model, Dict[Leaf, Dict[blendmat.BlendMat, Dict]]]]

    def add_leaf_mesh(self, pos, obj_name='leaf_simplex'):
        leaf = self.bsp.models[0].get_leaf_from_point(pos)
        verts, faces = leaf.simplex.to_mesh()

        mesh = bpy.data.meshes.new(obj_name)
        mesh.from_pydata(verts, [], faces)
        obj = bpy.data.objects.new(obj_name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.parent = self.map_obj


def _add_lights(lights_cfg, map_obj, obj_name_prefix):
    for obj_name, light_cfg in lights_cfg.items():
        obj_name = f'{obj_name_prefix}{obj_name}'
        data = bpy.data.lights.new(name=obj_name, type=light_cfg['type'])
        obj = bpy.data.objects.new(name=obj_name, object_data=data)
        data.energy = light_cfg['energy']
        data.color = light_cfg.get('color', (1, 1, 1))
        obj.location = light_cfg['location']
        obj.rotation_euler = light_cfg.get('rotation', (0, 0, 0))
        obj.parent = map_obj

        if light_cfg['type'] == 'SUN':
            data.angle = light_cfg['angle']
        bpy.context.scene.collection.objects.link(obj)


def load_bsp(pak_root, map_name, config):
    fs = pak.Filesystem(pak_root)
    fname = 'maps/{}.bsp'.format(map_name)
    bsp = Bsp(fs.open(fname))
    pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 255
    return add_bsp(bsp, pal, map_name, config)


def add_bsp(bsp, pal, map_name, config, obj_name_prefix=''):
    pal = np.concatenate([pal, np.ones(256)[:, None]], axis=1)

    if map_name.startswith('b_'):
        map_cfg = config['maps']['__bsp_model__']
    else:
        map_cfg = config['maps'][map_name]

    if map_cfg['do_materials']:
        mat_applier = _MaterialApplier(pal, bsp.textures_by_name, map_cfg)
    else:
        mat_applier = None

    map_obj = bpy.data.objects.new(f'{obj_name_prefix}{map_name}', None)
    bpy.context.scene.collection.objects.link(map_obj)

    fullbright_objects = {}
    model_objs = {}
    for model_id, model in enumerate(bsp.models):
        model_obj = _load_object(model_id, model, map_name, mat_applier, obj_name_prefix)
        model_obj.parent = map_obj
        model_objs[model_id] = model_obj

        if map_cfg['fullbright_object_overlay']:
            model_fullbright_objects = _load_fullbright_objects(
                model, map_name, pal, bsp.textures_by_name, mat_applier, map_cfg, obj_name_prefix
            )
        else:
            model_fullbright_objects = {}

        for obj in model_fullbright_objects.values():
            obj.parent = model_obj

        fullbright_objects.update(model_fullbright_objects)

    if mat_applier is not None:
        sample_as_light_info = mat_applier.sample_as_light_info
    else:
        sample_as_light_info = None

    _add_lights(map_cfg.get('lights', {}), map_obj, obj_name_prefix)

    return BlendBsp(bsp, map_obj, model_objs, fullbright_objects, sample_as_light_info)

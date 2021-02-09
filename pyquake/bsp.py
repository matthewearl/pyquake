# Copyright (c) 2019 Matthew Earl
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

__all__ = (
    'Bsp',
    'MalformedBspFile',
    'get_tex_coords',
)


import collections
import enum
import functools
import itertools
import logging
import struct
from typing import NamedTuple, Tuple, List, Iterable

import numpy as np

from . import boxpack
from . import ent
from . import simplex


def _listify(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return list(f(*args, **kwargs))
    return wrapped


class PlaneType(enum.Enum):
    AXIAL_X = 0
    AXIAL_Y = 1
    AXIAL_Z = 2
    NON_AXIAL_X = 3
    NON_AXIAL_Y = 4
    NON_AXIAL_Z = 5


class Plane(NamedTuple):
    normal: Tuple[float, float, float]
    dist: float
    plane_type: PlaneType


class BBox(NamedTuple):
    mins: Tuple[int, int, int]
    maxs: Tuple[int, int, int]


class Node(NamedTuple):
    bsp: "Bsp"
    plane_id: int
    child_ids: Tuple[int, int]
    bbox: BBox
    face_id: int
    num_faces: int

    @property
    def plane(self):
        return self.bsp.planes[self.plane_id]

    def child_is_leaf(self, child_num):
        return self.child_ids[child_num] < 0

    def get_child(self, child_num):
        if self.child_is_leaf(child_num):
            return self.bsp.leaves[-self.child_ids[child_num] - 1]
        else:
            return self.bsp.nodes[self.child_ids[child_num]]

    @property
    def leaves(self):
        for child_num in range(2):
            if self.child_is_leaf(child_num):
                yield self.get_child(child_num)
            else:
                yield from self.get_child(child_num).leaves

    def _get_childs_leaves_from_simplex(self, child_num: int, sx: simplex.Simplex) -> Iterable["Leaf"]:
        if self.child_is_leaf(child_num):
            l = self.get_child(child_num)
            if l.num_faces > 0:
                yield l
        else:
            yield from self.get_child(child_num).get_leaves_from_simplex(sx)

    def get_leaves_from_simplex(self, sx: simplex.Simplex) -> Iterable["Leaf"]:
        """Return an iterable of leaves under this node that intersect the given simplex."""
        p = np.concatenate([self.plane.normal, [-self.plane.dist]])

        if np.dot(sx.optimize(p[:-1]).pos, p[:-1]) + p[-1] < 0:
            # completely behind 
            yield from self._get_childs_leaves_from_simplex(1, sx)
        elif np.dot(sx.optimize(-p[:-1]).pos, p[:-1]) + p[-1] > 0:
            # completely infront
            yield from self._get_childs_leaves_from_simplex(0, sx)
        else:
            infront_sx = sx.add_constraint(p)
            yield from self._get_childs_leaves_from_simplex(0, infront_sx)
            behind_sx = sx.add_constraint(-p)
            yield from self._get_childs_leaves_from_simplex(1, behind_sx)

    def _generate_leaf_paths(self):
        for child_num in range(2):
            if self.child_is_leaf(child_num):
                leaf = self.get_child(child_num)
                yield leaf, [child_num]
            else:
                child_node = self.get_child(child_num)
                for leaf, path in child_node._generate_leaf_paths():
                    yield leaf, [child_num] + path


class Leaf(NamedTuple):
    bsp: "Bsp"
    contents: int
    vis_offset: int
    bbox: BBox
    face_list_idx: int
    num_faces: int

    @property
    def faces(self):
        return (self.bsp.faces[self.bsp.face_list[i]]
                for i in range(self.face_list_idx, self.face_list_idx + self.num_faces))

    @property
    @functools.lru_cache(None)
    @_listify
    def visible_leaves(self):
        i = self.vis_offset
        visdata = self.bsp.visdata
        leaf_idx = 1
        while leaf_idx < self.bsp.models[0].num_leaves:
            if visdata[i] == 0:
                leaf_idx += 8 * visdata[i + 1]
                i += 2
            else:
                for j in range(8):
                    if visdata[i] & (1 << j):
                        yield self.bsp.leaves[leaf_idx]
                    leaf_idx += 1
                i += 1

    @property
    @functools.lru_cache(None)
    @_listify
    def visible_faces(self):
        return (face for leaf in self.visible_leaves for face in leaf.faces)

    @property
    @functools.lru_cache(None)
    def id_(self):
        return self.bsp.leaves.index(self)

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)

    @property
    @functools.lru_cache(None)
    def simplex(self):
        model = self.bsp._leaf_to_model[self]
        path = self.bsp._leaf_to_path[self]
        node = model.node
        sx = simplex.Simplex.from_bbox(node.bbox.mins, node.bbox.maxs)
        for child_num in path:
            p = np.concatenate([node.plane.normal, [-node.plane.dist]])
            sx = sx.add_constraint(p if child_num == 0 else -p)
            node = node.get_child(child_num)
        return sx


class Face(NamedTuple):
    bsp: "Bsp"
    edge_list_idx: int
    num_edges: int
    texinfo_id: int
    lightmap_offset: int

    @property
    def vert_indices(self):
        for edge_id in self.bsp.edge_list[self.edge_list_idx:self.edge_list_idx + self.num_edges]:
            if edge_id < 0:
                v = self.bsp.edges[-edge_id][1]
            else:
                v = self.bsp.edges[edge_id][0]
            yield v

    @property
    def vertices(self):
        return (self.bsp.vertices[idx] for idx in self.vert_indices)

    @property
    def tex_coords(self):
        return [self.tex_info.vert_to_tex_coords(v) for v in self.vertices]

    @property
    def full_lightmap_tex_coords(self):
        return self.bsp._full_lightmap_tex_coords[self]

    @property
    def tex_info(self):
        return self.bsp.texinfo[self.texinfo_id]

    @property
    @functools.lru_cache(None)
    def plane(self):
        first_edge = None
        best_normal = None

        verts = list(self.vertices)
        for prev_vert, vert in zip(itertools.chain(verts[-1:], verts[:-1]), verts):
            edge = np.array(vert) - np.array(prev_vert)
            edge /= np.linalg.norm(edge)
            if first_edge is None:
                first_edge = edge
            else:
                normal = np.cross(edge, first_edge)
                if best_normal is None or np.linalg.norm(best_normal) < np.linalg.norm(normal):
                    best_normal = normal
            prev_vert = vert

        normal = normal / np.linalg.norm(normal)
        return normal, np.dot(verts[0], normal)

    @property
    @functools.lru_cache(None)
    @_listify
    def edge_planes(self):
        normal, _ = self.plane
        verts = list(self.vertices)
        for prev_vert, vert in zip(itertools.chain(verts[-1:], verts[:-1]), verts):
            edge = np.array(vert) - np.array(prev_vert)
            edge /= np.linalg.norm(edge)
            edge_normal = np.cross(normal, edge)
            yield edge_normal, np.dot(vert, edge_normal)

    @property
    @functools.lru_cache(None)
    def area(self):
        verts = np.array(list(self.vertices))
        v0 = verts[0]
        return 0.5 * sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0)) for v1, v2 in zip(verts[1:-1], verts[2:]))

    @property
    @functools.lru_cache(None)
    def centroid(self):
        return np.array(list(self.vertices)).mean(axis=0)

    @property
    def leaf(self):
        return self.bsp._face_to_leaf[self]

    @property
    def has_lightmap(self):
        return self.lightmap_offset != -1

    def _extract_local_lightmap(self):
        tex_coords = np.array(list(self.tex_coords))

        mins = np.floor(np.min(tex_coords, axis=0).astype(np.float32) / 16).astype(np.int)
        maxs = np.ceil(np.max(tex_coords, axis=0).astype(np.float32) / 16).astype(np.int)

        size = (maxs - mins) + 1

        lightmap = np.array(list(
            self.bsp.lightmap[self.lightmap_offset:
                              self.lightmap_offset + size[0] * size[1]]
        )).reshape((size[1], size[0]))

        tex_coords -= mins * 16
        tex_coords += 8
        tex_coords /= 16.

        return lightmap, tex_coords

class TexInfo(NamedTuple):
    bsp: "Bsp"
    vec_s: float
    dist_s: float
    vec_t: float
    dist_t: float
    texture_id: int
    flags: int

    @property
    def texture(self):
        return self.bsp.textures[self.texture_id]

    @property
    def texel_area(self):
        return np.linalg.norm(np.cross(self.vec_s, self.vec_t))

    def vert_to_tex_coords(self, vert):
        return [np.dot(vert, self.vec_s) + self.dist_s, np.dot(vert, self.vec_t) + self.dist_t]

    def tex_coords_to_vert(self, tc):
        return np.array(self.vec_s) * (tc[0] - self.dist_s) + np.array(self.vec_t) * (tc[1] - self.dist_t)


def _infront(point, plane_norm, plane_dist):
    return np.dot(point, plane_norm) - plane_dist >= 0


class Model(NamedTuple):
    bsp: "Bsp"
    first_face_idx: int
    num_faces: int
    num_leaves: int
    node_id: int

    @property
    def faces(self):
        return self.bsp.faces[self.first_face_idx:self.first_face_idx + self.num_faces]

    @property
    def node(self):
        return self.bsp.nodes[self.node_id]

    def get_simplex_from_point(self, point):
        sx = simplex.Simplex.from_bbox(self.node.bbox.mins, self.node.bbox.maxs)
        point = np.array(point)
        node = self.node
        while True:
            plane = node.plane
            child_num = 0 if _infront(point, plane.normal, plane.dist) else 1
            p = np.concatenate([plane.normal, [-plane.dist]])
            sx = sx.add_constraint(p if child_num == 0 else -p)
            if node.child_is_leaf(child_num):
                break
            node = node.get_child(child_num)

        return sx, node.get_child(child_num)

    def get_leaf_from_point(self, point):
        point = np.array(point)
        node = self.node
        while True:
            child_num = 0 if _infront(point, node.plane.normal, node.plane.dist) else 1
            child = node.get_child(child_num)
            if node.child_is_leaf(child_num):
                return child
            node = child

    def get_leaves_from_bbox(self, bbox: BBox) -> Iterable[Leaf]:
        """Return an iterable of leaves that intersect the given bounding box"""
        n = self.node
        sx = simplex.Simplex.from_bbox(bbox.mins, bbox.maxs)
        return n.get_leaves_from_simplex(sx)

    @property
    @functools.lru_cache(None)
    def id_(self):
        return self.bsp.models.index(self)

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)


class Texture(NamedTuple):
    name: str
    width: int
    height: int
    data: List[bytes]

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)


_DirEntry = collections.namedtuple('_DirEntry', ('offset', 'size'))


class MalformedBspFile(Exception):
    pass


class Bsp:
    """A BSP file parser, and interface to the information directly contained within."""

    @property
    @functools.lru_cache(None)
    def _face_to_leaf(self):
        return {face: leaf for leaf in self.leaves for face in leaf.faces}

    @property
    @functools.lru_cache(None)
    def _leaf_to_model(self):
        return {leaf: model for model in self.models for leaf in model.node.leaves}

    @property
    @functools.lru_cache(None)
    def _leaf_to_path(self):
        return {leaf: path for m in self.models for leaf, path in m.node._generate_leaf_paths()}

    @property
    @functools.lru_cache(None)
    def textures_by_name(self):
        return {t.name: t for t in self.textures.values()}

    @functools.lru_cache(1)
    def _make_full_lightmap(self, lightmap_size=(512, 512)):
        lightmaps = {face: face._extract_local_lightmap() for face in self.faces if face.has_lightmap}
        lightmaps = dict(reversed(sorted(lightmaps.items(), key=lambda x: x[1][0].shape[0] * x[1][0].shape[1])))

        box_packer = boxpack.BoxPacker(lightmap_size)
        for face, (lightmap, tex_coords) in lightmaps.items():
            if not box_packer.insert(face, (lightmap.shape[1], lightmap.shape[0])):
                raise Exception("Could not pack lightmaps into {} image".format(lightmap_size))

        lightmap_image = np.zeros((lightmap_size[1], lightmap_size[0]), dtype=np.uint8)
        tex_coords = {}
        for face, (x, y) in box_packer:
            lm, tc = lightmaps[face]
            lightmap_image[y:y + lm.shape[0], x:x + lm.shape[1]] = lm
            tex_coords[face] = (tc + (x, y)) / lightmap_size

        return lightmap_image, tex_coords

    @property
    def full_lightmap_image(self):
        lightmap_image, tex_coords = self._make_full_lightmap()
        return lightmap_image

    @property
    def _full_lightmap_tex_coords(self):
        lightmap_image, tex_coords = self._make_full_lightmap()
        return tex_coords

    def _read(self, f, n):
        b = f.read(n)
        if len(b) < n:
            raise MalformedBspFile("File ended unexpectedly")
        return b

    def _read_struct(self, f, struct_fmt):
        size = struct.calcsize(struct_fmt)
        out = struct.unpack(struct_fmt, self._read(f, size))
        return out

    def _read_lump(self, f, dir_entry, struct_fmt, post_func=None):
        size = struct.calcsize(struct_fmt)
        f.seek(dir_entry.offset)
        if dir_entry.size % size != 0:
            raise MalformedBspFile("Invalid lump size")
        out = [struct.unpack(struct_fmt, self._read(f, size)) for _ in range(0, dir_entry.size, size)]
        if post_func:
            out = [post_func(*x) for x in out]
        return out

    def _read_dir_entry(self, f, idx):
        fmt = "<II"
        size = struct.calcsize(fmt)
        f.seek(4 + size * idx)
        return _DirEntry(*struct.unpack(fmt, self._read(f, size)))

    def _read_texture(self, f, tex_offset):
        f.seek(tex_offset)
        name, width, height, *data_offsets = self._read_struct(f, "<16sLL4L")
        name = name[:name.index(b'\0')].decode('ascii')

        if width % 16 != 0 or height % 16 != 0:
            raise MalformedBspFile('Texture has invalid dimensions: {} x {}'.format(width, height))

        offset = 40
        data = []
        for i in range(4):
            if offset != data_offsets[i]:
                raise MalformedBspFile('Data offset is {} expected {}'.format(data_offsets[i], offset))
            mip_size = (width * height) >> (2 * i)
            data.append(self._read(f, mip_size))
            offset += mip_size
        return Texture(name, width, height, data)

    def _read_textures(self, f, texture_dir_entry):
        f.seek(texture_dir_entry.offset)
        num_textures, = self._read_struct(f, "<L")
        logging.debug('Loading %s textures', num_textures)
        tex_offsets = [self._read_struct(f, "<l")[0] for i in range(num_textures)]
        return {idx: self._read_texture(f, texture_dir_entry.offset + offs)
                for idx, offs in enumerate(tex_offsets)
                if offs != -1}

    def __init__(self, f):
        version, = struct.unpack("<I", self._read(f, 4))
        if version != 29:
            raise MalformedBspFile("Unsupported version {} (should be 29)".format(version))

        logging.debug("Reading vertices")
        self.vertices = self._read_lump(f, self._read_dir_entry(f, 3), "<fff")

        logging.debug("Reading edges")
        self.edges = self._read_lump(f, self._read_dir_entry(f, 12), "<HH")

        logging.debug("Reading edge list")
        self.edge_list = self._read_lump(f, self._read_dir_entry(f, 13), "<l", lambda x: x)

        logging.debug("Reading faces")
        def read_face(plane_id, side, edge_list_idx, num_edges, texinfo_id, typelight, baselight, light1, light2,
                      lightmap_offset):
            return Face(self, edge_list_idx, num_edges, texinfo_id, lightmap_offset)
        self.faces = self._read_lump(f, self._read_dir_entry(f, 7), "<HHLHHBBBBl", read_face)

        logging.debug("Reading texinfo")
        def read_texinfo(vs1, vs2, vs3, ds, vt1, vt2, vt3, dt, texture_id, flags):
            return TexInfo(self, (vs1, vs2, vs3), ds, (vt1, vt2, vt3), dt, texture_id, flags)
        self.texinfo = self._read_lump(f, self._read_dir_entry(f, 6), "<ffffffffLL", read_texinfo)

        logging.debug("Reading lightmap")
        lightmap_dir_entry = self._read_dir_entry(f, 8)
        f.seek(lightmap_dir_entry.offset)
        self.lightmap = self._read(f, lightmap_dir_entry.size)

        logging.debug("Reading models")
        def read_model(mins1, mins2, mins3, maxs1, maxs2, maxs3, o1, o2, o3, n1, n2, n3, n4, num_leaves, first_face_idx,
                       num_faces):
            return Model(self, first_face_idx, num_faces, num_leaves, n1)
        self.models = self._read_lump(f, self._read_dir_entry(f, 14), "<ffffffffflllllll", read_model)

        logging.debug("Reading textures")
        texture_dir_entry = self._read_dir_entry(f, 2)
        self.textures = self._read_textures(f, texture_dir_entry)

        logging.debug("Reading nodes")
        def read_node(plane_id, c1, c2, mins1, mins2, mins3, maxs1, maxs2, maxs3, face_id, num_faces):
            bbox = BBox((mins1, mins2, mins3), (maxs1, maxs2, maxs3))
            return Node(self, plane_id, (c1, c2), bbox, face_id, num_faces)
        self.nodes = self._read_lump(f, self._read_dir_entry(f, 5), "<lhhhhhhhhHH", read_node)

        logging.debug("Reading leaves")
        def read_leaf(contents, vis_offset, mins1, mins2, mins3, maxs1, maxs2, maxs3, face_list_idx, num_faces, l1, l2,
                      l3, l4):
            bbox = BBox((mins1, mins2, mins3), (maxs1, maxs2, maxs3))
            return Leaf(self, contents, vis_offset, bbox, face_list_idx, num_faces)
        self.leaves = self._read_lump(f, self._read_dir_entry(f, 10), "<llhhhhhhHHBBBB", read_leaf)

        logging.debug("Reading face list")
        self.face_list = self._read_lump(f, self._read_dir_entry(f, 11), "<H", lambda x: x)

        logging.debug("Reading planes")
        def read_plane(n1, n2, n3, d, plane_type):
            return Plane((n1, n2, n3), d, PlaneType(plane_type))
        self.planes = self._read_lump(f, self._read_dir_entry(f, 1), "<ffffl", read_plane)

        logging.debug("Reading entities")
        entity_dir_entry = self._read_dir_entry(f, 0)
        f.seek(entity_dir_entry.offset)
        b = self._read(f, entity_dir_entry.size)
        self.entities = ent.parse_entities(b[:b.index(b'\0')].decode('ascii'))

        logging.debug("Reading visdata")
        visinfo_dir_entry = self._read_dir_entry(f, 4)
        f.seek(visinfo_dir_entry.offset)
        b = self._read(f, visinfo_dir_entry.size)
        self.visdata = b



if __name__ == "__main__":
    import io
    import sys
    import logging

    import pak

    root_logger = logging.getLogger()
    root_logger.addHandler(logging.StreamHandler())
    root_logger.setLevel(logging.DEBUG)

    fs = pak.Filesystem(sys.argv[1])
    bsp = Bsp(io.BytesIO(fs[sys.argv[2]]))


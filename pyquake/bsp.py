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
import logging
import struct
from typing import NamedTuple, Tuple, List

import numpy as np

from . import ent


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


class BBoxShort(NamedTuple):
    mins: Tuple[int, int, int]
    maxs: Tuple[int, int, int]


class Node(NamedTuple):
    bsp: "Bsp"
    plane_id: int
    child_ids: Tuple[int, int]
    bbox: BBoxShort
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


class Leaf(NamedTuple):
    bsp: "Bsp"
    contents: int
    vis_offset: int
    bbox: BBoxShort
    face_list_idx: int
    num_faces: int

    @property
    def faces(self):
        return (self.bsp.faces[self.bsp.face_list[i]]
                for i in range(self.face_list_idx, self.face_list_idx + self.num_faces))


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
        return [get_tex_coords(self.tex_info, v) for v in self.vertices]

    @property
    def tex_info(self):
        return self.bsp.texinfo[self.texinfo_id]


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
        return self.bsp.textures[texture_id]


class Model(NamedTuple):
    bsp: "Bsp"
    first_face_idx: int
    num_faces: int
    node_id: int

    @property
    def faces(self):
        return self.bsp.faces[self.first_face_idx:self.first_face_idx + self.num_faces]

    @property
    def node(self):
        return self.bsp.nodes[self.node_id]


class Texture(NamedTuple):
    name: str
    width: int
    height: int
    data: bytes
    

_DirEntry = collections.namedtuple('_DirEntry', ('offset', 'size'))


class MalformedBspFile(Exception):
    pass


class Bsp:
    """A BSP file parser, and interface to the information directly contained within."""
    def _read(self, f, n):
        b = f.read(n)
        if len(b) < n:
            raise MalformedBspFile("File ended unexpectedly")
        return b

    def _read_struct(self, f, struct_fmt, post_func=None):
        size = struct.calcsize(struct_fmt)
        out = struct.unpack(struct_fmt, self._read(f, size))
        if post_func is not None:
            out = post_func(out)
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
        tex_offsets = [self._read_struct(f, "<L")[0] for i in range(num_textures)]
        return [self._read_texture(f, texture_dir_entry.offset + offs) for offs in tex_offsets]

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
            return Model(self, first_face_idx, num_faces, n1)
        self.models = self._read_lump(f, self._read_dir_entry(f, 14), "<ffffffffflllllll", read_model)

        logging.debug("Reading textures")
        texture_dir_entry = self._read_dir_entry(f, 2)
        self.textures = self._read_textures(f, texture_dir_entry)

        logging.debug("Reading nodes")
        def read_node(plane_id, c1, c2, mins1, mins2, mins3, maxs1, maxs2, maxs3, face_id, num_faces):
            bbox = BBoxShort((mins1, mins2, mins3), (maxs1, maxs2, maxs3))
            return Node(self, plane_id, (c1, c2), bbox, face_id, num_faces)
        self.nodes = self._read_lump(f, self._read_dir_entry(f, 5), "<lhhhhhhhhHH", read_node)

        logging.debug("Reading leaves")
        def read_leaf(contents, vis_offset, mins1, mins2, mins3, maxs1, maxs2, maxs3, face_list_idx, num_faces, l1, l2,
                      l3, l4):
            bbox = BBoxShort((mins1, mins2, mins3), (maxs1, maxs2, maxs3))
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


def get_tex_coords(tex_info, vert):
    return [np.dot(vert, tex_info.vec_s) + tex_info.dist_s, np.dot(vert, tex_info.vec_t) + tex_info.dist_t]


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


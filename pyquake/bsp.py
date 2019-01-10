# Copyright (c) 2018 Matthew Earl
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
)


import numpy as np

import collections
import logging
import struct


Face = collections.namedtuple('Face', ('edge_list_idx', 'num_edges', 'texinfo_id', 'lightmap_offset'))
TexInfo = collections.namedtuple('TexInfo', ('vec_s', 'dist_s', 'vec_t', 'dist_t', 'texture_id', 'animated'))
Model = collections.namedtuple('Model', ('first_face_idx', 'num_faces'))
Texture = collections.namedtuple('Texture', ('name', 'width', 'height', 'data'))

_DirEntry = collections.namedtuple('_DirEntry', ('offset', 'size'))


class MalformedBspFile(Exception):
    pass


class _BspFile:
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
            raise MalformedBspFile(f'Texture has invalid dimensions: {width} x {height}')

        offset = 40
        data = []
        for i in range(4):
            if offset != data_offsets[i]:
                raise MalformedBspFile(f'Data offset is {data_offsets[i]} expected {offset}')
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
            return Face(edge_list_idx, num_edges, texinfo_id, lightmap_offset)
        self.faces = self._read_lump(f, self._read_dir_entry(f, 7), "<HHLHHBBBBl", read_face)

        logging.debug("Reading texinfo")
        def read_texinfo(vs1, vs2, vs3, ds, vt1, vt2, vt3, dt, texture_id, flags):
            return TexInfo((vs1, vs2, vs3), ds, (vt1, vt2, vt3), dt, texture_id, flags)
        self.texinfo = self._read_lump(f, self._read_dir_entry(f, 6), "<ffffffffLL", read_texinfo)

        logging.debug("Reading lightmap")
        lightmap_dir_entry = self._read_dir_entry(f, 8)
        f.seek(lightmap_dir_entry.offset)
        self.lightmap = self._read(f, lightmap_dir_entry.size)

        logging.debug("Reading models")
        def read_model(mins1, mins2, mins3, maxs1, maxs2, maxs3, o1, o2, o3, n1, n2, n3, n4, num_leaves, first_face_idx,
                       num_faces):
            return Model(first_face_idx, num_faces)
        self.models = self._read_lump(f, self._read_dir_entry(f, 14), "<ffffffffflllllll", read_model)

        logging.debug("Reading textures")
        texture_dir_entry = self._read_dir_entry(f, 2)
        self.textures = self._read_textures(f, texture_dir_entry)


class Bsp(_BspFile):
    """An interface to a BSP file, with added convenience methods"""

    def iter_face_vert_indices(self, face_idx):
        face = self.faces[face_idx]
        for edge_id in self.edge_list[face.edge_list_idx:face.edge_list_idx + face.num_edges]:
            if edge_id < 0:
                v = self.edges[-edge_id][1]
            else:
                v = self.edges[edge_id][0]
            yield v

    def iter_face_verts(self, face_idx):
        return (self.vertices[idx] for idx in self.iter_face_vert_indices(face_idx))

    def iter_face_tex_coords(self, face_idx):
        tex_info = self.texinfo[self.faces[face_idx].texinfo_id]
        return [[np.dot(v, tex_info.vec_s) + tex_info.dist_s, np.dot(v, tex_info.vec_t) + tex_info.dist_t]
                    for v in self.iter_face_verts(face_idx)]


if __name__ == "__main__":
    import io
    import sys
    import logging

    import pak

    root_logger = logging.getLogger()
    root_logger.addHandler(logging.StreamHandler())
    root_logger.setLevel(logging.DEBUG)

    fs = pak.Filesystem(sys.argv[1])
    bsp = BspFile(io.BytesIO(fs[sys.argv[2]]))


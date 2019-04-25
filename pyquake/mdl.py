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


import collections
import enum
import struct
from dataclasses import dataclass
from typing import Sequence, NamedTuple

import numpy as np

from . import anorms


class SimpleFrame(NamedTuple):
    bbox_min: np.array
    bbox_max: np.array
    name: str
    frame_verts: np.array
    frame_normals: np.array


class FrameType(enum.IntEnum):
    SINGLE = 0
    GROUP = 1


class BaseFrame:
    frame_type: FrameType


@dataclass
class SingleFrame(BaseFrame):
    frame_type = FrameType.SINGLE

    frame: SimpleFrame


@dataclass
class GroupFrame(BaseFrame):
    frame_type = FrameType.GROUP

    bbox_min: np.ndarray
    bbox_max: np.ndarray
    times: np.ndarray
    frames: Sequence[SimpleFrame]


class AliasModel:
    def _read(self, f, n):
        b = f.read(n)
        if len(b) < n:
            raise Exception("File ended unexpectedly")
        return b

    def _read_struct(self, f, struct_fmt, post_func=None):
        size = struct.calcsize(struct_fmt)
        out = struct.unpack(struct_fmt, self._read(f, size))
        if post_func is not None:
            out = post_func(*out)
        return out

    def _read_header(self, f):
        (ident, version, sx, sy, sz, sox, soy, soz, bounding_radius, ex, ey, ez, num_skins, skin_width, skin_height,
            num_verts, num_tris, num_frames, sync_type, flags, size) = self._read_struct(f, "<LLffffffffffllllllllf")

        scale = np.array([sx, sy, sz])
        scale_origin = np.array([sox, soy, soz])
        eye_position = np.array([ex, ey, ez])

        return {'ident': ident,
                'version': version,
                'scale': scale,
                'scale_origin': scale_origin,
                'bounding_radius': bounding_radius,
                'eye_position': eye_position,
                'num_skins': num_skins,
                'skin_width': skin_width,
                'skin_height': skin_height,
                'num_verts': num_verts,
                'num_tris': num_tris,
                'num_frames': num_frames,
                'sync_type': sync_type,
                'flags': flags,
                'size': size}

    def _read_skin(self, f):
        group, = self._read_struct(f, "<L")
        if group != 0:
            raise Exception("Only single picture skins are supported")
        width = self.header['skin_width']
        height = self.header['skin_height']
        data = self._read(f, width * height)
        
        return np.array(list(data)).reshape((height, width))
        
    def _read_skins(self, f):
        return [self._read_skin(f) for _ in range(self.header['num_skins'])]

    def _read_tcs(self, f):
        dt = np.dtype(np.int32)
        dt = dt.newbyteorder('<')
        num_verts = self.header['num_verts']
        a = np.frombuffer(self._read(f, 4 * 3 * num_verts), dtype=dt).reshape([num_verts, 3])
        return a[:, 0], a[:, 1:]

    def _read_tris(self, f):
        dt = np.dtype(np.int32)
        dt = dt.newbyteorder('<')
        num_tris = self.header['num_tris']
        a = np.frombuffer(self._read(f, 4 * 4 * num_tris), dtype=dt).reshape([num_tris, 4])
        return a[:, 0], a[:, 1:]

    def _load_bbox(self, f):
        a = np.frombuffer(self._read(f, 4 * 2), dtype=np.uint8).reshape((2, 4))
        bbox_min, bbox_max = (self.header['scale'] * a[:, :3]) + self.header['scale_origin']
        return bbox_min, bbox_max

    def _load_trivertx(self, f, n):
        a = np.frombuffer(self._read(f, 4 * n), dtype=np.uint8).reshape((n, 4))
        pos = (self.header['scale'] * a[:, :3]) + self.header['scale_origin']
        normal = anorms.anorms[a[:, 3]]
        
        return pos, normal

    def _read_simple_frame(self, f) -> SimpleFrame:
        bbox_min, bbox_max = self._load_bbox(f)
        name, = self._read_struct(f, "16s")
        name = name[:name.index(b'\0')].decode('ascii')
        frame_verts, frame_normals = self._load_trivertx(f, self.header['num_verts'])

        return SimpleFrame(bbox_min, bbox_max, name, frame_verts, frame_normals)

    def _read_frame(self, f):
        frame_type = self._read_struct(f, "<L", FrameType)
        if frame_type == FrameType.SINGLE:
            frame = SingleFrame(self._read_simple_frame(f))
        elif frame_type == FrameType.GROUP:
            nb, = self._read_struct(f, "<L")
            bbox_min, bbox_max = self._load_bbox(f)

            times = np.frombuffer(self._read(f, 4 * nb), dtype=np.float32)
            simple_frames = [self._read_simple_frame(f) for _ in range(nb)]

            frame = GroupFrame(bbox_min, bbox_max, times, simple_frames)
        else:
            raise Exception(f"Invalid frame type {frame_type}")

        return frame

    def _read_frames(self, f):
        return [self._read_frame(f) for _ in range(self.header['num_frames'])]

    def __init__(self, f):
        self.header = self._read_header(f)
        self.skins = self._read_skins(f)
        self.on_seam, self.tcs = self._read_tcs(f)
        self.faces_front, self.tris = self._read_tris(f)
        self.frames = self._read_frames(f)


# Copyright (c) 2021 Matthew Earl
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


"""MD3 file parser"""


__all__ = (
    'Animation',
    'MalformedAnimationError',
    'MalformedMD3Error',
    'MD3',
    'MD3Surface',
    'PlayerAnimNumber',
    'PmoveFrames',
)


import dataclasses
import enum
import json
import re
import struct

import numpy as np

from . import tokenize


class MalformedMD3Error(Exception):
    pass


class MalformedAnimationError(Exception):
    pass


def _read(f, n):
    b = f.read(n)
    if len(b) < n:
        raise MalformedMD3Error("File ended unexpectedly")
    return b


def _read_struct(f, struct_fmt, post_func=None):
    size = struct.calcsize(struct_fmt)
    out = struct.unpack(struct_fmt, _read(f, size))
    if post_func is not None:
        out = post_func(*out)
    return out


class MD3Surface:
    def _read_tris(self, f, offset, num_triangles):
        f.seek(offset)
        return np.frombuffer(_read(f, 4 * 3 * num_triangles), dtype=np.int32).reshape((num_triangles, 3))

    def _read_verts(self, f, offset, num_verts):
        f.seek(offset)
        vert_dtype = np.dtype([('verts', np.int16, 3), ('lng', np.uint8), ('lat', np.uint8)])
        vert_data = (
            np.frombuffer(_read(f, vert_dtype.itemsize * num_verts * self.num_frames), dtype=vert_dtype)
                .reshape((self.num_frames, num_verts))
        )
        verts = vert_data['verts'] / 64.
        lat = vert_data['lat'] * np.pi / 128
        lng = vert_data['lng'] * np.pi / 128
        normals = np.stack([
            np.cos(lat) * np.sin(lng),
            np.sin(lat) * np.sin(lng),
            np.cos(lng)
        ], axis=-1)

        return verts, normals

    def _read_tcs(self, f, offset, num_verts):
        f.seek(offset)
        return np.frombuffer(_read(f, 4 * 2 * num_verts), dtype=np.float32).reshape((num_verts, 2))

    def _read_shaders(self, f, offset, num_shaders):
        f.seek(offset)
        shaders = {}

        for _ in range(num_shaders):
            name, index = _read_struct(f, '<64sl')
            name = name[:name.index(b'\0')].decode('ascii')
            shaders[index] = name

        return shaders

    def __init__(self, f, offset):
        f.seek(offset)
        (self.ident, name, flags, self.num_frames, num_shaders, num_verts, num_triangles, triangles_offset,
         shaders_offset, tc_offset, vert_offset, self.next_offset) = _read_struct(f, '<4s64sllllllllll')
        self.name = name[:name.index(b'\0')].decode('ascii')

        self.tris = self._read_tris(f, offset + triangles_offset, num_triangles)
        self.verts, self.normals = self._read_verts(f, offset + vert_offset, num_verts)
        self.tcs = self._read_tcs(f, offset + tc_offset, num_verts)
        self.shaders = self._read_shaders(f, offset + shaders_offset, num_shaders)


class MD3:
    def _decode_tag_name(self, file_tag):
        b = bytes(file_tag['name'])
        tag_name = b[:b.index(b'\0')].decode('ascii')
        return tag_name

    def _read_tags(self, f, offset, num_tags, num_frames):
        f.seek(offset)
        file_tag_dtype = np.dtype([('name', np.uint8, 64),
                                   ('origin', np.float32, 3),
                                   ('axis', np.float32, (3, 3))])
        file_tags = (np.frombuffer(_read(f, file_tag_dtype.itemsize * num_tags * num_frames),
                                    dtype=file_tag_dtype)
                      .reshape((num_frames, num_tags)))

        if len(file_tags) == 0:
            tags_dict = {}
        else:
            tag_names = [self._decode_tag_name(file_tag) for file_tag in file_tags[0]]

            tag_dtype = np.dtype([('origin', np.float32, 3),
                                  ('axis', np.float32, (3, 3))])
            tags_dict = {tag_name: np.zeros(num_frames, dtype=tag_dtype) for tag_name in tag_names}

            for frame_idx in range(num_frames):
                for tag_idx, tag_name in enumerate(tag_names):
                    file_tag = file_tags[frame_idx, tag_idx]
                    if self._decode_tag_name(file_tag) != tag_name:
                        raise MalformedMD3Error('Unexpected tag name')
                    tags_dict[tag_name][frame_idx]['origin'][:] = file_tag['origin']
                    tags_dict[tag_name][frame_idx]['axis'][:] = file_tag['axis']

        return tags_dict

    def __init__(self, f):
        (ident, version, name, flags, num_frames, num_tags, num_surfaces, num_skins, frames_offset, tags_offset,
         surfaces_offset, eof_offset) = _read_struct(f, "<4sl64slllllllll")
        self.name = name[:name.index(b'\0')].decode('ascii')

        if ident != b'IDP3': 
            raise MalformedMD3Error(f'Bad ident: {ident}')

        if version != 15:
            raise MalformedMD3Error(f'Only version 15 is supported, not {version}')

        self.surfaces = []
        for surface_idx in range(num_surfaces):
            surface = MD3Surface(f, surfaces_offset)
            surfaces_offset += surface.next_offset
            self.surfaces.append(surface)

        self.tags = self._read_tags(f, tags_offset, num_tags, num_frames)


class PlayerAnimNumber(enum.IntEnum):
    BOTH_DEATH1 = 0
    BOTH_DEAD1 = enum.auto()
    BOTH_DEATH2 = enum.auto()
    BOTH_DEAD2 = enum.auto()
    BOTH_DEATH3 = enum.auto()
    BOTH_DEAD3 = enum.auto()

    TORSO_GESTURE = enum.auto()

    TORSO_ATTACK = enum.auto()
    TORSO_ATTACK2 = enum.auto()

    TORSO_DROP = enum.auto()
    TORSO_RAISE = enum.auto()

    TORSO_STAND = enum.auto()
    TORSO_STAND2 = enum.auto()

    LEGS_WALKCR = enum.auto()
    LEGS_WALK = enum.auto()
    LEGS_RUN = enum.auto()
    LEGS_BACK = enum.auto()
    LEGS_SWIM = enum.auto()

    LEGS_JUMP = enum.auto()
    LEGS_LAND = enum.auto()

    LEGS_JUMPB = enum.auto()
    LEGS_LANDB = enum.auto()

    LEGS_IDLE = enum.auto()
    LEGS_IDLECR = enum.auto()

    LEGS_TURN = enum.auto()

    TORSO_GETFLAG = enum.auto()
    TORSO_GUARDBASE = enum.auto()
    TORSO_PATROL = enum.auto()
    TORSO_FOLLOWME = enum.auto()
    TORSO_AFFIRMATIVE = enum.auto()
    TORSO_NEGATIVE = enum.auto()

    MAX_ANIMATIONS = enum.auto()

    LEGS_BACKCR = enum.auto()
    LEGS_BACKWALK = enum.auto()
    FLAG_RUN = enum.auto()
    FLAG_STAND = enum.auto()
    FLAG_STAND2RUN = enum.auto()

    MAX_TOTALANIMATIONS = enum.auto()


@dataclasses.dataclass
class Animation:
    first_frame: int
    num_frames: int
    looping_frames: int
    fps: int


class AnimationInfo:
    def _read_directives(self, token_iter):
        try:
            while token_iter.has(1) and not token_iter.peek(1).s[0].isdigit():
                tok = next(token_iter).s
                if tok == "footsteps":
                    self.footsteps = next(token_iter).s
                elif tok == "headoffset":
                    self.head_offset = [float(next(token_iter).s) for _ in range(3)]
                elif tok == "sex":
                    self.sex = next(token_iter).s
                elif tok == "fixedlegs":
                    self.fixed_legs = True
                elif tok == "fixedtorso":
                    self.fixed_torso = True
                else:
                    raise MalformedMD3Error
        except StopIteration:
            raise MalformedAnimationError(f'Expected token on line {token_iter.line_num}')

    def _read_anims(self, token_iter):
        self.anims = []
        while token_iter.has(4):
            self.anims.append(Animation(
                int(next(token_iter).s),
                int(next(token_iter).s),
                int(next(token_iter).s),
                float(next(token_iter).s),
            ))

        # adjust leg only frames, as per CG_ParseAnimationFile.
        if len(self.anims) > PlayerAnimNumber.LEGS_WALKCR:
            skip = (self.anims[PlayerAnimNumber.LEGS_WALKCR].first_frame
                    - self.anims[PlayerAnimNumber.TORSO_GESTURE].first_frame)
            for anim_num in range(PlayerAnimNumber.LEGS_WALKCR,
                                  PlayerAnimNumber.TORSO_GETFLAG):
                if anim_num < len(self.anims):
                    self.anims[anim_num].first_frame -= skip

        if token_iter.has(1):
            raise MalformedAnimationError('Extra token at end of file')

    def __init__(self, f):
        s = f.read().decode('ascii')

        token_iter = tokenize.Tokenizer(s)
        self._read_directives(token_iter)
        self._read_anims(token_iter)


@dataclasses.dataclass
class PmoveFrames:
    times: np.ndarray
    leg_anim_idxs: np.ndarray
    torso_anim_idxs: np.ndarray
    origins: np.ndarray
    angles: np.ndarray

    @classmethod
    def from_dump(cls, f):
        """Parse a dump of leg and torso anims."""
        records = [json.loads(line.strip().split(' ', 1)[1])
                   for line in f.readlines() if line.startswith('@@@pmove_dump')]

        times = np.array([r['time'] for r in records]) * 1e-3
        leg_anim_idxs = np.array([r['legs_anim'] for r in records])
        torso_anim_idxs = np.array([r['torso_anim'] for r in records])
        origins = np.array([r['origin'] for r in records])
        angles = np.array([r['view_angle'] for r in records]) * np.pi / 180

        return cls(times, leg_anim_idxs, torso_anim_idxs, origins, angles)

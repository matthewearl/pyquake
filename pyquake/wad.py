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


__all__ = (
    'MalformedWadFile',
    'read_wad',
)


import dataclasses
import enum
import struct


class MalformedWadFile(Exception):
    pass


class _LumpType(enum.IntEnum):
    PALETTE = 64
    QTEX = 65
    QPIC = 66
    SOUND = 67
    MIPTEX = 68


@dataclasses.dataclass
class _WadInfo:
    magic: str
    num_lumps: int
    info_table_offset: int

    @classmethod
    def read(cls, f):
        header_fmt = "<4sii"
        wad_info = cls(*struct.unpack(header_fmt, f.read(struct.calcsize(header_fmt))))
        if wad_info.magic != b'WAD2':
            raise MalformedWadFile(f'Bad magic number: {wad_info.magic}')
        return wad_info


@dataclasses.dataclass
class _LumpInfo:
    offset: int
    disk_size: int
    size: int
    type_: _LumpType
    compression: bool
    name: str

    @classmethod
    def read(cls, f):
        entry_fmt = "<iiibbcc16s"
        header_size = struct.calcsize(entry_fmt)
        offset, disk_size, size, type_int, compression_int, p1, p2, name_bytes = (
            struct.unpack(entry_fmt, f.read(header_size))
        )

        lump_info = cls(offset, disk_size, size, _LumpType(type_int),
                        compression_int != 0, 
                        name_bytes[:name_bytes.index(b'\0')].decode('ascii').lower())

        if lump_info.compression:
            raise NotImplementedError("WAD compression not supported")
        elif lump_info.size != lump_info.disk_size:
            raise MalformedWadFile(f"Compression not enabled yet disk size ({lump_info.disk_size})  does not match size {lump_info.size}")

        return lump_info

    def read_data(self, f):
        f.seek(self.offset)
        return f.read(self.size)


def read_wad(f):
    wad_info = _WadInfo.read(f)
    f.seek(wad_info.info_table_offset)
    lump_infos = [_LumpInfo.read(f) for _ in range(wad_info.num_lumps)]
    return {li.name: li.read_data(f) for li in lump_infos}


if __name__ == "__main__":
    import io
    import sys
    import pprint

    import pak

    fs = pak.Filesystem(sys.argv[1])
    pprint.pprint(
        {k: len(v) for k, v in read_wad(fs.open(sys.argv[2])).items()}
    )

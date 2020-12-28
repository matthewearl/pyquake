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

__all__ = (
    'Filesystem',
    'MalformedPakFile',
)


import collections
import collections.abc
import glob
import io
import logging
import os
import pathlib
import struct


logger = logging.getLogger(__name__)
_PakEntry = collections.namedtuple('_PakEntry', ('pak_file', 'offset', 'size'))


class MalformedPakFile(Exception):
    pass


class Filesystem(collections.abc.Mapping):
    """Interface to a .pak file based filesystem."""

    def _read(self, f, n):
        b = f.read(n)
        if len(b) < n:
            raise MalformedPakFile("File ended unexpectedly")
        return b

    def _read_fname(self, f):
        fname = self._read(f, 56)
        if b'\0' in fname:
            fname = fname[:fname.index(b'\0')]
        return fname.decode('ascii')

    def _read_header(self, f):
        try:
            magic = self._read(f, 4)
            if magic != b"PACK":
                raise MalformedPakFile("Invalid magic number")
            return struct.unpack("<II", self._read(f, 8))
        except EOFError:
            raise MalformedPakFile("File too short")

    def _generate_entries(self, pak_file):
        with open(pak_file, "rb") as f:
            logger.info("Reading %s", pak_file)
            file_table_offset, file_table_size = self._read_header(f)
            f.seek(file_table_offset)
            i = 0
            while i < file_table_size:
                fname = self._read_fname(f)
                logger.debug("Indexed %s", fname)
                offset, size = struct.unpack("<II", self._read(f, 8))
                yield fname, _PakEntry(pak_file, offset, size)
                i += 64

    def __init__(self, game_dir):
        self._game_dir = pathlib.Path(game_dir).resolve()

        pak_files = sorted(glob.glob(os.path.join(game_dir, "*.pak")))
        self._index = {fname: entry for pak_file in pak_files for fname, entry in self._generate_entries(pak_file)}

    def __getitem__(self, fname):
        if fname in self._index:
            entry = self._index[fname]
            with open(entry.pak_file, "rb") as f:
                f.seek(entry.offset)
                return self._read(f, entry.size)
        else:
            file_path = (self._game_dir / fname).resolve()
            if self._game_dir not in file_path.parents:
                raise Exception(f'File path is not in game dir {file_path}')
            with file_path.open('rb') as f:
                return f.read()

    def open(self, fname):
        return io.BytesIO(self[fname])

    def __iter__(self):
        # TODO: This doesn't return files (directly) in the file system.  Maybe it should?
        return iter(self._index)

    def __len__(self):
        return len(self._index)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)
    fs = Filesystem(sys.argv[1])


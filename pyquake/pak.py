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


import argparse
import collections
import collections.abc
import glob
import io
import logging
import os
import pathlib
import struct
import sys
from typing import Optional


logger = logging.getLogger(__name__)
_PakEntry = collections.namedtuple('_PakEntry', ('pak_file', 'offset', 'size'))


class MalformedPakFile(Exception):
    pass


def _read(f, n):
    b = f.read(n)
    if len(b) < n:
        raise MalformedPakFile("File ended unexpectedly")
    return b


def _read_fname(f):
    fname = _read(f, 56)
    if b'\0' in fname:
        fname = fname[:fname.index(b'\0')]
    return fname.decode('ascii')


def _read_header(f):
    try:
        magic = _read(f, 4)
        if magic != b"PACK":
            raise MalformedPakFile("Invalid magic number")
        return struct.unpack("<II", _read(f, 8))
    except EOFError:
        raise MalformedPakFile("File too short")


def _generate_entries(pak_file):
    with open(pak_file, "rb") as f:
        logger.info("Reading %s", pak_file)
        file_table_offset, file_table_size = _read_header(f)
        f.seek(file_table_offset)
        i = 0
        while i < file_table_size:
            fname = _read_fname(f)
            logger.debug("Indexed %s", fname)
            offset, size = struct.unpack("<II", _read(f, 8))
            yield fname, _PakEntry(pak_file, offset, size)
            i += 64


def _read_entry(entry):
    with open(entry.pak_file, "rb") as f:
        f.seek(entry.offset)
        return _read(f, entry.size)


class Filesystem(collections.abc.Mapping):
    """Interface to a .pak file based filesystem."""

    def __init__(self, base_dir, game: Optional[str] = None):
        base_dir = pathlib.Path(base_dir).resolve()
        self._game_dirs = [(base_dir / "id1").resolve()]

        for game_dir in self._game_dirs:
            if game_dir.parent != base_dir:
                raise Exception(f'Game dir {game_dir} is not in {base_dir}')

        if game is not None:
            self._game_dirs.append((pathlib.Path(base_dir) / game).resolve())

        pak_files = [pak_path
                     for game_dir in self._game_dirs
                     for pak_path in sorted(glob.glob(os.path.join(game_dir, "*.pak")))]
        self._index = {fname: entry for pak_file in pak_files for fname, entry in _generate_entries(pak_file)}

    def __getitem__(self, fname):
        if fname in self._index:
            entry = self._index[fname]
            return _read_entry(entry)
        else:
            for game_dir in reversed(self._game_dirs):
                file_path = (game_dir / fname).resolve()
                if game_dir not in file_path.parents:
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


def pak_extract_main():
    parser = argparse.ArgumentParser(description='Extract / list pak archives')
    parser.add_argument('-l', '--list', action='store_const',
                        const=True, default=False,
                        help='list archive contents')
    parser.add_argument('-x', '--extract', action='store_const',
                        const=True, default=False,
                        help='extract archive contents')
    parser.add_argument('-f', '--file', required=False, help='extract a single file')
    parser.add_argument('pak_file_name', metavar='pak-file-name')
    parser.add_argument('target_dir', metavar='target-dir', nargs='?')
    parsed = parser.parse_args(sys.argv[1:])

    if (parsed.list + parsed.extract) != 1:
        parser.error('Exactly one of --list or --extract must be passed')

    if parsed.list:
        if parsed.target_dir is not None:
            parser.error('target-dir should only be passed with extracting')

        print(f'{"offset":>9}  {"size":>9}  {"filename"}')
        print(f'{"-" * 9}  {"-" * 9}  {"-" * 12}')
        for fname, entry in _generate_entries(parsed.pak_file_name):
            print(f'{entry.offset:>9}  {entry.size:>9}  {fname}')

    if parsed.extract:
        target_dir = os.getcwd() if parsed.target_dir is None else parsed.target_dir
        target_dir = os.path.join(target_dir, '')
        target_dir = os.path.realpath(target_dir)
        for rel_path, entry in _generate_entries(parsed.pak_file_name):
            if parsed.file is not None and rel_path != parsed.file:
                continue
            abs_path = os.path.realpath(os.path.join(target_dir, rel_path))
            if os.path.commonprefix((abs_path, target_dir)) != target_dir:
                raise Exception(
                    f'Directory traversal attack detected: {abs_path} is not in {target_dir}'
                )

            print(f'extracting {rel_path!r} to {abs_path!r}')
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, 'wb') as f:
                f.write(_read_entry(entry))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)
    fs = Filesystem(sys.argv[1])


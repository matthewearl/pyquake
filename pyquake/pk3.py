import collections
import contextlib
import glob
import os
import pathlib
import zipfile
from typing import Optional


def _get_entries(pk3_file):
    with zipfile.ZipFile(pk3_file) as zf:
        return zf.namelist()


class Filesystem(collections.abc.Mapping):
    """Interface to a .pk3 file based filesystem."""

    def __init__(self, base_dir, game: Optional[str] = None):
        base_dir = pathlib.Path(base_dir)
        self._game_dirs = [(base_dir / "baseq3").resolve()]

        for game_dir in self._game_dirs:
            if game_dir.parent != base_dir:
                raise Exception(f'Game dir {game_dir} is not in {base_dir}')

        if game is not None:
            self._game_dirs.append((pathlib.Path(base_dir) / game).resolve())

        pk3_files = [pk3_path
                     for game_dir in self._game_dirs
                     for pk3_path in sorted(glob.glob(os.path.join(game_dir, "*.pk3")))]

        self._index = {fname: pk3_file for pk3_file in pk3_files for fname in _get_entries(pk3_file)}
        # TODO:  Check search order matches game engine.

    @contextlib.contextmanager
    def open(self, fname):
        if fname in self._index:
            with zipfile.ZipFile(self._index[fname]) as zf:
                with zf.open(fname, 'r') as f:
                    yield f
        else:
            for game_dir in reversed(self._game_dirs):
                file_path = (game_dir / fname).resolve()
                if game_dir not in file_path.parents:
                    raise Exception(f'File path is not in game dir {file_path}')
                with file_path.open('rb') as f:
                    yield f.read()

    def __iter__(self):
        # TODO: This doesn't return files (directly) in the file system.  Maybe it should?
        return iter(self._index)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, fname):
        with self.open(fname):
            return fname.read()

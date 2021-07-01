__all__ = (
    'parse_portal_file',
    'Portal',
)


import dataclasses
import re
from typing import List, Tuple


_VERT_RE = re.compile(
    r" ?\((?P<x>-?\d+(\.\d*)?) "
    r"(?P<y>-?\d+(\.\d*)?) "
    r"(?P<z>-?\d+(\.\d*)?) \)"
    r"(?P<remainder>.*)"
)


@dataclasses.dataclass
class Portal:
    leaves: Tuple[int, int]
    winding: List[Tuple[float, float, float]]


def _read_vert(winding_str):
    m = re.match(_VERT_RE, winding_str)
    return (float(m['x']), float(m['y']), float(m['z'])), m['remainder']


def parse_portal_file(f):
    line_it = iter(x.decode('utf-8').strip() for x in f.readlines())

    if next(line_it) != "PRT1":
        raise Exception("Bad header")

    num_leaves = int(next(line_it))
    num_portals = int(next(line_it))

    portals = []
    for line in line_it:
        num_verts_str, leaf_1_str, leaf_2_str, winding_str = line.split(' ', 3)
        num_verts = int(num_verts_str)
        # Portal leaves are offset by one relative to BSP leaves.
        leaves = 1 + int(leaf_1_str), 1 + int(leaf_2_str)

        winding = []
        while winding_str:
            vert, winding_str = _read_vert(winding_str)
            winding.append(vert)

        if len(winding) != num_verts:
            raise Exception(f"Winding length ({len(winding)}) is "
                            f"different to indicated ({num_verts})")

        portals.append(Portal(leaves, winding))

    if len(portals) != num_portals:
        raise Exception("Number of portals ({len(portals)}) is different "
                        "to indicated")

    if not all(0 <= leaf_idx - 1 < num_leaves for p in portals for leaf_idx in p.leaves):
        raise Exception("Leaf index out of range")

    return portals


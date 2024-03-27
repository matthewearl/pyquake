import argparse
import logging

import numpy as np
import polytope
import scipy
import trimesh

import pyquake.pak
import pyquake.bsp


logger = logging.getLogger(__name__)


def _planes_to_convex_hull(planes):
    """Return a convex hull for a polytope."""
    planes = np.stack(planes)
    pt = polytope.Polytope(
        planes[:, :-1],
        -planes[:, -1]
    )
    interior_point = pt.chebXc
    halfspaces = np.concatenate([pt.A, -pt.b[:, None]], axis=1)
    hsi = scipy.spatial.HalfspaceIntersection(halfspaces, interior_point)
    return scipy.spatial.ConvexHull(hsi.intersections)


def _planes_to_mesh(planes):
    """Return a mesh for a polytope."""
    h = _planes_to_convex_hull(planes)
    tm = trimesh.Trimesh(h.points, h.simplices)
    trimesh.repair.fix_normals(tm)
    return tm


def _planes_to_bbox(planes):
    """Return a bounding box for a polytope.

    The polytope is specified as a set of bounding planes.
    """

    h = _planes_to_convex_hull(planes)
    return np.min(h.points, axis=0), np.max(h.points, axis=0)


def _get_leaf_paths(node):
    """Iterate all leaves and their paths.

    Returns an iterable of `(path, is_solid)` pairs for each leaf in the tree,
    where:
      - `path` is a sequence of (node, child_num) pairs with all the nodes that
        were encountered on the way to the leaf, and the side of the leaf that
        was taken
      - `is_empty` indicates whether the leaf is solid or not.
    """

    for child_num in range(2):
        if node.child_is_leaf(child_num):
            yield [(node, child_num)], node.child_is_empty(child_num)
        else:
            child_node = node.get_child(child_num)
            for path, is_empty in _get_leaf_paths(child_node):
                yield [(node, child_num)] + path, is_empty


def _path_to_planes(path):
    """Given a path to a leaf, return a set of bounding planes for the leaf."""
    return np.array([
        (-1 if child_num == 0 else 1)
        * np.concatenate([node.plane.normal, [-node.plane.dist]])
        for node, child_num in path
    ])


def _get_bbox(node):
    """Get a bounding box on all the empty leavess in a tree."""
    mins_list = []
    maxs_list = []
    for path, is_empty in _get_leaf_paths(node):
        if is_empty:
            single_mins, single_maxs = _planes_to_bbox(_path_to_planes(path))
            mins_list.append(single_mins)
            maxs_list.append(single_maxs)
    return np.min(mins_list, axis=0), np.max(maxs_list, axis=0)


def _planes_from_bbox(mins, maxs):
    """Return bounding planes for a bbox."""
    return np.array([[1, 0, 0, -maxs[0]],
                     [0, 1, 0, -maxs[1]],
                     [0, 0, 1, -maxs[2]],
                     [-1, 0, 0, mins[0]],
                     [0, -1, 0, mins[1]],
                     [0, 0, -1, mins[2]]])


def _get_leaf_meshes(node):
    """Return an iterable of `Trimesh`s in this tree.

    Also returns the path to each leaf along with the `Trimesh`.
    """

    logger.info('finding bounding box')
    mins, maxs = _get_bbox(node)
    mins -= 1
    maxs += 1

    logger.info('converting to meshes')
    bbox_planes = _planes_from_bbox(mins, maxs)
    for path, is_empty in _get_leaf_paths(node):
        if not is_empty:
            path_planes = _path_to_planes(path)
            planes = np.concatenate([bbox_planes, path_planes])
            yield path, _planes_to_mesh(planes)


def _convert_to_blender_coords(mesh):
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]
    mesh.vertices[:, 0] *= -1
    return mesh


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract BSP hulls to a GLB file.")
    parser.add_argument("base_dir", type=str,
                        help="Quake game root")
    parser.add_argument("map_name", type=str, help="Name of the map file.")
    parser.add_argument("-g", "--game", type=str,
                        default=None,
                        help="Optional mod sub-directory")
    parser.add_argument("-o", "--output", type=str,
                        required=True,
                        help="Output file name")
    parser.add_argument("-m", "--model", type=int,
                        default=0,
                        help="Model number")
    parser.add_argument("-H", "--hull", type=int,
                        default=1,
                        help="Hull number")
    args = parser.parse_args()


    logger.info('loading bsp file')
    fs = pyquake.pak.Filesystem(args.base_dir, args.game)
    with fs.open(f'maps/{args.map_name}.bsp') as f:
        b = pyquake.bsp.Bsp(f)

    model = b.models[args.model]
    if args.hull == 0:
        raise Exception("hull 0 not supported at the moment")
    else:
        node = model.get_clip_node(args.hull)
    scene = trimesh.Scene()
    for path, mesh in _get_leaf_meshes(node):
        scene.add_geometry(
            _convert_to_blender_coords(mesh),
            hex(sum((child_num << i) for i, (_, child_num) in enumerate(path)))[2:]
        )

    logger.info('exporting to glb')
    scene.export(args.output)

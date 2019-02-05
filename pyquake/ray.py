from typing import NamedTuple, Iterable

import numpy as np

from . import bsp


class Ray(NamedTuple):
    origin: np.ndarray
    dir: np.ndarray

def _infront(point, plane_norm, plane_dist):
    return np.dot(point, plane_norm) - plane_dist >= 0


class _FaceTextureCache:
    def __init__(self, pal: np.ndarray):
        self._pal = pal
        self._cache = {}

    def _load_face_texture(self, face: bsp.Face):
        texture = face.tex_info.texture
        array_im = self._pal[np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))]
        #array_im = array_im ** 0.8
        return array_im

    def get_face_texture(self, face: bsp.Face):
        if id(face) not in self._cache:
            self._cache[id(face)] = self._load_face_texture(face)
        return self._cache[id(face)]


def _sample_texture(texture_cache: _FaceTextureCache, face: bsp.Face, poi: np.ndarray):
    s, t = bsp.get_tex_coords(face.tex_info, poi)
    im = texture_cache.get_face_texture(face)
    s %= im.shape[1]
    t %= im.shape[0]
    return im[int(t), int(s)]


def _trace_leaves(is_leaf: bool, node: bsp.Node, ray: Ray, near_clip: float = 0., far_clip: float = np.inf):
    """Trace a ray through a BSP node, yielding all encountered leaves (in near to far order)."""
    if is_leaf:
        yield node, near_clip, far_clip
    else:
        plane = node.plane
        plane_norm = np.array(plane.normal)
        plane_dist = plane.dist

        beta = np.dot(plane_norm, ray.dir)
        near_child, far_child = (1, 0) if beta > 0 else (0, 1)

        if np.abs(beta) < 1e-5:
            child = 0 if _infront(ray.origin, plane_norm, plane_dist) else 1
            yield from _trace_leaves(node.child_is_leaf(child), node.get_child(child), ray, near_clip, far_clip)
        else:
            alpha = (plane_dist - np.dot(plane_norm, ray.origin)) / beta

            if near_clip <= alpha < far_clip:
                yield from _trace_leaves(node.child_is_leaf(near_child), node.get_child(near_child),
                                         ray, near_clip, alpha)
                yield from _trace_leaves(node.child_is_leaf(far_child), node.get_child(far_child),
                                         ray, alpha, far_clip)
            elif alpha < near_clip:
                yield from _trace_leaves(node.child_is_leaf(far_child), node.get_child(far_child),
                                         ray, near_clip, far_clip)
            else:
                yield from _trace_leaves(node.child_is_leaf(near_child), node.get_child(near_child),
                                         ray, near_clip, far_clip)


def _ray_face_intersect(face: bsp.Face, ray: Ray):
    face_norm, face_dist = face.plane

    beta = np.dot(face_norm, ray.dir)
    if abs(beta) >= 1e-5:
        ray_dist = (face_dist - np.dot(face_norm, ray.origin)) / beta
        if ray_dist >= 0:
            poi = ray.origin + ray.dir * ray_dist
            if all(_infront(poi, n, d) for n, d in face.edge_planes):
                return poi, ray_dist

    return None, np.inf


def _ray_faces_intersect(faces: Iterable[bsp.Face], ray: Ray, near_clip: float = 0., far_clip: float = np.inf):
    nearest_face, nearest_poi, nearest_dist = None, None, np.inf
    for face in faces:
        poi, ray_dist = _ray_face_intersect(face, ray)
        if ray_dist < nearest_dist and near_clip - 1e-3 <= ray_dist <= far_clip + 1e-3:
            nearest_face, nearest_poi, nearest_dist = face, poi, ray_dist
    return nearest_face, nearest_poi, nearest_dist


def _ray_bsp_intersect(model: bsp.Model, ray: Ray):
    nearest_leaf = None
    for leaf, near_clip, far_clip in _trace_leaves(False, model.node, ray):
        nearest_face, nearest_poi, nearest_dist = _ray_faces_intersect(leaf.faces, ray, near_clip, far_clip)
        if nearest_face is not None:
            nearest_leaf = leaf
            break

    return nearest_face, nearest_poi, nearest_dist, nearest_leaf


def raytracer_main2():
    import io
    import sys
    import logging

    import cv2

    from .bsp import Bsp
    from . import pak

    root_logger = logging.getLogger()
    root_logger.addHandler(logging.StreamHandler())
    root_logger.setLevel(logging.DEBUG)

    fs = pak.Filesystem(sys.argv[1])
    bsp = Bsp(io.BytesIO(fs[sys.argv[2]]))

    WIDTH, HEIGHT = 200, 200
    K_inv = np.matrix([[WIDTH / 2.,             0,  WIDTH  / 2],
                       [0,           HEIGHT / 2.,  HEIGHT / 2],
                       [0,                      0,  1]]).I
    K_inv = np.array(K_inv)

    player_start = next(iter(e for e in bsp.entities if e['classname'] == 'info_player_start'))
    ray_origin = np.array(player_start['origin']) + [0, 0, 21]

    rot = np.array([[1., 0.,  0.],
                    [0., 0.,  1.],
                    [0., -1., 0.]])

    pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 255
    texture_cache = _FaceTextureCache(pal)

    def pix_to_dir(x, y):
        ray_dir = rot @ K_inv @ np.array([x, y, 1])
        return ray_dir / np.linalg.norm(ray_dir)

    cv2.namedWindow("out")
    out = np.zeros((HEIGHT, WIDTH, 3))
    color_wheel = np.random.random((32, 3))
    color_wheel /= np.max(color_wheel, axis=1)[:, None]
    for y in range(HEIGHT):
        for x in range(WIDTH):
            ray = Ray(ray_origin, pix_to_dir(x, y))

            face, poi, dist, _ = _ray_bsp_intersect(bsp.models[0], ray)
            
            #color = (0., 0., 0.) if face is None else color_wheel[hash(face) % len(color_wheel)]
            color = (1, 0, 1) if face is None else _sample_texture(texture_cache, face, poi)
            out[y, x] = np.flip(color, axis=0)

        cv2.imshow("out", out)
        cv2.waitKey(1)

    cv2.destroyWindow("out")
    cv2.imwrite("out.png", out * 255.)


def raytracer_main():
    import cProfile
    cProfile.runctx('raytracer_main2()', globals(), locals(), 'stats3')

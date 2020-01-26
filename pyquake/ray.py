import functools
from typing import NamedTuple, Iterable

import numpy as np

from . import bsp


class Ray(NamedTuple):
    origin: np.ndarray
    dir: np.ndarray

def _infront(point, plane_norm, plane_dist):
    return np.dot(point, plane_norm) - plane_dist >= 0


class _GlowTexture(NamedTuple):
    im: np.ndarray
    sample_distribution: np.ndarray
    emissivity: float


    @classmethod
    def from_im(cls, im):
        dist = np.sum(im, axis=2)
        em = np.sum(dist)
        dist /= em
        return cls(im, dist, em)


class _FaceTextureCache:
    def __init__(self, pal: np.ndarray):
        self._pal = pal
        self._glow_pal = np.copy(pal)
        self._glow_pal[:-32, :] = 0.
        self._cache = {}

    def _load_face_texture(self, face: bsp.Face, glow: bool = False):
        texture = face.tex_info.texture
        indices = np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))
        if glow and np.all(indices < 255 - 32):
            return None
        pal = self._glow_pal if glow else self._pal
        array_im = pal[np.fromstring(texture.data[0], dtype=np.uint8).reshape((texture.height, texture.width))]
        #array_im = array_im ** 0.8
        return _GlowTexture.from_im(array_im) if glow else array_im

    def get_face_texture(self, face: bsp.Face, *, glow: bool = False):
        key = face, glow
        if key not in self._cache:
            self._cache[key] = self._load_face_texture(face, glow)
        return self._cache[key]


def _sample_texture(texture_cache: _FaceTextureCache, face: bsp.Face, poi: np.ndarray, glow=False):
    s, t = face.tex_info.vert_to_tex_coords(poi)
    if glow:
        im = texture_cache.get_face_texture(face, glow=True).im
    else:
        im = texture_cache.get_face_texture(face)
    s %= im.shape[1]
    t %= im.shape[0]
    return im[int(t), int(s)]


@functools.lru_cache(None)
def _get_light_faces_and_textures(texture_cache: _FaceTextureCache, leaf: bsp.Leaf):
    faces = leaf.visible_faces
    textures = [texture_cache.get_face_texture(face, glow=True) for face in faces]
    faces = [f for f, t in zip(faces, textures) if t is not None]
    textures = [t for t in textures if t is not None]
    return faces, textures
    return faces[12:13], textures[12:13]


def _sample_light_sources2(texture_cache: _FaceTextureCache, leaf: bsp.Leaf, normal: np.ndarray, poi: np.ndarray):
    #import ipdb; ipdb.set_trace()
    faces, textures = _get_light_faces_and_textures(texture_cache, leaf)

    # First choose a face to sample from
    em = np.array([t.emissivity for t in textures])
    ray_dir = np.array([face.centroid for face in faces]) - poi
    face_normals = np.stack([face.plane[0] for face in faces])
    distance = np.linalg.norm(ray_dir, axis=1)
    ray_dir /= distance[:, None]
    light_cos_angle = np.abs(np.sum(ray_dir * face_normals, axis=1))
    source_cos_angle = np.abs(np.sum(ray_dir * normal, axis=1))
    area = np.array([face.area for face in faces])

    face_probs = area * em * light_cos_angle * source_cos_angle / distance ** 2
    face_probs /= np.sum(face_probs)

    idx = np.random.choice(range(len(faces)), p=face_probs)
    light_face = faces[idx]
    light_texture = textures[idx]
    face_prob = face_probs[idx]

    # Then pick a random point on the face
    for i in range(20):
        vert_array = np.array(list(light_face.vertices))
        p = np.random.random((len(vert_array),))
        p /= np.sum(p)
        light_pos = p @ vert_array
        colour = _sample_texture(texture_cache, light_face, light_pos, glow=True)
        if not np.array_equal(colour, (0, 0, 0)):
            break

    return light_face, light_pos, colour, (i + 1) * face_prob / light_face.area


def _sample_light_sources(texture_cache: _FaceTextureCache, leaf: bsp.Leaf, normal: np.ndarray, poi: np.ndarray):
    faces = leaf.visible_faces
    textures = [texture_cache.get_face_texture(face, glow=True) for face in faces]
    faces = [f for f, t in zip(faces, textures) if t is not None]
    textures = [t for t in textures if t is not None]

    # First choose a face to sample from
    em = np.array([t.emissivity for t in textures])
    ray_dir = np.array([face.centroid for face in faces]) - poi
    face_normals = np.stack([face.plane[0] for face in faces])
    distance = np.linalg.norm(ray_dir, axis=1)
    ray_dir /= distance[:, None]
    light_cos_angle = np.maximum(np.sum(ray_dir * face_normals, axis=1), 0)
    source_cos_angle = np.maximum(-np.sum(ray_dir * normal, axis=1), 0)
    area = np.array([face.area for face in faces])

    #print(f"area {area}\n em {em}\n light_cos_angle {light_cos_angle}\n source_cos_angle {source_cos_angle}\n distance {distance}")
    face_probs = area * em * light_cos_angle * source_cos_angle / distance ** 2
    face_probs /= np.sum(face_probs)

    idx = np.random.choice(range(len(faces)), p=face_probs)
    light_face = faces[idx]
    light_texture = textures[idx]
    face_prob = face_probs[idx]

    # Sample a pixel on the face
    reshaped_im = light_texture.im.reshape((-1, 3))
    pixel_probs = light_texture.sample_distribution.ravel()
    idx = np.random.choice(range(reshaped_im.shape[0]), p=pixel_probs)
    colour = reshaped_im[idx]
    pixel_prob = pixel_probs[idx]

    # Work out the position
    s, t = idx // light_texture.im.shape[1], idx % light_texture.im.shape[1]
    light_pos = light_face.tex_info.tex_coords_to_vert((s, t))

    return light_face, light_pos, colour, face_prob * pixel_prob / light_face.tex_info.texel_area
    

def _sample_direct_light_contribution(model, texture_cache, face, poi, leaf):
    light_face, light_pos, light_colour, prob = _sample_light_sources2(texture_cache, leaf, face.plane[0], poi)

    ray_dir = light_pos - poi
    ray_len = np.linalg.norm(ray_dir)
    ray_dir /= ray_len
    ray = Ray(poi, ray_dir)
    trace_face, _, _, _ = _ray_bsp_intersect(model, ray, 1e-3)

    if trace_face != light_face:
        out = np.zeros((3,))
    else:
        scale = np.abs(np.dot(ray_dir, light_face.plane[0]) * np.dot(ray_dir, face.plane[0]))
        scale /= np.linalg.norm(light_pos - poi) ** 2
        scale /= prob
        out = 10 * light_colour * scale 

    return out


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
        if ray_dist < nearest_dist and near_clip - 1e-4 <= ray_dist <= far_clip + 1e-4:
            nearest_face, nearest_poi, nearest_dist = face, poi, ray_dist
    return nearest_face, nearest_poi, nearest_dist


def _ray_bsp_intersect(model: bsp.Model, ray: Ray, near_clip=0., far_clip=np.inf):
    nearest_leaf = None
    for leaf, near_clip, far_clip in _trace_leaves(False, model.node, ray, near_clip, far_clip):
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

    origin_leaf = bsp.models[0].get_leaf_from_point(ray_origin)
    visible_leaves = list(origin_leaf.visible_leaves)
    vis_faces = list({f for l in origin_leaf.visible_leaves for f in l.faces})
    print("Num visible leaves %s %s, num vis faces", len(visible_leaves), len(set(visible_leaves)), len(vis_faces))

    cv2.namedWindow("out")
    out = np.zeros((HEIGHT, WIDTH, 3))
    color_wheel = np.random.random((32, 3))
    color_wheel /= np.max(color_wheel, axis=1)[:, None]
    for y in range(HEIGHT):
        print(y)
        for x in range(WIDTH):
            #x, y = 150, 120
            ray = Ray(ray_origin, pix_to_dir(x, y))

            face, poi, dist, leaf = _ray_bsp_intersect(bsp.models[0], ray)
            #face, poi, dist = _ray_faces_intersect(vis_faces, ray)
            
            #color = (0., 0., 0.) if face is None else color_wheel[hash(face) % len(color_wheel)]
            #color = (1, 0, 1) if face is None else _sample_texture(texture_cache, face, poi)
            color = (1, 0, 1) if face is None else (1, 1, 1)

            if face is not None:
                # Sample light sources
                color = np.array([_sample_direct_light_contribution(
                                    bsp.models[0], texture_cache, face, poi, leaf)
                                    for i in range(1)]).mean(axis=0)
            out[y, x] = np.flip(color, axis=0)

        print(np.max(out[y]))
        cv2.imshow("out", out)
        cv2.waitKey(1)

    cv2.destroyWindow("out")
    cv2.imwrite("out.png", out * 255.)


def raytracer_main():
    profiler = 0
    if profiler == 0:
        raytracer_main2()
    elif profiler == 1:
        from line_profiler import LineProfiler
        profile = LineProfiler(_sample_light_sources2,
                               _FaceTextureCache.get_face_texture)
        profile.runctx('raytracer_main2()', globals(), locals())
        profile.print_stats()
    else:
        import cProfile
        cProfile.runctx('raytracer_main2()', globals(), locals(), 'stats5')

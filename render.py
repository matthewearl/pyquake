# Copyright (c) 2018 Matthew Earl
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

import copy
import ctypes
import hashlib
import logging
import os
import pickle
import time

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import ArrayDatatype as ADT
from pygame.locals import *
import numpy as np
import OpenGL.GL.shaders
import pygame
import scipy.interpolate

import boxpack
import proto


SCREEN_SIZE = (1600, 1200)

COLOR_CYCLE = [
    (1., 0., 0.),
    (0., 1., 0.),
    (0., 0., 1.),
    (0., 1., 1.),
    (1., 0., 1.),
    (1., 1., 0.),
]


def iter_face_verts(bsp_file, face_idx):
    face = bsp_file.faces[face_idx]
    for edge_id in bsp_file.edge_list[face.edge_list_idx:face.edge_list_idx + face.num_edges]:
        if edge_id < 0:
            v = bsp_file.edges[-edge_id][1]
        else:
            v = bsp_file.edges[edge_id][0]
        yield bsp_file.vertices[v]


def face_has_lightmap(bsp_file, face_idx):
    face = bsp_file.faces[face_idx]
    return face.lightmap_offset != -1


def extract_lightmap_texture(bsp_file, face_idx):
    face = bsp_file.faces[face_idx]

    tex_info = bsp_file.texinfo[face.texinfo_id]

    tex_coords = np.array([[np.dot(v, tex_info.vec_s) + tex_info.dist_s,
                               np.dot(v, tex_info.vec_t) + tex_info.dist_t]
                                    for v in iter_face_verts(bsp_file, face_idx)])

    mins = np.floor(np.min(tex_coords, axis=0).astype(np.float32) / 16).astype(np.int)
    maxs = np.ceil(np.max(tex_coords, axis=0).astype(np.float32) / 16).astype(np.int)

    size = (maxs - mins) + 1

    lightmap = np.array(list(bsp_file.lightmap[face.lightmap_offset:
                                          face.lightmap_offset + size[0] * size[1]])).reshape((size[1], size[0]))

    tex_coords -= mins * 16
    tex_coords += 8
    tex_coords /= 16.

    return lightmap, tex_coords


def make_full_lightmap(bsp_file, lightmap_size=(512, 512)):
    logging.info("Making lightmap")
    lightmaps = {face_idx: extract_lightmap_texture(bsp_file, face_idx)
                    for face_idx in range(len(bsp_file.faces))
                    if face_has_lightmap(bsp_file, face_idx)}

    lightmaps = dict(reversed(sorted(lightmaps.items(), key=lambda x: x[1][0].shape[0] * x[1][0].shape[1])))

    box_packer = boxpack.BoxPacker(lightmap_size)
    for face_idx, (lightmap, tex_coords) in lightmaps.items():
        if not box_packer.insert(face_idx, (lightmap.shape[1], lightmap.shape[0])):
            raise Exception("Could not pack lightmaps into {} image".format(lightmap_size))

    lightmap_image = np.zeros((lightmap_size[1], lightmap_size[0]), dtype=np.uint8)
    tex_coords = {}
    for face_idx, (x, y) in box_packer:
        lm, tc = lightmaps[face_idx]
        lightmap_image[y:y + lm.shape[0], x:x + lm.shape[1]] = lm
        tex_coords[face_idx] = (tc + (x, y)) / lightmap_size

    return lightmap_image, tex_coords


def iter_face_verts(bsp_file, face_idx):
    face = bsp_file.faces[face_idx]
    for edge_id in bsp_file.edge_list[face.edge_list_idx:face.edge_list_idx + face.num_edges]:
        if edge_id < 0:
            v = bsp_file.edges[-edge_id][1]
        else:
            v = bsp_file.edges[edge_id][0]
        yield bsp_file.vertices[v]


def get_face_coords(bsp_file):
    return {face_idx: np.array([v for v in iter_face_verts(bsp_file, face_idx)])
                for face_idx in range(len(bsp_file.faces))}

class FpsDisplay:
    def __init__(self):
        self._frames_since_last_display = 0
        self._last_display_time = time.perf_counter()

    def new_frame(self):
        self._frames_since_last_display += 1
        t = time.perf_counter()
        if t - self._last_display_time > 1.:
            print("FPS: {:.2f}".format(self._frames_since_last_display / (t - self._last_display_time)))
            self._frames_since_last_display = 0
            self._last_display_time = t


class _DemoCacheMiss(Exception):
    pass


class DemoView:
    def __init__(self, demo_file_name):
        cache_fname = self._get_cache_filename(demo_file_name)

        try:
            view_angles, positions, times, self.map_name = self._check_cache(cache_fname)
            logging.info("Read demo %s from cache %s", demo_file_name, cache_fname)
        except _DemoCacheMiss:
            logging.info("Reading demo %s", demo_file_name)
            data = list(self._view_gen(demo_file_name))
            view_angles, positions, times = (np.array([x[i] for x in data]) for i in range(3))
            self._set_cache(cache_fname, view_angles, positions, times, self.map_name)

        self._view_angle_interp = scipy.interpolate.interp1d(times, view_angles,
                                                             axis=0,
                                                             bounds_error=False,
                                                             fill_value=(view_angles[0], view_angles[-1]))
        self._pos_interp = scipy.interpolate.interp1d(times, positions,
                                                      axis=0,
                                                      bounds_error=False,
                                                      fill_value=(positions[0], positions[-1]))

    def _get_cache_filename(self, demo_file_name):
        with open(demo_file_name, "rb") as f:
            s = f.read()
        return os.path.join("democache", "{}.pickle".format(hashlib.sha1(s).hexdigest()))

    def _check_cache(self, cache_fname):
        try:
            with open(cache_fname, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise _DemoCacheMiss

    def _set_cache(self, cache_fname, view_angles, positions, times, map_name):
        with open(cache_fname, "wb") as f:
            pickle.dump((view_angles, positions, times, map_name), f)

    def _patch_vec(self, old_vec, update):
        return tuple(v if u is None else u for v, u in zip(old_vec, update))
                
    def _view_gen(self, demo_file_name):
        with open(demo_file_name, "rb") as f:
            time = None
            entity_num_to_model_num = {}
            for view_angle, msg in proto.read_demo_file(f):
                if msg.msg_type == proto.ServerMessageType.SERVERINFO:
                    self.map_name = msg.models[0]
                    model_nums = [idx + 1 for idx, model_name in enumerate(msg.models)
                                          if model_name[0] == '*' or idx == 0]
                    model_num_idx = {model_num: idx + 1 for idx, model_num in enumerate(model_nums)}
                    pos = np.zeros((1 + len(model_nums), 3))
                if msg.msg_type == proto.ServerMessageType.SPAWNBASELINE and msg.model_num in model_nums:
                    entity_num_to_model_num[msg.entity_num] = msg.model_num
                if (msg.msg_type in (proto.ServerMessageType.UPDATE, proto.ServerMessageType.SPAWNBASELINE)  and
                    (msg.entity_num == 1 or msg.entity_num in entity_num_to_model_num)):
                    if msg.entity_num == 1:
                        idx = 0
                    else:
                        idx = model_num_idx[entity_num_to_model_num[msg.entity_num]]
                    pos[idx] = self._patch_vec(pos[idx], msg.origin)
                elif msg.msg_type == proto.ServerMessageType.TIME:
                    if time is not None and all(x is not None for x in pos):
                        yield view_angle, copy.copy(pos), time
                    time = msg.time

    def get_view_at_time(self, t):
        all_pos = self._pos_interp(t)
        return self._view_angle_interp(t), all_pos[0], all_pos[1:]


class Renderer:
    def __init__(self, bsp_file, demo_views):
        self._bsp_file = bsp_file
        self._bsp_model_origins = np.zeros((len(bsp_file.models), 3))
        self._demo_views = demo_views
        self._lightmap_image, self._lightmap_texcoords = make_full_lightmap(self._bsp_file)
        self._lightmap_image = np.stack([self._lightmap_image] * 3, axis=2)
        self._face_coords = get_face_coords(self._bsp_file)
        
    def resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, float(width) / height, 10., 20000.)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def _setup_textures(self):
        self._lightmap_texture_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self._lightmap_texture_id)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glTexImage2Df(GL_TEXTURE_2D, 0, 3, 0, GL_RGB, self._lightmap_image / 255.)

    def _setup_buffer_objects(self):
        vertex_array = [v for face_idx in range(len(self._bsp_file.faces))
                          if face_idx in self._lightmap_texcoords
                          for v in self._face_coords[face_idx]]
        texcoord_array = [v for face_idx in range(len(self._bsp_file.faces))
                            if face_idx in self._lightmap_texcoords
                            for v in self._lightmap_texcoords[face_idx]]
        model_faces = {i for m in self._bsp_file.models[1:]
                         for i in range(m.first_face_idx, m.first_face_idx + m.num_faces)}
        color_array = [[0, 1, 0] if face_idx in model_faces else [1, 1, 1]
                         for face_idx in range(len(self._bsp_file.faces))
                         if face_idx in self._lightmap_texcoords
                         for v in self._lightmap_texcoords[face_idx]]

        # Array of indices into the vertex array such that rendering vertices in this order with GL_TRIANGLES will
        # produce all of the models in the map.
        vert_idx = 0
        index_array = []
        for face_idx in range(len(self._bsp_file.faces)):
            if face_idx in self._lightmap_texcoords:
                num_verts_in_face = len(self._face_coords[face_idx])
                for i in range(1, num_verts_in_face - 1):
                    index_array.extend([vert_idx, vert_idx + i, vert_idx + i + 1])
                vert_idx += num_verts_in_face

        # Make a dict from faces indices to (idx, count) pairs such that index_array[idx:idx + count] gives the vertices
        # to render this face.
        vert_idx = 0
        self._face_to_idx = {}
        for face_idx in range(len(self._bsp_file.faces)):
            if face_idx in self._lightmap_texcoords:
                num_verts_in_face = len(self._face_coords[face_idx])
                self._face_to_idx[face_idx] = (vert_idx, 3 * (num_verts_in_face - 2))
                vert_idx += 3 * (num_verts_in_face - 2)
            else:
                self._face_to_idx[face_idx] = (vert_idx, 0)

        vertex_array = np.array(vertex_array, dtype=np.float32)
        texcoord_array = np.array(texcoord_array, dtype=np.float32)
        color_array = np.array(color_array, dtype=np.float32)
        index_array = np.array(index_array, dtype=np.uint32)
        self._index_array_len = len(index_array)

        pos_bo, texcoord_bo, color_bo, self._index_bo = glGenBuffers(4)
        
        # Set up the vertex position buffer
        glBindBuffer(GL_ARRAY_BUFFER, pos_bo);
        glBufferData(GL_ARRAY_BUFFER, ADT.arrayByteCount(vertex_array), vertex_array, GL_STATIC_DRAW)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 12, None)

        # Set up the tex coord buffer
        glBindBuffer(GL_ARRAY_BUFFER, texcoord_bo);
        glBufferData(GL_ARRAY_BUFFER, ADT.arrayByteCount(texcoord_array), texcoord_array, GL_STATIC_DRAW);
        glClientActiveTexture(GL_TEXTURE0)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glTexCoordPointer(2, GL_FLOAT, 8, None)

        # Set up the color buffer
        glBindBuffer(GL_ARRAY_BUFFER, color_bo);
        glBufferData(GL_ARRAY_BUFFER, ADT.arrayByteCount(color_array), color_array, GL_STATIC_DRAW);
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, 0, None)

        # Set up the index buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._index_bo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ADT.arrayByteCount(index_array), index_array, GL_STATIC_DRAW)

    def _get_time(self):
        return (time.perf_counter() - self._game_start_time)

    def _draw_bsp(self):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._lightmap_texture_id)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._index_bo)
        _, _, model_pos = self._demo_views[0].get_view_at_time(self._get_time())

        for idx, m in enumerate(self._bsp_file.models):
            start_idx, _ = self._face_to_idx[m.first_face_idx]
            end_idx, n = self._face_to_idx[m.first_face_idx + m.num_faces - 1]
            end_idx += n
            glPushMatrix()
            #print(idx, model_pos[idx])
            glTranslate(*model_pos[idx])
            glDrawElements(GL_TRIANGLES, end_idx - start_idx, GL_UNSIGNED_INT, ctypes.c_void_p(4 * start_idx))
            glPopMatrix()

    def _show_demo_views(self):
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)

        for idx, demo_view in enumerate(self._demo_views[1:]):
            #glColor3f(*COLOR_CYCLE[idx % len(COLOR_CYCLE)])
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            if idx == 0:
                glColor4f(1, 1, 0, 1.)
            else:
                glColor4f(1, 0, 0, 0.05)
            try:
                view_angle, pos, _ = demo_view.get_view_at_time(self._get_time())
            except StopIteration:
                continue
            glPushMatrix()
            glTranslate(*pos)
            glScalef(30, 30, 30);
            glBegin(GL_TRIANGLE_FAN)
            for theta in np.arange(0, 1, 0.25) * 2 * np.pi:
                glVertex3f(np.sin(theta), np.cos(theta), 0.)
            for theta in np.arange(0, 1, 0.25) * 2 * np.pi:
                glVertex3f(np.sin(theta),  0., np.cos(theta))
            for theta in np.arange(0, 1, 0.25) * 2 * np.pi:
                glVertex3f(0., np.sin(theta), np.cos(theta))
            glEnd()
            glPopMatrix()

        glPopAttrib()

    def _set_view_from_demo(self):
        view_angle, pos, _ = self._demo_views[0].get_view_at_time(self._get_time())

        glRotatef(-view_angle[0], 0, 1, 0)
        glRotatef(-view_angle[1], 0, 0, 1)
        glTranslate(*-np.array(pos))

    def _draw_frame(self):
        self._fps_display.new_frame()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        #glUseProgram(self._shader)

        glLoadIdentity()
        glRotatef (-90,  1, 0, 0)
        glRotatef (90,  0, 0, 1)

        self._set_view_from_demo()
        #glRotatef(90, 0, -1, 0)
        #glTranslate(*-(self._centre + [0, 0, 3000]))

        self._draw_bsp()

        self._show_demo_views()

    def run(self):
        self._fps_display = FpsDisplay()
        self._game_start_time = time.perf_counter()

        pygame.init()
        screen = pygame.display.set_mode(SCREEN_SIZE, HWSURFACE | OPENGL | DOUBLEBUF)
        self._setup_textures()
        self._setup_buffer_objects()
        glEnable(GL_DEPTH_TEST)
        self.resize(*SCREEN_SIZE)

        self._centre = np.array(self._bsp_file.vertices).mean(axis=0)

        x = 0.
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                if event.type == KEYUP and event.key == K_ESCAPE:
                    return                
                if event.type == KEYUP and event.key == ord('n'):
                    self._game_start_time += 1.0
                if event.type == KEYUP and event.key == ord('m'):
                    self._game_start_time -= 1.0
            self._draw_frame()
            pygame.display.flip()

            x += 1.


if __name__ == "__main__":
    import io
    import sys
    import logging

    import bsp
    import pak

    root_logger = logging.getLogger()
    root_logger.addHandler(logging.StreamHandler())
    root_logger.setLevel(logging.DEBUG)

    fs = pak.Filesystem(sys.argv[1])

    demo_views = [DemoView(fname) for fname in sys.argv[2:]]
    map_name = demo_views[0].map_name
    demo_views = [dv for dv in demo_views if dv.map_name == map_name]
    bsp_file = bsp.BspFile(io.BytesIO(fs[demo_views[0].map_name]))

    renderer = Renderer(bsp_file, demo_views)
    renderer.run()



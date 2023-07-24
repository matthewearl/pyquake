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

import ctypes
import logging
import time

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import ArrayDatatype as ADT
from pygame.locals import *
import numpy as np
import pygame

from . import demo


logger = logging.getLogger(__name__)


REFERENCE_RADIUS = 100
SCREEN_SIZE = (1600, 1200)

COLOR_CYCLE = [
    (1., 0., 0.),
    (0., 1., 0.),
    (0., 0., 1.),
    (0., 1., 1.),
    (1., 0., 1.),
    (1., 1., 0.),
]


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


class Timer:
    def __init__(self):
        self._last_perf_counter = time.perf_counter()
        self.time = 0.
        self.timescale = 1.0
        self.paused = False

    def update(self):
        pc = time.perf_counter()
        if not self.paused:
            self.time += (pc - self._last_perf_counter) * self.timescale
        self._last_perf_counter = pc

    def shift(self, offset):
        self.time += offset

    def change_speed(self, inc):
        self.timescale *= 1.1 ** inc
        logger.info("Timescale set to %.2f", self.timescale)


class Renderer:
    def __init__(self, bsp, demo_views):
        self._bsp = bsp
        self._bsp_model_origins = np.zeros((len(bsp.models), 3))
        self._demo_views = demo_views
        self._lightmap_image = np.sum(self._bsp.full_lightmap_image, axis=0)
        self._lightmap_image = np.stack([self._lightmap_image] * 3, axis=2)
        self._first_person = False
        self._timer = Timer()
        self._demo_offsets = np.zeros((len(demo_views),), dtype=np.float32)
        
    def resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, float(width) / height, 10., 20000.)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def _draw_speed(self):
        delta_t = 0.05
        current_time = self._get_time()
        _, p1, _ = self._demo_views[0].get_view_at_time(current_time)
        _, p2, _ = self._demo_views[0].get_view_at_time(current_time + delta_t)

        p1[2] = p2[2] - 0.
        speed = np.linalg.norm(p1 - p2) / delta_t
        s = "Speed: {:.1f}".format(speed)
        print(s)

    def _sync_demos_to_reference(self):
        current_time = self._get_time()
        _, reference_pos, _ = self._demo_views[0].get_view_at_time(current_time)
        reference_pos[2] = 0.

        for idx, dv in enumerate(self._demo_views):
            if idx == 0:
                continue
            for t in np.arange(0, dv.end_time, 0.2):
                _, pos, _ = dv.get_view_at_time(t)
                pos[2] = 0.
                if np.linalg.norm(pos - reference_pos) < REFERENCE_RADIUS:
                    self._demo_offsets[idx] = t - current_time
                    break
            for t in np.arange(t - 0.2, t, 0.02):
                _, pos, _ = dv.get_view_at_time(t)
                pos[2] = 0.
                if np.linalg.norm(pos - reference_pos) < REFERENCE_RADIUS:
                    self._demo_offsets[idx] = t - current_time
                    break

    def _setup_textures(self):
        self._lightmap_texture_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self._lightmap_texture_id)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glTexImage2Df(GL_TEXTURE_2D, 0, 3, 0, GL_RGB, self._lightmap_image / 255.)

    def _setup_buffer_objects(self):
        vertex_array = [v for face in self._bsp.faces
                          if face.has_any_lightmap
                          for v in face.vertices]
        texcoord_array = [v for face in self._bsp.faces
                            if face.has_any_lightmap
                            for v in face.full_lightmap_tex_coords]
        model_faces = {i for m in self._bsp.models[1:]
                         for i in range(m.first_face_idx, m.first_face_idx + m.num_faces)}
        color_array = [[0, 1, 0] if face.id_ in model_faces else [1, 1, 1]
                         for face in self._bsp.faces
                         if face.has_any_lightmap
                         for v in face.vertices]
        assert len({len(vertex_array), len(texcoord_array), len(color_array)}) == 1

        # Array of indices into the vertex array such that rendering vertices in this order with GL_TRIANGLES will
        # produce all of the models in the map.
        vert_idx = 0
        index_array = []
        for face in self._bsp.faces:
            if face.has_any_lightmap:
                num_verts_in_face = face.num_edges
                for i in range(1, num_verts_in_face - 1):
                    index_array.extend([vert_idx, vert_idx + i, vert_idx + i + 1])
                vert_idx += num_verts_in_face

        # Make a dict from faces indices to (idx, count) pairs such that index_array[idx:idx + count] gives the vertices
        # to render this face.
        vert_idx = 0
        self._face_to_idx = {}
        for face in self._bsp.faces:
            if face.has_any_lightmap:
                num_verts_in_face = face.num_edges
                self._face_to_idx[face.id_] = (vert_idx, 3 * (num_verts_in_face - 2))
                vert_idx += 3 * (num_verts_in_face - 2)
            else:
                self._face_to_idx[face.id_] = (vert_idx, 0)

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
        return self._timer.time

    def _draw_bsp(self):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._lightmap_texture_id)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._index_bo)
        _, _, model_pos = self._demo_views[0].get_view_at_time(self._get_time())

        for idx, m in enumerate(self._bsp.models):
            start_idx, _ = self._face_to_idx[m.first_face_idx]
            end_idx, n = self._face_to_idx[m.first_face_idx + m.num_faces - 1]
            end_idx += n
            glPushMatrix()
            glTranslate(*model_pos[idx])
            glDrawElements(GL_TRIANGLES, end_idx - start_idx, GL_UNSIGNED_INT, ctypes.c_void_p(4 * start_idx))
            glPopMatrix()

    def _draw_circle(self):
        glBegin(GL_TRIANGLE_FAN)
        for theta in np.arange(0, 1, 0.05) * 2 * np.pi:
            glVertex3f(np.sin(theta), np.cos(theta), 0.)
        glEnd()

    def _draw_cross(self):
        glBegin(GL_LINES)
        glVertex3f(-1, 0, 0)
        glVertex3f(1, 0, 0)
        glVertex3f(0, -1, 0)
        glVertex3f(0, +1, 0)
        glVertex3f(0, 0, -1)
        glVertex3f(0, 0, +1)
        glEnd()

    def _show_demo_views(self):
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        current_time = self._get_time()

        for idx, demo_view in enumerate(self._demo_views):
            if self._first_person and idx == 0:
                continue
            #glColor3f(*COLOR_CYCLE[idx % len(COLOR_CYCLE)])
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            try:
                view_angle, pos, _ = demo_view.get_view_at_time(current_time + self._demo_offsets[idx])
            except StopIteration:
                continue
            glPushMatrix()
            glTranslate(*pos)
            if idx == 0:
                glColor4f(1, 1, 0, 0.2)
                glScalef(*([REFERENCE_RADIUS] * 3))
                self._draw_circle()
            else:
                glColor4f(1, 0, 0, 0.5)
                glScalef(30, 30, 30);
                self._draw_cross()
            glPopMatrix()

        glPopAttrib()

    def _set_view_from_demo(self):
        view_angle, pos, _ = self._demo_views[0].get_view_at_time(self._get_time())

        glRotatef(-view_angle[0], 0, 1, 0)
        glRotatef(-view_angle[1], 0, 0, 1)
        glTranslate(*-np.array(pos))

    def _set_view_to_player_start(self):
        player_start = next(iter(e for e in self._bsp.entities if e['classname'] == 'info_player_start'))
        glRotatef(-player_start['angle'], 0, 0, 1)
        glTranslate(*-np.array(player_start['origin']))

    def _draw_frame(self):
        self._fps_display.new_frame()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        #glUseProgram(self._shader)

        glLoadIdentity()
        glRotatef (-90,  1, 0, 0)
        glRotatef (90,  0, 0, 1)

        if self._first_person:
            self._set_view_from_demo()
        else:
            glRotatef(90, 0, -1, 0)
            glTranslate(*-(self._centre + [0, 0, 3000]))

        self._draw_bsp()
        self._show_demo_views()
        self._draw_speed()

    def run(self):
        self._fps_display = FpsDisplay()
        self._game_start_time = time.perf_counter()

        pygame.init()
        screen = pygame.display.set_mode(SCREEN_SIZE, HWSURFACE | OPENGL | DOUBLEBUF)
        self._setup_textures()
        self._setup_buffer_objects()
        glEnable(GL_DEPTH_TEST)
        self.resize(*SCREEN_SIZE)

        self._centre = np.array(self._bsp.vertices).mean(axis=0)

        x = 0.
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                if event.type == KEYUP and event.key == K_ESCAPE:
                    return                
                if event.type == KEYUP and event.key == ord(' '):
                    self._timer.paused = not self._timer.paused
                    if self._timer.paused:
                        print("Paused at {:.2f}".format(self._get_time()))
                if event.type == KEYUP and event.key == ord('f'):
                    self._first_person = not self._first_person
                if event.type == KEYUP and event.key == ord('n'):
                    self._timer.shift(-1)
                if event.type == KEYUP and event.key == ord('m'):
                    self._timer.shift(1)
                if event.type == KEYUP and event.key == ord('j'):
                    self._timer.change_speed(-1)
                if event.type == KEYUP and event.key == ord('k'):
                    self._timer.change_speed(1)
                if event.type == KEYUP and event.key == ord('x'):
                    self._sync_demos_to_reference()
            self._draw_frame()
            pygame.display.flip()

            x += 1.
            self._timer.update()

def demo_viewer_main():
    import io
    import sys

    from .bsp import Bsp
    from . import pak

    logging.basicConfig(level=logging.DEBUG)

    fs = pak.Filesystem(sys.argv[1])

    def load_dv(fname, fetch_model_positions):
        try:
            dv = demo.DemoView(fname, fetch_model_positions)
        except demo.NoServerInfoInDemo:
            return None
        return dv

    demo_views = [load_dv(fname, (idx == 0)) for idx, fname in enumerate(sys.argv[2:])]
    demo_views = [x for x in demo_views if x is not None]
    map_name = demo_views[0].map_name
    demo_views = [dv for dv in demo_views if dv.map_name == map_name]
    bsp = Bsp(io.BytesIO(fs[demo_views[0].map_name]))

    renderer = Renderer(bsp, demo_views)
    renderer.run()

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

import logging
import time

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

import boxpack


SCREEN_SIZE = (1600, 1200)


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


class Renderer:
    def __init__(self, bsp_file):
        self._bsp_file = bsp_file
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

        glTexImage2Df(GL_TEXTURE_2D, 0, 3, 0, GL_RGB, (self._lightmap_image / 255.) ** 0.5)

    def _get_time(self):
        return time.perf_counter() - self._game_start_time

    def _draw_bsp(self):
        #glColor([255., 1., 0.])
        #glBegin(GL_LINES)
        #for e1, e2 in self._bsp_file.edges:
        #    glVertex(self._bsp_file.vertices[e1])
        #    glVertex(self._bsp_file.vertices[e2])
        #glEnd()

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._lightmap_texture_id)
        for face_idx in range(len(self._bsp_file.faces)):
            if face_idx not in self._lightmap_texcoords:
                continue
            glBegin(GL_TRIANGLE_FAN)
            for vert_coords, tex_coords in zip(self._face_coords[face_idx], self._lightmap_texcoords[face_idx]):
                glTexCoord2f(*tex_coords)
                glVertex3f(*vert_coords)
            glEnd()

    def _draw_frame(self):
        self._fps_display.new_frame()
        glLoadIdentity()
        glRotatef (-90,  1, 0, 0)
        glRotatef (90,  0, 0, 1)
        
        glRotatef(90, 0, -1, 0)
        glTranslate(*-(self._centre + [0, 0, 3000 - 100 * self._get_time()]))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        self._draw_bsp()

        self._last_frame_time = time.perf_counter()

    def run(self):
        self._fps_display = FpsDisplay()
        self._game_start_time = time.perf_counter()

        pygame.init()
        screen = pygame.display.set_mode(SCREEN_SIZE, HWSURFACE | OPENGL | DOUBLEBUF)
        self._setup_textures()
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
    bsp_file = bsp.BspFile(io.BytesIO(fs[sys.argv[2]]))

    renderer = Renderer(bsp_file)
    renderer.run()


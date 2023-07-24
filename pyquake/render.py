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


import contextlib
import ctypes
import logging

import numpy as np
from OpenGL import GL
from OpenGL.arrays import ArrayDatatype as ADT


logger = logging.getLogger(__name__)


class Renderer:
    def __init__(self, bsp):
        self._bsp = bsp

    def _setup_textures(self):
        lightmap_image = np.sum(self._bsp.full_lightmap_image, axis=0)
        lightmap_image = np.stack([lightmap_image] * 3, axis=2)

        self._lightmap_texture_id = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._lightmap_texture_id)
        GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)

        GL.glTexImage2Df(GL.GL_TEXTURE_2D, 0, 3, 0, GL.GL_RGB, lightmap_image / 255.)

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

        pos_bo, texcoord_bo, color_bo, self._index_bo = GL.glGenBuffers(4)
        
        # Set up the vertex position buffer
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, pos_bo);
        GL.glBufferData(GL.GL_ARRAY_BUFFER, ADT.arrayByteCount(vertex_array), vertex_array, GL.GL_STATIC_DRAW)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 12, None)

        # Set up the tex coord buffer
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, texcoord_bo);
        GL.glBufferData(GL.GL_ARRAY_BUFFER, ADT.arrayByteCount(texcoord_array), texcoord_array, GL.GL_STATIC_DRAW);
        GL.glClientActiveTexture(GL.GL_TEXTURE0)
        GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
        GL.glTexCoordPointer(2, GL.GL_FLOAT, 8, None)

        # Set up the color buffer
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, color_bo);
        GL.glBufferData(GL.GL_ARRAY_BUFFER, ADT.arrayByteCount(color_array), color_array, GL.GL_STATIC_DRAW);
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glColorPointer(3, GL.GL_FLOAT, 0, None)

        # Set up the index buffer
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._index_bo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, ADT.arrayByteCount(index_array), index_array, GL.GL_STATIC_DRAW)

    def draw_model(self, model_idx: int, pos):
        m = self._bsp.models[model_idx]
        start_idx, _ = self._face_to_idx[m.first_face_idx]
        end_idx, n = self._face_to_idx[m.first_face_idx + m.num_faces - 1]
        end_idx += n
        GL.glPushMatrix()
        GL.glTranslate(*pos)
        GL.glDrawElements(GL.GL_TRIANGLES, end_idx - start_idx, GL.GL_UNSIGNED_INT, ctypes.c_void_p(4 * start_idx))
        GL.glPopMatrix()

    @contextlib.contextmanager
    def setup_frame(self):
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._lightmap_texture_id)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._index_bo)
        yield
        # TODO: unbind / disable?

    def setup(self):
        self._setup_textures()
        self._setup_buffer_objects()

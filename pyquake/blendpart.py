# Copyright (c) 2021 Matthew Earl
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
    'create_explosion',
    'get_particle_root',
)


import functools

import bpy
import bmesh

from . import blendmat


def _create_icosphere(diameter, obj_name):
    mesh = bpy.data.meshes.new(obj_name)
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, diameter=diameter)
    bm.to_mesh(mesh)
    bm.free()

    return obj


@functools.lru_cache(None)
def get_particle_root():
    return bpy.data.objects.new('particle_root', None)


@functools.lru_cache(None)
def _get_explosion_particle_object():
    obj = _create_icosphere(2, 'explosion_particle')
    obj.parent = get_particle_root()
    obj.data.materials.append(blendmat.setup_explosion_particle_material('explosion').mat)

    return obj


def create_explosion(start_time, obj_name, pos, fps):
    emitter = _create_icosphere(4, obj_name)

    emitter.modifiers.new("explosion_particle_system", type='PARTICLE_SYSTEM')

    assert len(emitter.particle_systems) == 1
    part = emitter.particle_systems[0].settings

    part.frame_start = int(round(start_time * fps))
    part.frame_end = int(round((start_time + 0.1) * fps))
    part.lifetime = int(round(fps * 0.2))
    part.lifetime_random = 0.9
    part.factor_random = 0.5
    part.render_type = 'OBJECT'
    part.instance_object = _get_explosion_particle_object()
    part.particle_size = 1
    part.effector_weights.gravity = 0.01

    emitter.show_instancer_for_render = False

    return emitter


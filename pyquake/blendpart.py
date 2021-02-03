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
import numpy as np

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


def _create_cuboid(mins, maxs, obj_name):
    mesh = bpy.data.meshes.new(obj_name)
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    mins = np.array(mins)
    maxs = np.array(maxs)
    size = maxs - mins
    centre = 0.5 * (maxs + mins)

    bm = bmesh.new()
    d = bmesh.ops.create_cube(bm, size=1)
    verts = d['verts']
    bmesh.ops.scale(bm, vec=size, verts=verts)
    bmesh.ops.translate(bm, vec=centre, verts=verts)
    bm.to_mesh(mesh)
    bm.free()

    return obj


@functools.lru_cache(None)
def get_particle_root():
    obj = bpy.data.objects.new('particle_root', None)
    bpy.context.scene.collection.objects.link(obj)
    return obj


@functools.lru_cache(None)
def _get_explosion_particle_object():
    obj = _create_icosphere(1, 'explosion_particle')
    obj.parent = get_particle_root()
    obj.data.materials.append(blendmat.setup_explosion_particle_material('explosion').mat)

    obj.hide_render = True

    return obj


@functools.lru_cache(None)
def _get_teleport_particle_object():
    obj = _create_icosphere(1, 'teleport_particle')
    obj.parent = get_particle_root()
    obj.data.materials.append(blendmat.setup_teleport_particle_material('teleport').mat)

    obj.hide_render = True

    return obj


def create_explosion(start_time, obj_name, pos, fps):
    emitter = _create_icosphere(4, obj_name)

    emitter.modifiers.new("explosion_particle_system", type='PARTICLE_SYSTEM')

    assert len(emitter.particle_systems) == 1
    part = emitter.particle_systems[0].settings

    part.frame_start = int(round(start_time * fps))
    part.frame_end = int(round((start_time + 0.1) * fps))
    part.lifetime = int(round(fps))
    part.lifetime_random = 0.9
    part.normal_factor = 0
    part.factor_random = 0.5
    part.render_type = 'OBJECT'
    part.instance_object = _get_explosion_particle_object()
    part.particle_size = 1
    part.size_random = 0.9
    part.effector_weights.gravity = 0.01

    emitter.show_instancer_for_render = False
    emitter.parent = get_particle_root()
    emitter.location = pos

    return emitter

def create_teleport(start_time, obj_name, pos, fps):
    emitter = _create_cuboid((-16, -16, -24), (16, 16, 32), obj_name)
    emitter.parent = get_particle_root()

    emitter.modifiers.new("teleport_particle_system", type='PARTICLE_SYSTEM')

    assert len(emitter.particle_systems) == 1
    part = emitter.particle_systems[0].settings

    part.frame_start = int(round(start_time * fps))
    part.frame_end = int(round((start_time + 0.1) * fps))
    part.lifetime = int(round(fps * 0.5))
    part.lifetime_random = 0.5
    part.emit_from = 'VOLUME'
    part.normal_factor = 0.02
    part.factor_random = 0.02
    part.render_type = 'OBJECT'
    part.instance_object = _get_teleport_particle_object()
    part.particle_size = 1
    part.size_random = 0.9
    part.effector_weights.gravity = 0.00

    emitter.show_instancer_for_render = False
    emitter.parent = get_particle_root()
    emitter.location = pos

    return emitter


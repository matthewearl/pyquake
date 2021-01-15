# Copyright (c) 2020 Matthew Earl
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
import math
import re
from dataclasses import dataclass
from typing import List, Tuple

import bpy
import numpy as np

from . import proto, bsp, mdl, blendmdl, blendbsp


logger = logging.getLogger(__name__)
Vec3 = Tuple[float, float, float]


def _patch_vec(old_vec: Vec3, update):
    return tuple(v if u is None else u for v, u in zip(old_vec, update))


def _quake_to_blender_angles(quake_angles: Vec3) -> Vec3:
    return (math.pi * (90. - quake_angles[0]) / 180,
            math.pi * quake_angles[2] / 180,
            math.pi * (quake_angles[1] - 90.) / 180)


@dataclass(frozen=True)
class AliasModelEntityFrame:
    time: float
    origin: Vec3
    angles: Vec3
    frame: int
    skin_idx: int

    def update(self, time, origin_update, angles_update, frame_update, skin_idx_update) -> "AliasModelEntityFrame":
        return AliasModelEntityFrame(
            time=time,
            origin=_patch_vec(self.origin, origin_update),
            angles=_patch_vec(self.angles, angles_update),
            frame=self.frame if frame_update is None else frame_update,
            skin_idx=self.skin_idx if skin_idx_update is None else skin_idx_update,
        )


@dataclass
class AliasModelEntity:
    model_path: str
    path: List[AliasModelEntityFrame]


class AliasModelAnimator:
    def __init__(self, world_obj, fs, pal, fps=30):
        self._world_obj = world_obj
        self._model_paths = None
        self._entities = {}
        self._static_entities = []
        self._fs = fs
        self._pal = pal
        self._fps = fps
        self._final_time = None

        self.entity_objs = {}
        self.static_entity_objs = {}

    def handle_parsed(self, view_angles, parsed, time):
        if parsed.msg_type == proto.ServerMessageType.SERVERINFO:
            self._model_paths = parsed.models

        if parsed.msg_type == proto.ServerMessageType.SPAWNSTATIC:
            model_path = self._model_paths[parsed.model_num - 1]
            if model_path.endswith('.mdl'):
                current_frame = AliasModelEntityFrame(
                    time=time,
                    origin=parsed.origin,
                    angles=parsed.angles,
                    frame=parsed.frame,
                    skin_idx=parsed.skin,
                )
                self._static_entities.append(AliasModelEntity(
                    model_path=model_path,
                    path=[current_frame],
                ))

        if parsed.msg_type == proto.ServerMessageType.SPAWNBASELINE:
            model_path = self._model_paths[parsed.model_num - 1]
            if model_path.endswith('.mdl'):
                current_frame = AliasModelEntityFrame(
                    time=time,
                    origin=parsed.origin,
                    angles=parsed.angles,
                    frame=parsed.frame,
                    skin_idx=parsed.skin,
                )
                self._entities[parsed.entity_num] = AliasModelEntity(
                    model_path=model_path,
                    path=[current_frame],
                )

        if parsed.msg_type == proto.ServerMessageType.UPDATE:
            if parsed.entity_num in self._entities:
                ame = self._entities[parsed.entity_num]
                ame.path.append(ame.path[-1].update(time, parsed.origin, parsed.angle, parsed.frame, parsed.skin))

        self._final_time = time

    def _path_to_model_name(self, mdl_path):
        m = re.match(r"progs/([A-Za-z0-9-_]*)\.mdl", mdl_path)
        if m is None:
            raise Exception("Unexpected model path {mdl_path}")
        return m.group(1)

    def done(self):
        logger.info("Loading models")
        alias_models = {p: mdl.AliasModel(self._fs.open(p)) for p in self._model_paths if p.endswith('.mdl')}

        logger.info("Creating models in blender")
        for entity_num, ame in self._entities.items():
            frames = [(fr.time, fr.frame) for fr in ame.path if fr.time is not None]
            frames = [frames[i] for i in range(len(frames)) if i == 0 or frames[i][1] != frames[i - 1][1]]

            bm = blendmdl.add_model(alias_models[ame.model_path],
                                    self._pal,
                                    self._path_to_model_name(ame.model_path),
                                    f"ent{entity_num}",
                                    frames,
                                    [fr.skin_idx for fr in ame.path][-1],
                                    self._final_time,
                                    static=False,
                                    fps=self._fps)
            bm.obj.parent = self._world_obj

            self.entity_objs[entity_num] = bm.obj

            for fr in ame.path:
                if fr.time is not None:
                    bm.obj.location = fr.origin
                    blender_frame = int(round(self._fps * fr.time))
                    bm.obj.keyframe_insert('location', frame=blender_frame)
                    bm.obj.rotation_euler = (0., 0., fr.angles[1])  # ¯\_(ツ)_/¯
                    bm.obj.keyframe_insert('rotation_euler', frame=blender_frame)

        for idx, ame in enumerate(self._static_entities):
            assert len(ame.path) == 1
            frame, = ame.path
            print(frame)
            print(ame.model_path)
            bm = blendmdl.add_model(alias_models[ame.model_path],
                                    self._pal,
                                    self._path_to_model_name(ame.model_path),
                                    f"static{idx}",
                                    [(frame.time, frame.frame)],
                                    [fr.skin_idx for fr in ame.path][-1],
                                    self._final_time,
                                    static=True,
                                    fps=self._fps)
            bm.obj.parent = self._world_obj
            self.static_entity_objs[idx] = bm.obj

            bm.obj.location = frame.origin


class LevelAnimator:
    def __init__(self, world_obj, fs, pal, config, fps=30):
        self._world_obj = world_obj
        self._config = config
        self._fs = fs
        self._pal = pal
        self._fps = fps
        self._view_entity = None
        self._map_name = None
        self._num_models = None
        self._entity_origins = {}
        self._entity_to_model = {}
        self._bb = None

        demo_cam = bpy.data.cameras.new(name="demo_cam")
        demo_cam.lens = 18.0
        self._demo_cam_obj = bpy.data.objects.new(name="demo_cam", object_data=demo_cam)
        bpy.context.scene.collection.objects.link(self._demo_cam_obj)
        self._demo_cam_obj.parent = self._world_obj

    def handle_parsed(self, view_angles, parsed, time):
        if parsed.msg_type == proto.ServerMessageType.SERVERINFO:
            map_path = parsed.models[0]
            self._map_name = re.match(r"maps/([a-zA-Z0-9_]+).bsp", map_path).group(1)

            b = bsp.Bsp(self._fs.open(map_path))
            self._bb = blendbsp.add_bsp(b, self._pal, self._map_name, self._config)
            self._bb.map_obj.parent = self._world_obj

            self._num_models = 1 + sum((m[0] == '*') for m in parsed.models)

        if parsed.msg_type == proto.ServerMessageType.SETVIEW:
            self._view_entity = parsed.viewentity

        if parsed.msg_type == proto.ServerMessageType.SPAWNBASELINE:
            if parsed.model_num < self._num_models:
                self._entity_origins[parsed.entity_num] = parsed.origin
                self._entity_to_model[parsed.entity_num] = parsed.model_num

            if parsed.entity_num == self._view_entity:
                self._view_origin = parsed.origin

        if parsed.msg_type == proto.ServerMessageType.UPDATE:
            frame = int(self._fps * time)
            if parsed.entity_num in self._entity_origins:
                model_num = self._entity_to_model[parsed.entity_num]
                self._entity_origins[parsed.entity_num] = _patch_vec(self._entity_origins[parsed.entity_num],
                                                                     parsed.origin)

                model_obj = self._bb.model_objs[model_num - 1]
                model_obj.location = self._entity_origins[parsed.entity_num]
                model_obj.keyframe_insert('location', frame=frame)

            if parsed.entity_num == self._view_entity:
                self._view_origin = _patch_vec(self._view_origin, parsed.origin)
                view_origin = self._view_origin[:2] + (self._view_origin[2] + 22,)
                self._demo_cam_obj.location = view_origin
                self._demo_cam_obj.keyframe_insert('location', frame=frame)
                self._demo_cam_obj.rotation_euler = _quake_to_blender_angles(view_angles)
                self._demo_cam_obj.keyframe_insert('rotation_euler', frame=frame)

                self._bb.set_visible_sample_as_light(view_origin)
                self._bb.insert_sample_as_light_visibility_keyframe(frame)

    def done(self):
        pass


def add_demo(demo_file, fs, config, fps=30, world_obj_name='demo',
             load_level=True, relative_time=False):
    pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 255
    world_obj = bpy.data.objects.new(world_obj_name, None)
    bpy.context.scene.collection.objects.link(world_obj)

    if load_level:
        level_animator = LevelAnimator(world_obj, fs, pal, config, fps)
    am_animator = AliasModelAnimator(world_obj, fs, pal, fps)

    time = None
    first_time = None
    for view_angles, parsed in proto.read_demo_file(demo_file):
        if parsed.msg_type == proto.ServerMessageType.TIME:
            if first_time is None:
                first_time = parsed.time
            if relative_time:
                time = parsed.time - first_time
            else:
                time = parsed.time
        if load_level:
            level_animator.handle_parsed(view_angles, parsed, time)
        am_animator.handle_parsed(view_angles, parsed, time)

    if load_level:
        level_animator.done()
    am_animator.done()

    return world_obj, am_animator.entity_objs

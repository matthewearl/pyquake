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


import dataclasses
import functools
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


def _fix_angles(old_angles, new_angles, degrees=False):
    # Undo wrapping of the yaw
    if degrees:
        t = 360
    else:
        t = 2 * np.pi

    old_yaw = old_angles[1] / t
    new_yaw = new_angles[1] / t
    i = old_yaw // 1
    j = np.argmin(np.abs(np.array([i - 1, i, i + 1]) + new_yaw - old_yaw))
    return (new_angles[0], (new_yaw + i + j - 1) * t, new_angles[2])


@dataclass(frozen=True)
class AliasModelEntityFrame:
    time: float
    origin: Vec3
    angles: Vec3
    frame: int
    skin_idx: int
    visible: bool

    def update(self, time, origin_update, angles_update, frame_update, skin_idx_update) -> "AliasModelEntityFrame":

        new_angles = _fix_angles(self.angles, _patch_vec(self.angles, angles_update))
        return AliasModelEntityFrame(
            time=time,
            origin=_patch_vec(self.origin, origin_update),
            angles=new_angles,
            frame=self.frame if frame_update is None else frame_update,
            skin_idx=self.skin_idx if skin_idx_update is None else skin_idx_update,
            visible=True,
        )

    def set_invisible(self):
        return dataclasses.replace(self, visible=False)


@dataclass
class AliasModelEntity:
    model_path: str
    path: List[AliasModelEntityFrame]


@dataclass
class ManagedObject:
    fps: float

    def _get_blender_frame(self, time):
        return int(round(self.fps * time))

    def add_pose_keyframe(self, pose_num: int, time: float):
        raise NotImplementedError

    def add_visible_keyframe(self, visible: bool, time: float):
        raise NotImplementedError

    def add_origin_keyframe(self, origin: Vec3, time: float):
        raise NotImplementedError

    def add_angles_keyframe(self, angles: Vec3, time: float):
        raise NotImplementedError


@dataclass
class AliasModelManagedObject(ManagedObject):
    bm: BlendMdl

    def add_pose_keyframe(self, pose_num: int, time: float):
        bm.add_pose_keyframe(pose_num, time, self.fps)

    def add_visible_keyframe(self, visible: bool, time: float):
        blender_frame = self.get_blender_frame(time)
        for sub_obj in self.sub_objs:
            sub_obj.hide_render = not visible
            sub_obj.keyframe_insert('hide_render', frame=blender_frame)
            sub_obj.hide_viewport = not visible
            sub_obj.keyframe_insert('hide_viewport', frame=blender_frame)

    def add_origin_keyframe(self, origin: Vec3, time: float):
        self.bm.obj.location = location
        self.bm.obj.insert_keyframe('location', self._get_blender_frame(time))

    def add_angles_keyframe(self, angles: Vec3, time: float):
        self.bm.obj.rotation_euler = location
        if self.bm.am.header['flags'] & mdl.ModelFlags.ROTATE:
            self.bm.obj.rotation_euler.z = time * 100. * np.pi / 180
        self.bm.obj.insert_keyframe('rotation_euler', self._get_blender_frame(time))


@dataclass
class BspModelManagedObject(ManagedObject):
    obj: bpy_types.Object

    def add_pose_keyframe(self, pose_num: int, time: float):
        pass

    def add_visible_keyframe(self, visible: bool, time: float):
        pass

    def add_origin_keyframe(self, origin: Vec3, time: float):
        self.obj.location = location
        self.obj.insert_keyframe('location', self._get_blender_frame(time))

    def add_angles_keyframe(self, angles: Vec3, time: float):
        self.obj.rotation_euler = location
        self.obj.insert_keyframe('rotation_euler', self._get_blender_frame(time))


@dataclass
class NullManagedObject(ManagedObject):
    def add_pose_keyframe(self, pose_num: int, time: float):
        pass

    def add_visible_keyframe(self, visible: bool, time: float):
        pass

    def add_origin_keyframe(self, origin: Vec3, time: float):
        pass

    def add_angles_keyframe(self, angles: Vec3, time: float):
        pass


class ObjectManager:
    def __init__(self, fs, config, fps, world_obj_name='demo', load_level=True, relative_time=False):
        assert not relative_time, "Not yet supported"

        self._fs = fs
        self._fps = fps
        self._num_statics = 0
        self._config = config

        self._pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 255

        self._bb: Optional[blendbsp.BlendBsp] = None
        self._objs: Dict[Tuple[int, int], ManagedObject] = {}
        self._model_paths: Optional[List[str]] = None


    def set_model_paths(self, model_paths: List[str]):
        self._model_paths = model_paths

        map_path = self._model_paths[0]
        b = bsp.Bsp(self._fs.open(map_path))
        map_name = re.match(r"maps/([a-zA-Z0-9_]+).bsp", map_path).group(1)
        self._bb = blendbsp.add_bsp(b, self._pal, map_name, self._config)

    @functools.lru_cache(None)
    def _load_alias_model(self, model_path):
        return mdl.AliasModel(self._fs.open(model_path))

    def create_static_object(self, model_num):
        am = mdl.AliasModel(self._fs.open(model_path))
        bm = blendmdl.add_model(am,
                                self._pal,
                                self._path_to_model_name(model_path),
                                f"static{self._num_statics}",
                                [(frame.time, frame.frame)],
                                [fr.skin_idx for fr in ame.path][-1],
                                self._final_time,
                                self._mdls_cfg,
                                static=True,
                                fps=self._fps)
        bm.obj.parent = self._world_obj
        bm.obj.location = frame.origin

        return bm

    def _create_managed_object(self, entity_num, model_num, skin_num):
        model_path = self._model_paths[model_num - 1]
        if model_path.startswith('*'):
            map_model_idx = int(model_path[1:])
            managed_obj = BspModelManagedObject(self._fps, self._bb.model_objs[map_model_idx])
        elif model_path.endswith('.mdl'):
            am = self._load_alias_model(model_path)
            base_name = model_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
            bm = blendmdl.add_model(am,
                                    self._pal,
                                    f'ent{entity_num}_{base_name}',
                                    skin_num,
                                    self._config['models'])
            managed_obj = AliasModelManagedObject(self._fps, bm)
        else:
            managed_obj = NullManagedObject(self._fps)

        return managed_obj

    def update(self, time, prev_entities, entities, prev_updated, updated):
        blender_frame = int(round(self._fps * time))

        # Hide any objects that weren't updated in this frame, or whose model changed.
        for entity_num in prev_updated:
            if (prev_entities[entity_num].model_num != entities[entity_num.model_num]
                    or entity_num not in updated):
                self._objs[entity_num, prev_entities[entity_num].model_num].add_visible_keyframe(
                    False, time
                )

        for entity_num in updated:
            # Create managed objects where we don't already have one for the given entity num / model num.
            ent = entities[entity_num]
            model_num = entities[entity_num].model_num
            key = entity_num, model_num
            if key not in self._model_paths:
                obj = self._create_managed_object(entity_num, model_num, ent.skin)
                self._objs[entity_num, model_num] = obj
            else:
                obj = self._objs[entity_num, model_num]

            # Update position / rotation / pose
            obj.add_origin_keyframe(ent.origin, time)
            obj.add_angles_keyframe(ent.angles, time)
            obj.add_pose_keyframe(ent.frame, time)

        # Unhide objects that were updated this frame, or whose model changed.
        for entity_num in updated:
            if (prev_updated is None or entity_num not in prev_updated or
                (entity_num in prev_entities and prev_entities[entity_num].model_num !=
                    entities[entity_num].model_num)):
            self._objs[entity_num, entities[entity_num].model_num].add_visible_keyframe(
                True, time
            )


@dataclass
class EntityInfo:
    """Information for a single entity slot"""
    model_num: int
    frame: int
    skin: int
    origin: Vec3
    angles: Vec3

    def update(self, msg: proto.ServerMessageUpdate, baseline: "EntityInfo"):
        def none_or(a, b):
            return a if a is not None else b

        angles = _fix_angles(self.angles, _patch_vec(baseline.angles, msg.angles))

        return dataclasses.replace(
            baseline,
            model_num=none_or(msg.model_num, baseline.model_num),
            frame=none_or(msg.frame, baseline.frame),
            skin=none_or(msg.skin, baseline.skin),
            origin=_patch_vec(baseline.origin, msg.origin),
            angles=angles
        )



_DEFAULT_BASELINE = EntityInfo(0, 0, 0, (0, 0, 0), (0, 0, 0))

class DemoAnimator:
    def __init__(self, demo_file, fs, config, fps=30, world_obj_name='demo',
                 load_level=True, relative_time=False):
        assert not relative_time, "Not yet supported"

        self._demo_file = demo_file
        self._fs = fs
        self._config = config
        self._fps = fps
        self._load_level = load_level
        self._world_obj_name = world_obj_name
        self._relative_time = relative_time

        self.static_mats: List[Vec3, Set[bpy.types.Material]] = []

        self._world_obj = bpy.data.objects.new(self._world_obj_name, None)
        bpy.context.scene.collection.objects.link(self._world_obj)

        self._process_messages(proto.read_demo_file(demo_file))

    def _load_blend_bsp(map_path):
        map_name = re.match(r"maps/([a-zA-Z0-9_]+).bsp", map_path).group(1)
        b = bsp.Bsp(self._fs.open(map_path))
        return blendbsp.add_bsp(b, self._pal, self._map_name, self._config)

    def _path_to_model_name(self, mdl_path):
        m = re.match(r"progs/([A-Za-z0-9-_]*)\.mdl", mdl_path)
        if m is None:
            raise Exception("Unexpected model path {mdl_path}")
        return m.group(1)

    def _process_messages(self, msg_iter):
        baseline_entities: Dict[int, EntityInfo] = collections.defaultdict(lambda: _DEFAULT_BASELINE)
        entities: Dict[int, EntityInfo] = {}
        fixed_view_angles: Vec3 = (0, 0, 0)
        objs: Dict[int, bpy_types.Object] = {}
        model_paths: Optional[List[str]] = None
        bb: Optional[BlendBsp] = None
        view_entity: Optional[int] = None
        prev_updated = None

        while True:
            time = None
            update_done = True
            updated = set()
            prev_entities = entities

            while not update_done:
                update_done, view_angles, parsed = next(msg_iter)

                fixed_view_angles = _fix_angles(fixed_view_angles, view_angles, degrees=True)

                if parsed.msg_type == proto.ServerMessageType.TIME:
                    if time is not None:
                        raise Exception("Multiple time messages per update")
                    time = msg.time

                if parsed.msg_type == proto.ServerMessageType.SERVERINFO:
                    if model_paths is not None:
                        raise Exception("ServerInfo already received")

                    model_paths = parsed.models
                    bb = _load_blend_bsp(parsed.models[0])
                    bb.map_obj.parent = self._world_obj

                if parsed.msg_type == proto.ServerMessageType.SETVIEW:
                    view_entity = parsed.viewentity

                if parsed.msg_type == proto.ServerMessageType.SETANGLE:
                    fixed_view_angles = parsed.view_angles

                if parsed.msg_type == proto.ServerMessageType.SPAWNSTATIC:
                    model_path = model_paths[parsed.model_num - 1]
                    bm = self._create_static_object(
                    bm.obj.parent = self._world_obj
                    bm.obj.location = frame.origin
                    if bm.sample_as_light_mats:
                        self.static_mats.append((frame.origin, bm.sample_as_light_mats))

                if parsed.msg_type == proto.ServerMessageType.SPAWNBASELINE:
                    baseline_entities[parsed.entity_num] = EntityInfo(
                        model_num=parsed.model_num,
                        frame=parsed.frame,
                        skin=parsed.skin,
                        origin=parsed.origin,
                        angles=parsed.angles,
                    )

                if parsed.msg_type == proto.ServerMessageType.UPDATE:
                    baseline = baseline_entitities[parsed.entity_num]
                    prev_info = entities.get(parsed.entity_num, baseline)
                    entities[parsed.entity_num] = prev_info.update(parsed, baseline)
                    updated.add(parsed.entity_num)

            prev_update = updated


class AliasModelAnimator:
    def __init__(self, world_obj, fs, pal, mdls_cfg, fps=30):
        self._world_obj = world_obj
        self._model_paths = None
        self._entities = {}
        self._static_entities = []
        self._fs = fs
        self._pal = pal
        self._fps = fps
        self._final_time = None
        self._mdls_cfg = mdls_cfg

        self.entity_objs = {}
        self.static_entity_objs = {}
        self.static_mats = []

    def handle_parsed(self, view_angles, parsed, time, update_done):
        if parsed.msg_type == proto.ServerMessageType.TIME:
            # Set every (baselined) entity to invisible, until an update is seen.
            for entity_num, ame in self._entities.items():
                ame.path.append(ame.path[-1].set_invisible())

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
                    visible=True,
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
                    visible=False,
                )
                self._entities[parsed.entity_num] = AliasModelEntity(
                    model_path=model_path,
                    path=[current_frame],
                )

        if parsed.msg_type == proto.ServerMessageType.UPDATE:
            if parsed.entity_num in self._entities:
                ame = self._entities[parsed.entity_num]
                ame.path[-1] = ame.path[-1].update(time, parsed.origin, parsed.angle, parsed.frame, parsed.skin)

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
                                    self._mdls_cfg,
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

                    if bm.am.header['flags'] & mdl.ModelFlags.ROTATE:
                        bm.obj.rotation_euler.z = fr.time * 100. * np.pi / 180

                    bm.obj.keyframe_insert('rotation_euler', frame=blender_frame)
                    for sub_obj in bm.sub_objs:
                        sub_obj.hide_render = not fr.visible
                        sub_obj.keyframe_insert('hide_render', frame=blender_frame)
                        sub_obj.hide_viewport = not fr.visible
                        sub_obj.keyframe_insert('hide_viewport', frame=blender_frame)

        for idx, ame in enumerate(self._static_entities):
            assert len(ame.path) == 1
            frame, = ame.path
            bm = blendmdl.add_model(alias_models[ame.model_path],
                                    self._pal,
                                    self._path_to_model_name(ame.model_path),
                                    f"static{idx}",
                                    [(frame.time, frame.frame)],
                                    [fr.skin_idx for fr in ame.path][-1],
                                    self._final_time,
                                    self._mdls_cfg,
                                    static=True,
                                    fps=self._fps)
            bm.obj.parent = self._world_obj
            self.static_entity_objs[idx] = bm.obj

            bm.obj.location = frame.origin
            if bm.sample_as_light_mats:
                self.static_mats.append((frame.origin, bm.sample_as_light_mats))


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
        self._view_path = []
        self._fixed_view_angles = (0, 0, 0)

        demo_cam = bpy.data.cameras.new(name="demo_cam")
        demo_cam.lens = 18.0
        self._demo_cam_obj = bpy.data.objects.new(name="demo_cam", object_data=demo_cam)
        bpy.context.scene.collection.objects.link(self._demo_cam_obj)
        self._demo_cam_obj.parent = self._world_obj

    def handle_parsed(self, view_angles, parsed, time, update_done):
        self._fixed_view_angles = _fix_angles(self._fixed_view_angles, view_angles, degrees=True)

        if parsed.msg_type == proto.ServerMessageType.SETANGLE:
            self._fixed_view_angles = parsed.view_angles

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
                self._view_path.append((time, view_origin))
                self._demo_cam_obj.location = view_origin
                self._demo_cam_obj.keyframe_insert('location', frame=frame)

                self._demo_cam_obj.rotation_euler = _quake_to_blender_angles(self._fixed_view_angles)
                self._demo_cam_obj.keyframe_insert('rotation_euler', frame=frame)

                self._bb.set_visible_sample_as_light(view_origin, bounces=2)
                self._bb.insert_sample_as_light_visibility_keyframe(frame)

    def done(self, am_animator: AliasModelAnimator):
        @functools.lru_cache(None)
        def leaf_from_pos(pos):
            return self._bb.bsp.models[0].get_leaf_from_point(pos)

        static_mats = am_animator.static_mats
        for time, view_origin in self._view_path:
            blender_frame = int(round(self._fps * time))
            vis_leaves = self._bb.get_visible_leaves(view_origin, bounces=1)
            for pos, mats in static_mats:
                is_vis = leaf_from_pos(pos) in vis_leaves
                for mat in mats:
                    mat.cycles.sample_as_light = is_vis
                    mat.cycles.keyframe_insert('sample_as_light', frame=blender_frame)


def add_demo(demo_file, fs, config, fps=30, world_obj_name='demo',
             load_level=True, relative_time=False):
    pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 255
    world_obj = bpy.data.objects.new(world_obj_name, None)
    bpy.context.scene.collection.objects.link(world_obj)

    if load_level:
        level_animator = LevelAnimator(world_obj, fs, pal, config, fps)
    am_animator = AliasModelAnimator(world_obj, fs, pal, config['models'], fps)

    time = None
    first_time = None
    for update_done, view_angles, parsed in proto.read_demo_file(demo_file):
        if parsed.msg_type == proto.ServerMessageType.TIME:
            if first_time is None:
                first_time = parsed.time
            if relative_time:
                time = parsed.time - first_time
            else:
                time = parsed.time
        if load_level:
            level_animator.handle_parsed(view_angles, parsed, time, update_done)
        am_animator.handle_parsed(view_angles, parsed, time, update_done)

    am_animator.done()
    if load_level:
        level_animator.done(am_animator)

    return world_obj, am_animator.entity_objs

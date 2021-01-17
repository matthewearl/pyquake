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


@dataclass
class _EntityInfo:
    """Information for a single entity slot"""
    model_num: int
    frame: int
    skin: int
    origin: Vec3
    angles: Vec3

    def update(self, msg: proto.ServerMessageUpdate, baseline: "_EntityInfo"):
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


_DEFAULT_BASELINE = _EntityInfo(0, 0, 0, (0, 0, 0), (0, 0, 0))


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
    def __init__(self, fs, config, fps, world_obj_name='demo', load_level=True):
        assert load_level, "Not yet supported"

        self._fs = fs
        self._fps = fps
        self._config = config

        self._pal = np.fromstring(fs['gfx/palette.lmp'], dtype=np.uint8).reshape(256, 3) / 255

        self._bb: Optional[blendbsp.BlendBsp] = None
        self._objs: Dict[Tuple[int, int], ManagedObject] = {}
        self._model_paths: Optional[List[str]] = None
        self._view_entity_num: Optional[int] = None

        self._static_objects: List[ManagedObject] = []
        self._static_object_leaves: List[bsp.Leaf] = []

        self._world_obj = bpy.data.objects.new(self._world_obj_name, None)
        bpy.context.scene.collection.objects.link(self._world_obj)

        demo_cam = bpy.data.cameras.new(name="demo_cam")
        demo_cam.lens = 18.0
        self._demo_cam_obj = bpy.data.objects.new(name="demo_cam", object_data=demo_cam)
        bpy.context.scene.collection.objects.link(self._demo_cam_obj)
        self._demo_cam_obj.parent = self._world_obj

    @functools.lru_cache(1024)
    def _leaf_from_pos(pos):
        return self._bb.bsp.models[0].get_leaf_from_point(pos)

    def set_model_paths(self, model_paths: List[str]):
        if self._model_paths is not None:
            raise Exception("Model paths already set")
        self._model_paths = model_paths

        map_path = self._model_paths[0]
        b = bsp.Bsp(self._fs.open(map_path))
        map_name = re.match(r"maps/([a-zA-Z0-9_]+).bsp", map_path).group(1)
        self._bb = blendbsp.add_bsp(b, self._pal, map_name, self._config)
        self._bb.map_obj.parent = self._world_obj

    def _path_to_model_name(self, mdl_path):
        m = re.match(r"progs/([A-Za-z0-9-_]*)\.mdl", mdl_path)
        if m is None:
            raise Exception("Unexpected model path {mdl_path}")
        return m.group(1)

    @functools.lru_cache(None)
    def _load_alias_model(self, model_path):
        return mdl.AliasModel(self._fs.open(model_path))

    def set_view_entity(self, entity_num):
        self._view_entity_num = entity_num

    def create_static_object(self, model_num, frame, origin, angles, frame, skin):
        am = mdl.AliasModel(self._fs.open(model_path))
        bm = blendmdl.add_model(am,
                                self._pal,
                                self._path_to_model_name(model_path),
                                f"static{len(self._static_objects)}",
                                skin,
                                self._config['models'],
                                static_pose_num=frame)
        bm.obj.parent = self._world_obj
        bm.obj.location = frame.origin

        self._static_objects.append(bm)
        self._static_object_leaves.append(self._leaf_from_pos(origin))

    def _create_managed_object(self, entity_num, model_num, skin_num):
        model_path = self._model_paths[model_num - 1]
        if model_path.startswith('*'):
            map_model_idx = int(model_path[1:])
            managed_obj = BspModelManagedObject(self._fps, self._bb.model_objs[map_model_idx])
        elif model_path.endswith('.mdl'):
            am = self._load_alias_model(model_path)
            model_name = self._path_to_model_name(model_path),
            bm = blendmdl.add_model(am,
                                    self._pal,
                                    f'ent{entity_num}_{model_name}',
                                    skin_num,
                                    self._config['models'])
            bm.obj.parent = self._world_obj
            managed_obj = AliasModelManagedObject(self._fps, bm)
        else:
            managed_obj = NullManagedObject(self._fps)

        return managed_obj

    def update(self, time, prev_entities, entities, prev_updated, updated, view_angles):
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

        # Set sample_as_light materials.
        view_origin = entities[self._view_entity_num].origin
        self._bb.set_visible_sample_as_light(view_origin, bounces=1)
        vis_leaves = self._bb.get_visible_leaves(view_origin, bounces=1)
        for bm, leaf in zip(self._static_objects, self._static_object_leaves):
            is_vis = leaf in vis_leaves
            for mat in bm.sample_as_light_mats:
                mat.cycles.sample_as_light = is_vis
                mat.cycles.keyframe_insert('sample_as_light', frame=blender_frame)

        # Pose camera
        self._demo_cam_obj.location = view_origin
        self._demo_cam_obj.keyframe_insert('location', frame=blender_frame)
        self._demo_cam_obj.rotation_euler = _quake_to_blender_angles(view_angles)
        self._demo_cam_obj.keyframe_insert('rotation_euler', frame=blender_frame)

    def done(self, final_time: float):
        # Animate static objects
        for bm in self._static_objects:
            bm.done(final_time, self._fps)


def add_demo(demo_file, fs, config, fps=30, world_obj_name='demo',
             load_level=True, relative_time=False):
    assert not relative_time, "Not yet supported"

    baseline_entities: Dict[int, _EntityInfo] = collections.defaultdict(lambda: _DEFAULT_BASELINE)
    entities: Dict[int, _EntityInfo] = {}
    fixed_view_angles: Vec3 = (0, 0, 0)
    prev_updated = None
    demo_done = False
    obj_mgr = ObjectManager(fs, config, fps, world_obj_name, load_level)

    msg_iter = proto.read_demo_file(demo_file))

    while not demo_done:
        time = None
        update_done = False
        updated = set()
        prev_entities = entities

        while not update_done:
            try:
                update_done, view_angles, parsed = next(msg_iter)
            except StopIteration:
                demo_done = True
                break

            fixed_view_angles = _fix_angles(fixed_view_angles, view_angles, degrees=True)

            if parsed.msg_type == proto.ServerMessageType.TIME:
                if time is not None:
                    raise Exception("Multiple time messages per update")
                time = msg.time
                if time >= 5:
                    return

            if parsed.msg_type == proto.ServerMessageType.SERVERINFO:
                obj_mgr.set_model_paths(parsed.models)

            if parsed.msg_type == proto.ServerMessageType.SETVIEW:
                obj_mgr.set_view_entity(parsed.viewentity)

            if parsed.msg_type == proto.ServerMessageType.SETANGLE:
                fixed_view_angles = parsed.view_angles

            if parsed.msg_type == proto.ServerMessageType.SPAWNSTATIC:
                obj_mgr.create_static_object(
                    parsed.model_num, parsed.frame, parsed.origin, parsed.angles,
                    parsed.frame, parsed.skin
                )

            if parsed.msg_type == proto.ServerMessageType.SPAWNBASELINE:
                baseline_entities[parsed.entity_num] = _EntityInfo(
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

        obj_mgr.update(time, prev_entities, entities, prev_updated, updated, fixed_view_angles)
        prev_updated = updated

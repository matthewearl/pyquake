import math
import re

import bpy

from . import proto


def _patch_vec(old_vec, update):
    return tuple(v if u is None else u for v, u in zip(old_vec, update))


def animate_level(demo_file, fps=30):
    entity_origins = {}
    entity_to_model = {}

    demo_cam = bpy.data.cameras.new(name="demo_cam")
    demo_cam.lens = 18.0
    demo_cam_obj = bpy.data.objects.new(name="demo_cam", object_data=demo_cam)
    bpy.context.scene.collection.objects.link(demo_cam_obj)

    for view_angles, parsed in proto.read_demo_file(demo_file):
        if parsed.msg_type == proto.ServerMessageType.SERVERINFO:
            map_name = re.match(r"maps/([a-zA-Z0-9_]+).bsp", parsed.models[0]).group(1)
            num_models = 1 + sum((m[0] == '*') for m in parsed.models)
            demo_cam_obj.parent = bpy.data.objects[map_name]

        if parsed.msg_type == proto.ServerMessageType.SETVIEW:
            view_entity = parsed.viewentity

        if parsed.msg_type == proto.ServerMessageType.SPAWNBASELINE:
            if parsed.model_num < num_models:
                entity_origins[parsed.entity_num] = parsed.origin
                entity_to_model[parsed.entity_num] = parsed.model_num

            if parsed.entity_num == view_entity:
                view_origin = parsed.origin

        if parsed.msg_type == proto.ServerMessageType.TIME:
            time = parsed.time

        if parsed.msg_type == proto.ServerMessageType.UPDATE:
            if parsed.entity_num in entity_origins:
                model_num = entity_to_model[parsed.entity_num]
                entity_origins[parsed.entity_num] = _patch_vec(entity_origins[parsed.entity_num], parsed.origin)

                model_obj = bpy.data.objects[f"{map_name}_{model_num - 1}"]
                model_obj.location = entity_origins[parsed.entity_num]
                model_obj.keyframe_insert('location', frame=int(fps * time))

            if parsed.entity_num == view_entity:
                view_origin = _patch_vec(view_origin, parsed.origin)
                demo_cam_obj.location = view_origin
                demo_cam_obj.keyframe_insert('location', frame=int(fps * time))
                demo_cam_obj.rotation_euler = (math.pi * (90. - view_angles[0]) / 180,
                                               math.pi * view_angles[2] / 180,
                                               math.pi * (view_angles[1] - 90.) / 180)
                demo_cam_obj.keyframe_insert('rotation_euler', frame=int(fps * time))

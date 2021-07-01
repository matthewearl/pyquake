import collections
from typing import Iterable

import bpy

from .mapping import prt


def leaf_objects_from_portals(portals: Iterable[prt.Portal]):
    leaf_to_portals = collections.defaultdict(list)
    for portal in portals:
        for leaf_idx in portal.leaves:
            leaf_to_portals[leaf_idx].append(portal)

    parent_obj = bpy.data.objects.new('leaf_parent', None)
    bpy.context.scene.collection.objects.link(parent_obj)

    for leaf_idx, portals in leaf_to_portals.items():
        verts = []
        faces = []
        for p in portals:
            faces.append([len(verts) + i for i in range(len(p.winding))])
            verts.extend(p.winding)

        obj_name = f'leaf{leaf_idx}'
        mesh = bpy.data.meshes.new(obj_name)
        mesh.from_pydata(verts, [], faces)

        obj = bpy.data.objects.new(obj_name, mesh)
        bpy.context.scene.collection.objects.link(obj)

        obj.parent = parent_obj

    return parent_obj

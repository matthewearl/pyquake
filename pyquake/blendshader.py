__all__ = (
    'im_from_file',
    'setup_diffuse_material',
)


import io

import bpy
import numpy as np
import PIL.Image
import simplejpeg


def im_from_file(f, name):
    b = f.read()

    if simplejpeg.is_jpeg(b):
        # For some reason PIL doesn't like reading JPEGs from within Blender.  Use `simplejpeg` in this case.
        a = simplejpeg.decode_jpeg(b) / 255.
    else:
        bf = io.BytesIO(b)
        pil_im = PIL.Image.open(bf)
        a = (np.frombuffer(pil_im.tobytes(), dtype=np.uint8)
             .reshape((pil_im.size[1], pil_im.size[0], -1))) / 255.


    if a.shape[-1] == 3:
        a = np.concatenate([
            a,
            np.ones((a.shape[0], a.shape[1]))[:, :, None]
        ], axis=2)
    elif a.shape[-1] != 4:
        raise Exception('Only RGB and RGBA images are supported')

    blend_im = bpy.data.images.new(name, alpha=True, width=a.shape[1], height=a.shape[0])
    blend_im.pixels = a.ravel()
    blend_im.pack()

    return blend_im


def _new_mat(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    while nodes:
        nodes.remove(nodes[0])
    while links:
        links.remove(links[0])

    return mat, nodes, links


def setup_diffuse_material(im: bpy.types.Image, mat_name: str):
    mat, nodes, links = _new_mat(mat_name)

    output_node = nodes.new('ShaderNodeOutputMaterial')

    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    links.new(output_node.inputs['Surface'], diffuse_node.outputs['BSDF'])

    texture_node = nodes.new('ShaderNodeTexImage')
    texture_node.image = im
    texture_node.interpolation = 'Closest'
    links.new(diffuse_node.inputs['Color'], texture_node.outputs['Color'])

    return mat


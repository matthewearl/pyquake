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


__all__ = (
    'array_ims_from_indices',
    'im_from_array',
    'new_mat',
    'setup_diffuse_material',
    'setup_fullbright_material',
    'setup_transparent_fullbright_material',
)


import bpy
import numpy as np


def im_from_array(name, array_im):
    im = bpy.data.images.new(name, width=array_im.shape[1], height=array_im.shape[0])
    im.pixels = np.ravel(array_im)
    im.pack()
    return im


def array_ims_from_indices(pal, im_indices, gamma=1.0, light_tint=(1, 1, 1, 1), force_fullbright=False):
    if force_fullbright:
        fullbright_array = np.full_like(im_indices, True)
    else:
        fullbright_array = (im_indices >= 224)

    array_im = pal[im_indices]
    array_im = array_im ** gamma

    if np.any(fullbright_array):
        fullbright_array_im = array_im * fullbright_array[..., None]
        fullbright_array_im *= light_tint
        fullbright_array_im = np.clip(fullbright_array_im, 0., 1.)
    else:
        fullbright_array_im = None

    return array_im, fullbright_array_im, fullbright_array


def new_mat(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    while nodes:
        nodes.remove(nodes[0])
    while links:
        links.remove(links[0])

    return mat, nodes, links


def setup_diffuse_material(nodes, links, im):
    texture_node = nodes.new('ShaderNodeTexImage')
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes.new('ShaderNodeOutputMaterial')

    texture_node.image = im
    texture_node.interpolation = 'Closest'
    links.new(diffuse_node.inputs['Color'], texture_node.outputs['Color'])
    links.new(output_node.inputs['Surface'], diffuse_node.outputs['BSDF'])


def setup_fullbright_material(nodes, links, im, glow_im, strength):
    texture_node = nodes.new('ShaderNodeTexImage')
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    add_node = nodes.new('ShaderNodeAddShader')
    glow_texture_node = nodes.new('ShaderNodeTexImage')
    emission_node = nodes.new('ShaderNodeEmission')

    texture_node.image = im
    texture_node.interpolation = 'Closest'
    glow_texture_node.image = glow_im
    glow_texture_node.interpolation = 'Closest'

    emission_node.inputs['Strength'].default_value = strength

    links.new(diffuse_node.inputs['Color'], texture_node.outputs['Color'])
    links.new(emission_node.inputs['Color'], glow_texture_node.outputs['Color'])
    links.new(add_node.inputs[0], diffuse_node.outputs['BSDF'])
    links.new(add_node.inputs[1], emission_node.outputs['Emission'])
    links.new(output_node.inputs['Surface'], add_node.outputs['Shader'])


def setup_transparent_fullbright_material(nodes, links, im, glow_im, strength):
    texture_node = nodes.new('ShaderNodeTexImage')
    emission_node = nodes.new('ShaderNodeEmission')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    glow_texture_node = nodes.new('ShaderNodeTexImage')
    mix_node = nodes.new('ShaderNodeMixShader')
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')

    texture_node.image = im
    texture_node.interpolation = 'Closest'
    glow_texture_node.image = glow_im
    glow_texture_node.interpolation = 'Closest'

    emission_node.inputs['Strength'].default_value = strength

    links.new(emission_node.inputs['Color'], texture_node.outputs['Color'])
    links.new(mix_node.inputs[0], glow_texture_node.outputs['Color'])
    links.new(mix_node.inputs[1], transparent_node.outputs['BSDF'])
    links.new(mix_node.inputs[2], emission_node.outputs['Emission'])
    links.new(output_node.inputs['Surface'], mix_node.outputs['Shader'])


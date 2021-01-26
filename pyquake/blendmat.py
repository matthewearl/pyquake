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

from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple

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


@dataclass(eq=False)
class BlendMatImagePair:
    im: bpy.types.Image
    fullbright_im: Optional[bpy.types.Image]


@dataclass(eq=False)
class BlendMatImages:
    frames: List[BlendMatImagePair]
    alt_frames: List[BlendMatImagePair]

    @classmethod
    def from_single_diffuse(cls, im: bpy.types.Image):
        return cls(
            frames=[BlendMatImagePair(im, None)],
            alt_frames=[]
        )

    @classmethod
    def from_single_pair(cls, im: bpy.types.Image, fullbright_im: bpy.types.Image):
        return cls(
            frames=[BlendMatImagePair(im, fullbright_im)],
            alt_frames=[]
        )

    @property
    def any_fullbright(self):
        return any(p.fullbright_im is not None for l in [self.frames, self.alt_frames] for p in l)


@dataclass(eq=False)
class BlendMat:
    mat: bpy.types.Material
    _frame_input: Optional[bpy.types.NodeSocketFloatFactor]
    _time_input: Optional[bpy.types.NodeSocketFloatFactor]

    def add_time_keyframe(self, time: float, blender_frame: int):
        self._time_input.default_value = time
        self._time_input.keyframe_insert('default_value', frame=blender_frame)

    def add_frame_keyframe(self, frame: int, blender_frame: int):
        self._frame_input.default_value = frame
        self._frame_input.keyframe_insert('default_value', frame=blender_frame)

    def add_sample_as_light_keyframe(self, sample_as_light: bool, blender_frame: int):
        self.mat.cycles.sample_as_light = sample_as_light
        self.mat.cycles.keyframe_insert('sample_as_light', frame=blender_frame)


def _setup_image_nodes(ims: Iterable[Optional[bpy.types.Image]], nodes, links) -> \
        Tuple[bpy.types.NodeSocketColor, List[bpy.types.NodeSocketFloatFactor]]:
    texture_nodes = []
    for im in ims:
        if im is not None:
            texture_node = nodes.new('ShaderNodeTexImage')
            texture_node.image = im
            texture_node.interpolation = 'Closest'
            texture_nodes.append(texture_node)
        else:
            texture_nodes.append(None)

    if len(texture_nodes) == 1:
        if texture_nodes[0] is None:
            out = None, []
        else:
            out = texture_nodes[0].outputs['Color'], []
    elif len(texture_nodes) > 1:
        if texture_nodes[0] is None:
            prev_output = None
        else:
            prev_output = texture_nodes[0].outputs['Color']

        mul_node = nodes.new('ShaderNodeMath')
        mul_node.operation = 'MULTIPLY'
        mul_node.inputs[1].default_value = 10
        time_input = mul_node.inputs[0]

        mod_node = nodes.new('ShaderNodeMath')
        mod_node.operation = 'MODULO'
        links.new(mod_node.inputs[0], mul_node.outputs['Value'])
        mod_node.inputs[1].default_value = len(texture_nodes)

        floor_node = nodes.new('ShaderNodeMath')
        floor_node.operation = 'FLOOR'
        links.new(floor_node.inputs[0], mod_node.outputs['Value'])
        frame_output = floor_node.outputs['Value']

        for frame_num, texture_node in enumerate(texture_nodes[1:], 1):
            sub_node = nodes.new('ShaderNodeMath')
            sub_node.operation = 'SUBTRACT'
            sub_node.inputs[0].default_value = frame_num
            links.new(sub_node.inputs[1], frame_output)

            mix_node = nodes.new('ShaderNodeMixRGB')
            if texture_node is not None:
                links.new(mix_node.inputs['Color1'], texture_node.outputs['Color'])
            else:
                mix_node.inputs['Color1'].default_value = (0, 0, 0, 1)
            if prev_output is None:
                mix_nodes.inputs['Color2'].default_value = (0, 0, 0, 1)
            else:
                links.new(mix_node.inputs['Color2'], prev_output)

            links.new(mix_node.inputs['Fac'], sub_node.outputs['Value'])

            prev_output = mix_node.outputs['Color']

        out = prev_output, [time_input]
    else:
        raise ValueError('No images passed')

    return out


def _setup_alt_image_nodes(ims: BlendMatImages, nodes, links, fullbright: False) -> \
        Tuple[bpy.types.NodeSocketColor,
              List[bpy.types.NodeSocketFloatFactor],
              List[bpy.types.NodeSocketFloatFactor]]:
    main_output, main_time_inputs = _setup_image_nodes(
        ((im_pair.fullbright_im if fullbright else im_pair.im)
            for im_pair in ims.frames),
        nodes, links
    )

    if not ims.alt_frames:
        out = main_output, main_time_inputs, []
    else:
        alt_output, alt_time_inputs = _setup_image_nodes(
            ((im_pair.fullbright_im if fullbright else im_pair.im)
                for im_pair in ims.alt_frames),
            nodes, links
        )

        mix_node = nodes.new('ShaderNodeMixRGB')
        if main_output is not None:
            links.new(mix_node.inputs['Color1'], main_output)
        else:
            mix_node.inputs['Color1'].default_value = (0, 0, 0, 1)

        if alt_output is not None:
            links.new(mix_node.inputs['Color2'], alt_output)
        else:
            mix_node.inputs['Color2'].default_value = (0, 0, 0, 1)

        out = (mix_node.outputs['Color'],
               main_time_inputs + alt_time_inputs,
               [mix_node.inputs['Fac']])

    return out


def _create_value_node(inputs, nodes, links):
    value_node = nodes.new('ShaderNodeValue')
    for inp in inputs:
        links.new(inp, value_node.outputs['Value'])
    return value_node.outputs['Value']


def setup_diffuse_material(ims: BlendMatImages, mat_name: str):
    mat, nodes, links = _new_mat(mat_name)

    im_output, time_inputs, frame_inputs = _setup_alt_image_nodes(ims, nodes, links, fullbright=False)

    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes.new('ShaderNodeOutputMaterial')

    links.new(diffuse_node.inputs['Color'], im_output)
    links.new(output_node.inputs['Surface'], diffuse_node.outputs['BSDF'])

    return BlendMat(
        mat,
        _create_value_node(time_inputs, nodes, links) if time_inputs else None,
        _create_value_node(frame_inputs, nodes, links) if frame_inputs else None
    )


def setup_fullbright_material(ims: BlendMatImages, mat_name: str, strength: float):
    mat, nodes, links = _new_mat(mat_name)

    diffuse_im_output, diffuse_time_inputs, diffuse_frame_inputs = _setup_alt_image_nodes(
        ims, nodes, links, fullbright=False
    )
    fullbright_im_output, fullbright_time_inputs, fullbright_frame_inputs = _setup_alt_image_nodes(
            ims, nodes, links, fullbright=True
    )
    time_inputs = diffuse_time_inputs + fullbright_time_inputs
    frame_inputs = diffuse_frame_inputs + fullbright_frame_inputs

    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    add_node = nodes.new('ShaderNodeAddShader')
    emission_node = nodes.new('ShaderNodeEmission')

    emission_node.inputs['Strength'].default_value = strength

    links.new(diffuse_node.inputs['Color'], diffuse_im_output)
    links.new(emission_node.inputs['Color'], fullbright_im_output)
    links.new(add_node.inputs[0], diffuse_node.outputs['BSDF'])
    links.new(add_node.inputs[1], emission_node.outputs['Emission'])
    links.new(output_node.inputs['Surface'], add_node.outputs['Shader'])

    return BlendMat(
        mat,
        _create_value_node(time_inputs, nodes, links) if time_inputs else None,
        _create_value_node(frame_input, nodes, links) if frame_inputs else None
    )


def setup_transparent_fullbright_material(ims: BlendMatImages, mat_name: str, strength: float):
    mat, nodes, links = _new_mat(mat_name)

    diffuse_im_output, diffuse_time_inputs, diffuse_frame_inputs = _setup_alt_image_nodes(
        ims, nodes, links, fullbright=False
    )
    fullbright_im_output, fullbright_time_inputs, fullbright_frame_inputs = _setup_alt_image_nodes(
        ims, nodes, links, fullbright=True
    )
    time_inputs = diffuse_time_inputs + fullbright_time_inputs
    frame_inputs = diffuse_frame_inputs + fullbright_frame_inputs

    emission_node = nodes.new('ShaderNodeEmission')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    mix_node = nodes.new('ShaderNodeMixShader')
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')

    emission_node.inputs['Strength'].default_value = strength

    links.new(emission_node.inputs['Color'], diffuse_im_output)
    links.new(mix_node.inputs[0], fullbright_im_output)
    links.new(mix_node.inputs[1], transparent_node.outputs['BSDF'])
    links.new(mix_node.inputs[2], emission_node.outputs['Emission'])
    links.new(output_node.inputs['Surface'], mix_node.outputs['Shader'])

    return BlendMat(
        mat,
        _create_value_node(time_inputs, nodes, links) if time_inputs else None,
        _create_value_node(frame_inputs, nodes, links) if frame_inputs else None
    )

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
    'setup_diffuse_material',
    'setup_flat_material',
    'setup_fullbright_material',
    'setup_lightmap_material',
    'setup_light_style_node_groups',
    'setup_sky_material',
    'setup_transparent_fullbright_material',
)


from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict

import bpy
import numpy as np


_MAX_LIGHT_STYLES = 64


def im_from_array(name, array_im):
    im = bpy.data.images.new(name, alpha=True, width=array_im.shape[1], height=array_im.shape[0])
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


def setup_light_style_node_groups():
    groups = {}
    for style_idx in range(_MAX_LIGHT_STYLES):
        group = bpy.data.node_groups.new(f'style_{style_idx}', 'ShaderNodeTree')
        group.outputs.new('NodeSocketFloat', 'Value')
        input_node = group.nodes.new('NodeGroupInput')
        output_node = group.nodes.new('NodeGroupOutput')
        value_node = group.nodes.new('ShaderNodeValue')
        group.links.new(output_node.inputs['Value'], value_node.outputs['Value'])

        groups[style_idx] = group

    return groups


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

    @property
    def width(self):
        return self.frames[0].im.size[0]

    @property
    def height(self):
        return self.frames[0].im.size[1]

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

    @property
    def is_animated(self):
        return len(self.frames) > 1 or len(self.alt_frames) > 1

    @property
    def is_posable(self):
        return len(self.alt_frames) > 0


@dataclass(eq=False)
class BlendMat:
    mat: bpy.types.Material

    def add_time_keyframe(self, time: float, blender_frame: int):
        time_input = self.mat.node_tree.nodes['time'].outputs['Value']
        time_input.default_value = time
        time_input.keyframe_insert('default_value', frame=blender_frame)

        fcurve = self.mat.node_tree.animation_data.action.fcurves.find(
            'nodes["time"].outputs[0].default_value'
        )
        fcurve.keyframe_points[-1].interpolation = 'LINEAR'

    def add_frame_keyframe(self, frame: int, blender_frame: int):
        frame_input = self.mat.node_tree.nodes['frame'].outputs['Value']
        frame_input.default_value = frame
        frame_input.keyframe_insert('default_value', frame=blender_frame)

    def add_sample_as_light_keyframe(self, sample_as_light: bool, blender_frame: int):
        self.mat.cycles.sample_as_light = sample_as_light
        self.mat.cycles.keyframe_insert('sample_as_light', frame=blender_frame)

    @property
    def is_animated(self):
        return 'time' in self.mat.node_tree.nodes

    @property
    def is_posable(self):
        return 'frame' in self.mat.node_tree.nodes


def _setup_warp_uv_single(nodes, links, dim1_output, dim2_output, size1, size2):
    mul_node = nodes.new('ShaderNodeMath')
    mul_node.operation = 'MULTIPLY'
    mul_node.inputs[1].default_value = size2 * 2 * np.pi / 128
    links.new(mul_node.inputs[0], dim2_output)

    add_node = nodes.new('ShaderNodeMath')
    add_node.operation = 'ADD'
    links.new(add_node.inputs[0], mul_node.outputs['Value'])

    sine_node = nodes.new('ShaderNodeMath')
    sine_node.operation = 'SINE'
    links.new(sine_node.inputs[0], add_node.outputs['Value'])

    mul2_node = nodes.new('ShaderNodeMath')
    mul2_node.operation = 'MULTIPLY'
    mul2_node.inputs[1].default_value = 8 / size1
    links.new(mul2_node.inputs[0], sine_node.outputs['Value'])

    add2_node = nodes.new('ShaderNodeMath')
    add2_node.operation = 'ADD'
    links.new(add2_node.inputs[0], mul2_node.outputs['Value'])
    links.new(add2_node.inputs[1], dim1_output)

    return [add_node.inputs[1]], add2_node.outputs['Value']


def _setup_warp_uv(nodes, links, width, height):
    uv_node = nodes.new('ShaderNodeUVMap')

    sep_node = nodes.new('ShaderNodeSeparateXYZ')
    links.new(sep_node.inputs['Vector'], uv_node.outputs['UV'])

    u_time_inputs, u_output = _setup_warp_uv_single(
        nodes, links,
        sep_node.outputs['X'], sep_node.outputs['Y'],
        width, height
    )

    v_time_inputs, v_output = _setup_warp_uv_single(
        nodes, links,
        sep_node.outputs['Y'], sep_node.outputs['X'],
        height, width
    )

    combine_node = nodes.new('ShaderNodeCombineXYZ')
    links.new(combine_node.inputs['X'], u_output)
    links.new(combine_node.inputs['Y'], v_output)

    return u_time_inputs + v_time_inputs, combine_node.outputs['Vector']


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
            time_inputs = []
            uv_inputs = []
            colour_output = None
        else:
            time_inputs = []
            uv_inputs = [texture_nodes[0].inputs['Vector']]
            colour_output = texture_nodes[0].outputs['Color']
    elif len(texture_nodes) > 1:
        if texture_nodes[0] is None:
            prev_output = None
        else:
            prev_output = texture_nodes[0].outputs['Color']

        mul_node = nodes.new('ShaderNodeMath')
        mul_node.operation = 'MULTIPLY'
        # empirically measured frame time to be 0.24s
        # TODO: Find out why this is the case in the Quake source code.
        mul_node.inputs[1].default_value = 1. / 0.24
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

        time_inputs = [time_input]
        uv_inputs = [tn.inputs['Vector'] for tn in texture_nodes if tn is not None]
        colour_output = prev_output
    else:
        raise ValueError('No images passed')

    return time_inputs, uv_inputs, colour_output


def _reduce(node_type: str, operation: str, it: Iterable[bpy.types.NodeSocket], nodes, links):
    iter_ = iter(it)
    accum = next(iter_)

    for ns in iter_:
        op_node = nodes.new(node_type)
        op_node.operation = operation
        links.new(op_node.inputs[0], accum)
        links.new(op_node.inputs[1], ns)
        accum = op_node.outputs[0]

    return accum


def _setup_alt_image_nodes(ims: BlendMatImages, nodes, links, warp: bool, fullbright: bool) -> \
        Tuple[bpy.types.NodeSocketColor,
              List[bpy.types.NodeSocketFloatFactor],
              List[bpy.types.NodeSocketFloatFactor]]:
    main_time_inputs, main_uv_inputs, main_output = _setup_image_nodes(
        ((im_pair.fullbright_im if fullbright else im_pair.im)
            for im_pair in ims.frames),
        nodes, links
    )

    time_inputs = main_time_inputs
    uv_inputs = main_uv_inputs
    frame_inputs = []
    if not ims.alt_frames:
        output = main_output
    else:
        alt_time_inputs, alt_uv_inputs, alt_output = _setup_image_nodes(
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

        output = mix_node.outputs['Color']
        time_inputs += alt_time_inputs
        uv_inputs += alt_uv_inputs
        frame_inputs += [mix_node.inputs['Fac']]

    if warp:
        warp_time_inputs, uv_output = _setup_warp_uv(nodes, links, ims.width, ims.height)
        for uv_input in uv_inputs:
            links.new(uv_input, uv_output)
        time_inputs += warp_time_inputs

    return output, time_inputs, frame_inputs


def _create_value_node(inputs, nodes, links, name):
    value_node = nodes.new('ShaderNodeValue')
    value_node.name = name
    for inp in inputs:
        links.new(inp, value_node.outputs['Value'])


def _create_inputs(frame_inputs, time_inputs, nodes, links):
    if len(frame_inputs) > 0:
        _create_value_node(frame_inputs, nodes, links, 'frame')
    if len(time_inputs) > 0:
        _create_value_node(time_inputs, nodes, links, 'time')


def setup_sky_material(ims: BlendMatImages, mat_name):
    image = ims.frames[0].im

    mat, nodes, links = _new_mat(mat_name)

    output_node = nodes.new('ShaderNodeOutputMaterial')

    mix_node = nodes.new('ShaderNodeMixShader')
    links.new(output_node.inputs['Surface'], mix_node.outputs['Shader'])

    light_path_node = nodes.new('ShaderNodeLightPath')
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')
    emission_node = nodes.new('ShaderNodeEmission')
    emission_node.inputs['Strength'].default_value = 0.25
    links.new(mix_node.inputs['Fac'], light_path_node.outputs['Is Camera Ray'])
    links.new(mix_node.inputs[1], transparent_node.outputs['BSDF'])
    links.new(mix_node.inputs[2], emission_node.outputs['Emission'])

    mix_rgb_node = nodes.new('ShaderNodeMixRGB')
    links.new(emission_node.inputs['Color'], mix_rgb_node.outputs['Color'])

    front_texture_node = nodes.new('ShaderNodeTexImage')
    front_texture_node.image = image
    back_texture_node = nodes.new('ShaderNodeTexImage')
    back_texture_node.image = image
    links.new(mix_rgb_node.inputs['Color1'], back_texture_node.outputs['Color'])
    links.new(mix_rgb_node.inputs['Color2'], front_texture_node.outputs['Color'])
    links.new(mix_rgb_node.inputs['Fac'], front_texture_node.outputs['Alpha'])

    wrap_back_node = nodes.new('ShaderNodeVectorMath')
    wrap_front_node = nodes.new('ShaderNodeVectorMath')
    wrap_back_node.operation = 'WRAP'
    wrap_back_node.inputs[1].default_value = (0.5, 0, 0)
    wrap_back_node.inputs[2].default_value = (1, 1, 1)
    wrap_front_node.operation = 'WRAP'
    wrap_front_node.inputs[1].default_value = (0, 0, 0)
    wrap_front_node.inputs[2].default_value = (0.5, 1, 1)
    links.new(back_texture_node.inputs['Vector'], wrap_back_node.outputs[0])
    links.new(front_texture_node.inputs['Vector'], wrap_front_node.outputs[0])

    add_back_node = nodes.new('ShaderNodeVectorMath')
    add_front_node = nodes.new('ShaderNodeVectorMath')
    add_back_node.operation = 'ADD'
    add_front_node.operation = 'ADD'
    links.new(wrap_back_node.inputs[0], add_back_node.outputs[0])
    links.new(wrap_front_node.inputs[0], add_front_node.outputs[0])

    vec_mul2_node = nodes.new('ShaderNodeVectorMath')
    back_vel_node = nodes.new('ShaderNodeVectorMath')
    front_vel_node = nodes.new('ShaderNodeVectorMath')
    vec_mul2_node.operation = 'MULTIPLY'
    back_vel_node.operation = 'MULTIPLY'
    front_vel_node.operation = 'MULTIPLY'
    vec_mul2_node.inputs[1].default_value = (3, 3, 3)
    back_vel_node.inputs[1].default_value = (.125, .125, .125)
    front_vel_node.inputs[1].default_value = (.25, .25, .25)
    links.new(add_back_node.inputs[0], vec_mul2_node.outputs['Vector'])
    links.new(add_front_node.inputs[0], vec_mul2_node.outputs['Vector'])
    links.new(add_back_node.inputs[1], back_vel_node.outputs['Vector'])
    links.new(add_front_node.inputs[1], front_vel_node.outputs['Vector'])

    _create_value_node([back_vel_node.inputs[0], front_vel_node.inputs[0]],
                       nodes, links, 'time')

    normalize_node = nodes.new('ShaderNodeVectorMath')
    normalize_node.operation = 'NORMALIZE'
    links.new(vec_mul2_node.inputs[0], normalize_node.outputs['Vector'])

    vec_mul_node = nodes.new('ShaderNodeVectorMath')
    vec_mul_node.operation = 'MULTIPLY'
    vec_mul_node.inputs[1].default_value = (-1, -1, -3)
    links.new(normalize_node.inputs['Vector'], vec_mul_node.outputs['Vector'])

    geometry_node = nodes.new('ShaderNodeNewGeometry')
    links.new(vec_mul_node.inputs[0], geometry_node.outputs['Incoming'])

    return BlendMat(mat)


def setup_diffuse_material(ims: BlendMatImages, mat_name: str, warp: bool):
    mat, nodes, links = _new_mat(mat_name)

    im_output, time_inputs, frame_inputs = _setup_alt_image_nodes(ims, nodes, links, warp=warp, fullbright=False)
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes.new('ShaderNodeOutputMaterial')

    links.new(diffuse_node.inputs['Color'], im_output)
    links.new(output_node.inputs['Surface'], diffuse_node.outputs['BSDF'])

    _create_inputs(frame_inputs, time_inputs, nodes, links)

    return BlendMat(mat)


def setup_fullbright_material(ims: BlendMatImages, mat_name: str,
                              strength: float, cam_strength: float, warp: bool):
    mat, nodes, links = _new_mat(mat_name)

    diffuse_im_output, diffuse_time_inputs, diffuse_frame_inputs = _setup_alt_image_nodes(
        ims, nodes, links, warp=warp, fullbright=False
    )
    fullbright_im_output, fullbright_time_inputs, fullbright_frame_inputs = _setup_alt_image_nodes(
            ims, nodes, links, warp=warp, fullbright=True
    )
    time_inputs = diffuse_time_inputs + fullbright_time_inputs
    frame_inputs = diffuse_frame_inputs + fullbright_frame_inputs

    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    add_node = nodes.new('ShaderNodeAddShader')
    emission_node = nodes.new('ShaderNodeEmission')

    if strength == cam_strength:
        emission_node.inputs['Strength'].default_value = strength
    else:
        map_range_node = nodes.new('ShaderNodeMapRange')
        map_range_node.inputs['From Min'].default_value = 0
        map_range_node.inputs['From Max'].default_value = 1
        map_range_node.inputs['To Min'].default_value = strength
        map_range_node.inputs['To Max'].default_value = cam_strength
        links.new(emission_node.inputs['Strength'], map_range_node.outputs['Result'])

        light_path_node = nodes.new('ShaderNodeLightPath')
        links.new(map_range_node.inputs['Value'], light_path_node.outputs['Is Camera Ray'])

    links.new(diffuse_node.inputs['Color'], diffuse_im_output)
    links.new(emission_node.inputs['Color'], fullbright_im_output)
    links.new(add_node.inputs[0], diffuse_node.outputs['BSDF'])
    links.new(add_node.inputs[1], emission_node.outputs['Emission'])
    links.new(output_node.inputs['Surface'], add_node.outputs['Shader'])

    _create_inputs(frame_inputs, time_inputs, nodes, links)

    return BlendMat(mat)


def setup_transparent_fullbright_material(ims: BlendMatImages, mat_name: str, strength: float, warp: bool):
    mat, nodes, links = _new_mat(mat_name)

    diffuse_im_output, diffuse_time_inputs, diffuse_frame_inputs = _setup_alt_image_nodes(
        ims, nodes, links, warp=warp, fullbright=False
    )
    fullbright_im_output, fullbright_time_inputs, fullbright_frame_inputs = _setup_alt_image_nodes(
        ims, nodes, links, warp=warp, fullbright=True
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

    _create_inputs(frame_inputs, time_inputs, nodes, links)

    return BlendMat(mat)


def setup_explosion_particle_material(mat_name):
    mat, nodes, links = _new_mat(mat_name)

    output_node = nodes.new('ShaderNodeOutputMaterial')

    mix_node = nodes.new('ShaderNodeMixShader')
    links.new(output_node.inputs['Surface'], mix_node.outputs['Shader'])

    emission_node = nodes.new('ShaderNodeEmission')
    emission_node.inputs['Strength'].default_value = 10
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')
    links.new(mix_node.inputs[1], emission_node.outputs['Emission'])
    links.new(mix_node.inputs[2], transparent_node.outputs['BSDF'])

    blackbody_node = nodes.new('ShaderNodeBlackbody')
    links.new(emission_node.inputs['Color'], blackbody_node.outputs['Color'])

    map_range_node = nodes.new('ShaderNodeMapRange')
    map_range_node.inputs['From Min'].default_value = 0
    map_range_node.inputs['From Max'].default_value = 1
    map_range_node.inputs['To Min'].default_value = 4000
    map_range_node.inputs['To Max'].default_value = 500
    links.new(blackbody_node.inputs['Temperature'], map_range_node.outputs['Result'])

    div_node = nodes.new('ShaderNodeMath')
    div_node.operation = 'DIVIDE'
    links.new(map_range_node.inputs['Value'], div_node.outputs['Value'])
    links.new(mix_node.inputs[0], div_node.outputs['Value'])

    particle_info_node = nodes.new('ShaderNodeParticleInfo')
    links.new(div_node.inputs[0], particle_info_node.outputs['Age'])
    links.new(div_node.inputs[1], particle_info_node.outputs['Lifetime'])

    return BlendMat(mat)


def setup_teleport_particle_material(mat_name):
    mat, nodes, links = _new_mat(mat_name)

    output_node = nodes.new('ShaderNodeOutputMaterial')

    mix_node = nodes.new('ShaderNodeMixShader')
    links.new(output_node.inputs['Surface'], mix_node.outputs['Shader'])

    emission_node = nodes.new('ShaderNodeEmission')
    emission_node.inputs['Strength'].default_value = .5
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')
    links.new(mix_node.inputs[1], emission_node.outputs['Emission'])
    links.new(mix_node.inputs[2], transparent_node.outputs['BSDF'])

    color_ramp_node = nodes.new('ShaderNodeValToRGB')
    div_node = nodes.new('ShaderNodeMath')
    div_node.operation = 'DIVIDE'
    links.new(emission_node.inputs['Color'], color_ramp_node.outputs['Color'])
    links.new(mix_node.inputs[0], div_node.outputs['Value'])

    particle_info_node = nodes.new('ShaderNodeParticleInfo')
    links.new(div_node.inputs[0], particle_info_node.outputs['Age'])
    links.new(div_node.inputs[1], particle_info_node.outputs['Lifetime'])
    links.new(color_ramp_node.inputs['Fac'], particle_info_node.outputs['Random'])

    return BlendMat(mat)


def setup_lightmap_material(mat_name: str, ims: BlendMatImages,
                            lightmap_ims: List[bpy.types.Image], lightmap_uv_layer_name: str,
                            warp: bool,
                            lightmap_styles: Tuple[int],
                            style_node_groups: Dict[int, bpy.types.ShaderNodeTree]):
    mat, nodes, links = _new_mat(mat_name)

    im_output, time_inputs, frame_inputs = _setup_alt_image_nodes(ims, nodes, links, warp=warp, fullbright=False)
    output_node = nodes.new('ShaderNodeOutputMaterial')

    color_mul_node = nodes.new('ShaderNodeVectorMath')
    color_mul_node.operation = 'MULTIPLY'
    links.new(output_node.inputs['Surface'], color_mul_node.outputs['Vector'])
    links.new(color_mul_node.inputs[0], im_output)

    uv_node = nodes.new('ShaderNodeUVMap')
    uv_node.uv_map = lightmap_uv_layer_name

    lightmap_outputs = []
    for lightmap_idx in (idx for idx in range(4) if lightmap_styles[idx] != 255):
        lightmap_mul_node = nodes.new('ShaderNodeVectorMath')
        lightmap_mul_node.operation = 'MULTIPLY'
        lightmap_outputs.append(lightmap_mul_node.outputs[0])

        lightmap_texture_node = nodes.new('ShaderNodeTexImage')
        lightmap_texture_node.image = lightmap_ims[lightmap_idx]
        lightmap_texture_node.interpolation = 'Linear'
        group_node = nodes.new('ShaderNodeGroup')
        group_node.node_tree = style_node_groups[lightmap_styles[lightmap_idx]]
        links.new(lightmap_mul_node.inputs[0], lightmap_texture_node.outputs['Color'])
        links.new(lightmap_mul_node.inputs[1], group_node.outputs['Value'])

        links.new(lightmap_texture_node.inputs['Vector'], uv_node.outputs['UV'])

    links.new(color_mul_node.inputs[1],
              _reduce('ShaderNodeVectorMath', 'ADD', lightmap_outputs, nodes, links))

    _create_inputs(frame_inputs, time_inputs, nodes, links)

    return BlendMat(mat)


def setup_flat_material(mat_name: str, ims: BlendMatImages, warp: bool):
    mat, nodes, links = _new_mat(mat_name)

    im_output, time_inputs, frame_inputs = _setup_alt_image_nodes(ims, nodes, links, warp=warp, fullbright=False)

    output_node = nodes.new('ShaderNodeOutputMaterial')
    links.new(output_node.inputs['Surface'], im_output)
    _create_inputs(frame_inputs, time_inputs, nodes, links)

    color_mul_node = nodes.new('ShaderNodeVectorMath')
    color_mul_node.operation = 'MULTIPLY'
    links.new(output_node.inputs['Surface'], color_mul_node.outputs['Vector'])

    links.new(color_mul_node.inputs[0], im_output)
    color_mul_node.inputs[1].default_value = (0.25, 0.25, 0.25) if warp else (0, 0, 0)

    return BlendMat(mat)

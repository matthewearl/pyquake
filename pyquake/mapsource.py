import json
import string
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import parsley



grammar_source = r"""
ws = ( ' ' | '\t' )*
string = '"' (~'"' anything)*:c '"' -> ''.join(c)

texture_char = :x ?(x in (string.ascii_lowercase + string.ascii_uppercase + string.digits) + '_*+') -> x
texture_name = texture_char+:c -> ''.join(c)

digit = :x ?(x in '0123456789') -> x
digits = <digit*>
digit1_9 = :x ?(x in '123456789') -> x

intPart = (digit1_9:first digits:rest -> first + rest) | digit
floatPart :sign :ds = <'.' digits>:tail -> float(sign + ds + tail)
float = spaces ('-' | -> ''):sign (intPart:ds (floatPart(sign ds) | -> float(sign + ds)))
int = spaces ('-' | -> ''):sign (intPart:ds -> int(sign + ds))

prop = ws string:k ws string:v ws '\n' -> (k, v)
vec = '(' (ws (float | int):f -> f)*:fs ws ')' -> tuple(fs)

plane = ws vec:v1 ws vec:v2 ws vec:v3  ws texture_name:t ws int:x_off ws int:y_off ws float:rot_angle ws float:x_scale ws float:y_scale ws '\n'
    -> Plane(np.stack([v1, v2, v3]), t, (x_off, y_off), rot_angle, (x_scale, y_scale))

brush = ws '{' ws '\n' plane*:p '}' ws '\n' -> p

entity = ws '{' ws '\n' ws prop*:pr brush*:bs ws '}' ws '\n'  -> Entity(dict(pr), bs)

map = entity*:e -> e
"""


@dataclass
class Plane:
    verts: np.ndarray
    texture_name: str
    off: Tuple[int, int]
    rot_angle: float
    scale: Tuple[float, float]


@dataclass
class Entity:
    props: Dict[str, str]
    brushes: List[List[Plane]]


def parse(f):
    g = parsley.makeGrammar(
        grammar_source,
        {'string': string, 'Plane': Plane, 'np': np, 'Entity': Entity}
    )

    map_source = f.read()
    entities = g(map_source).map()

    return entities


def extract_lights_main():
    fname, = sys.argv[1:]

    with open(fname) as f:
        entities = parse(f)

    light_props = [e.props for e in entities if e.props.get('classname') == 'light']
    light_cfg = {
        f'light{i}': {
            'type': 'POINT',
            'location': [float(x) for x in p['origin'].split(' ')],
            'energy': float(p.get('light', -1))
        }
        for i, p in enumerate(light_props)
    }

    json.dump(light_cfg, sys.stdout, indent=4)

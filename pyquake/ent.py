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
    "parse_entities",
    "InvalidEntityString",
)


import re


class InvalidEntityString(Exception):
    def __init__(self, msg, line_num, lines):
        super().__init__(f"Line {line_num}: {msg}")
        self.line_num = line_num
        self.lines = lines


_OPEN_LINE = re.compile(r'{')
_CLOSE_LINE = re.compile(r'}')
_DEF_LINE = re.compile(r'"(?P<key>[^"]*)" *"(?P<value>[^"]*)"')


# TODO:  Field defs should be read from a progs.dat.
_VEC_KEYS = ['origin']
_FLOAT_KEYS = ['angle']


def _parse_val(key, val):
    if key in _VEC_KEYS:
        return tuple(float(x) for x in val.split(' '))
    elif key in _FLOAT_KEYS:
        return float(val)
    else:
        return val


def parse_entities(ent_str):
    lines = [l.strip() for l in ent_str.split('\n')]
    lines = [l for l in lines if l]

    entities = []
    line_num = 0
    while line_num < len(lines):
        if re.match(_OPEN_LINE, lines[line_num]) is None:
            raise InvalidEntityString(f"Expected opening brace, not {lines[line_num]!r}", line_num, lines)
        line_num += 1
        entity = {}
        entity_done = False
        while not entity_done and line_num < len(lines):
            m = re.match(_DEF_LINE, lines[line_num])
            if m is not None:
                k, v = m.group('key'), m.group('value')
                entity[k] = _parse_val(k, v)
            elif re.match(_CLOSE_LINE, lines[line_num]):
                entity_done = True
            else:
                raise InvalidEntityString(f"Expected closing brace or definition, not {lines[line_num]!r}",
                                          line_num, lines)
            line_num += 1
        entities.append(entity)
    return entities


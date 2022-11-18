from __future__ import annotations

import abc
import dataclasses
import enum
import struct

from typing import List


_MAX_PARMS = 8


def _read_struct(fmt, f):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(size))


class _Formattable(abc.ABC):
    def format(self) -> str:
        raise NotImplementedError


@dataclasses.dataclass
@_Formattable.register
class Function:
    progs: Progs = dataclasses.field(repr=False)
    first_statement: int
    parm_start: int
    locals_: int
    
    profile: int

    s_name: int
    s_file: int

    parm_size: List[int]

    @classmethod
    def load(cls, f):
        args = _read_struct("<LLLLLL", f)
        num_parms, = _read_struct("<L", f)
        parm_size = _read_struct("<" + "B" * _MAX_PARMS, f)
        return cls(None, *args, parm_size[:num_parms])

    @property
    def name(self):
        return self.progs.read_string(self.s_name)

    @property
    def file(self):
        return self.progs.read_string(self.s_file)

    def format(self) -> str:
        return self.name


class Op(enum.IntEnum):
    DONE = 0
    MUL_F = enum.auto()
    MUL_V = enum.auto()
    MUL_FV = enum.auto()
    MUL_VF = enum.auto()
    DIV_F = enum.auto()
    ADD_F = enum.auto()
    ADD_V = enum.auto()
    SUB_F = enum.auto()
    SUB_V = enum.auto()
    EQ_F = enum.auto()
    EQ_V = enum.auto()
    EQ_S = enum.auto()
    EQ_E = enum.auto()
    EQ_FNC = enum.auto()
    NE_F = enum.auto()
    NE_V = enum.auto()
    NE_S = enum.auto()
    NE_E = enum.auto()
    NE_FNC = enum.auto()
    LE = enum.auto()
    GE = enum.auto()
    LT = enum.auto()
    GT = enum.auto()
    LOAD_F = enum.auto()
    LOAD_V = enum.auto()
    LOAD_S = enum.auto()
    LOAD_ENT = enum.auto()
    LOAD_FLD = enum.auto()
    LOAD_FNC = enum.auto()
    ADDRESS = enum.auto()
    STORE_F = enum.auto()
    STORE_V = enum.auto()
    STORE_S = enum.auto()
    STORE_ENT = enum.auto()
    STORE_FLD = enum.auto()
    STORE_FNC = enum.auto()
    STOREP_F = enum.auto()
    STOREP_V = enum.auto()
    STOREP_S = enum.auto()
    STOREP_ENT = enum.auto()
    STOREP_FLD = enum.auto()
    STOREP_FNC = enum.auto()
    RETURN = enum.auto()
    NOT_F = enum.auto()
    NOT_V = enum.auto()
    NOT_S = enum.auto()
    NOT_ENT = enum.auto()
    NOT_FNC = enum.auto()
    IF = enum.auto()
    IFNOT = enum.auto()
    CALL0 = enum.auto()
    CALL1 = enum.auto()
    CALL2 = enum.auto()
    CALL3 = enum.auto()
    CALL4 = enum.auto()
    CALL5 = enum.auto()
    CALL6 = enum.auto()
    CALL7 = enum.auto()
    CALL8 = enum.auto()
    STATE = enum.auto()
    GOTO = enum.auto()
    AND = enum.auto()
    OR = enum.auto()
    BITAND = enum.auto()
    BITOR = enum.auto()


class Type(enum.IntEnum):
    BAD = -1
    VOID = enum.auto()
    STRING = enum.auto()
    FLOAT = enum.auto()
    VECTOR = enum.auto()
    ENTITY = enum.auto()
    FIELD = enum.auto()
    FUNCTION = enum.auto()
    POINTER = enum.auto()

    def format(self):
        if self == Type.POINTER:
            return "void*"
        else:
            return self.name.lower()


_BINARY_OPS = {
    Op.ADD_F: ('+', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.SUB_F: ('-', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.MUL_F: ('*', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.DIV_F: ('/', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.ADD_V: ('+', Type.VECTOR, Type.VECTOR, Type.VECTOR),
    Op.SUB_V: ('-', Type.VECTOR, Type.VECTOR, Type.VECTOR),
    Op.MUL_V: ('*', Type.VECTOR, Type.VECTOR, Type.VECTOR),
    Op.MUL_VF: ('*vf', Type.VECTOR, Type.FLOAT, Type.VECTOR),
    Op.MUL_FV: ('*fv', Type.VECTOR, Type.VECTOR, Type.FLOAT),
    Op.BITAND: ('&', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.BITOR: ('|', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.GE: ('>=', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.LE: ('<=', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.GT: ('>', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.LT: ('<', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.AND: ('&&', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.OR: ('||', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.EQ_F: ('==', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.EQ_V: ('==', Type.FLOAT, Type.VECTOR, Type.VECTOR),
    Op.EQ_S: ('==', Type.FLOAT, Type.STRING, Type.STRING),
    Op.EQ_E: ('==', Type.FLOAT, Type.ENTITY, Type.ENTITY),
    Op.EQ_FNC: ('==', Type.FLOAT, Type.FUNCTION, Type.FUNCTION),
    Op.NE_F: ('!=', Type.FLOAT, Type.FLOAT, Type.FLOAT),
    Op.NE_V: ('!=', Type.FLOAT, Type.VECTOR, Type.VECTOR),
    Op.NE_S: ('!=', Type.FLOAT, Type.STRING, Type.STRING),
    Op.NE_E: ('!=', Type.FLOAT, Type.ENTITY, Type.ENTITY),
    Op.NE_FNC: ('!=', Type.FLOAT, Type.FUNCTION, Type.FUNCTION),
}


_STORE_OPS = {
    Op.STOREP_F: (True, Type.FLOAT),
    Op.STOREP_ENT: (True, Type.ENTITY),
    Op.STOREP_FLD: (True, Type.FIELD),
    Op.STOREP_S: (True, Type.STRING),
    Op.STOREP_FNC: (True, Type.FUNCTION),
    Op.STOREP_V: (True, Type.VECTOR),
    Op.STORE_F: (False, Type.FLOAT),
    Op.STORE_ENT: (False, Type.ENTITY),
    Op.STORE_FLD: (False, Type.FIELD),
    Op.STORE_S: (False, Type.STRING),
    Op.STORE_FNC: (False, Type.FUNCTION),
    Op.STORE_V: (False, Type.VECTOR),
}


_LOAD_OPS = {
    Op.LOAD_F: Type.FLOAT,
    Op.LOAD_ENT: Type.ENTITY,
    Op.LOAD_FLD: Type.FIELD,
    Op.LOAD_S: Type.STRING,
    Op.LOAD_FNC: Type.FUNCTION,
    Op.LOAD_V: Type.VECTOR,
}


_CALL_OPS = [
    Op.CALL0, Op.CALL1, Op.CALL2, Op.CALL3, Op.CALL4, Op.CALL5, Op.CALL6,
    Op.CALL7, Op.CALL8
]


_NOT_OPS = {
    Op.NOT_F: Type.FLOAT,
    Op.NOT_V: Type.VECTOR,
    Op.NOT_S: Type.STRING,
    Op.NOT_ENT: Type.ENTITY,
    Op.NOT_FNC: Type.FUNCTION,
}


@dataclasses.dataclass
class Statement:
    progs: Progs
    op: Op
    a: int
    b: int
    c: int

    @classmethod
    def load(cls, f):
        op_int, *args = _read_struct("<Hhhh", f)
        return cls(None, Op(op_int), *args)

    def _format_arg(self, offset, type_, pointer=False) -> str:
        if pointer:
            return f"*({type_.format()}*)*(void**){offset}"
        try:
            global_def = self.progs.find_global_def(offset, type_)
        except KeyError:
            out = f"*({type_.format()}*){offset}"
        else:
            if global_def.name != "IMMEDIATE":
                out = global_def.name
            else:
                val = self.progs.read_global(offset, global_def.type_)
                if isinstance(val, _Formattable):
                    out = val.format()
                else:
                    out = repr(val)

            if global_def.type_ != type_:
                out = f"*({type_.format()}*)&{out}"
        return out

    def format(self, num):
        if self.op in _BINARY_OPS:
            op_str, c_type, a_type, b_type = _BINARY_OPS[self.op]
            out = (f"{self._format_arg(self.c, c_type)} = "
                   f"{self._format_arg(self.a, a_type)} {op_str} "
                   f"{self._format_arg(self.b, b_type)}")
        elif self.op in _NOT_OPS:
            type_ = _NOT_OPS[self.op]
            out = (f"{self._format_arg(self.c, type_)} = "
                   f"!{self._format_arg(self.a, type_)}")
        elif self.op in _STORE_OPS:
            pointer, type_ = _STORE_OPS[self.op]
            out = (f"{self._format_arg(self.b, type_, pointer)} = "
                   f"{self._format_arg(self.a, type_)}")
        elif self.op in _LOAD_OPS:
            type_ = _LOAD_OPS[self.op]
            out = (f"{self._format_arg(self.c, type_)} = "
                   f"edicts[{self._format_arg(self.a, Type.ENTITY)}]."
                   f"{self._format_arg(self.b, Type.FIELD)}")
        elif self.op in _CALL_OPS:
            out = f"{self._format_arg(self.a, Type.FUNCTION)}()"
        elif self.op in (Op.IF, Op.IFNOT):
            invert = "!" if self.op == Op.IFNOT else ""
            out = (f"if ({invert}{self._format_arg(self.a, Type.FLOAT)}) "
                   f" goto {num + self.b}")
        elif self.op == Op.GOTO:
            out = f" goto {num + self.a}"
        elif self.op == Op.RETURN:
            out = "return"
        elif self.op == Op.ADDRESS:
            out = (f"{self._format_arg(self.c, Type.POINTER, False)} = "
                   f"&edicts[{self._format_arg(self.a, Type.ENTITY)}]."
                   f"{self._format_arg(self.b, Type.FIELD)}")
        elif self.op == Op.STATE:
            out = (f"@state(frame={self._format_arg(self.a, Type.FLOAT)}, "
                   f"think={self._format_arg(self.b, Type.FUNCTION)})")
        elif self.op == Op.DONE:
            out = "done"
        else:
            raise Exception(f"Unsupported op {self.op.name}")

        out = out + ";"

        out = f"{num:<8n} {out:50s}" + str((self.op, self.a, self.b, self.c))
        return out


@dataclasses.dataclass
@_Formattable.register
class Definition:
    progs: Progs = dataclasses.field(repr=False)
    type_: int
    save_global: bool
    ofs: int
    s_name: int

    @classmethod
    def load(cls, f):
        type_, ofs, s_name = _read_struct("<HHl", f)
        save_global = (type_ & (1 << 15)) != 0
        type_ &= ~(1 << 15)
        return cls(None, Type(type_), save_global, ofs, s_name)

    @property
    def name(self):
        return self.progs.read_string(self.s_name)

    def format(self) -> str:
        return self.name
    

@dataclasses.dataclass
class Progs:
    version: int
    crc: int
    strings: str
    globals_: bytes
    functions: List[Function]
    statements: List[Statement]
    global_defs: List[Definition]
    field_defs: List[Definition]

    def read_string(self, i):
        out = self.strings[i:]
        if '\0' in out:
            out = out[:out.index('\0')]
        return out

    def read_global(self, offset, type_: Type):
        offset = offset * 4
        if type_ == Type.STRING:
            out = self.read_string(
                struct.unpack('<L', self.globals_[offset:offset + 4])[0]
            )
        elif type_ == Type.FLOAT:
            out, = struct.unpack('<f', self.globals_[offset:offset + 4])
        elif type_ == Type.VECTOR:
            out = struct.unpack('<fff', self.globals_[offset:offset + 12])
        elif type_ == Type.ENTITY:
            out, = struct.unpack('<L', self.globals_[offset:offset + 4])
        elif type_ == Type.FUNCTION:
            func_idx, = struct.unpack('<L', self.globals_[offset:offset + 4])
            out = self.functions[func_idx]
        elif type_ == Type.FIELD:
            ofs, = struct.unpack('<L', self.globals_[offset:offset + 4])
            out = next(iter(d for d in self.field_defs if d.ofs == ofs))
        else:
            out = f"Unhandled type: {type_}"

        return out

    @classmethod
    def load(cls, f):
        version, crc = _read_struct("<LL", f)

        # Load lump offsets
        offsets = {}
        for key in ['statements', 'global_defs', 'field_defs', 'functions',
                    'strings', 'globals']:
            offset, count = _read_struct("<LL", f)
            offsets[key] = {'offset': offset, 'count': count}

        # Read strings
        f.seek(offsets['strings']['offset'])
        strings = f.read(offsets['strings']['count']).decode('ascii')

        # Read globals
        f.seek(offsets['globals']['offset'])
        globals_ = f.read(4 * offsets['globals']['count'])

        # Read functions
        f.seek(offsets['functions']['offset'])
        functions = [Function.load(f)
                     for _ in range(offsets['functions']['count'])]

        # Read statements
        f.seek(offsets['statements']['offset'])
        statements = [Statement.load(f)
                      for _ in range(offsets['statements']['count'])]

        # Read global defs
        f.seek(offsets['global_defs']['offset'])
        global_defs = [Definition.load(f)
                       for _ in range(offsets['global_defs']['count'])]

        # Read field defs
        f.seek(offsets['field_defs']['offset'])
        field_defs = [Definition.load(f)
                       for _ in range(offsets['field_defs']['count'])]

        return cls(version, crc, strings, globals_, functions, statements,
                   global_defs, field_defs)

    def find_global_def(self, offset: int, type_: Type):
        if (offset, type_) in self._global_def_type_dict:
            out = self._global_def_type_dict[offset, type_]
        else:
            out = self._global_def_dict[offset]
        return out

    def __post_init__(self):
        for function in self.functions:
            function.progs = self

        for statement in self.statements:
            statement.progs = self

        for global_def in self.global_defs:
            global_def.progs = self

        for field_def in self.field_defs:
            field_def.progs = self

        self._global_def_type_dict = {
            (global_def.ofs, global_def.type_): global_def
            for global_def in self.global_defs
        }

        self._global_def_dict = {
            global_def.ofs: global_def
            for global_def in self.global_defs
        }


def dump_progs_main():
    import argparse
    import sys

    from . import pak

    parser = argparse.ArgumentParser(description='Dump a progs.dat')
    parser.add_argument('-b', '--base-dir', help='base dir containing id1/ etc')
    parser.add_argument('-g', '--game', help='sub-dir within game dir',
                        default=None)
    parsed = parser.parse_args(sys.argv[1:])

    fs = pak.Filesystem(parsed.base_dir, parsed.game)
    with fs.open('progs.dat') as f:
        pr = Progs.load(f)

        print('------- FUNCTIONS --------')
        for func in pr.functions:
            print(func.name, func.file)

        statement_funcs = {
            func.first_statement: func for func in pr.functions
        }

        print('------- STATEMENTS --------')
        for num, statement in enumerate(pr.statements):
            if num in statement_funcs:
                func = statement_funcs[num]
                print("//", func.file, ':', func.name)
            print(statement.format(num))

        print('------- GLOBAL DEFS --------')
        for d in pr.global_defs: 
            print((d.type_, d.name, d.ofs, pr.read_global(d.ofs, d.type_)))

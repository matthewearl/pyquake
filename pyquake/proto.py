# Copyright (c) 2018 Matthew Earl
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
    'MalformedNetworkData',
    'ServerMessage',
)


import enum
import inspect
import math
import struct


class MalformedNetworkData(Exception):
    pass


def _read(f, n):
    s = f.read(n)
    if len(s) != n:
        raise MalformedNetworkData
    return s


class TempEntityTypes(enum.IntEnum):
    SPIKE = 0
    SUPERSPIKE = 1
    GUNSHOT = 2
    EXPLOSION = 3
    TAREXPLOSION = 4
    LIGHTNING1 = 5
    LIGHTNING2 = 6
    WIZSPIKE = 7
    KNIGHTSPIKE = 8
    LIGHTNING3 = 9
    LAVASPLASH = 10
    TELEPORT = 11
    EXPLOSION2 = 12
    BEAM = 13


class ServerMessageType(enum.Enum):
    BAD = 0
    NOP = 1
    DISCONNECT = 2
    UPDATESTAT = 3
    VERSION = 4
    SETVIEW = 5
    SOUND = 6
    TIME = 7
    PRINT = 8
    STUFFTEXT = 9
    SETANGLE = 10
    SERVERINFO = 11
    LIGHTSTYLE = 12
    UPDATENAME = 13
    UPDATEFRAGS = 14
    CLIENTDATA = 15
    STOPSOUND = 16
    UPDATECOLORS = 17
    PARTICLE = 18
    DAMAGE = 19
    SPAWNSTATIC = 20
    SPAWNBINARY = 21
    SPAWNBASELINE = 22
    TEMP_ENTITY = 23
    SETPAUSE = 24
    SIGNONNUM = 25
    CENTERPRINT = 26
    KILLEDMONSTER = 27
    FOUNDSECRET = 28
    SPAWNSTATICSOUND = 29
    INTERMISSION = 30
    FINALE = 31
    CDTRACK = 32
    SELLSCREEN = 33
    CUTSCENE = 34
    UPDATE = 128


class ItemFlags(enum.IntFlag):
    SHOTGUN = 1
    SUPER_SHOTGUN = 2
    NAILGUN = 4
    SUPER_NAILGUN = 8
    GRENADE_LAUNCHER = 16
    ROCKET_LAUNCHER = 32
    LIGHTNING = 64
    SUPER_LIGHTNING = 128
    SHELLS = 256
    NAILS = 512
    ROCKETS = 1024
    CELLS = 2048
    AXE = 4096
    ARMOR1 = 8192
    ARMOR2 = 16384
    ARMOR3 = 32768
    SUPERHEALTH = 65536
    KEY1 = 131072
    KEY2 = 262144
    INVISIBILITY = 524288
    INVULNERABILITY = 1048576
    SUIT = 2097152
    QUAD = 4194304
    SIGIL1 = (1<<28)
    SIGIL2 = (1<<29)
    SIGIL3 = (1<<30)
    SIGIL4 = (1<<31)


class _UpdateFlags(enum.IntFlag):
    MOREBITS = (1<<0)
    ORIGIN1 = (1<<1)
    ORIGIN2 = (1<<2)
    ORIGIN3 = (1<<3)
    ANGLE2 = (1<<4)
    STEP = (1<<5)
    FRAME = (1<<6)
    SIGNAL = (1<<7)
    ANGLE1 = (1<<8)
    ANGLE3 = (1<<9)
    MODEL = (1<<10)
    COLORMAP = (1<<11)
    SKIN = (1<<12)
    EFFECTS = (1<<13)
    LONGENTITY = (1<<14)


class _ClientDataFlags(enum.IntFlag):
    VIEWHEIGHT = 1<<0
    IDEALPITCH = 1<<1
    PUNCH1 = 1<<2
    PUNCH2 = 1<<3
    PUNCH3 = 1<<4
    VELOCITY1 = 1<<5
    VELOCITY2 = 1<<6
    VELOCITY3 = 1<<7
    UNUSED8 = 1<<8
    ITEMS = 1<<9
    ONGROUND = 1<<10
    INWATER = 1<<11
    WEAPONFRAME = 1<<12
    ARMOR = 1<<13
    WEAPON = 1<<14


class _SoundFlags(enum.IntFlag):
    VOLUME = (1<<0)
    ATTENUATION = (1<<1)
    LOOPING = (1<<2)


_MESSAGE_CLASSES = {}
def _register_server_message(cls):
    _MESSAGE_CLASSES[cls.msg_type] = cls


_DEFAULT_VIEW_HEIGHT = 22
_DEFAULT_SOUND_PACKET_ATTENUATION = 1.0
_DEFAULT_SOUND_PACKET_VOLUME = 255


class ServerMessage:
    field_names = None

    def __init__(self, *args, **kwargs):
        sig = inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                    for n in self.field_names])
        bound_args = sig.bind(*args, **kwargs)
        for key, val in bound_args.arguments.items():
            setattr(self, key, val)

    def __repr__(self):
        return "{}({})".format(
                    self.__class__.__name__,
                    ", ".join("{}={!r}".format(k, getattr(self, k)) for k in self.field_names))

    @classmethod
    def _parse_struct(cls, fmt, m):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, m[:size]), m[size:]

    @classmethod
    def _parse_string(cls, m):
        if b'\0' not in m:
            raise MalformedNetworkData('Null terminator not found')
        idx = m.index(b'\0')
        return m[:idx].decode('latin'), m[idx + 1:]

    @classmethod
    def _parse_coord(cls, m):
        (c,), m = cls._parse_struct("<h", m)
        return c / 8., m

    @classmethod
    def _parse_angle(cls, m):
        b, m = m[0], m[1:]
        return b * math.pi / 128., m

    @classmethod
    def _parse_tuple(cls, n, el_parser, m):
        l = []
        for _ in range(n):
            x, m = el_parser(m)
            l.append(x)
        return tuple(l), m

    @classmethod
    def _parse_angles(cls, m):
        return cls._parse_tuple(3, cls._parse_angle, m)

    @classmethod
    def _parse_coords(cls, m):
        return cls._parse_tuple(3, cls._parse_coord, m)

    @classmethod
    def _parse_optional(cls, bit, flags, fmt, m, post_func=None, default=None):
        if bit & flags:
            (val,), m = cls._parse_struct(fmt, m)
            if post_func:
                val = post_func(val)
            return val, m
        else:
            return default, m

    @classmethod
    def parse_message(cls, m):
        msg_type_int = m[0]

        if msg_type_int & _UpdateFlags.SIGNAL:
            return ServerMessageUpdate.parse(m)

        try:
            msg_type = ServerMessageType(msg_type_int)
        except ValueError:
            raise MalformedNetworkData("Invalid message type {}".format(msg_type_int))
        
        try:
            msg_cls = _MESSAGE_CLASSES[msg_type]
        except KeyError:
            raise MalformedNetworkData("No handler for message type {}".format(msg_type))
        return msg_cls.parse(m[1:])
        
    @classmethod
    def parse(cls, m):
        raise NotImplementedError


class StructServerMessage(ServerMessage):
    @classmethod
    def parse(cls, m):
        vals, m = cls._parse_struct(cls.fmt, m)
        return cls(**dict(zip(cls.field_names, vals))), m


class ServerMessageUpdate(ServerMessage):
    msg_type = ServerMessageType.UPDATE
    field_names = (
        'entity_num',
        'model_num',
        'frame',
        'colormap',
        'skin',
        'effects',
        'origin',
        'angle',
        'step',
    )

    @classmethod
    def parse(cls, m):
        flags, m = _UpdateFlags(m[0]), m[1:]
        assert flags & _UpdateFlags.SIGNAL


        if flags & _UpdateFlags.MOREBITS:
            more_flags, m = m[0], m[1:]
            flags |= (more_flags << 8)

        (entity_num,), m = cls._parse_struct("<H" if flags & _UpdateFlags.LONGENTITY else "<B", m)
        model_num, m = cls._parse_optional(_UpdateFlags.MODEL, flags, "<B", m)
        frame, m = cls._parse_optional(_UpdateFlags.FRAME, flags, "<B", m)
        colormap, m = cls._parse_optional(_UpdateFlags.COLORMAP, flags, "<B", m)
        skin, m = cls._parse_optional(_UpdateFlags.SKIN, flags, "<B", m)
        effects, m = cls._parse_optional(_UpdateFlags.EFFECTS, flags, "<B", m)

        fix_coord = lambda c: c / 8.
        fix_angle = lambda a: a * math.pi / 128.
        origin1, m = cls._parse_optional(_UpdateFlags.ORIGIN1, flags, "<h", m, fix_coord)
        angle1, m = cls._parse_optional(_UpdateFlags.ANGLE1, flags, "<B", m, fix_angle)
        origin2, m = cls._parse_optional(_UpdateFlags.ORIGIN2, flags, "<h", m, fix_coord)
        angle2, m = cls._parse_optional(_UpdateFlags.ANGLE2, flags, "<B", m, fix_angle)
        origin3, m = cls._parse_optional(_UpdateFlags.ORIGIN3, flags, "<h", m, fix_coord)
        angle3, m = cls._parse_optional(_UpdateFlags.ANGLE3, flags, "<B", m, fix_angle)
        origin = (origin1, origin2, origin3)
        angle = (angle1, angle2, angle3)

        step = bool(flags & _UpdateFlags.STEP)

        return cls(entity_num,
                   model_num,
                   frame,
                   colormap,
                   skin,
                   effects,
                   origin,
                   angle,
                   step), m


class NoFieldsServerMessage(ServerMessage):
    field_names = ()

    @classmethod
    def parse(cls, m):
        return cls(), m


@_register_server_message
class ServerMessageNop(NoFieldsServerMessage):
    msg_type = ServerMessageType.NOP


@_register_server_message
class ServerMessageFoundSecret(NoFieldsServerMessage):
    msg_type = ServerMessageType.FOUNDSECRET


@_register_server_message
class ServerMessagePrint(ServerMessage):
    field_names = ('string',)
    msg_type = ServerMessageType.PRINT

    @classmethod
    def parse(cls, m):
        s, m = cls._parse_string(m)
        return cls(s), m


@_register_server_message
class ServerMessageCenterPrint(ServerMessage):
    field_names = ('string',)
    msg_type = ServerMessageType.CENTERPRINT

    @classmethod
    def parse(cls, m):
        s, m = cls._parse_string(m)
        return cls(s), m


@_register_server_message
class ServerMessageCutScene(ServerMessage):
    field_names = ('string',)
    msg_type = ServerMessageType.CUTSCENE

    @classmethod
    def parse(cls, m):
        s, m = cls._parse_string(m)
        return cls(s), m


@_register_server_message
class ServerMessageStuffText(ServerMessage):
    field_names = ('string',)
    msg_type = ServerMessageType.STUFFTEXT

    @classmethod
    def parse(cls, m):
        s, m = cls._parse_string(m)
        return cls(s), m


@_register_server_message
class ServerMessageSpawnStaticSound(ServerMessage):
    field_names = ("origin", "sound_num", "vol", "atten")
    msg_type = ServerMessageType.SPAWNSTATICSOUND

    @classmethod
    def parse(cls, m):
        origin, m = cls._parse_coords(m)
        (sound_num, vol, atten), m = cls._parse_struct("<BBB", m)

        return cls(origin, sound_num, vol, atten), m


@_register_server_message
class ServerMessageCdTrack(StructServerMessage):
    fmt = "<BB"
    field_names = ("track", "loop")
    msg_type = ServerMessageType.CDTRACK


@_register_server_message
class ServerMessageSetView(StructServerMessage):
    fmt = "<H"
    field_names = ("viewentity",)
    msg_type = ServerMessageType.SETVIEW


@_register_server_message
class ServerMessageSignOnNum(StructServerMessage):
    fmt = "<B"
    field_names = ("num",)
    msg_type = ServerMessageType.SIGNONNUM


@_register_server_message
class ServerMessageSetPause(StructServerMessage):
    fmt = "<B"
    field_names = ("paused",)
    msg_type = ServerMessageType.SETPAUSE

@_register_server_message
class ServerMessageSpawnBaseline(ServerMessage):
    field_names = ("entity_num", "model_num", "frame", "colormap", "skin", "origin", "angles")
    msg_type = ServerMessageType.SPAWNBASELINE

    @classmethod
    def parse(cls, m):
        (entity_num, model_num, frame, colormap, skin), m = cls._parse_struct("<HBBBB", m)
        origin, angles = [], []
        for _ in range(3):
            o, m = cls._parse_coord(m)
            a, m = cls._parse_angle(m)
            origin.append(o)
            angles.append(a)
        return cls(entity_num, model_num, frame, colormap, skin, tuple(origin), tuple(angles)), m


@_register_server_message
class ServerMessageSpawnStatic(ServerMessage):
    field_names = ("model_num", "frame", "colormap", "skin", "origin", "angles")
    msg_type = ServerMessageType.SPAWNSTATIC

    @classmethod
    def parse(cls, m):
        (model_num, frame, colormap, skin), m = cls._parse_struct("<BBBB", m)
        origin, angles = [], []
        for _ in range(3):
            o, m = cls._parse_coord(m)
            a, m = cls._parse_angle(m)
            origin.append(o)
            angles.append(a)
        return cls(model_num, frame, colormap, skin, tuple(origin), tuple(angles)), m


@_register_server_message
class ServerMessageTime(StructServerMessage):
    fmt = "<f"
    field_names = ("time",)
    msg_type = ServerMessageType.TIME


@_register_server_message
class ServerMessageUpdateName(ServerMessage):
    msg_type = ServerMessageType.UPDATENAME
    field_names = ('client_num', 'name')

    @classmethod
    def parse(cls, m):
        client_num, m = m[0], m[1:]
        name, m = cls._parse_string(m)
        return cls(client_num, name), m


@_register_server_message
class ServerMessageUpdateFrags(StructServerMessage):
    fmt = "<BH"
    field_names = ("client_num", "count")
    msg_type = ServerMessageType.UPDATEFRAGS


@_register_server_message
class ServerMessageUpdateColors(StructServerMessage):
    fmt = "<BB"
    field_names = ("client_num", "color")
    msg_type = ServerMessageType.UPDATECOLORS


@_register_server_message
class ServerMessageLightStyle(ServerMessage):
    field_names = ('index', 'style')
    msg_type = ServerMessageType.LIGHTSTYLE

    @classmethod
    def parse(cls, m):
        index, m = m[0], m[1:]
        style, m = cls._parse_string(m)
        return cls(index, style), m


@_register_server_message
class ServerMessageUpdateStat(StructServerMessage):
    fmt = "<BI"
    field_names = ('index', 'value')
    msg_type = ServerMessageType.UPDATESTAT


@_register_server_message
class ServerMessageSetAngle(ServerMessage):
    field_names = ('view_angles',)
    msg_type = ServerMessageType.SETANGLE

    @classmethod
    def parse(cls, m):
        view_angles, m = cls._parse_angles(m)
        return cls(view_angles), m


@_register_server_message
class ServerMessageServerInfo(ServerMessage):
    field_names = ('protocol_version', 'max_clients', 'game_type', 'level_name', 'models', 'sounds')
    msg_type = ServerMessageType.SERVERINFO

    @classmethod
    def _parse_string_list(cls, m):
        l = []
        while True:
            s, m = cls._parse_string(m)
            if not s:
                break
            l.append(s)
        return l, m

    @classmethod
    def parse(cls, m):
        (protocol_version, max_clients, game_type), m = cls._parse_struct("<IBB", m)
        if protocol_version != 15:
            raise MalformedNetworkData("Invaid protocol version {}, only 15 is supported".format(protocol_version))
        level_name, m = cls._parse_string(m)
        models, m = cls._parse_string_list(m)
        sounds, m = cls._parse_string_list(m)

        return cls(protocol_version, max_clients, game_type, level_name, models, sounds), m


@_register_server_message
class ServerMessageClientData(ServerMessage):
    field_names = (
        'view_height',
        'ideal_pitch',
        'punch_angles',
        'm_velocity',
        'items',
        'on_ground',
        'in_water',
        'weapon_frame',
        'armor',
        'weapon_model_index',
        'health',
        'ammo',
        'shells',
        'nails',
        'rockets',
        'cells',
        'active_weapon',
    )
    msg_type = ServerMessageType.CLIENTDATA

    @classmethod
    def parse(cls, m):
        (flags_int,), m = cls._parse_struct("<H", m)
        flags = _ClientDataFlags(flags_int)

        view_height, m = cls._parse_optional(_ClientDataFlags.VIEWHEIGHT, flags, "<B", m,
                                             default=_DEFAULT_VIEW_HEIGHT)
        ideal_pitch, m = cls._parse_optional(_ClientDataFlags.IDEALPITCH, flags, "<B", m, default=0)

        fix_velocity = lambda v: v * 16
        punch1, m = cls._parse_optional(_ClientDataFlags.PUNCH1, flags, "<B", m, default=0)
        m_velocity1, m = cls._parse_optional(_ClientDataFlags.VELOCITY1, flags, "<b", m, fix_velocity,
                                             default=0)
        punch2, m = cls._parse_optional(_ClientDataFlags.PUNCH2, flags, "<B", m, default=0)
        m_velocity2, m = cls._parse_optional(_ClientDataFlags.VELOCITY2, flags, "<b", m, fix_velocity,
                                             default=0)
        punch3, m = cls._parse_optional(_ClientDataFlags.PUNCH3, flags, "<B", m, default=0)
        m_velocity3, m = cls._parse_optional(_ClientDataFlags.VELOCITY3, flags, "<b", m, fix_velocity,
                                             default=0)
        punch_angles = (punch1, punch2, punch3)
        m_velocity = (m_velocity1, m_velocity2, m_velocity3)

        (items_int,), m = cls._parse_struct("<I", m)
        items = ItemFlags(items_int)

        on_ground = bool(flags & _ClientDataFlags.ONGROUND)
        in_water = bool(flags & _ClientDataFlags.INWATER)

        weapon_frame, m = cls._parse_optional(_ClientDataFlags.WEAPONFRAME, flags, "<B", m, default=0)
        armor, m = cls._parse_optional(_ClientDataFlags.ARMOR, flags, "<B", m, default=0)
        weapon_model_index, m = cls._parse_optional(_ClientDataFlags.WEAPON, flags, "<B", m, default=0)

        (health, ammo, shells, nails, rockets, cells, active_weapon), m = cls._parse_struct("<HBBBBBB", m)
        active_weapon = ItemFlags(active_weapon)

        return cls(
            view_height,
            ideal_pitch,
            punch_angles,
            m_velocity,
            items,
            on_ground,
            in_water,
            weapon_frame,
            armor,
            weapon_model_index,
            health,
            ammo,
            shells,
            nails,
            rockets,
            cells,
            active_weapon
        ), m


@_register_server_message
class ServerMessageSound(ServerMessage):
    field_names = ('volume', 'attenuation', 'entity_num', 'channel', 'sound_num', 'pos')
    msg_type = ServerMessageType.SOUND

    @classmethod
    def parse(cls, m):
        flags, m = _SoundFlags(m[0]), m[1:]

        volume, m = cls._parse_optional(_SoundFlags.VOLUME, flags, "<B", m,
                                        default=_DEFAULT_SOUND_PACKET_VOLUME)
        attenuation, m = cls._parse_optional(_SoundFlags.ATTENUATION, flags, "<B", m, lambda b: b / 64.,
                                             default=_DEFAULT_SOUND_PACKET_ATTENUATION)

        (t,), m = cls._parse_struct("<H", m)
        entity_num = t >> 3
        channel = t & 7

        sound_num, m = m[0], m[1:]
        pos, m = cls._parse_coords(m)
        
        return cls(volume, attenuation, entity_num, channel, sound_num, pos), m


@_register_server_message
class ServerMessageParticle(ServerMessage):
    field_names = ('origin', 'direction', 'count', 'color')
    msg_type = ServerMessageType.PARTICLE

    @classmethod
    def parse(cls, m):
        origin, m = cls._parse_coords(m)
        
        direction, m = cls._parse_struct("<bbb", m)
        direction = tuple(x / 16. for x in direction)

        count, m = m[0], m[1:]
        if count == 255:
            count = 1024

        color, m = m[0], m[1:]

        return cls(origin, direction, count, color), m


@_register_server_message
class ServerMessageTempEntity(ServerMessage):
    field_names = ('temp_entity_type', 'entity_num', 'origin', 'end', 'color_start', 'color_length')
    msg_type = ServerMessageType.TEMP_ENTITY

    @classmethod
    def parse(cls, m):
        temp_entity_type, m = TempEntityTypes(m[0]), m[1:]

        if temp_entity_type in (TempEntityTypes.LIGHTNING1, TempEntityTypes.LIGHTNING2, TempEntityTypes.LIGHTNING3,
                                TempEntityTypes.BEAM):
            (entity_num,), m = cls._parse_struct("<H", m)
            origin, m = cls._parse_coords(m)
            end, m = cls._parse_coords(m)
        else:
            origin, m = cls._parse_coords(m)
            end = None
            entity_num = None

        if temp_entity_type == TempEntityTypes.EXPLOSION2:
            color_start, color_length, m = m[0], m[1], m[2:]
        else:
            color_start, color_length = None, None

        return cls(temp_entity_type, entity_num, origin, end, color_start, color_length), m


@_register_server_message
class ServerMessageKilledMonster(NoFieldsServerMessage):
    msg_type = ServerMessageType.KILLEDMONSTER


@_register_server_message
class ServerMessageIntermission(NoFieldsServerMessage):
    msg_type = ServerMessageType.INTERMISSION


@_register_server_message
class ServerMessageFinale(ServerMessage):
    field_names = ('string',)
    msg_type = ServerMessageType.FINALE

    @classmethod
    def parse(cls, m):
        s, m = cls._parse_string(m)
        return cls(s), m


@_register_server_message
class ServerMessageDisconnect(NoFieldsServerMessage):
    msg_type = ServerMessageType.DISCONNECT


@_register_server_message
class ServerMessageDamage(ServerMessage):
    field_names = ('armor', 'blood', 'origin')
    msg_type = ServerMessageType.DAMAGE

    @classmethod
    def parse(cls, m):
        armor, blood, m = m[0], m[1], m[2:]
        origin, m = cls._parse_coords(m)
        return cls(armor, blood, origin), m


def read_demo_file(f):
    while _read(f, 1) != b'\n':
        pass

    demo_header_fmt = "<Ifff"
    demo_header_size = struct.calcsize(demo_header_fmt)
    
    while True:
        d = f.read(demo_header_size)
        if len(d) == 0:
            break
        if len(d) < demo_header_size:
            raise MalformedNetworkData
        msg_len, *view_angles = struct.unpack(demo_header_fmt, d)
        msg = _read(f, msg_len)
        while msg:
            parsed, msg = ServerMessage.parse_message(msg)
            yield view_angles, parsed


def demo_parser_main():
    import sys
    with open(sys.argv[1], "rb") as f:
        for msg in read_demo_file(f):
            print(msg)


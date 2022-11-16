import datetime
import enum
import logging
import pathlib
import re
import sys

from . import proto


logger = logging.getLogger(__name__)
_SKILL_RE = r'.*Playing on (?P<skill>\w+) skill'
_TIME_RE = r'(exact time was|The recorded time was) (?P<time>\S+)'


class Stat(enum.IntEnum):
    TOTALSECRETS = 11
    TOTALMONSTERS = 12
    SECRETS = 13
    MONSTERS = 14


def _format_time(seconds):
    frac = seconds * 1e5
    seconds = frac // int(1e5)
    frac = frac % int(1e5)

    minutes = seconds // 60
    seconds = seconds % 60

    minutes = int(minutes)
    seconds = int(seconds)
    frac = int(frac)

    out = f'{seconds:02d}.{frac:05d}'
    if minutes != 0:
        out = f'{minutes}:{out}'
    return out


def demo_stats_entrypoint():
    demo_path = pathlib.Path(sys.argv[1])
    server_info_received = False
    stats = None
    time = None
    finish_time = None
    player_name = None
    joequake_time = None
    skill = None
    print_buf = ""

    def print_stats():
        formatted_time = _format_time(finish_time)

        if formatted_time != joequake_time:
            logger.warning('Joequake finish time (%s) does not match finish '
                           'time (%s)',
                          joequake_time, formatted_time)

        dt = datetime.datetime.fromtimestamp(demo_path.stat().st_ctime)

        print('---------------------------------------------')
        print(f'Runner  : {player_name}')
        print()
        print(f'Map     : {map_file} - {map_name}')
        if skill is not None:
            print(f'Skill   : {skill}')
        print(f'Kills   : {stats[Stat.MONSTERS]}/{stats[Stat.TOTALMONSTERS]}')
        print(f'Secrets : {stats[Stat.SECRETS]}/{stats[Stat.TOTALSECRETS]}')
        if finish_time is not None:
            print(f'Time    : {formatted_time}')
        print(f'Date    : {dt.day}/{dt.month}/{dt.year}')
        print('---------------------------------------------')
        print()

    with demo_path.open('rb') as f:
        for msg_end, view_angle, msg in proto.read_demo_file(f):
            if msg.msg_type == proto.ServerMessageType.SERVERINFO:
                stats = {}
                kills = 0
                secrets = 0
                finish_time = None
                player_name = None
                joequake_time = None
                skill = None
                map_file = msg.models[0].rsplit('/', 1)[1].split('.', 1)[0]
                map_name = msg.level_name

                if server_info_received:
                    print_stats()
                server_info_received = True

            if msg.msg_type == proto.ServerMessageType.TIME:
                time = msg.time

            if msg.msg_type in (proto.ServerMessageType.INTERMISSION,
                                proto.ServerMessageType.FINALE):
                if finish_time is None:
                    finish_time = time

            if msg.msg_type == proto.ServerMessageType.UPDATESTAT:
                stats[msg.index] = msg.value

            if msg.msg_type == proto.ServerMessageType.KILLEDMONSTER:
                stats[Stat.MONSTERS] += 1

            if msg.msg_type == proto.ServerMessageType.FOUNDSECRET:
                stats[Stat.SECRETS] += 1

            if msg.msg_type == proto.ServerMessageType.UPDATENAME:
                if msg.client_num != 0:
                    raise Exception("Co-op demos not supported")
                player_name = msg.name

            if msg.msg_type == proto.ServerMessageType.PRINT:
                print_buf += msg.string
                if '\n' in print_buf:
                    lines = print_buf.split('\n')
                    for line in lines[:-1]:
                        if (m := re.match(_TIME_RE, line)):
                            joequake_time = m.group('time')
                        if (m := re.match(_SKILL_RE, line)):
                            skill = m.group('skill')
                    print_buf = lines[-1]

    print_stats()

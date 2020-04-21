# Copyright (c) 2019 Matthew Earl
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

import asyncio
import collections
import logging
import math
import struct
import time

import numpy as np

from . import aiodgram
from . import proto
from . import dgram


logger = logging.getLogger()


class ClientError(Exception):
    pass


def _make_cmd_body(cmd):
    return b'\x04' + cmd.encode('ascii') + b'\0'


def _encode_angle(angle):
    return int((angle * 128 / math.pi) % 256)


def _make_move_body(pitch, yaw, roll, forward, side, up, buttons, impulse):
    return struct.pack("<BfBBBhhhBB",
                       3, 0.,
                       _encode_angle(pitch), _encode_angle(yaw), _encode_angle(roll),
                       forward, side, up, buttons, impulse)


def _patch_vec(old_vec, update):
    return tuple(v if u is None else u for v, u in zip(old_vec, update))


class Demo:
    def __init__(self):
        self._msgs = []
        self._angles = []
        self.record_pending = False
        self.recording_started = False
        self.recording_complete = False

    def start_recording(self):
        self.record_pending = True

    def stop_recording(self):
        self.record_pending = self.recording_started = False
        self.recording_complete = True

    def add_message(self, angles, msg, has_server_info):
        if self.record_pending:
            if has_server_info:
                self.record_pending = False
                self.recording_started = True
        elif self.recording_started:
            if has_server_info:
                self.recording_started = False
                self.recording_complete = True
        elif self.recording_complete:
            raise Exception("add_message called when recording is complete")

        if self.recording_started:
            self._msgs.append(msg)
            self._angles.append(angles)

    def _parse_messages(self):
        for msg in self._msgs:
            parsed_msgs = []
            remaining_msg = msg
            while remaining_msg:
                parsed, remaining_msg = proto.ServerMessage.parse_message(remaining_msg)
                parsed_msgs.append(parsed)
            yield parsed_msgs

    def dump(self, f, *, angle_calculator=None):
        f.write(b'-1\n')   # cd track
        demo_header_fmt = "<Ifff"

        if angle_calculator is not None:
            all_angles = angle_calculator.calculate(self._parse_messages())
        else:
            all_angles = self._angles

        for msg, angles in zip(self._msgs, all_angles):
            f.write(struct.pack(demo_header_fmt, len(msg), *(180. * a / math.pi for a in angles)))
            f.write(msg)


class _BaseAngleCalculator:
    def _angle_difference(self, a, b):
        t = (a - b) % (np.pi * 2.)
        return min(t, 2. * np.pi - t)

    def _get_movement_yaws(self, parsed_msg_lists):
        view_entity = None
        pos = np.zeros((3,))
        time = 0.
        move_time = None
        yaw = 0.

        for parsed_msgs in parsed_msg_lists:
            for parsed_msg in parsed_msgs:
                if parsed_msg.msg_type == proto.ServerMessageType.SETVIEW:
                    view_entity = parsed_msg.viewentity

                if parsed_msg.msg_type == proto.ServerMessageType.SPAWNBASELINE:
                    if view_entity is None:
                        raise ClientError("View entity not set but spawnbaseline received")
                    pos = parsed_msg.origin

                if parsed_msg.msg_type == proto.ServerMessageType.TIME:
                    time = parsed_msg.time

                if (parsed_msg.msg_type == proto.ServerMessageType.UPDATE and parsed_msg.entity_num == view_entity):
                    new_pos = np.array(_patch_vec(pos, parsed_msg.origin))
                    if move_time is not None:
                        vel = (new_pos - pos) / (time - move_time)
                        if np.linalg.norm(vel) > 50:
                            yaw = np.arctan2(vel[1], vel[0])
                    move_time = time
                    pos = new_pos

            yield yaw


class AngleCalculatorSmoothed(_BaseAngleCalculator):
    def _angle_cost(self, movement_yaw, yaw):
        diff = self._angle_difference(movement_yaw, yaw)
        if diff < np.pi / 8:
            return 0
        return 0.4

    def calculate(self, parsed_msg_lists):
        paths = [([], 0.)]
        for j, movement_yaw in enumerate(self._get_movement_yaws(parsed_msg_lists)):
            new_paths = []
            for yaw in np.linspace(0., 2. * np.pi, 8, endpoint=False):
                c = self._angle_cost(movement_yaw, yaw)
                new_dists = [int(bool(prev_path) and prev_path[-1] != yaw) + c + prev_dist
                             for prev_path, prev_dist in paths]
                i = np.argmin(new_dists)
                new_paths.append((paths[i][0] + [yaw], new_dists[i]))
            paths = new_paths

        best_path = min(paths, key=lambda p: p[1])[0]
        return ((0., yaw, 0.) for yaw in best_path)


class AngleCalculatorHysteresis(_BaseAngleCalculator):
    def __init__(self, hysteresis_angle):
        self._hysteresis_angle = hysteresis_angle

    def calculate(self, parsed_msg_lists):
        yaw = 0.
        for movement_yaw in self._get_movement_yaws(parsed_msg_lists):
            if self._angle_difference(movement_yaw, yaw) > self._hysteresis_angle:
                yaw = np.pi * round((4 * movement_yaw / np.pi)) / 4
            yield (0., yaw, 0.)


class AsyncClient:
    def __init__(self, conn):
        self._conn = conn
        self._spawned_fut = asyncio.Future()
        self.level_name = None
        self.view_entity = None
        self.level_finished = False
        self.time = None
        self._moved_fut = collections.defaultdict(asyncio.Future)
        self.center_print_queue = asyncio.Queue()
        self.origins = {}
        self.angles = (0., 0., 0.)
        self.velocity = (0., 0., 0.)

        self._demos = []

    @property
    def player_origin(self):
        return self.origins[self.view_entity]

    def record_demo(self):
        d = Demo()
        self._demos.append(d)
        d.start_recording()
        return d

    async def _read_messages(self):
        while True:
            remaining_msg = msg = await self._conn.read_message()

            has_server_info = False
            while remaining_msg:
                parsed, remaining_msg = proto.ServerMessage.parse_message(remaining_msg)
                logger.debug("Got message: %s", parsed)

                # Player goes "unspawned" when server info received (see SV_SendServerinfo).
                if parsed.msg_type == proto.ServerMessageType.SERVERINFO:
                    self.level_name = parsed.level_name
                    self.view_entity = None
                    self.level_finished = False
                    has_server_info = True

                # Handle sign-on.
                if parsed.msg_type == proto.ServerMessageType.SIGNONNUM:
                    if parsed.num == 1:
                        await self._conn.send_reliable(_make_cmd_body("prespawn"))
                    elif parsed.num == 2:
                        body = (_make_cmd_body('name "pyquake"\n') +
                                _make_cmd_body("color 0 0\n") +
                                _make_cmd_body("spawn "))
                        await self._conn.send_reliable(body)
                    elif parsed.num == 3:
                        await self._conn.send_reliable(_make_cmd_body("begin"))
                        logger.info("Spawned")
                        self._spawned_fut.set_result(None)
                        self._spawned_fut = asyncio.Future()

                # Set view entity
                if parsed.msg_type == proto.ServerMessageType.SETVIEW:
                    self.view_entity = parsed.viewentity

                # Set view angle
                if parsed.msg_type == proto.ServerMessageType.SETANGLE:
                    self.angles = parsed.view_angles

                # Set velocity
                if parsed.msg_type == proto.ServerMessageType.CLIENTDATA:
                    self.velocity = parsed.m_velocity

                # Update entity positions
                if parsed.msg_type == proto.ServerMessageType.SPAWNBASELINE:
                    if self.view_entity is None:
                        raise ClientError("View entity not set but spawnbaseline received")
                    self.origins[parsed.entity_num] = parsed.origin
                if parsed.msg_type == proto.ServerMessageType.UPDATE:
                    ent_num = parsed.entity_num
                    if ent_num in self.origins:
                        self.origins[ent_num] = _patch_vec(
                                self.origins[ent_num], parsed.origin)

                        if parsed.entity_num in self._moved_fut:
                            self._moved_fut[ent_num].set_result(self.origins[ent_num])
                        self._moved_fut[ent_num] = asyncio.Future()

                if parsed.msg_type == proto.ServerMessageType.PRINT:
                    logger.info("Print: %s", parsed.string)
                if parsed.msg_type == proto.ServerMessageType.CENTERPRINT:
                    logger.info("Center print: %s", parsed.string)
                    await self.center_print_queue.put(parsed.string)

                if parsed.msg_type == proto.ServerMessageType.TIME:
                    self.time = parsed.time

                if parsed.msg_type == proto.ServerMessageType.INTERMISSION:
                    self.level_finished = True

            self._demos = [d for d in self._demos if not d.recording_complete]
            for demo in self._demos:
                demo.add_message(self.angles, msg, has_server_info)

    async def wait_for_movement(self, entity_num):
        return await self._moved_fut[entity_num]

    async def wait_until_spawn(self):
        await self._spawned_fut

    @classmethod
    async def connect(cls, host, port):
        """Connect to the given host and port, and start listening for messages.

        At the point this coroutine returns, no messages have yet been read.

        """
        conn = await aiodgram.DatagramConnection.connect(host, port)
        client = cls(conn)
        asyncio.create_task(client._read_messages()).add_done_callback(
                lambda fut: fut.result)
        return client

    def move(self, pitch, yaw, roll, forward, side, up, buttons, impulse):
        self.angles = (pitch, yaw, roll)
        self._conn.send(_make_move_body(pitch, yaw, roll,
                                        forward, side, up, buttons, impulse))

    async def send_command(self, cmd):
        await self._conn.send_reliable(_make_cmd_body(cmd))

    async def disconnect(self):
        self._conn.disconnect()
        await self._conn.wait_until_disconnected()


async def _monitor_movements(client):
    while True:
        origin = await client.wait_for_movement(client.view_entity)
        logger.debug("Player moved to %s", origin)


async def _perf_benchmark(client):
    """Benchmark for measuring the move-rate with sync_movements 1."""
    start = time.perf_counter()
    for i in range(10000):
        client.move(0, 0, 0, 400, 0, 0, 0, 0)
        await client.wait_for_movement()
    logger.info("Took %s seconds", time.perf_counter() - start)


async def _aioclient():
    host, port = "localhost", 26000

    client = await AsyncClient.connect(host, port)
    demo = client.record_demo()
    logger.info("Connected to %s %s", host, port)

    asyncio.ensure_future(_monitor_movements(client)).add_done_callback(
            lambda fut: fut.result)

    try:
        await client.wait_until_spawn()

        for i in range(3):
            client.move(*client._angles, 320, 0, 0, 0, 0)
            await asyncio.sleep(1)
            client.move(*client._angles, -320, 0, 0, 0, 0)
            await asyncio.sleep(1)
            if i == 1:
                await client.send_command("kill")
                await asyncio.sleep(1)

        demo.stop_recording()
        from pprint import pprint

        def decode_msg(msg):
            while msg:
                parsed, msg = proto.ServerMessage.parse_message(msg)
                yield parsed
        pprint(list(zip(demo._angles, (list(decode_msg(msg)) for msg in demo._msgs))))
        with open("pyquake.dem", "wb") as f:
            demo.dump(f)
    finally:
        await client.disconnect()
        demo.stop_recording()


def aioclient_main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
    asyncio.run(_aioclient())


def client_main():
    logging.basicConfig(level=logging.INFO)

    host, port = "localhost", 26000

    conn = dgram.DatagramConnection.connect(host, port)
    spawned = False

    try:
        for msg in conn.iter_messages():
            while msg:
                parsed, msg = proto.ServerMessage.parse_message(msg)
                logger.debug("Got message: %s", parsed)

                if parsed.msg_type == proto.ServerMessageType.SIGNONNUM:
                    if parsed.num == 1:
                        conn.send(_make_cmd_body("prespawn"), reliable=True)
                    elif parsed.num == 2:
                        body = (_make_cmd_body('name "pyquake"\n') +
                                _make_cmd_body("color 0 0\n") +
                                _make_cmd_body("spawn "))
                        conn.send(body, reliable=True)
                    elif parsed.num == 3:
                        conn.send(_make_cmd_body("begin"), reliable=True)
                elif parsed.msg_type == proto.ServerMessageType.UPDATECOLORS:
                    spawned = True
                elif spawned:
                    conn.send(_make_move_body(0, 0, 0, 0, 0, 0, 3, 0))
    except KeyboardInterrupt:
        conn.disconnect()

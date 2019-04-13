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
import struct
import time

from . import aiodgram
from . import proto
from . import dgram


logger = logging.getLogger()


class ClientError(Exception):
    pass


def _make_cmd_body(cmd):
    return b'\x04' + cmd.encode('ascii') + b'\0'


def _make_move_body(yaw, pitch, roll, forward, side, up, buttons, impulse):
    return struct.pack("<BfBBBhhhBB",
                       3, 0.,
                       pitch, yaw, roll,
                       forward, side, up, buttons, impulse)


def _patch_vec(old_vec, update):
    return tuple(v if u is None else u for v, u in zip(old_vec, update))


class ZeroLengthQueue:
    """A Queue-like object that blocks on `put` until the item has been read."""
    def __init__(self):
        self._put_fut = asyncio.Future()
        self._get_fut = None

    async def put(self, x):
        self._put_fut.set_result(x)
        self._get_fut = asyncio.Future()
        await self._get_fut
        self._get_fut = None
        self._put_fut = asyncio.Future()

    async def get(self):
        x = await self._put_fut
        self._get_fut.set_result(None)
        return x


class Demo:
    def __init__(self):
        self.queue = ZeroLengthQueue()
        self.server_info_queue = ZeroLengthQueue()

        self._msgs = []
        self._angles = []
        self._record_task = None
        self.recording_complete = False

    async def _record(self):
        try:
            logger.info("Waiting for server info")
            await self.server_info_queue.get()
            logger.info("Recording started")

            get_server_info_task = asyncio.create_task(self.server_info_queue.get())
            get_msg_task = asyncio.create_task(self.queue.get())
            while not get_server_info_task.done():
                done, pending = await asyncio.wait([get_server_info_task, get_msg_task],
                                                   return_when=asyncio.FIRST_COMPLETED)
                assert len(done) == 1
                if get_msg_task.done():
                    angles, msg = get_msg_task.result()
                    self._angles.append(angles)
                    self._msgs.append(msg)

                    get_msg_task = asyncio.create_task(self.queue.get())
        finally:
            logger.info("Recording finished (level ended)")
            self.recording_complete = True

        get_msg_task.cancel()
        try:
            await get_msg_task
        except asyncio.CancelledError:
            pass

    async def start_recording(self):
        self._record_task = asyncio.create_task(self._record())

    async def stop_recording(self):
        self._record_task.cancel()
        try:
            await self._record_task
        except asyncio.CancelledError:
            logger.info("Recording finished (stop record called)")

    def dump(self, f, *, recompute_angles=False):
        f.write(b'-1\n')   # cd track
        demo_header_fmt = "<Ifff"
        for msg, angles in zip(self._msgs, self._angles):
            f.write(struct.pack(demo_header_fmt, len(msg), *angles))
            f.write(msg)


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
        self._angles = (0., 0., 0.)
        
        self._demos = []

    @property
    def player_origin(self):
        return self.origins[self.view_entity]

    async def record_demo(self):
        d = Demo()
        self._demos.append(d)
        await d.start_recording()
        return d

    async def _read_messages(self):
        while True:
            remaining_msg = msg = await self._conn.read_message()

            while remaining_msg:
                parsed, remaining_msg = proto.ServerMessage.parse_message(remaining_msg)
                logger.debug("Got message: %s", parsed)

                # Player goes "unspawned" when server info received (see SV_SendServerinfo).
                if parsed.msg_type == proto.ServerMessageType.SERVERINFO:
                    self.level_name = parsed.level_name
                    self.view_entity = None
                    self.level_finished = False

                    await asyncio.gather(*(demo.server_info_queue.put(None) for demo in self._demos))
                    self._demos = [d for d in self._demos if not d.recording_complete]

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
                        logging.info("Spawned")
                        self._spawned_fut.set_result(None)
                        self._spawned_fut = asyncio.Future()

                # Set view entity
                if parsed.msg_type == proto.ServerMessageType.SETVIEW:
                    self.view_entity = parsed.viewentity 

                # Set view angle
                if parsed.msg_type == proto.ServerMessageType.SETANGLE:
                    self._angles = parsed.view_angles

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
                    logging.info("Print: %s", parsed.string)
                if parsed.msg_type == proto.ServerMessageType.CENTERPRINT:
                    logging.info("Center print: %s", parsed.string)
                    await self.center_print_queue.put(parsed.string)

                if parsed.msg_type == proto.ServerMessageType.TIME:
                    self.time = parsed.time

                if parsed.msg_type == proto.ServerMessageType.INTERMISSION:
                    self.level_finished = True

            for demo in self._demos:
                await demo.queue.put((self._angles, msg))

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

    def move(self, yaw, pitch, roll, forward, side, up, buttons, impulse):
        self._angles = (yaw, pitch, roll)
        self._conn.send(_make_move_body(yaw, pitch, roll,
                                  forward, side, up, buttons, impulse))

    async def send_command(self, cmd):
        await self._conn.send_reliable(_make_cmd_body(cmd))

    async def disconnect(self):
        self._conn.disconnect()
        await self._conn.wait_until_disconnected()


async def _monitor_movements(client):
    while True:
        origin = await client.wait_for_movement()
        logging.debug("Player moved to %s", origin)


async def _perf_benchmark(client):
    """Benchmark for measuring the move-rate with sync_movements 1."""
    start = time.perf_counter()
    for i in range(10000):
        client.move(0, 0, 0, 400, 0, 0, 0, 0)
        origin = await client.wait_for_movement()
    logging.info("Took %s seconds", time.perf_counter() - start)


async def _aioclient():
    host, port = "localhost", 26000

    client = await AsyncClient.connect(host, port)
    demo = await client.record_demo()
    logger.info("Connected to %s %s", host, port)

    asyncio.ensure_future(_monitor_movements(client)).add_done_callback(
            lambda fut: fut.result)

    try:
        await client.wait_until_spawn()

        for i in range(3):
            client.move(0, 0, 0, 100, 0, 0, 0, 0)
            await asyncio.sleep(1)
            client.move(0, 0, 0, -100, 0, 0, 0, 0)
            await asyncio.sleep(1)
            if i == 1:
                await client.send_command("kill")
                await asyncio.sleep(1)
        
        await demo.stop_recording()
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
        await demo.stop_recording()


def aioclient_main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')
    asyncio.run(_aioclient())


def client_main():
    logging.getLogger().setLevel(logging.INFO)

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


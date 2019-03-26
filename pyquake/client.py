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


class AsyncClient:
    def __init__(self, conn):
        self._conn = conn
        self._spawned_fut = asyncio.Future()
        self.level_name = None
        self.view_entity = None
        self.player_origin = None
        self.time = None
        self._moved_fut = asyncio.Future()

    async def _read_messages(self):
        while True:
            msg = await self._conn.read_message()
            while msg:
                parsed, msg = proto.ServerMessage.parse_message(msg)
                logger.debug("Got message: %s", parsed)

                # Player goes "unspawned" when server info received (see SV_SendServerinfo).
                if parsed.msg_type == proto.ServerMessageType.SERVERINFO:
                    self.level_name = parsed.level_name
                    self.view_entity = None

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

                # Update player's position
                if parsed.msg_type == proto.ServerMessageType.SPAWNBASELINE:
                    if self.view_entity is None:
                        raise ClientError("View entity not set but spawnbaseline received")
                    if parsed.entity_num == self.view_entity:
                        self.player_origin = parsed.origin
                if parsed.msg_type == proto.ServerMessageType.UPDATE:
                    if self.view_entity is None:
                        raise ClientError("View entity not set but update received")
                    if parsed.entity_num == self.view_entity:
                        self.player_origin = _patch_vec(self.player_origin, parsed.origin)
                        self._moved_fut.set_result(self.player_origin)
                        self._moved_fut = asyncio.Future()

                if parsed.msg_type == proto.ServerMessageType.TIME:
                    self.time = parsed.time

    async def wait_for_movement(self):
        return await self._moved_fut

    async def wait_until_spawn(self):
        await self._spawned_fut

    @classmethod
    async def connect(cls, host, port):
        conn = await aiodgram.DatagramConnection.connect(host, port)
        client = cls(conn)
        asyncio.create_task(client._read_messages()).add_done_callback(
                lambda fut: fut.result)
        return client

    def move(self, yaw, pitch, roll, forward, side, up, buttons, impulse):
        self._conn.send(_make_move_body(yaw, pitch, roll,
                                  forward, side, up, buttons, impulse))

    async def send_command(self, cmd):
        await self._conn.send_reliable(_make_cmd_body(cmd))

    async def disconnect(self):
        self._conn.disconnect()


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
    logger.info("Connected to %s %s", host, port)

    asyncio.ensure_future(_monitor_movements(client)).add_done_callback(
            lambda fut: fut.result)

    try:
        await client.wait_until_spawn()

        while True:
            for i in range(255):
                client.move(i, 0, 0, 0, 0, 0, 0, 0)
                await asyncio.sleep(1/255.)
            await asyncio.sleep(1)
    finally:
        await client.disconnect()


def aioclient_main():
    logging.basicConfig(level=logging.INFO)
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


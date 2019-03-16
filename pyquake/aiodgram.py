# Copyright (c) 2019 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
import enum
import logging
import socket
import struct


logger = logging.getLogger(__name__)


_MAX_DATAGRAM_BODY = 32000


class _NetFlags(enum.IntFlag):
    DATA = 0x1
    ACK = 0x2
    NAK = 0x4
    EOM = 0x8
    UNRELIABLE = 0x10
    CTL = 0x8000


_SUPPORTED_FLAG_COMBINATIONS = (
    #_NetFlags.DATA,    # Should work but I don't think it's been seen in the wild
    _NetFlags.DATA | _NetFlags.EOM,
    _NetFlags.UNRELIABLE,
    _NetFlags.ACK,
)

class DatagramError(Exception):
    pass


class _GenericUdpProtocol(asyncio.Protocol):
    def __init__(self, loop):
        self._loop = loop
        self._connected_future = asyncio.Future()
        self._transport = None
        self._recv_queue = asyncio.Queue()

    def connection_made(self, transport):
        self._transport = transport
        self._connected_future.set_result(None)

    async def wait_until_connected(self):
        await self._connected_future

    def datagram_received(self, data, addr):
        asyncio.ensure_future(self._recv_queue.put((data, addr)), loop=self._loop)
        
    async def recvfrom(self):
        data, addr = await self._recv_queue.get()
        logger.debug("Received from %s: %r (%s)", addr, data, data.hex())
        return data, addr

    def sendto(self, data, addr):
        logger.debug("Sending: %r (%s)", data, data.hex())
        self._transport.sendto(data, addr)

    def connection_lost(self, exc):
        assert False, "UDP cannot close connection?"


def _raise_cb(fut):
    fut.result()


class DatagramConnection:
    def __init__(self, udp_protocol):
        self._host = None
        self._port = None

        self._udp = udp_protocol

        self._send_reliable_queue = asyncio.Queue()
        self._send_reliable_ack_queue = asyncio.Queue()
        self._message_queue = asyncio.Queue()

        self._unreliable_send_seq = 0

    async def _monitor_queues(self):
        while True:
            await asyncio.sleep(1)
            logging.debug("Dgram Queue lengths: send=%s ack=%s out=%s",
                          self._send_reliable_queue.qsize(),
                          self._send_reliable_ack_queue.qsize(),
                          self._message_queue.qsize())
                         
    async def _connect(self, host, port):
        host = socket.gethostbyname(host)

        await self._udp.wait_until_connected()

        # Request a connection, and wait for a response
        response_received = False
        while not response_received:
            logger.info("Sending connection request...")
            body = b'\x01QUAKE\x00\x03'
            header_fmt = ">HH"
            header_size = struct.calcsize(header_fmt)
            header = struct.pack(header_fmt, _NetFlags.CTL, len(body) + header_size)
            self._udp.sendto(header + body, (host, port))
            try:
                packet, addr = await asyncio.wait_for(self._udp.recvfrom(), 1.0)
                response_received = True
            except asyncio.TimeoutError:
                pass

        # Parse the response.
        if addr != (host, port):
            raise DatagramError("Spoofed packet received")
        netflags, size = struct.unpack(header_fmt, packet[:header_size])
        netflags = _NetFlags(netflags)
        body = packet[header_size:]
        if size != len(packet):
            raise DatagramError("Invalid packet size")
        if netflags != _NetFlags.CTL:
            raise DatagramError(f"Unexpected net flags: {netflags}")

        # All going well, the body should contain a new port to communicate with.
        if body[0] != 0x81:
            raise DatagramError(f"Expected CCREP_ACCEPT message, not {body[0]}")
        if len(body) != 5:
            raise DatagramError(f"Unexpected packet length {len(body)}")
        self._port, = struct.unpack("<L", body[1:])
        self._host = host
        logger.info("Connected")

        # Spin up required tasks.
        self._send_reliable_task = asyncio.create_task(self._send_reliable_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._send_reliable_task.add_done_callback(lambda fut: fut.result())
        self._recv_task.add_done_callback(lambda fut: fut.result())

        asyncio.create_task(self._monitor_queues()).add_done_callback(
                lambda fut: fut.result())

    @classmethod
    async def connect(cls, host, port):
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
                lambda: _GenericUdpProtocol(loop),
                family=socket.AF_INET)

        conn = cls(protocol)
        await conn._connect(host, port)
        return conn

    async def wait_until_disconnected(self):
        await asyncio.gather(self._send_reliable_task, self._recv_task)

    def _encap_packet(self, netflags, seq_num, payload):
        logger.debug("Sending packet: %s, %s, %s", netflags, seq_num, payload)
        header_fmt = ">HHL"
        header_size = struct.calcsize(header_fmt)
        header = struct.pack(header_fmt,
                             netflags, len(payload) + header_size, seq_num)
        return header + payload

    async def _send_reliable_loop(self):
        send_seq = 0

        while True:
            # Wait for the next message to be sent.
            data, fut = await self._send_reliable_queue.get()

            # Split the message into packets up to the maximum allowed.
            while data:
                # Send the packet
                payload, data = data[:_MAX_DATAGRAM_BODY], data[_MAX_DATAGRAM_BODY:]
                netflags = _NetFlags.DATA
                if not data:
                    netflags |= _NetFlags.EOM
                packet = self._encap_packet(netflags, send_seq, payload)
                self._udp.sendto(packet, (self._host, self._port))

                # Wait for an ACK
                while True:
                    ack_seq = await self._send_reliable_ack_queue.get()
                    if ack_seq != send_seq:
                        logger.warning("Stale ACK received")
                    else:
                        break

                send_seq += 1

            # Let the caller know the result
            fut.set_result(None)

    async def send_reliable(self, data):
        acked_future = asyncio.Future()
        await self._send_reliable_queue.put((data, acked_future))
        await acked_future

    def send(self, data):
        if len(data) > _MAX_DATAGRAM_BODY:
            raise DatagramError(f"Datagram too big: {len(body)}")
        packet = self._encap_packet(_NetFlags.UNRELIABLE,
                                    self._unreliable_send_seq,
                                    data)
        self._udp.sendto(packet, (self._host, self._port))
        self._unreliable_send_seq += 1

    def _send_ack(self, seq_num):
        packet = self._encap_packet(_NetFlags.ACK, seq_num, b'')
        self._udp.sendto(packet, (self._host, self._port))

    async def _recv_loop(self):
        header_fmt = ">HHL"
        header_size = struct.calcsize(header_fmt)

        recv_seq = 0
        unreliable_recv_seq = 0
        reliable_msg = b''

        while True:
            packet, addr = await self._udp.recvfrom()
            if addr != (self._host, self._port):
                raise DatagramError("Spoofed packet received")

            netflags, size, seq_num = struct.unpack(header_fmt, packet[:header_size])
            netflags = _NetFlags(netflags)
            body = packet[header_size:]
            logger.debug("Received packet: %s %s %s", netflags, seq_num, body)

            if len(packet) != size:
                raise DatagramError(f"Packet size {len(packet)} does not "
                                     "match header {size}")

            if netflags not in _SUPPORTED_FLAG_COMBINATIONS:
                raise DatagramError(f"Unsupported flag combination: {netflags}")

            if _NetFlags.UNRELIABLE in netflags:
                if seq_num < unreliable_recv_seq:
                    logger.warning("Stale unreliable message received")
                else:
                    if seq_num != unreliable_recv_seq:
                        logger.warning("Skipped %s unreliable messages",
                                        self._unreliable_recv_seq - seq_num)
                    unreliable_recv_seq = seq_num + 1
                    await self._message_queue.put(body)
            elif _NetFlags.ACK in netflags:
                await self._send_reliable_ack_queue.put(seq_num)
            elif _NetFlags.DATA in netflags:
                self._send_ack(seq_num)
                if seq_num != recv_seq:
                    logger.warning("Duplicate reliable message received")
                else:
                    reliable_msg += body
                    recv_seq += 1

                if _NetFlags.EOM in netflags:
                    await self._message_queue.put(reliable_msg)
                    reliable_msg = b''

    async def read_message(self):
        return await self._message_queue.get()

    def disconnect(self):
        self.send(b'\x02')


async def _async_main():
    loop = asyncio.get_running_loop()
    conn = await DatagramConnection.connect("localhost", 26000)

    await conn.wait_until_disconnected()


def main():
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(_async_main())


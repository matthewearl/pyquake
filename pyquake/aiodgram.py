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

    async def datagram_received(self, data, addr):
        await self._recv_queue.put((data, addr))
        
    async def recvfrom(self):
        return await self._recv_queue.get()

    def sendto(self, data, addr):
        self._transport.sendto(data, addr)

    def connection_lost(self, exc):
        assert False, "UDP cannot close connection?"


class DatagramConnection(_GenericUdpProtocol):
    def __init__(self, loop, udp_protocol):
        self._loop = loop

        self._host = None
        self._port = None

        self._udp = udp_protocol

        self._send_reliable_queue = asyncio.Queue()
        self._send_reliable_ack_queue = asyncio.Queue()
        self._message_queue = asyncio.Queue()

        self._unreliable_send_seq = 0

    async def _connect(self, host, port):
        host = socket.gethostbyname(host)

        await self._udp.wait_until_connected()

        # Request a connection
        body = b'\x01QUAKE\x00\x03'
        header_fmt = ">HH"
        header_size = struct.calcsize(header_fmt)
        header = struct.pack(header_fmt, _NetFlags.CTL, len(body) + header_size)
        self._udp.sendto(header + body, (host, port)

        # Wait for, and parse the response.
        packet, addr = await self._udp.recvfrom()
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

    @classmethod
    async def connect(cls, host, port, loop):
        transport, protocol = await loop.create_datagram_endpoint(
                lambda: _GenericUdpProtocol(loop))

        conn = cls(host, port, loop, protocol)
        await conn._connect(host, port)
        return conn

    def _encap_packet(self, netflags, seq_num, payload):
        logging.debug("Sending packet: %s, %s, %s", netflags, seq_num, payload)
        header_fmt = ">HHL"
        header_size = struct.calcsize(header_fmt)
        header = struct.pack(header_fmt,
                             netflags, len(payload) + header_size, seq_num)
        return header + payload

    async def _send_reliable_task(self):
        send_seq = 0

        while True:
            # Wait for the next message to be sent.
            data, fut = await self._send_reliable_queue.get()

            # Split the message into packets up to the maximum allowed.
            while data:
                # Send the packet
                payload, data = data[:_MAX_DATAGRAM_BODY], data[_MAX_DATAGRAM_BODY:]
                netflags = NetFlags.DATA
                if not data:
                    netflags |= NetFlags.EOM
                packet = self._encap_packet(netflags, send_seq, payload)
                self._udp.sendto(packet, (self.host, self.port))

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
        self._udp.sendto(packet, (self.host, self.port))
        self._unreliable_send_seq += 1

    def _send_ack(self, seq_num):
        packet = self._encap_packet(_NetFlags.ACK, seq_num, b'')
        self._udp.sendto(packet, (self.host, self.port))

    async def _recv_task(self):
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
            logging.debug("Received packet: %s %s %s", netflags, seq_num, body)

            if len(packet) != size:
                raise DatagramError(f"Packet size {len(packet)} does not "
                                     "match header {size}")

            if netflags not in _SUPPORTED_FLAG_COMBINATIONS:
                raise DatagramError(f"Unsupported flag combination: {netflags}")

            if _NetFlags.UNRELIABLE in netflags:
                if seq_num < unreliable_recv_seq:
                    logging.warning("Stale unreliable message received")
                else:
                    if seq_num != unreliable_recv_seq:
                        logging.warning("Skipped %s unreliable messages",
                                        self._unreliable_recv_seq - seq_num)
                    unreliable_recv_seq = seq_num + 1
                    await self._message_queue.put(body)
            elif _NetFlags.ACK in netflags:
                await self._send_reliable_ack_queue.put(seq_num)
            elif _NetFlags.DATA in netflags:
                self._send_ack(seq_num)
                if seq_num != self._recv_seq:
                    logging.warning("Duplicate reliable message received")
                else:
                    reliable_msg += body
                    self._recv_seq += 1

                if _NetFlags.EOM in netflags:
                    await self._message_queue.put(reliable_msg)
                    reliable_msg = b''


async def _quake_protocol(host, port, loop):
    transport, protocol = loop.create_datagram_endpoint(
            lambda: _GenericUdpProtocol(loop))

    await protocol.connected

    body = b'\x01QUAKE\x00\x03'
    header_fmt = ">HH"
    header_size = struct.calcsize(header_fmt)
    header = struct.pack(header_fmt, _NetFlags.CTL, len(body) + header_size)
    protocol.sendto(header + body, (host, port))

    data, addr = await protocol.recvfrom()
    if addr != (host, port):
        raise DatagramError("Spoofed packet received")
    netflags, size = struct.unpack(header_fmt, packet[:header_size])
    netflags = _NetFlags(netflags)
    body = packet[header_size:]
    if size != len(packet):
        raise DatagramError("Invalid packet size")

    if netflags != _NetFlags.CTL:
        raise DatagramError(f"Unexpected net flags: {netflags}")

    if body[0] != 0x81:
        raise NetworkError(f"Expected CCREP_ACCEPT message, not {body[0]}")
    if len(body) != 5:
        raise NetworkError(f"Unexpected packet length {len(body)}")

    new_port, = struct.unpack("<L", body[1:])
    return new_port


class DatagramConnection:
    def __init__(self, loop):
        self._sock = None
        self._host = None
        self._port = None

        self._send_seq = 0
        self._ack_seq = 0
        self._recv_seq = 0
        self._unreliable_send_seq = 0
        self._unreliable_recv_seq = 0

        self._loop = loop

        self.can_send = True

    async def _connect(self, host, port):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._host = socket.gethostbyname(host)
        self._port = port

        body = b'\x01QUAKE\x00\x03'
        header_fmt = ">HH"
        header_size = struct.calcsize(header_fmt)
        header = struct.pack(header_fmt, _NetFlags.CTL, len(body) + header_size)
        self._sock.sendto(header + body, (host, port))

        packet = await self._loop.sock_recv(self._sock, 1024)

        netflags, size = struct.unpack(header_fmt, packet[:header_size])
        netflags = _NetFlags(netflags)
        body = packet[header_size:]
        if size != len(packet):
            raise DatagramError("Invalid packet size")

        if netflags != _NetFlags.CTL:
            raise DatagramError(f"Unexpected net flags: {netflags}")

        if body[0] != 0x81:
            raise NetworkError(f"Expected CCREP_ACCEPT message, not {body[0]}")
        if len(body) != 5:
            raise NetworkError(f"Unexpected packet length {len(body)}")

        self._port, = struct.unpack("<L", body[1:])
        logger.debug(f"Got new port: {self._port}")

    @classmethod
    async def connect(cls, host, port, loop):
        conn = cls(loop)
        await conn._connect(host, port)


async def _async_main():
    loop = asyncio.get_running_loop()
    #conn = await DatagramConnection.connect("localhost", 26000, loop)
    await _quake_protocol("localhost", 26000, loop)


def main():
    asyncio.run(_async_main())


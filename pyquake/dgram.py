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

import enum
import io
import logging
import socket
import struct


logger = logging.getLogger(__name__)


_MAX_DATAGRAM_BODY = 32000


class NetFlags(enum.IntFlag):
    DATA = 0x1
    ACK = 0x2
    NAK = 0x4
    EOM = 0x8
    UNRELIABLE = 0x10
    CTL = 0x8000


_UNSUPPORTED_FLAG_COMBINATIONS = (
    #NetFlags.DATA,
    NetFlags.DATA | NetFlags.EOM,
    NetFlags.UNRELIABLE,
    NetFlags.ACK,
)


class DatagramError(Exception):
    pass


class DatagramConnection:
    def __init__(self, sock, host, port):
        self._sock = sock
        self._host = socket.gethostbyname(host)
        self._port = port

        self._send_seq = 0
        self._ack_seq = 0
        self._recv_seq = 0
        self._unreliable_send_seq = 0
        self._unreliable_recv_seq = 0

        self._send_buffer = b''

        self.can_send = True

    @classmethod
    def connect(cls, host, port):
        host = socket.gethostbyname(host)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        body = b'\x01QUAKE\x00\x03'
        header_fmt = ">HH"
        header_size = struct.calcsize(header_fmt)
        header = struct.pack(header_fmt, NetFlags.CTL, len(body) + header_size)
        sock.sendto(header + body, (host, port))

        packet, addr = sock.recvfrom(1024)
        if addr != (host, port):
            raise DatagramError("Spoofed packet received")
        netflags, size = struct.unpack(header_fmt, packet[:header_size])
        netflags = NetFlags(netflags)
        body = packet[header_size:]
        if size != len(packet):
            raise DatagramError("Invalid packet size")

        if netflags != NetFlags.CTL:
            raise DatagramError(f"Unexpected net flags: {netflags}")

        if body[0] != 0x81:
            raise NetworkError(f"Expected CCREP_ACCEPT message, not {body[0]}")
        if len(body) != 5:
            raise NetworkError(f"Unexpected packet length {len(body)}")

        new_port, = struct.unpack("<L", body[1:])
        return cls(sock, host, new_port)

    def _send_packet(self, netflags, seq_num, body):
        logging.debug("Sending packet: %s, %s, %s", netflags, seq_num, body)
        header_fmt = ">HHL"
        header_size = struct.calcsize(header_fmt)
        header = struct.pack(header_fmt, netflags, len(body) + header_size, seq_num)
        self._sock.sendto(header + body, (self._host, self._port))

    def _send_chunk(self):
        body = self._send_buffer[:_MAX_DATAGRAM_BODY]
        netflags = NetFlags.DATA
        if len(body) <= _MAX_DATAGRAM_BODY:
            netflags |= NetFlags.EOM
        self._send_packet(netflags, self._send_seq, body)

    def send(self, body, reliable=False):
        if not self.can_send:
            raise ValueError("can_send is False")

        if reliable:
            self._send_buffer = body
            self._send_chunk()
        else:
            if len(body) > _MAX_DATAGRAM_BODY:
                raise DatagramConnection(f"Datagram too big: {len(body)}")
            self._send_packet(NetFlags.UNRELIABLE, self._unreliable_send_seq, body)
            self._unreliable_send_seq += 1

    def iter_messages(self):
        header_fmt = ">HHL"
        header_size = struct.calcsize(header_fmt)

        recv_buffer = b''

        while True:
            packet, addr = self._sock.recvfrom(_MAX_DATAGRAM_BODY + header_size)
            if addr != (self._host, self._port):
                raise DatagramError("Spoofed packet received")

            netflags, size, seq_num = struct.unpack(header_fmt, packet[:header_size])
            netflags = NetFlags(netflags)
            body = packet[header_size:]
            logging.debug("Received packet: %s %s %s", netflags, seq_num, body)

            assert len(packet) == size

            if netflags not in _UNSUPPORTED_FLAG_COMBINATIONS:
                raise DatagramError(f"Unsupported flag combination: {netflags}")

            if NetFlags.UNRELIABLE in netflags:
                if seq_num < self._unreliable_recv_seq:
                    logging.warning("Stale unreliable message received")
                else:
                    if seq_num  != self._unreliable_recv_seq:
                        logging.warning("Skipped %s unreliable messages",
                                        self._unreliable_recv_seq - seq_num)
                    self._unreliable_recv_seq = seq_num + 1
                    yield body
            elif NetFlags.ACK in netflags:
                if seq_num != self._send_seq:
                    logging.warning("Stale ACK received")
                elif seq_num != self._ack_seq:
                    logging.warning("Duplicate ACK received")
                else:
                    self._send_seq += 1
                    self._send_buffer = self._send_buffer[_MAX_DATAGRAM_BODY:]
                    self._ack_seq += 1
                    if self._send_buffer:
                        self._send_chunk()
            elif NetFlags.DATA in netflags:
                self._send_packet(NetFlags.ACK, seq_num, b'')
                if seq_num != self._recv_seq:
                    logging.warning("Duplicate reliable message received")
                else:
                    self._recv_seq += 1
                    recv_buffer += body
                    if NetFlags.EOM in netflags:
                        yield recv_buffer
                        recv_buffer = b''


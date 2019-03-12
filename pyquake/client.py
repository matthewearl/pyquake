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

import logging
import struct

from . import proto
from . import dgram


logger = logging.getLogger()


class NetworkError(Exception):
    pass


def _make_cmd_body(cmd):
    return b'\x04' + cmd.encode('ascii') + b'\0'


def _make_move_body(yaw, pitch, roll, forward, side, up, buttons, impulse):
    return struct.pack("<BfBBBhhhBB",
                       3, 0.,
                       pitch, yaw, roll,
                       forward, side, up, buttons, impulse)


def client_main():
    logging.getLogger().setLevel(logging.WARNING)

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


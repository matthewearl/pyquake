import asyncio
import errno
import logging
import multiprocessing
import os
import socket
import subprocess

import gym
import numpy as np

from .. import client
from .. import demo
from .. import progress


logger = logging.getLogger(__name__)


_TIME_LIMIT = 15.
_QUAKE_EXE = os.path.expanduser("~/Quakespasm/quakespasm/Quake/quakespasm")
_QUAKE_OPTION_ARGS = [
    '-protocol', '15',
    '-dedicated', '1',
    '-basedir', os.path.expanduser('~/.quakespasm'),
    #'+host_framerate', '0.013888',
    '+host_framerate', '0.1',
    '+sys_ticrate', '0.0',
    '+sync_movements', '1',
    '+nomonsters', '1',
    '+map', 'e1m1',
]


def _get_quake_args(port_num):
    return [_QUAKE_EXE, '-port', str(port_num)] + _QUAKE_OPTION_ARGS


def _get_player_origins(demo_file):
    dv = demo.DemoView(demo_file, fetch_model_positions=False)
    times = np.arange(0, dv.end_time, 0.05)
    origins = np.stack([dv.get_view_at_time(t)[1] for t in times])
    return origins, times


def _get_free_udp_port(start_port_num, num_ports):
    for port_num in range(start_port_num, start_port_num + num_ports):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("localhost", port_num))
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
        else:
            return port_num
        finally:
            sock.close()


class _Client(multiprocessing.Process):
    def __init__(self, host, port, in_q, out_q):
        super().__init__()

        self._host = host
        self._port = port
        self._in_q = in_q
        self._out_q = out_q
        self._client = None
        self._client_exists = asyncio.Future()

    async def _read_from_queue(self):
        loop = asyncio.get_running_loop()

        while True:
            func, *args = await loop.run_in_executor(None, lambda: self._in_q.get())
            await getattr(self, func)(*args)

    async def _read_movements(self):
        while True:
            pos = await self._client.wait_for_movement()
            self._out_q.put((self._client.time, np.array(pos)))

    async def _run_coro(self):
        self._client = await client.AsyncClient.connect(self._host, self._port)
        self._client_exists.set_result(None)
        logger.info("Connected to %s %s", self._host, self._port)
        try:
            asyncio.ensure_future(self._read_movements()).add_done_callback(
                    lambda fut: fut.result)
            await self._client.wait_until_spawn()
            logger.info("Spawned")
            await self._read_from_queue()
        finally:
            await self._client.disconnect()

    async def move(self, yaw, pitch, roll, forward, side, up, buttons, impulse):
        await self._client_exists
        self._client.move(yaw, pitch, roll, forward, side, up, buttons, impulse)

    async def kill(self):
        await self._client_exists
        await self._client.send_command("kill")
        await self._client.wait_until_spawn()
        
    def run(self):
        asyncio.run(self._run_coro())


class GuidedEnv(gym.Env):
    def __init__(self, demo_file):
        guide_origins, _ = _get_player_origins(demo_file)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(11,),
                                                dtype=np.float32)

        self.action_space = self.get_action_space()
        self._pm = progress.ProgressMap(guide_origins, 250)

        port = _get_free_udp_port(26000, 1000)
        self._server_proc = subprocess.Popen(
                _get_quake_args(port),
                stdin=subprocess.PIPE)

        self._send_q = multiprocessing.Queue()
        self._recv_q = multiprocessing.Queue()
        self._client_proc = _Client("localhost", port, self._send_q, self._recv_q)
        self._client_proc.start()

        self._paths = []

        self._reset_episode()

    def _reset_episode(self):
        self._prev_progress = 0.
        self._prev_dist = 0.
        self._pos = None
        self._paths.append([])

    def get_action_space(self):
        """Get the action space for this environment."""
        raise NotImplementedError

    def convert_action(self, a):
        """Convert an OpenAI action into `move` command parameters."""
        raise NotImplementedError

    def step(self, a):
        while not self._recv_q.empty():
            logger.warning("Removing superfluous queue item")
            self._recv_q.get()

        self._send_q.put(("move",) + self.convert_action(a))
        time, new_pos = self._recv_q.get()

        if self._pos is not None:
            self._vel = new_pos - self._pos
        else:
            self._vel = np.zeros_like(new_pos)
        self._pos = new_pos

        assert self._recv_q.empty()

        (closest_point,), (progress,) = self._pm.get_progress(np.array([self._pos]))
        dist = (np.linalg.norm(closest_point - self._pos) / 32) ** 2
        reward = (progress - self._prev_progress) - 0 * (dist - self._prev_dist)
        self._prev_progress = progress
        self._prev_dist = dist

        #state = np.array([np.concatenate([self._pos, self._vel, [time]])])
        dir_ = self._pm.get_dir(progress)
        offset = self._pos - closest_point
        state = np.concatenate([offset,
                                self._vel,
                                dir_,
                                [progress],
                               [time]])

        state = state.squeeze()
        done = time > _TIME_LIMIT
        logger.debug("time:%.2f progress %.2f pos:%s vel:%s",
                     time, progress, self._pos, self._vel)

        info = {'time': time,
                'pos': self._pos,
                'vel': self._vel,
                'progress': progress,
                'offset': offset,
                'dist': dist,
                'dir': dir_}
        self._paths[-1].append(info)
        return state, reward, done, info

    def reset(self):
        self._send_q.put(("kill",))
        self._reset_episode()
        obs, _, _, _ = self.step(2)
        return obs

    def render(self):
        print(self._pos, self._vel)
        
    def close(self):
        self._client_proc.terminate()
        self._client_proc.join()
        self._server_proc.terminate()
        self._server_proc.wait()

class ArrowsOnlyGuidedEnv(GuidedEnv):
    def get_action_space(self):
        return gym.spaces.Discrete(4)

    def convert_action(self, a):
        if a == 0:
            v = (10000, 0)
        elif a == 1:
            v = (-10000, 0)
        elif a == 2:
            v = (0, 10000)
        elif a == 3:
            v = (0, -10000)
        
        return (0, 0, 0, *v, 0, 0, 0)


class NoJumpGuidedEnv(GuidedEnv):
    key_to_dir = [(0, -700), (400, -700), (400, 0), (400, 700), (0, 700)]
    def get_action_space(self):
        return gym.spaces.Tuple(
                    gym.spaces.Discrete(5),
                    gym.spaces.Box(0, 256, (1,)))
                    
    def convert_action(self, a):
        keys, yaw = a
        yaw = int(yaw % 256)
        dir_ = self.key_to_dir[keys]
        return (yaw, 0, 0, *dir_, 0, 0, 0)


gym.envs.registration.register(
    id='pyquake-guided-v0',
    entry_point='pyquake.rl.env:GuidedEnv',
)

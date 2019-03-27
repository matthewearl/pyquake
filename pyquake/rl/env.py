import asyncio
import errno
import logging
import multiprocessing
import os
import socket
import subprocess
import threading

import gym
import numpy as np

from .. import client
from .. import demo
from .. import progress


logger = logging.getLogger(__name__)


_TIME_LIMIT = 30.
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


class AsyncEnv(multiprocessing.Process):
    def __init__(self):
        super().__init__()
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self._server_proc = None

        self._reset_per_episode_state()

    async def _handle_rpc(self):
        loop = asyncio.get_running_loop()

        while True:
            func, args = await loop.run_in_executor(None, lambda: self.child_conn.recv())
            return_val = await getattr(self, func)(*args)
            self.child_conn.send(return_val)

    async def _run_coro(self):
        port = _get_free_udp_port(26000, 1000)
        server_proc = await asyncio.create_subprocess_exec(*_get_quake_args(port),
                                stdin=subprocess.PIPE)
        self._client = await client.AsyncClient.connect("localhost", port)
        logger.info("Connected to %s %s", "localhost", port)
        try:
            await self._client.wait_until_spawn()
            await self._handle_rpc()
        finally:
            await self._client.disconnect()
            server_proc.terminate()
            await server_proc.wait()

    def run(self):
        async def create_and_run_task():
            self._coro = asyncio.create_task(self._run_coro())
            try:
                await self._coro
            except asyncio.CancelledError:
                pass
        asyncio.run(create_and_run_task())

    async def step(self, a):
        raise NotImplementedError

    def _get_initial_observation(self):
        raise NotImplementedError

    def _reset_per_episode_state(self):
        raise NotImplementedError

    async def reset(self):
        spawn_fut = self._client.wait_until_spawn()
        await self._client.send_command("kill")
        await spawn_fut
        self._reset_per_episode_state()
        return self._get_initial_observation()

    async def close(self):
        self._coro.cancel()


class AsyncEnvAdaptor(gym.Env):
    """Turn an async env into a gym env."""
    def __init__(self, async_env):
        self._async_env_proc = async_env

        self.action_space = self._async_env_proc.action_space
        self.observation_space = self._async_env_proc.observation_space

        self._rpc_lock = threading.Lock()

        self._async_env_proc.start()

    def _make_rpc_call(self, method, args):
        with self._rpc_lock:
            self._async_env_proc.parent_conn.send((method, args))
            result = self._async_env_proc.parent_conn.recv()
        return result

    def step(self, a):
        return self._make_rpc_call('step', (a,))

    def reset(self):
        return self._make_rpc_call('reset', ())

    def render(self):
        return self._make_rpc_call('render', ())

    def close(self):
        self._make_rpc_call('close', ())
        self._async_env_proc.join()


class AsyncGuidedEnv(AsyncEnv):
    key_to_dir = [(0, -1000), (1000, -1000), (1000, 0), (1000, 1000), (0, 1000),
                  (-1000, 1000), (-1000, 0), (-1000, -1000)]
    action_space = gym.spaces.Discrete(8)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)

    def __init__(self, demo_file):
        super().__init__()

        guide_origins, _ = _get_player_origins(demo_file)
        self._pm = progress.ProgressMap(guide_origins, 250)

    def _action_to_move_args(self, a):
        return (0, 0, 0, *self.key_to_dir[a], 0, 0, 0)

    async def step(self, a):
        self._client.move(*self._action_to_move_args(a))
        await self._client.wait_for_movement()
        return self._get_step_return_and_update()

    def _reset_per_episode_state(self):
        self._prev_progress = 0.
        self._prev_dist = 0.
        self._old_pos = None

    def _get_step_return_and_update(self):
        pos = np.array(self._client.player_origin)
        if self._old_pos is not None:
            vel = pos - self._old_pos
        else:
            vel = np.zeros_like(pos)

        (closest_point,), (progress,) = self._pm.get_progress(np.array([pos]))
        dir_ = self._pm.get_dir(progress)
        offset = pos - closest_point
        obs = np.concatenate([offset, vel, dir_, [progress], [self._client.time]])

        reward = (progress - self._prev_progress)

        done = self._client.time > _TIME_LIMIT

        info = {'time': self._client.time,
                'pos': pos,
                'vel': vel,
                'progress': progress,
                'offset': offset,
                'dir': dir_}

        self._old_pos = pos
        self._prev_progress = progress

        return obs, reward, done, info

    def _get_initial_observation(self):
        obs, reward, done, info = self._get_step_return_and_update()
        return obs


gym.envs.registration.register(
    id='pyquake-arrows-only-v0',
    entry_point='pyquake.rl.env:AsyncGuidedEnv',
)

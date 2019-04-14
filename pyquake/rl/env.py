import asyncio
import errno
import logging
import multiprocessing
import os
import pickle
import socket
import subprocess
import threading

import gym
import numpy as np

from .. import client
from .. import demo
from .. import progress


logger = logging.getLogger(__name__)


_TIME_LIMIT = 35.
_QUAKE_EXE = os.path.expanduser("~/quakespasm/quakespasm/Quake/quakespasm")
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
        self._client = None

    async def _handle_rpc(self):
        loop = asyncio.get_running_loop()

        while True:
            func, args = await loop.run_in_executor(None, lambda: self.child_conn.recv())
            return_val = await getattr(self, func)(*args)
            self.child_conn.send(return_val)

    async def _run_coro(self):
        # TODO: Obtain a file lock around checking the port and creating the server, to avoid race conditions, and get
        # rid of the os.getpid() hack.
        import random
        port = 26000 + random.randint(0, 1000 - 1)
        port = _get_free_udp_port(port, 1000)

        server_proc = await asyncio.create_subprocess_exec(
                                *_get_quake_args(port),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                stdin=subprocess.PIPE)
        self._client = await client.AsyncClient.connect("localhost", port)
        logger.info("Connected to %s %s", "localhost", port)
        try:
            self._before_spawn()
            await self._client.wait_until_spawn()
            self._reset_per_episode_state()
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

    async def _get_initial_observation(self):
        raise NotImplementedError

    def _reset_per_episode_state(self):
        raise NotImplementedError

    def _on_episode_end(self):
        raise NotImplementedError

    def _before_spawn(self):
        raise NotImplementedError

    async def reset(self):
        self._on_episode_end()
        self._before_spawn()
        spawn_fut = self._client.wait_until_spawn()
        await self._client.send_command("kill")
        await spawn_fut
        obs = await self._get_initial_observation()
        self._reset_per_episode_state()
        return obs

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
        self._paths = []
        self._current_path = None

    def _make_rpc_call(self, method, args):
        with self._rpc_lock:
            self._async_env_proc.parent_conn.send((method, args))
            result = self._async_env_proc.parent_conn.recv()
        return result

    def step(self, a):
        obs, reward, done, info = self._make_rpc_call('step', (a,))
        if self._current_path is None:
            self._current_path = []
            self._paths.append(self._current_path)
        self._current_path.append(info)
        return obs, reward, done, info

    def reset(self):
        self._current_path = None
        return self._make_rpc_call('reset', ())

    def render(self):
        return self._make_rpc_call('render', ())

    def close(self):
        self._make_rpc_call('close', ())
        self._async_env_proc.join()


class AsyncGuidedEnv(AsyncEnv):
    key_to_dir = [(0, -1000),     # 0: Forward
                  (1000, -1000),  # 1: Forward-right
                  (1000, 0),      # 2: Right
                  (1000, 1000),   # 3: Back-right
                  (0, 1000),      # 4: Back
                  (-1000, 1000),  # 5: Back-left
                  (-1000, 0),     # 6: Left
                  (-1000, -1000)] # 7: Forward-left
    action_space = gym.spaces.Discrete(8)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)

    center_print_rewards = {
    }

    center_print_progress = {
    }

    movement_rewards = {
        25: 3000,  # bridge button
        62: 3000,  # spiral button 1
        60: 3000,  # spiral button 2
        61: 3000,  # spiral button 3
    }

    movement_progress = {
        25: 4088,
        62: 5795,  # spiral button 1
        60: 6273,  # spiral button 2
        61: 6734,  # spiral button 3
    }

    def __init__(self, demo_file):
        super().__init__()

        guide_origins, _ = _get_player_origins(demo_file)
        self._pm = progress.ProgressMap(guide_origins, 250)

        self._highest_reward = None
        self._total_reward = None
        self._demo = None

    def _action_to_move_args(self, a):
        return (0, 0, 0, *self.key_to_dir[a], 0, 0, 0)

    async def step(self, a):
        self._client.move(*self._action_to_move_args(a))
        await self._client.wait_for_movement(self._client.view_entity)
        obs, reward, done, info = await self._get_step_return_and_update()
        self._total_reward += reward
        return obs, reward, done, info

    def _before_spawn(self):
        self._demo = self._client.record_demo()

    def _on_episode_end(self):
        self._demo.stop_recording()

        if self._highest_reward is None or self._total_reward > self._highest_reward:
            self._highest_reward = self._total_reward
            with open(f"demos/reward_{self._highest_reward:08.02f}.demo.pickle", "wb") as f:
                pickle.dump(self._demo, f)

    def _reset_per_episode_state(self):
        #self._prev_progress = 5464.89   # spiral
        #self._prev_progress = 3688.33527211  # draw bridge
        self._prev_progress = 0.
        self._prev_dist = 0.
        self._old_pos = None
        self._center_prints_seen = set()
        self._moved = set()
        self._total_reward = 0.

    def _limit_progress_by_center_prints(self, progress):
        for k, v in self.center_print_progress.items():
            if k not in self._center_prints_seen:
                progress = min(progress, v)
        return progress

    def _limit_progress_by_moved(self, progress):
        for k, v in self.movement_progress.items():
            moved = (self._client.origins[k] != (0., 0., 0.))
            if not moved:
                progress = min(progress, v)
        return progress

    def _get_movement_rewards(self):
        reward = 0
        for k, v in self.movement_rewards.items():
            if (k not in self._moved and
                    self._client.origins[k] != (0., 0., 0.)):
                reward += v
                self._moved |= {k}

        return reward

    async def _get_center_print_reward(self):
        print_queue = self._client.center_print_queue
        reward = 0
        while print_queue.qsize():
            string = await print_queue.get()
            for k, v in self.center_print_rewards.items():
                if k in string:
                    logging.info("Center print reward: %r %s", k, v)
                    reward += v
                    self._center_prints_seen |= {k}
        return reward

    async def _get_step_return_and_update(self):
        pos = np.array(self._client.player_origin)
        if self._old_pos is not None:
            vel = pos - self._old_pos
        else:
            vel = np.zeros_like(pos)

        (closest_point,), (progress,) = self._pm.get_progress(np.array([pos]))
        progress = self._limit_progress_by_center_prints(progress)
        progress = self._limit_progress_by_moved(progress)
        if self._client.level_finished:
            logger.warning("LEVEL FINISHED %s", self._client.time)
            progress = self._pm.get_distance()
        closest_point = self._pm.get_pos(progress)
        dir_ = self._pm.get_dir(progress)
        offset = pos - closest_point
        dist = np.linalg.norm(offset)
        obs = np.concatenate([pos, vel,
                              [k in self._moved
                                  for k in self.movement_progress],
                              [self._client.time]])
        #obs = np.concatenate([offset, vel, dir_,
        #                      [progress],
        #                      [len(self._moved)],
        #                      [len(self._center_prints_seen)],
        #                      [self._client.time]])

        reward = ((progress - self._prev_progress) +
                  self._get_movement_rewards() +
                  await self._get_center_print_reward() -
                  1. * (dist - self._prev_dist))
        if self._client.level_finished:
            reward += 100

        done = self._client.time > _TIME_LIMIT

        info = {'time': self._client.time,
                'pos': pos,
                'vel': vel,
                'progress': progress,
                'offset': offset,
                'dir': dir_,
                'obs': obs,
                'reward': reward,
                'moved': list(self._moved),
                #'center_prints_seen': list(self._center_prints_seen),
                'finished': self._client.level_finished}

        self._old_pos = pos
        self._prev_progress = progress
        self._prev_dist = dist

        return obs, reward, done, info

    async def _get_initial_observation(self):
        obs, reward, done, info = await self._get_step_return_and_update()
        return obs


gym.envs.registration.register(
    id='pyquake-v0',
    entry_point='pyquake.rl.env:AsyncEnvAdaptor',
)

# Copyright (c) 2018 Matthew Earl
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


__all__ = (
        'DemoView',
        'NoServerInfoInDemo',
        'ViewGen',
)


import contextlib
import copy
import functools
import hashlib
import json
import logging
import os
import pickle
import re
import sys

import colorama
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import inotify_simple

from . import proto
from . import progress


logger = logging.getLogger(__name__)


class NoServerInfoInDemo(Exception):
    pass


class _DemoCacheMiss(Exception):
    pass


def _patch_vec(old_vec, update):
    return tuple(v if u is None else u for v, u in zip(old_vec, update))


class ViewGen:
    def __init__(self, demo_file, fetch_model_positions):
        self.complete = None
        self.map_name = None
        self._demo_file = demo_file
        self._fetch_model_positions = fetch_model_positions
        self._iter_called = False

    def _view_gen_unwrapped(self):
        # Like _view_gen, except the final frame is an intermission frame. This is dropped from `_view_gen`.
        time = None
        entity_num_to_model_num = {}
        serverinfo_rcvd = False
        self.complete = False
        for view_angle, msg in proto.read_demo_file(self._demo_file):
            if msg.msg_type == proto.ServerMessageType.SERVERINFO:
                if serverinfo_rcvd:
                    break
                self.map_name = msg.models[0]
                if self._fetch_model_positions:
                    model_nums = [idx + 1 for idx, model_name in enumerate(msg.models)
                                          if model_name[0] == '*' or idx == 0]
                else:
                    model_nums = []
                model_num_idx = {model_num: idx + 1 for idx, model_num in enumerate(model_nums)}
                pos = np.zeros((1 + len(model_nums), 3))
                serverinfo_rcvd = True

            if not serverinfo_rcvd:
                continue

            if msg.msg_type == proto.ServerMessageType.SPAWNBASELINE and msg.model_num in model_nums:
                entity_num_to_model_num[msg.entity_num] = msg.model_num
            if (msg.msg_type in (proto.ServerMessageType.UPDATE, proto.ServerMessageType.SPAWNBASELINE)  and
                (msg.entity_num == 1 or msg.entity_num in entity_num_to_model_num)):
                if msg.entity_num == 1:
                    idx = 0
                else:
                    idx = model_num_idx[entity_num_to_model_num[msg.entity_num]]
                pos[idx] = _patch_vec(pos[idx], msg.origin)
            elif msg.msg_type == proto.ServerMessageType.TIME:
                if time is not None and all(x is not None for x in pos):
                    yield view_angle, copy.copy(pos), time
                time = msg.time
            elif msg.msg_type == proto.ServerMessageType.INTERMISSION:
                self.complete = True
                break

        if not serverinfo_rcvd:
            raise NoServerInfoInDemo("No server info message in {}".format(demo_file_name))

    def _view_gen(self):
        """Generate view angles, player positions, and map entity positions.

        Angles/positions are generated between the first server info message and the intermission message.

        """
        prev_t = None
        for t in self._view_gen_unwrapped():
            if prev_t is not None:
                yield t
            prev_t = t

    def __iter__(self):
        if self._iter_called:
            raise Exception("Cannot iterate a ViewGen twice")
        return self._view_gen()


class DemoView:
    def __init__(self, demo_file_name, fetch_model_positions=True):
        cache_fname = self._get_cache_filename(demo_file_name, fetch_model_positions)

        try:
            logger.info("Reading demo %s from cache %s", demo_file_name, cache_fname)
            view_angles, positions, times, self.map_name, self.complete = self._check_cache(cache_fname)
        except _DemoCacheMiss:
            logger.info("Reading demo %s", demo_file_name)
            with open(demo_file_name, 'rb') as f:
                view_gen = ViewGen(f, fetch_model_positions)
                data = list(view_gen)
                self.map_name = view_gen.map_name
                self.complete = view_gen.complete
            view_angles, positions, times = (np.array([x[i] for x in data]) for i in range(3))
            self._set_cache(cache_fname, view_angles, positions, times, self.map_name)

        self._view_angle_interp = scipy.interpolate.interp1d(times, view_angles,
                                                             axis=0,
                                                             bounds_error=False,
                                                             fill_value=(view_angles[0], view_angles[-1]))
        self._pos_interp = scipy.interpolate.interp1d(times, positions,
                                                      axis=0,
                                                      bounds_error=False,
                                                      fill_value=(positions[0], positions[-1]))

        self._end_time = times[-1]

    def _get_cache_filename(self, demo_file_name, fetch_model_positions=True):
        with open(demo_file_name, "rb") as f:
            s = f.read()
        return os.path.join("democache", "{}{}.pickle".format(hashlib.sha1(s).hexdigest(),
                                                              "_no_models" if not fetch_model_positions else ""))

    def _check_cache(self, cache_fname):
        try:
            with open(cache_fname, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise _DemoCacheMiss

    def _set_cache(self, cache_fname, view_angles, positions, times, map_name):
        with open(cache_fname, "wb") as f:
            pickle.dump((view_angles, positions, times, map_name, self.complete), f)

    def get_view_at_time(self, t):
        all_pos = self._pos_interp(t)
        return self._view_angle_interp(t), all_pos[0], all_pos[1:]

    @property
    def end_time(self):
        return self._end_time


class _WaitingFile:
    """A file-like object that waits for a blocks for a file to be written to when it runs out of data to read"""
    def __init__(self, dm, name, mode):
        self._f = open(f"{dm.dname}/{name}", mode)
        self._dm = dm
        self._name = name
        self._closed = False

    def close(self):
        self._f.close()

    def read(self, n):
        pos = self._f.tell()
        sz = os.stat(self._f.fileno()).st_size
        while not self._closed and sz < pos + n:
            ev = None
            while (ev is None or ev.name != self._name or
                   not ev.mask & (inotify_simple.flags.CLOSE_WRITE | inotify_simple.flags.MODIFY)):
                ev = self._dm.wait_for_event()

            if ev.mask == inotify_simple.flags.CLOSE_WRITE:
                logger.info("File %s closed", self._name)
                self._closed = True
            elif ev.mask == inotify_simple.flags.MODIFY:
                sz = os.stat(self._f.fileno()).st_size
            else:
                assert False
        return self._f.read(n)


class _DirMonitor:
    """Read pertinent inotify events from a directory, one at a time"""
    def __init__(self, dname):
        self._ino = inotify_simple.INotify()
        self._wd = self._ino.add_watch(dname, inotify_simple.flags.CREATE |
                                        inotify_simple.flags.MODIFY |
                                        inotify_simple.flags.CLOSE_WRITE)
        self._unhandled_events = []
        self.dname = dname

    def wait_for_event(self):
        """Return an event from the queue"""
        while not self._unhandled_events:
            self._unhandled_events.extend(self._ino.read(read_delay=500))
        ev = self._unhandled_events.pop(0)
        return ev

    def close(self):
        self._ino.rm_watch(self._wd)

    @classmethod
    @contextlib.contextmanager
    def open(cls, dname):
        dm = cls(dname)
        try:
            yield dm
        finally:
            dm.close()

    @contextlib.contextmanager
    def open_file(self, name, mode):
        wf = _WaitingFile(self, name, mode)
        try:
            yield wf
        finally:
            wf.close()


def _detect_first_movements():
    moved_now = None
    first_pos = None
    moved_before = set()
    while True:
        pos = yield moved_now
        if first_pos is None:
            first_pos = pos
        moved_now = set(np.where(np.any(pos != first_pos, axis=1))[0]) - moved_before
        moved_before |= moved_now


@functools.lru_cache(None)
def _load_reference_demo(demo_file_name, expected_map_name, checkpoints):
    checkpoints = dict(checkpoints)
    logger.info('Loading reference times from %s', demo_file_name)

    move_detector = _detect_first_movements()
    assert next(move_detector) is None

    positions = []
    times = []

    with open(demo_file_name, 'rb') as f:
        view_gen = ViewGen(f, fetch_model_positions=True)
        ref_checkpoint_times = {}
        last_checkpoint_time = 0
        for view_angles, pos, time in view_gen:
            positions.append(pos[0])
            times.append(time)
            if view_gen.map_name is not None and view_gen.map_name != expected_map_name:
                raise Exception(f"Reference demo is for map {view_gen.map_name}, not {expected_map_name}")
            moved_now = move_detector.send(pos)
            for model_num in moved_now:
                if model_num in checkpoints:
                    checkpoint_name = checkpoints[model_num]
                    ref_checkpoint_times[checkpoint_name] = {'time': time, 'segment_time': time - last_checkpoint_time}
                    last_checkpoint_time = time

        logger.info('Loaded reference times (%s)', ref_checkpoint_times)

    return ref_checkpoint_times, np.stack(times), np.stack(positions)


def _format_time_delta(d):
    if d > 0.1:
        col = colorama.Fore.RED
    elif d < -0.1:
        col = colorama.Fore.GREEN
    else:
        col = colorama.Fore.WHITE
    return f"{col}{colorama.Style.BRIGHT}{d:+.2f}{colorama.Style.RESET_ALL}"


class _ComparisonPlot:
    def __init__(self, num_old_runs=5):
        self._num_old_runs = num_old_runs

        self._runs = []
        self._fig, _ = plt.subplots()
        self._fig.set_figwidth(12)
        self._fig.set_figheight(12)
        self._map_name = None
        self._pm_cache = {}
        self._pm = None
        self._checkpoint_dists = {}
        self._ref_dist_times = None

    def update_plot(self):
        ax, = self._fig.axes

        ax.clear()
        
        if self._ref_dist_times is not None:
            ax.grid(True)
            ax.tick_params(axis='x', labelrotation=90)
            ax.set_xticks(np.arange(0, len(self._ref_dist_times), 200))

            ax.set_xlabel('distance')
            ax.set_ylabel('time ahead of reference')

            for i, r in enumerate(self._runs):
                if i == len(self._runs) - 1:
                    color = 'red'
                    alpha = 1.0
                else:
                    color = 'gray'
                    alpha = i / (len(self._runs)  - 1)
                ax.plot(r, color=color, alpha=alpha)

            for label, t in self._checkpoint_dists.items():
                ax.axvline(t, alpha=0.5)
                ax.text(t + 0.1, 0, label, rotation=90)

        plt.show(block=False)
        plt.pause(0.001)

    def _distance_to_time(self, pm, times, origins):
        _, dists = pm.get_progress(origins)
        return np.interp(np.arange(dists[-1]), dists, times)

    def set_ref(self, ref_checkpoint_times, ref_times, ref_origins, map_name):
        if map_name != self._map_name:
            # Changed map
            self._runs = []
            self._map_name = map_name

        if id(ref_origins) not in self._pm_cache:
            self._pm_cache[id(ref_origins)] = progress.ProgressMap(np.stack(ref_origins), 250)
        self._pm = self._pm_cache[id(ref_origins)]

        self._ref_dist_times = self._distance_to_time(self._pm, ref_times, ref_origins)

        for label, d in ref_checkpoint_times.items():
            t = d['time']
            self._checkpoint_dists[label] = np.interp(t, self._ref_dist_times, np.arange(len(self._ref_dist_times)))

    def add_run(self, times, origins):
        dist_times = self._distance_to_time(self._pm, times, origins)
        n = min(len(dist_times), len(self._ref_dist_times))
        self._runs.append(dist_times[:n] - self._ref_dist_times[:n])
        self._runs = self._runs[-self._num_old_runs:]
        self.update_plot()


def _monitor_demo_file(f, demo_dir_name, checkpoint_info, comparison_plot):
    last_checkpoint_time = 0
    checkpoints = None

    move_detector = _detect_first_movements()
    assert next(move_detector) is None

    view_gen = ViewGen(f, fetch_model_positions=True)
    times = []
    positions = []
    for view_angles, pos, time in view_gen:
        times.append(time)
        positions.append(pos[0])
        map_name = view_gen.map_name
        if map_name and checkpoints is None:
            if map_name in checkpoint_info['checkpoints']:
                checkpoints = {model_num: name for name, model_num in checkpoint_info['checkpoints'][map_name].items()}
            else:
                logger.warning("No checkpoints found for map %s", view_gen.map_name)
                checkpoints = {}

            if map_name in checkpoint_info['refs']:
                demo_file_name = os.path.expanduser(checkpoint_info['refs'][map_name])
                ref_checkpoint_times, ref_times, ref_origins = _load_reference_demo(demo_file_name, map_name,
                                                                                    tuple(checkpoints.items()))
                comparison_plot.set_ref(ref_checkpoint_times, ref_times, ref_origins, map_name)
            else:
                ref_checkpoint_times, ref_times, ref_origins = {}, None, None

        moved_now = move_detector.send(pos)
        if moved_now:
            logger.debug("Models %s moved at %s", moved_now, time)
        if checkpoints is not None:
            for model_num in moved_now:
                if model_num in checkpoints:
                    checkpoint_name = checkpoints[model_num]
                    if checkpoint_name not in ref_checkpoint_times:
                        raise Exception(f"Checkpoint {checkpoint_name} not found in ref times: {ref_checkpoint_times}")
                    segment_time = time - last_checkpoint_time
                    delta_time = time - ref_checkpoint_times[checkpoint_name]['time']
                    delta_segment_time = segment_time - ref_checkpoint_times[checkpoint_name]['segment_time']
                    print(f"checkpoint: {checkpoint_name:>20} "
                          f"segment time: {segment_time:.2f} ({_format_time_delta(delta_segment_time)}) "
                          f"time: {time:.2f} ({_format_time_delta(delta_time)})")
                    last_checkpoint_time = time

    logger.debug('Plotting run')
    if ref_times is not None:
        comparison_plot.add_run(np.stack(times), np.stack(positions))

    logger.debug('Waiting for underlying file to close')
    while len(f.read(1 << 20)) == (1 << 20):
        pass

    logger.info('Finished reading file')


_DEMO_RE = r'(x|demo(\d)+)\.dem'

def monitor_demos():
    import sys

    dname, checkpoint_info_fname = sys.argv[1:]

    with open(checkpoint_info_fname) as f:
        checkpoint_info = json.load(f)

    cp = _ComparisonPlot()
    cp.update_plot()

    logging.basicConfig(level=logging.INFO)

    logger.info("Watching directory %s", dname)
    with _DirMonitor.open(dname) as dm:
        while True:
            logger.info("Waiting for file to be created")
            ev = None
            while (ev is None or
                   not ev.mask & (inotify_simple.flags.CREATE | inotify_simple.flags.MODIFY) or
                   not re.match(_DEMO_RE, ev.name)):
                ev = dm.wait_for_event()
            name = ev.name
            logger.info('File %s created. Opening.', name)
            with dm.open_file(name, 'rb') as f:
                _monitor_demo_file(f, dname, checkpoint_info, cp)


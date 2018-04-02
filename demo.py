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
)


import copy
import hashlib
import logging
import os
import pickle

import numpy as np
import scipy.interpolate

import proto


class NoServerInfoInDemo(Exception):
    pass


class _DemoCacheMiss(Exception):
    pass


class DemoView:
    def __init__(self, demo_file_name, fetch_model_positions=True):
        cache_fname = self._get_cache_filename(demo_file_name, fetch_model_positions)

        try:
            logging.info("Reading demo %s from cache %s", demo_file_name, cache_fname)
            view_angles, positions, times, self.map_name, self.complete = self._check_cache(cache_fname)
        except _DemoCacheMiss:
            logging.info("Reading demo %s", demo_file_name)
            data = list(self._view_gen(demo_file_name, fetch_model_positions))
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

    def _patch_vec(self, old_vec, update):
        return tuple(v if u is None else u for v, u in zip(old_vec, update))
                
    def _view_gen(self, demo_file_name, fetch_model_positions):
        """Generate view angles, player positions, and map entity positions.

        Angles/positions are genrated between the first server info message and the intermission message.

        """
        prev_t = None
        for t in self._view_gen_unwrapped(demo_file_name, fetch_model_positions):
            if prev_t is not None:
                yield t
            prev_t = t
        
    def _view_gen_unwrapped(self, demo_file_name, fetch_model_positions):
        # Like _view_gen, except the final frame is an intermission frame. This is dropped from `_view_gen`.
        with open(demo_file_name, "rb") as f:
            time = None
            entity_num_to_model_num = {}
            serverinfo_rcvd = False
            self.complete = False
            for view_angle, msg in proto.read_demo_file(f):
                if msg.msg_type == proto.ServerMessageType.SERVERINFO:
                    if serverinfo_rcvd:
                        break
                    self.map_name = msg.models[0]
                    if fetch_model_positions:
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
                    pos[idx] = self._patch_vec(pos[idx], msg.origin)
                elif msg.msg_type == proto.ServerMessageType.TIME:
                    if time is not None and all(x is not None for x in pos):
                        yield view_angle, copy.copy(pos), time
                    time = msg.time
                elif msg.msg_type == proto.ServerMessageType.INTERMISSION:
                    self.complete = True
                    break
        if not serverinfo_rcvd:
            raise NoServerInfoInDemo("No server info message in {}".format(demo_file_name))

    def get_view_at_time(self, t):
        all_pos = self._pos_interp(t)
        return self._view_angle_interp(t), all_pos[0], all_pos[1:]

    @property
    def end_time(self):
        return self._end_time



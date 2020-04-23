# Copyright (c) 2020 Matthew Earl
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

import numpy as np

import gym


class TestEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,),
                                                dtype=np.float32)

        self.reset()

    @property
    def state(self):
        return np.array([self._pos, self._vel, self._time])

    def step(self, a):
        self._time += 1
        done = self._time > 10

        # Acceleration due to input
        if a == 3:
            self._vel = min(self._vel + 32, 42)
        elif a == 2:
            self._vel = max(self._vel - 32, -42)

        # Friction
        if self._vel > 0:
            self._vel = max(self._vel - 10, 0)
        elif self._vel < 0:
            self._vel = min(self._vel + 10, 0)

        reward = self._vel
        self._pos += self._vel
        return self.state, reward, done, {}

    def reset(self):
        self._time = 0
        self._vel = 0
        self._pos = 0

        return self.state


gym.envs.registration.register(
    id='pyquake-testenv-v0',
    entry_point='pyquake.rl.testenv:TestEnv',
)


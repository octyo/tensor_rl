"""N-dimensional discrete navigation environment.

A grid-world generalized to N dimensions. The agent starts at a random position
and must reach a fixed goal. Each dimension has a configurable size.

Observation: one-hot encoding per dimension, concatenated.
  For dims=(5,5,5,5,5,5,5,5,5,5), obs is 50-dimensional (10 dims × 5 values).
  Structured obs is a tuple of per-dimension one-hot vectors.

Actions: 2*N (move +1 or -1 along each dimension, clamped to grid bounds).

Reward: +1.0 at goal, with optional distance-based shaping. Episode ends on goal or max_steps.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NDimNavEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, dims=(3,) * 10, max_steps=200, goal=None, reward_shaping=True):
        super().__init__()
        self.dims = tuple(dims)
        self.n_dims = len(dims)
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping

        if goal is not None:
            self.goal = tuple(goal)
        else:
            self.goal = tuple(d - 1 for d in dims)

        self.action_space = spaces.Discrete(2 * self.n_dims)

        obs_size = sum(dims)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.pos = None
        self.steps = 0
        self._prev_dist = None

    def _obs(self):
        parts = []
        for i, d in enumerate(self.dims):
            oh = np.zeros(d, dtype=np.float32)
            oh[self.pos[i]] = 1.0
            parts.append(oh)
        return np.concatenate(parts)

    def _obs_structured(self):
        parts = []
        for i, d in enumerate(self.dims):
            oh = np.zeros(d, dtype=np.float32)
            oh[self.pos[i]] = 1.0
            parts.append(oh)
        return parts

    def _manhattan(self):
        return sum(abs(p - g) for p, g in zip(self.pos, self.goal))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = list(self.np_random.integers(0, d, size=1)[0] for d in self.dims)
        while tuple(self.pos) == self.goal:
            self.pos = list(self.np_random.integers(0, d, size=1)[0] for d in self.dims)
        self.steps = 0
        self._prev_dist = self._manhattan()
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        dim_idx = action // 2
        direction = 1 if action % 2 == 0 else -1
        self.pos[dim_idx] = max(0, min(self.dims[dim_idx] - 1, self.pos[dim_idx] + direction))

        done = tuple(self.pos) == self.goal
        truncated = self.steps >= self.max_steps
        if done:
            reward = 1.0
        elif self.reward_shaping:
            dist = self._manhattan()
            reward = 0.05 * (self._prev_dist - dist)
            self._prev_dist = dist
        else:
            reward = 0.0
        return self._obs(), reward, done, truncated, {}


def make_ndim_env(dims=(3,) * 10, max_steps=200, reward_shaping=True):
    env = NDimNavEnv(dims=dims, max_steps=max_steps, reward_shaping=reward_shaping)
    return env

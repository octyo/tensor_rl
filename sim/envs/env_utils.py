import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
import minigrid


def make_env_cnn(env_id: str):
    """Creates an environment with spatial (image-like) observations for CNN inputs.

    Returns (env, obs_shape) where obs_shape is channels-first (C, H, W).

    For MiniGrid: applies ImgObsWrapper (H, W, C) and transposes to (C, H, W).
    For other envs: falls back to flat obs wrapped as (1, 1, N) — CNN degenerates to linear.

    The returned env yields observations in channels-first format (C, H, W).
    """
    env = gym.make(env_id)
    if "MiniGrid" in env_id:
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
        h, w, c = env.observation_space.shape   # (H, W, C)
        obs_shape = (c, h, w)                   # channels-first for Conv2d
        # Wrap to transpose obs automatically
        env = _ChannelsFirstWrapper(env)
    else:
        env = FlattenObservation(env)
        n = env.observation_space.shape[0]
        obs_shape = (1, 1, n)                   # degenerate spatial shape
    return env, obs_shape


class _NormalizeWrapper(gym.ObservationWrapper):
    """Divide observation by a fixed scale so values land in [0, 1].

    MiniGrid ImgObsWrapper returns uint8 values in [0, 10] (object/color/state encodings).
    Scaling to [0, 1] stabilises Q-network training.
    """
    def __init__(self, env, scale: float = 10.0):
        super().__init__(env)
        self._scale = scale
        low  = self.observation_space.low.astype(np.float32)  / scale
        high = self.observation_space.high.astype(np.float32) / scale
        self.observation_space = gym.spaces.Box(low, high,
                                                dtype=np.float32)

    def observation(self, obs):
        return obs.astype(np.float32) / self._scale


class _ChannelsFirstWrapper(gym.ObservationWrapper):
    """Transpose MiniGrid (H, W, C) observations to channels-first (C, H, W) and normalise."""

    def __init__(self, env, scale: float = 10.0):
        super().__init__(env)
        self._scale = scale
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1)).astype(np.float32) / self._scale


def make_env(env_id: str):
    """Creates a Gym/MiniGrid environment with observations flattened to a 1D float vector.

    For MiniGrid: applies ImgObsWrapper then normalises values to [0, 1] (÷10).
    """
    env = gym.make(env_id)
    if "MiniGrid" in env_id:
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
        env = _NormalizeWrapper(env)
    env = FlattenObservation(env)
    return env


def make_env_structured(env_id: str):
    """Creates an environment that preserves multi-dimensional observation structure.

    For MiniGrid: applies ImgObsWrapper (gives (7,7,3) image) but NOT FlattenObservation.
    Returns (env, mode_dims) where mode_dims is the observation shape tuple.

    For non-MiniGrid envs: falls back to the flat path (same as make_env) and returns a
    1D shape. Structured TN layers are not applicable to such environments.
    """
    env = gym.make(env_id)
    if "MiniGrid" in env_id:
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
        env = _NormalizeWrapper(env)
        mode_dims = env.observation_space.shape  # (7, 7, 3)
    else:
        env = FlattenObservation(env)
        mode_dims = env.observation_space.shape  # 1D fallback
    return env, mode_dims

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
import minigrid


def make_env(env_id: str):
    """Creates a Gym/MiniGrid environment with observations flattened to a 1D vector."""
    env = gym.make(env_id)
    if "MiniGrid" in env_id:
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
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
        mode_dims = env.observation_space.shape  # (7, 7, 3)
    else:
        env = FlattenObservation(env)
        mode_dims = env.observation_space.shape  # 1D fallback
    return env, mode_dims

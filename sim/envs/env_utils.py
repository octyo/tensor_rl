import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
import minigrid

def make_env(env_id: str):
    """
    Creates a Gym or Minigrid environment and applies necessary wrappers 
    to flatten the observation space into a 1D vector.
    """
    # If using Minigrid, we might need gym-minigrid wrappers, 
    # but for gym > 0.26 or gymnasium, FlattenObservation usually works.
    env = gym.make(env_id)
    
    if "MiniGrid" in env_id:
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
    
    # We want flat 1D inputs for our Linear and Tensor layers
    env = FlattenObservation(env)
    
    return env

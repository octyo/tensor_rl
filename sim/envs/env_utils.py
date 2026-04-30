import gym
import numpy as np
from gym.wrappers import FlattenObservation

def make_env(env_id: str):
    """
    Creates a Gym or Minigrid environment and applies necessary wrappers 
    to flatten the observation space into a 1D vector.
    """
    # If using Minigrid, we might need gym-minigrid wrappers, 
    # but for gym > 0.26 or gymnasium, FlattenObservation usually works.
    env = gym.make(env_id)
    
    # We want flat 1D inputs for our Linear and Tensor layers
    env = FlattenObservation(env)
    
    return env

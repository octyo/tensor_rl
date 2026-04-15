import minigrid
import gymnasium as gym
import numpy as np
from collections import defaultdict

env = gym.make("MiniGrid-BlockedUnlockPickup-v0")

# Simple Tabular Q-Learning (Reinforcement Learning) setup
q_table = defaultdict(lambda: np.zeros(env.action_space.n))

learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1

def get_state(obs):
    # MiniGrid observations are dicts. We extract 'image' and make it a hashable tuple.
    return tuple(obs["image"].flatten())

# Training loop following standard Gymnasium syntax
for episode in range(500):
    obs, info = env.reset()
    state = get_state(obs)
    
    terminated, truncated = False, False
    
    while not (terminated or truncated):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
            
        # Standard Gymnasium step
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = get_state(next_obs)
        
        # Q-learning update step
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        
        # Q(s,a) = Q(s,a) + alpha * (R + gamma * max Q(s',a') - Q(s,a))
        new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
        q_table[state][action] = new_value
        
        # Update current state
        state = next_state

env.close()
print("Simple Q-Learning finished running over the environment.")

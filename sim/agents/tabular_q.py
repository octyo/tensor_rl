import numpy as np
import random


class TabularQAgent:
    def __init__(self, state_space_size, action_space_size, lr=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.action_space_size = action_space_size
        self.steps = 0

    def epsilon(self) -> float:
        progress = min(1.0, self.steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: int, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon():
            return random.randint(0, self.action_space_size - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        best_next = np.max(self.q_table[next_state]) if not done else 0.0
        target = reward + self.gamma * best_next
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        self.steps += 1


class SarsaAgent(TabularQAgent):
    """On-policy TD control — update uses the action actually taken in next_state."""

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool,
               next_action: int = None):
        if next_action is None:
            next_action = self.select_action(next_state)
        next_q = self.q_table[next_state, next_action] if not done else 0.0
        target = reward + self.gamma * next_q
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        self.steps += 1


class TensorizedTabularQAgent:
    def __init__(self):
        # Tensor representation framework (CP/Tucker/TT applied directly to Q-table tensor)
        pass

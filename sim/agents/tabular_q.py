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


class TensorizedTabularQAgent(TabularQAgent):
    """CP-decomposed Q-table: Q(s, a) = factor_s[s] · factor_a[a].
    The full Q matrix is factor_s @ factor_a.T of shape (S, A).
    TD updates apply a rank-1 gradient step on the relevant rows of each factor.
    """

    def __init__(self, state_space_size, action_space_size, rank=4, lr=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000):
        # Skip TabularQAgent.__init__ to avoid allocating the full Q-table
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.rank = rank
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps = 0

        scale = 1.0 / np.sqrt(state_space_size)
        self.factor_s = np.random.uniform(-scale, scale, (state_space_size, rank))
        self.factor_a = np.random.uniform(-scale, scale, (action_space_size, rank))

    @property
    def q_table(self):
        """Reconstruct full Q matrix on demand (for compatibility with base class)."""
        return self.factor_s @ self.factor_a.T

    def _q(self, state: int, action: int) -> float:
        return float(self.factor_s[state] @ self.factor_a[action])

    def select_action(self, state: int, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon():
            return random.randint(0, self.action_space_size - 1)
        q_row = self.factor_s[state] @ self.factor_a.T  # shape (A,)
        return int(np.argmax(q_row))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        current_q = self._q(state, action)
        best_next = float(np.max(self.factor_s[next_state] @ self.factor_a.T)) if not done else 0.0
        target = reward + self.gamma * best_next
        delta = target - current_q

        # Capture pre-update factors for the symmetric gradient step
        fs = self.factor_s[state].copy()
        fa = self.factor_a[action].copy()

        self.factor_s[state] += self.lr * delta * fa
        self.factor_a[action] += self.lr * delta * fs

        self.steps += 1


class SarsaTensorAgent(TensorizedTabularQAgent):
    """CP-decomposed Q-table with SARSA (on-policy) TD target."""

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool,
               next_action: int = None):
        if next_action is None:
            next_action = self.select_action(next_state)

        current_q = self._q(state, action)
        next_q = self._q(next_state, next_action) if not done else 0.0
        target = reward + self.gamma * next_q
        delta = target - current_q

        fs = self.factor_s[state].copy()
        fa = self.factor_a[action].copy()

        self.factor_s[state] += self.lr * delta * fa
        self.factor_a[action] += self.lr * delta * fs

        self.steps += 1

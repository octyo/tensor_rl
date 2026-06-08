from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Simple MLP used to approximate Q(s, a) for discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, int] = (128, 128)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.model = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class Transition:
    """Container for one experience tuple."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=float(done),
            )
        )

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)


class DQNAgent:
    """
    Minimal DQN agent for Gym-like environments with discrete action spaces.

    Main components:
    - Online Q-network
    - Target Q-network
    - Experience replay buffer
    - Epsilon-greedy exploration
    - One-step TD training

    Notes:
    - This is a baseline implementation close to original DQN.
    - No Double DQN, no prioritized replay, no dueling head.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, int] = (128, 128),
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100_000,
        target_update_interval: int = 1_000,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)

        self.target_update_interval = target_update_interval

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.q_network = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.train_steps = 0
        self.total_env_steps = 0

        # Extension hook: user can set a callable to transform a transition before storage.
        # Signature:
        #   f(state, action, reward, next_state, done) -> (state, action, reward, next_state, done)
        self.replay_store_transform: Optional[
            Callable[[np.ndarray, int, float, np.ndarray, float], Tuple[np.ndarray, int, float, np.ndarray, float]]
        ] = None

        # Extension hook: user can set a callable to transform sampled batch tensors.
        # Signature:
        #   f(batch_dict) -> batch_dict
        # where batch_dict contains: states, actions, rewards, next_states, dones
        self.replay_batch_transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None

    def epsilon(self) -> float:
        """Linear epsilon schedule based on total environment steps."""
        progress = min(1.0, self.total_env_steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not deterministic and random.random() < self.epsilon():
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition in replay memory."""
        done_f = float(done)

        if self.replay_store_transform is not None:
            state, action, reward, next_state, done_f = self.replay_store_transform(
                state,
                action,
                reward,
                next_state,
                done_f,
            )

        self.replay_buffer.push(state, action, reward, next_state, done_f)

    def _prepare_batch(self, transitions: List[Transition]) -> Dict[str, torch.Tensor]:
        """Convert a list of transitions to PyTorch tensors."""
        states = torch.as_tensor(
            np.array([t.state for t in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.array([t.action for t in transitions], dtype=np.int64),
            dtype=torch.int64,
            device=self.device,
        )
        rewards = torch.as_tensor(
            np.array([t.reward for t in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.as_tensor(
            np.array([t.next_state for t in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            np.array([t.done for t in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        batch = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

        if self.replay_batch_transform is not None:
            batch = self.replay_batch_transform(batch)

        return batch

    def update_target_network(self) -> None:
        """Hard update: copy online network weights into target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train_step(self) -> Optional[float]:
        """
        Sample one replay batch and run one DQN optimization step.

        Returns:
            Scalar loss value if an update was performed, otherwise None.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = self._prepare_batch(transitions)

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Current Q-values for selected actions: Q(s, a)
        q_values = self.q_network(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_max = next_q_values.max(dim=1).values
            targets = rewards + self.gamma * next_q_max * (1.0 - dones)

        loss = nn.functional.mse_loss(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.update_target_network()

        return float(loss.item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Convenience wrapper to store transitions from environment interaction."""
        self.store_transition(state, action, reward, next_state, done)

    def step_env(self, env, state: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Run one environment step using epsilon-greedy policy.

        Supports Gym-style and Gymnasium-style `step` outputs.
        """
        action = self.select_action(state)
        step_out = env.step(action)

        if len(step_out) == 5:
            next_state, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            next_state, reward, done, info = step_out

        self.observe(state, action, reward, next_state, done)
        self.total_env_steps += 1

        return np.asarray(next_state, dtype=np.float32), float(reward), bool(done), info


def reset_env(env, seed: Optional[int] = None) -> np.ndarray:
    """
    Reset helper compatible with Gym and Gymnasium reset signatures.
    """
    if seed is None:
        reset_out = env.reset()
    else:
        try:
            reset_out = env.reset(seed=seed)
        except TypeError:
            reset_out = env.reset()

    if isinstance(reset_out, tuple):
        state, _ = reset_out
    else:
        state = reset_out
    return np.asarray(state, dtype=np.float32)


def train_dqn(
    env,
    agent: DQNAgent,
    num_episodes: int,
    max_steps_per_episode: int = 1_000,
    initial_seed: Optional[int] = None,
    fixed_seed_per_episode: bool = False,
) -> List[float]:
    """
    Simple training loop.

    Returns:
        Episode returns list.
    """
    returns: List[float] = []

    for episode_idx in range(num_episodes):
        if initial_seed is None:
            episode_seed = None
        elif fixed_seed_per_episode:
            episode_seed = initial_seed
        else:
            episode_seed = initial_seed + episode_idx

        state = reset_env(env, seed=episode_seed)
        episode_return = 0.0

        for _ in range(max_steps_per_episode):
            next_state, reward, done, _ = agent.step_env(env, state)
            state = next_state
            episode_return += reward

            agent.train_step()

            if done:
                break

        returns.append(float(episode_return))

    return returns

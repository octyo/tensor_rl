import random
import numpy as np
import torch
import torch.nn as nn

from agents.dqn_agent import DQNAgent
from config import LR, GAMMA, BUFFER_SIZE, BATCH_SIZE, REPLAY_MIN_SIZE, EPSILON_START, EPSILON_END, TAU


class DoubleDQNAgent(DQNAgent):
    """Double DQN: online network selects next action, target network evaluates it.
    Uses Polyak (soft) target updates instead of hard copies.
    Returns (loss, mean_max_q, grad_norm) from update().
    """

    def __init__(self, q_network: nn.Module, action_dim: int, lr=LR, gamma=GAMMA,
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 replay_min_size=REPLAY_MIN_SIZE,
                 epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                 epsilon_decay_steps=10000, tau=TAU, device="cpu"):
        super().__init__(q_network, lr=lr, gamma=gamma, buffer_size=buffer_size,
                         batch_size=batch_size, replay_min_size=replay_min_size,
                         epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                         epsilon_decay_steps=epsilon_decay_steps,
                         target_update_interval=int(1e9),  # disabled — we use soft updates
                         device=device)
        self.action_dim = action_dim
        self.tau = tau

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon():
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            # unsqueeze(0) is shape-agnostic: works for flat (147,) and structured (7,7,3).
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_network(state_t).argmax(dim=1).item()

    def _soft_update(self):
        for target_p, online_p in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_p.data.copy_(target_p.data + self.tau * (online_p.data - target_p.data))

    def update(self):
        """Returns (loss, mean_max_q, grad_norm). Returns (0, 0, 0) if buffer not ready."""
        if len(self.replay_buffer) < self.replay_min_size:
            return 0.0, 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        q_values = self.q_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            # Double DQN: online selects action, target evaluates
            next_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states_t).gather(1, next_actions)
            target_q_values = rewards_t + (1 - dones_t) * self.gamma * next_q
            self.last_mean_q = self.q_network(states_t).max(dim=1)[0].mean().item()

        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.last_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), max_norm=10.0).item()
        self.optimizer.step()

        self._soft_update()
        self.train_steps += 1

        return loss.item(), self.last_mean_q, self.last_grad_norm

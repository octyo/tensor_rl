import copy
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (LR, GAMMA, BUFFER_SIZE, BATCH_SIZE, REPLAY_MIN_SIZE,
                    EPSILON_START, EPSILON_END, TARGET_UPDATE_INTERVAL)


class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done


class DQNAgent:
    def __init__(self, q_network: nn.Module, lr=LR, gamma=GAMMA,
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 replay_min_size=REPLAY_MIN_SIZE,
                 epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                 epsilon_decay_steps=10000, target_update_interval=TARGET_UPDATE_INTERVAL,
                 device="cpu"):
        self.device = torch.device(device)
        self.q_network = q_network.to(self.device)
        self.target_network = copy.deepcopy(q_network).to(self.device)
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_min_size = replay_min_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_interval = target_update_interval

        self.train_steps = 0

    def epsilon(self) -> float:
        progress = min(1.0, self.train_steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon():
            # Fallback introspection — subclasses should pass action_dim directly
            last = self.q_network.model[-1]
            action_dim = getattr(last, 'out_features', None) or last.bias.shape[0]
            return random.randint(0, action_dim - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_network(state_t).argmax(dim=1).item()

    def update(self):
        if len(self.replay_buffer) < self.replay_min_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        q_values = self.q_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(1, keepdim=True)[0]
            target_q_values = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

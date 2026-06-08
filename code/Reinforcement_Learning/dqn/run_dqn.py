from __future__ import annotations

import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from boilerplate_test import SimpleEnv
from .dqn_agent import DQNAgent, train_dqn


ACTION_MAP = np.array([0, 1, 2, 3, 5], dtype=np.int64)
MINIGRID_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = MINIGRID_DIR / "blocked_unlock_pickup_dqn.pt"
RETURNS_PATH = MINIGRID_DIR / "blocked_unlock_pickup_dqn_returns.npy"


class FlattenedActionSubsetEnv:
    def __init__(self, env: SimpleEnv):
        self.env = env
        self.action_space = SimpleNamespace(n=len(ACTION_MAP))

    def _flatten_observation(self, observation) -> np.ndarray:
        if isinstance(observation, tuple):
            observation = observation[0]

        image = np.asarray(observation["image"], dtype=np.float32).reshape(-1) / 10.0
        direction = int(observation["direction"])
        direction_one_hot = np.zeros(4, dtype=np.float32)
        direction_one_hot[direction] = 1.0
        return np.concatenate([image, direction_one_hot]).astype(np.float32)

    def reset(self):
        return self._flatten_observation(self.env.reset())

    def step(self, action: int):
        step_out = self.env.step(int(ACTION_MAP[action]))

        if len(step_out) == 5:
            next_state, reward, terminated, truncated, info = step_out
            return self._flatten_observation(next_state), reward, terminated, truncated, info

        next_state, reward, done, info = step_out
        return self._flatten_observation(next_state), reward, done, info

    def close(self):
        return self.env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on SimpleEnv.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=1_000, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed for reproducibility.")
    parser.add_argument(
        "--overfit-one-episode",
        action="store_true",
        help="Deterministic overfit debug mode: train on one fixed-seed episode only.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        set_global_seed(args.seed)

    env = FlattenedActionSubsetEnv(SimpleEnv())
    state_dim = env.reset().shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=(128, 128),
        lr=1e-3,
        gamma=0.99,
        buffer_size=50_000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=25_000,
        target_update_interval=500,
    )

    num_episodes = 1 if args.overfit_one_episode else args.episodes
    fixed_seed_per_episode = bool(args.overfit_one_episode and args.seed is not None)

    if args.overfit_one_episode:
        if args.seed is None:
            print("Overfit mode enabled without --seed; run remains stochastic.")
        else:
            print(f"Overfit mode enabled: training on one deterministic episode with seed={args.seed}.")

    print(f"Training DQN on SimpleEnv for {num_episodes} episodes...")
    returns = train_dqn(
        env,
        agent,
        num_episodes=num_episodes,
        max_steps_per_episode=args.max_steps,
        initial_seed=args.seed,
        fixed_seed_per_episode=fixed_seed_per_episode,
    )

    torch.save(
        {
            "state_dict": agent.q_network.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_map": ACTION_MAP,
        },
        CHECKPOINT_PATH,
    )
    np.save(RETURNS_PATH, np.asarray(returns, dtype=np.float32))

    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
    print(f"Saved episode returns to {RETURNS_PATH}")
    env.close()


if __name__ == "__main__":
    main()
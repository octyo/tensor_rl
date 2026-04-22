from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..boilerplate_test import SimpleEnv
from .dqn_agent import DQNAgent
from .run_dqn import CHECKPOINT_PATH, FlattenedActionSubsetEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN greedily on SimpleEnv.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1_000,
        help="Max steps per episode before truncating evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to a checkpoint saved by run_dqn.py.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes in a human-view window.",
    )
    return parser.parse_args()


def load_agent(checkpoint_path: Path) -> tuple[DQNAgent, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=(128, 128),
        epsilon_start=0.0,
        epsilon_end=0.0,
    )
    agent.q_network.load_state_dict(checkpoint["state_dict"])
    agent.q_network.eval()
    agent.update_target_network()

    return agent, state_dim, action_dim


def run_greedy_episode(env: FlattenedActionSubsetEnv, agent: DQNAgent, max_steps: int) -> tuple[float, bool, int]:
    state = env.reset()
    episode_return = 0.0
    solved = False

    for step in range(1, max_steps + 1):
        action = agent.select_action(state, deterministic=True)
        step_out = env.step(action)

        if len(step_out) == 5:
            next_state, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            next_state, reward, done, _ = step_out

        state = next_state
        episode_return += float(reward)
        if reward > 0:
            solved = True

        if done:
            return episode_return, solved, step

    return episode_return, solved, max_steps


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run run_dqn.py first to train and save a model.")
        return

    agent, _, _ = load_agent(args.checkpoint)
    render_mode = "human" if args.render else None
    env = FlattenedActionSubsetEnv(SimpleEnv(render_mode=render_mode))

    returns = []
    solved_count = 0
    steps_list = []

    try:
        for episode in range(1, args.episodes + 1):
            episode_return, solved, steps = run_greedy_episode(env, agent, args.max_steps)
            returns.append(episode_return)
            solved_count += int(solved)
            steps_list.append(steps)
            print(
                f"Episode {episode:02d}/{args.episodes}: "
                f"return={episode_return:.3f}, solved={solved}, steps={steps}"
            )
    finally:
        env.close()

    print("\nSummary:")
    print(f"  mean_return={float(np.mean(returns)):.3f}")
    print(f"  success_rate={solved_count / args.episodes:.2%}")
    print(f"  mean_steps={float(np.mean(steps_list)):.1f}")


if __name__ == "__main__":
    main()
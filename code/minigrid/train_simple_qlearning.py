from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import random
import statistics

from boilerplate_test import SimpleEnv


def encode_state(obs: dict, env: SimpleEnv) -> tuple[int, ...]:
    """Build a hashable state representation for tabular Q-learning."""
    image_state = tuple(obs["image"].flatten().tolist())

    # Extra task-specific signals to speed up learning in this map.
    carrying_key = int(env.unwrapped.carrying is not None)
    door = env.unwrapped.grid.get(5, 6)
    door_open = int(getattr(door, "is_open", False))
    door_locked = int(getattr(door, "is_locked", False))

    return image_state + (carrying_key, door_open, door_locked)


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def train_q_learning(
    episodes: int = 3000,
    alpha: float = 0.15,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.997,
    seed: int = 42,
) -> tuple[dict[tuple[int, ...], list[float]], list[float], list[int]]:
    env = SimpleEnv(render_mode=None)
    rng = random.Random(seed)
    q_table: dict[tuple[int, ...], list[float]] = defaultdict(
        lambda: [0.0] * env.action_space.n
    )

    epsilon = epsilon_start
    returns: list[float] = []
    successes: list[int] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = encode_state(obs, env)

        terminated = False
        truncated = False
        total_reward = 0.0

        prev_carrying = int(env.unwrapped.carrying is not None)
        prev_door_open = int(getattr(env.unwrapped.grid.get(5, 6), "is_open", False))

        while not (terminated or truncated):
            if rng.random() < epsilon:
                action = int(env.action_space.sample())
            else:
                action = argmax(q_table[state])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = encode_state(next_obs, env)

            # Mild shaping: reward progress toward key/door/goal.
            carrying_now = int(env.unwrapped.carrying is not None)
            door_open_now = int(getattr(env.unwrapped.grid.get(5, 6), "is_open", False))

            shaped_reward = reward
            if carrying_now and not prev_carrying:
                shaped_reward += 0.15
            if door_open_now and not prev_door_open:
                shaped_reward += 0.25

            prev_carrying = carrying_now
            prev_door_open = door_open_now

            best_next = max(q_table[next_state])
            td_target = shaped_reward + gamma * best_next * (0.0 if terminated else 1.0)
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state
            total_reward += reward

        returns.append(total_reward)
        successes.append(int(terminated and total_reward > 0.0))
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (ep + 1) % 200 == 0:
            avg_return = statistics.mean(returns[-200:])
            success_rate = statistics.mean(successes[-200:])
            print(
                f"Episode {ep + 1:4d} | "
                f"epsilon={epsilon:.3f} | "
                f"avg_return(200)={avg_return:.3f} | "
                f"success_rate(200)={success_rate:.2%}"
            )

    env.close()
    return q_table, returns, successes


def evaluate_policy(
    q_table: dict[tuple[int, ...], list[float]], episodes: int = 5, seed: int = 1000
) -> None:
    env = SimpleEnv(render_mode="human")

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = encode_state(obs, env)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = argmax(q_table[state])
            obs, reward, terminated, truncated, _ = env.step(action)
            state = encode_state(obs, env)
            total_reward += reward

        print(f"[Eval] Episode {ep + 1}: reward={total_reward:.3f}")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q-learning on the custom MiniGrid game")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.997)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("q_table_simple_env.pkl"),
        help="Where to save the learned Q-table",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="If > 0, run greedy evaluation with rendering after training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_table, _, _ = train_q_learning(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    payload = {"q_table": dict(q_table), "meta": vars(args)}
    with args.save_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Saved Q-table to {args.save_path.resolve()}")

    if args.eval_episodes > 0:
        evaluate_policy(q_table, episodes=args.eval_episodes)


if __name__ == "__main__":
    main()

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from boilerplate_test import SimpleEnv

ACTION_MAP = np.array([0, 1, 2, 3, 5], dtype=np.int64)
NUM_ACTIONS = len(ACTION_MAP)
MODEL_PATH = Path(__file__).with_name("blocked_unlock_pickup_q_table.pkl")


def get_state(env):
    u = env.unwrapped
    door = u.grid.get(5, 6)
    carrying = u.carrying

    return (
        int(u.agent_pos[0]),
        int(u.agent_pos[1]),
        int(u.agent_dir),
        int(carrying is not None),
        int(getattr(door, "is_open", False)),
        int(getattr(door, "is_locked", False)),
    )


def argmax_random_tie(values):
    max_v = np.max(values)
    max_idxs = np.flatnonzero(values == max_v)
    return int(np.random.choice(max_idxs))


def run_episode(env, q_table):
    env.reset()
    state = get_state(env)

    terminated, truncated = False, False
    episode_return = 0.0
    steps = 0
    solved = False

    while not (terminated or truncated):
        action_idx = argmax_random_tie(q_table[state])
        action = int(ACTION_MAP[action_idx])

        _, reward, terminated, truncated, _ = env.step(action)
        state = get_state(env)

        episode_return += reward
        steps += 1

        if reward > 0:
            solved = True

    return episode_return, solved, steps


def load_q_table(path):
    with path.open("rb") as file:
        loaded_q_table = pickle.load(file)

    return defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float32), loaded_q_table)


def main():
    if not MODEL_PATH.exists():
        print(f"No saved Q-table found at {MODEL_PATH}")
        print("Run blocked_unlock_pickup.py first to train and save one.")
        return

    q_table = load_q_table(MODEL_PATH)
    env = SimpleEnv(render_mode="human")

    print(f"Loaded Q-table from {MODEL_PATH}")
    print("Showing greedy demo runs:")

    try:
        for episode in range(3):
            episode_return, solved, steps = run_episode(env, q_table)
            print(
                f"  demo {episode + 1}: return={episode_return:.3f}, "
                f"solved={solved}, steps={steps}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()

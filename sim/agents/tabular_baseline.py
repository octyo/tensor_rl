#!/usr/bin/env python3
"""
Tabular Q-learning baseline  (Milestone M1 of the CP-tensor project roadmap).

Algorithm  : Algorithm 1 from the roadmap, lines 14-19 replaced by the standard
             tabular update  Q[s,a] ← Q[s,a] + lr·δ  (small lr, no normalisation).
Environment: deterministic (rows × cols) grid world.
             State  = (row, col)  — factored, mirroring the CP tensor index.
             Action ∈ {UP, DOWN, LEFT, RIGHT}.
             Reward = -1 per step; 0 at goal (bottom-right corner, terminal).
Reference  : value_iteration_q() computes the exact Q* via dynamic programming,
             which enables the relative value-error metric ||Q̂ − Q*|| / ||Q*||.

Exports
-------
GridWorld           lightweight environment (no gym dependency)
value_iteration_q   exact Q* via DP
TabularGridQAgent   dense Q[row, col, action], ε-greedy, online Q-learning
train_agent         episode loop (returns episode_returns and value_errors)
run_experiment      multi-seed runner → dict ready for JSON serialisation
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from envs.gridworld import GridWorld, NUM_ACTIONS

# ── Exact Q* via value iteration ──────────────────────────────────────────────

def value_iteration_q(env: GridWorld, gamma: float = 0.99,
                      theta: float = 1e-9) -> np.ndarray:
    """Compute the exact optimal Q-function for *env* via dynamic programming.

    Iterates  Q*(s,a) ← r(s,a) + γ · max_{a'} Q*(s',a')
    until the max change in a sweep is below *theta*.

    Returns
    -------
    Q_star : ndarray of shape (rows, cols, NUM_ACTIONS)
    """
    Q = np.zeros((env.rows, env.cols, NUM_ACTIONS))

    for _ in range(100_000):
        delta = 0.0
        for state in env.nonterminal_states():
            r, c = state
            for a in range(NUM_ACTIONS):
                ns, reward = env.transition(state, a)
                nr, nc = ns
                if ns == env.goal:
                    target = reward
                else:
                    target = reward + gamma * float(np.max(Q[nr, nc]))
                diff = abs(Q[r, c, a] - target)
                Q[r, c, a] = target
                delta = max(delta, diff)
        if delta < theta:
            break

    return Q


# ── Tabular Q-learning agent ──────────────────────────────────────────────────

class TabularGridQAgent:
    """Dense Q[row, col, action] with ε-greedy online Q-learning.

    Update rule (Algorithm 1, tabular version):
        δ       = r + γ · max_{a'} Q[s',a'] − Q[s,a]
        Q[s,a] ← Q[s,a] + lr · δ
    ε decays multiplicatively at the end of each episode.
    """

    def __init__(self, rows: int, cols: int,
                 lr: float = 0.1, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995):
        self.rows = rows
        self.cols = cols
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((rows, cols, NUM_ACTIONS))

    @property
    def param_count(self) -> int:
        return self.Q.size  # rows × cols × NUM_ACTIONS

    def select_action(self, state: tuple, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        r, c = state
        return int(np.argmax(self.Q[r, c]))

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, done: bool) -> float:
        r, c = state
        nr, nc = next_state
        best_next = 0.0 if done else float(np.max(self.Q[nr, nc]))
        target = reward + self.gamma * best_next
        delta = target - self.Q[r, c, action]
        self.Q[r, c, action] += self.lr * delta
        return delta

    def decay_epsilon(self):
        """Multiplicative ε-decay (called once per episode, end of episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_agent(agent: TabularGridQAgent, env: GridWorld,
                n_episodes: int = 500, max_steps: int = 2000,
                q_star: np.ndarray | None = None) -> dict:
    """Episode loop implementing Algorithm 1 (tabular variant).

    Parameters
    ----------
    agent      : TabularGridQAgent to train (mutated in-place)
    env        : GridWorld environment
    n_episodes : number of training episodes
    max_steps  : step cap per episode (prevents infinite loops early in training)
    q_star     : optional exact Q*; if supplied, value error is tracked per episode

    Returns
    -------
    dict with keys:
      episode_returns  list[float]   total reward per episode
      value_errors     list[float]   ||Q̂ − Q*||_F / ||Q*||_F  (empty if q_star is None)
    """
    episode_returns: list[float] = []
    value_errors: list[float] = []
    q_star_norm = float(np.linalg.norm(q_star)) if q_star is not None else None

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        agent.decay_epsilon()
        episode_returns.append(total_reward)

        if q_star is not None:
            err = float(np.linalg.norm(agent.Q - q_star)) / (q_star_norm + 1e-12)
            value_errors.append(err)

    return {"episode_returns": episode_returns, "value_errors": value_errors}


# ── Multi-seed experiment runner ───────────────────────────────────────────────

def run_experiment(rows: int = 5, cols: int = 5, n_episodes: int = 500,
                   n_seeds: int = 5, lr: float = 0.1, gamma: float = 0.99,
                   epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                   epsilon_decay: float = 0.995, verbose: bool = True) -> dict:
    """Run the tabular baseline over *n_seeds* random seeds.

    Computes Q* once (shared across seeds) and tracks per-episode value error.

    Returns
    -------
    dict with keys:
      config        hyperparameters
      q_star        exact Q* as nested list (rows × cols × NUM_ACTIONS)
      param_count   number of parameters in the Q-table
      seeds         list of per-seed metric dicts from train_agent()
    """
    env = GridWorld(rows=rows, cols=cols)

    if verbose:
        print(f"Computing Q* for {rows}x{cols} grid world ...", end=" ", flush=True)
    q_star = value_iteration_q(env, gamma=gamma)
    if verbose:
        print("done.")

    results: dict = {
        "config": {
            "rows": rows, "cols": cols, "n_episodes": n_episodes,
            "n_seeds": n_seeds, "lr": lr, "gamma": gamma,
            "epsilon_start": epsilon_start, "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
        },
        "q_star": q_star.tolist(),
        "param_count": rows * cols * NUM_ACTIONS,
        "seeds": [],
    }

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)
        agent = TabularGridQAgent(rows, cols, lr=lr, gamma=gamma,
                                  epsilon_start=epsilon_start,
                                  epsilon_min=epsilon_min,
                                  epsilon_decay=epsilon_decay)
        metrics = train_agent(agent, env, n_episodes=n_episodes, q_star=q_star)
        results["seeds"].append(metrics)

        if verbose:
            final_ret = float(np.mean(metrics["episode_returns"][-20:]))
            final_err = metrics["value_errors"][-1] if metrics["value_errors"] else float("nan")
            print(f"  seed {seed}:  final return {final_ret:7.2f}  |  "
                  f"value error {final_err:.4f}")

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Tabular Q-learning baseline (M1)")
    p.add_argument("--rows",     type=int,   default=5)
    p.add_argument("--cols",     type=int,   default=5)
    p.add_argument("--episodes", type=int,   default=2000)
    p.add_argument("--seeds",    type=int,   default=5)
    p.add_argument("--lr",       type=float, default=0.1)
    p.add_argument("--gamma",    type=float, default=0.99)
    p.add_argument("--eps-start",  type=float, default=1.0,   dest="epsilon_start")
    p.add_argument("--eps-min",    type=float, default=0.05,  dest="epsilon_min")
    p.add_argument("--eps-decay",  type=float, default=0.995, dest="epsilon_decay")
    p.add_argument("--save", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'data', 'tabular_baseline_results.json'))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print("=" * 60)
    print(f"Tabular Q-learning baseline  ({args.rows}x{args.cols} grid)")
    print(f"Episodes : {args.episodes}   Seeds : {args.seeds}")
    print(f"lr={args.lr}  gamma={args.gamma}  eps: {args.epsilon_start}->{args.epsilon_min}"
          f" (decay {args.epsilon_decay})")
    print(f"Q-table parameters: {args.rows * args.cols * NUM_ACTIONS}")
    print("=" * 60)

    results = run_experiment(
        rows=args.rows, cols=args.cols,
        n_episodes=args.episodes, n_seeds=args.seeds,
        lr=args.lr, gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)

    all_final_returns = [float(np.mean(s["episode_returns"][-20:])) for s in results["seeds"]]
    all_final_errors  = [s["value_errors"][-1] for s in results["seeds"]]

    print("\nSummary")
    print(f"  Final return : {np.mean(all_final_returns):.3f} +/- {np.std(all_final_returns):.3f}")
    print(f"  Value error  : {np.mean(all_final_errors):.4f} +/- {np.std(all_final_errors):.4f}")
    print(f"  Parameters   : {results['param_count']}  "
          f"({args.rows}x{args.cols}x{NUM_ACTIONS})")
    print(f"\nResults saved: {args.save}")

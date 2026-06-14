#!/usr/bin/env python3
"""
CP Q-learning agent  (Milestones M2 / M3 of the CP-tensor project roadmap).

Implements Algorithm 1 from the roadmap exactly:
  - CP representation   Q[i_1,...,i_N] = sum_r prod_d A_d[i_d, r]
  - entry()             read one cell without building the full tensor   (Sec. 3b)
  - entry_grad()        leave-one-out product for each factor row         (Sec. 3c)
  - td_update()         NLMS semi-gradient step  (lr ~ 0.3-0.5)          (Sec. 3d)
  - Initialisation      N(0, sigma^2),  sigma = R^(-1/(2N))              (Sec. 3e)

Environment  : same GridWorld as tabular_baseline.py (reused directly).
Axes (N=3)   : row factor  A_0 (rows x R)
               col factor  A_1 (cols x R)
               action factor A_2 (NUM_ACTIONS x R)

Run standalone for a rank sweep and comparison with the tabular baseline:
    python cp_agent.py --ranks 1 2 4 8 16 --episodes 2000 --seeds 5
"""

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.gridworld import GridWorld, NUM_ACTIONS
from tabular_baseline import value_iteration_q


# ── CP helper functions (Section 3b / 3c / 3d) ───────────────────────────────

def entry(factors: list, index: tuple) -> float:
    """Q[index] = sum_r prod_d A_d[index_d, r].

    Never builds the full tensor: takes one row from each factor matrix,
    multiplies element-wise over the rank axis, then sums.
    """
    prod = np.ones(factors[0].shape[1])
    for d, i in enumerate(index):
        prod *= factors[d][i]
    return float(prod.sum())


def entry_grad(factors: list, index: tuple) -> list:
    """Gradient of Q[index] w.r.t. each factor row.

    grad_d = prod_{d' != d} A_{d'}[index_{d'}]   (leave-one-out product)
    Returns a list of N vectors, each of shape (R,).
    """
    rows = [factors[d][index[d]] for d in range(len(factors))]
    grads = []
    for d in range(len(factors)):
        loo = np.ones(factors[0].shape[1])
        for d2, row in enumerate(rows):
            if d2 != d:
                loo = loo * row
        grads.append(loo)
    return grads


def td_update(factors: list, sa_index: tuple, target: float,
              lr: float, eps: float = 1e-8,
              delta_clip: float = 10.0) -> float:
    """NLMS semi-gradient update (Algorithm 1, lines 15-19).

    delta = target - Q[sa_index]
    norm2 = sum_d ||grad_d||^2 + eps
    A_d[idx_d] += (lr * delta / norm2) * grad_d

    Dividing by norm2 keeps the actual change in Q approximately lr*delta
    regardless of rank or number of axes — this is why lr ~ 0.3-0.5 works here
    while a naive update would diverge.

    Implementation notes:
    - scale = lr*delta/norm2 is computed as a scalar first so that the
      subsequent vector multiply never overflows float64.
    - delta is soft-clipped to [-delta_clip, delta_clip].  At low rank the CP
      tensor can momentarily represent Q-values far outside the true range;
      without clipping these large errors feed back into huge intermediate
      products (lr * delta * grads[d]) before the division, causing inf/NaN.
    """
    delta = float(np.clip(target - entry(factors, sa_index), -delta_clip, delta_clip))
    grads = entry_grad(factors, sa_index)
    norm2 = sum(float((g * g).sum()) for g in grads) + eps
    scale = lr * delta / norm2          # scalar — multiply vector by this, not vice-versa
    for d in range(len(factors)):
        factors[d][sa_index[d]] += scale * grads[d]
    return delta


# ── CP Q-learning agent ───────────────────────────────────────────────────────

class CPGridQAgent:
    """CP-decomposed Q-tensor for the deterministic grid world.

    Q[row, col, action] = sum_r A_row[row,r] * A_col[col,r] * A_act[action,r]

    Factor matrices
    ---------------
    factors[0]  A_row   shape (rows, R)
    factors[1]  A_col   shape (cols, R)
    factors[2]  A_act   shape (NUM_ACTIONS, R)

    Initialisation: N(0, sigma^2) with sigma = R^(-1/(2N)), N=3.
    Update: NLMS semi-gradient; lr should be 0.3-0.5 (not the small lr of tabular).
    """

    def __init__(self, rows: int, cols: int, rank: int = 4,
                 lr: float = 0.4, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995):
        self.rows = rows
        self.cols = cols
        self.rank = rank
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        N = 3  # row, col, action axes
        sigma = rank ** (-1.0 / (2 * N))
        axis_sizes = [rows, cols, NUM_ACTIONS]
        self.factors = [np.random.normal(0.0, sigma, (size, rank))
                        for size in axis_sizes]

    @property
    def param_count(self) -> int:
        return sum(f.size for f in self.factors)

    def _q_all_actions(self, state: tuple) -> np.ndarray:
        """Q[state, :] for all actions efficiently — shape (NUM_ACTIONS,).

        h = A_row[row] * A_col[col]   (R,)
        Q[state, a] = h @ A_act[a]  for all a  →  h @ A_act.T
        """
        r, c = state
        h = self.factors[0][r] * self.factors[1][c]
        return h @ self.factors[2].T  # (NUM_ACTIONS,)

    def select_action(self, state: tuple, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(self._q_all_actions(state)))

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, done: bool) -> float:
        """Algorithm 1, lines 11-19: compute TD target then NLMS update."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self._q_all_actions(next_state)))

        sa_index = (state[0], state[1], action)
        return td_update(self.factors, sa_index, target, self.lr)

    def decay_epsilon(self):
        """Algorithm 1, line 21: multiplicative decay at end of each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def to_dense(self) -> np.ndarray:
        """Reconstruct full Q-tensor of shape (rows, cols, NUM_ACTIONS).

        Used only for computing the value error ||Q - Q*|| / ||Q*||.
        Never called during training.
        """
        return np.einsum('ir,jr,kr->ijk',
                         self.factors[0], self.factors[1], self.factors[2])


# ── Training loop ──────────────────────────────────────────────────────────────

def train_agent(agent: CPGridQAgent, env: GridWorld,
                n_episodes: int = 500, max_steps: int = 2000,
                q_star: np.ndarray | None = None) -> dict:
    """Episode loop — identical structure to tabular_baseline.train_agent.

    Returns
    -------
    dict with keys:
      episode_returns  list[float]   total reward per episode
      value_errors     list[float]   ||Q_dense - Q*||_F / ||Q*||_F  per episode
                                     (empty list if q_star is None)
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
            q_dense = agent.to_dense()
            err = float(np.linalg.norm(q_dense - q_star)) / (q_star_norm + 1e-12)
            value_errors.append(err)

    return {"episode_returns": episode_returns, "value_errors": value_errors}


# ── Single-rank experiment ─────────────────────────────────────────────────────

def run_experiment(rows: int = 5, cols: int = 5, rank: int = 4,
                   n_episodes: int = 500, n_seeds: int = 5,
                   lr: float = 0.4, gamma: float = 0.99,
                   epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                   epsilon_decay: float = 0.995,
                   q_star: np.ndarray | None = None,
                   verbose: bool = True) -> dict:
    """Run CP Q-learning at a single rank over n_seeds random seeds.

    Parameters
    ----------
    q_star : precomputed Q* array; if None, it is computed here.

    Returns
    -------
    dict with keys config, q_star, rank, param_count, seeds.
    """
    env = GridWorld(rows=rows, cols=cols)

    if q_star is None:
        if verbose:
            print(f"Computing Q* for {rows}x{cols} grid world ...", end=" ", flush=True)
        q_star = value_iteration_q(env, gamma=gamma)
        if verbose:
            print("done.")

    n_params = rank * (rows + cols + NUM_ACTIONS)

    results: dict = {
        "config": {
            "rows": rows, "cols": cols, "rank": rank,
            "n_episodes": n_episodes, "n_seeds": n_seeds,
            "lr": lr, "gamma": gamma,
            "epsilon_start": epsilon_start, "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
        },
        "q_star": q_star.tolist(),
        "rank": rank,
        "param_count": n_params,
        "seeds": [],
    }

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)
        agent = CPGridQAgent(rows, cols, rank=rank, lr=lr, gamma=gamma,
                             epsilon_start=epsilon_start,
                             epsilon_min=epsilon_min,
                             epsilon_decay=epsilon_decay)
        metrics = train_agent(agent, env, n_episodes=n_episodes, q_star=q_star)
        results["seeds"].append(metrics)

        if verbose:
            final_ret = float(np.mean(metrics["episode_returns"][-20:]))
            final_err = metrics["value_errors"][-1] if metrics["value_errors"] else float("nan")
            print(f"  rank={rank} seed {seed}:  final return {final_ret:7.2f}  |  "
                  f"value error {final_err:.4f}")

    return results


# ── Rank sweep (Experiment A) ──────────────────────────────────────────────────

def run_rank_sweep(rows: int = 5, cols: int = 5,
                   ranks: list[int] | None = None,
                   n_episodes: int = 500, n_seeds: int = 5,
                   lr: float = 0.4, gamma: float = 0.99,
                   epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                   epsilon_decay: float = 0.995,
                   verbose: bool = True) -> dict:
    """Sweep CP rank and collect metrics for comparison with the tabular baseline.

    Returns
    -------
    dict  {rank: experiment_result}  — each value is the output of run_experiment().
    The same Q* is shared across all ranks (computed once).
    """
    if ranks is None:
        ranks = [1, 2, 4, 8, 16]

    env = GridWorld(rows=rows, cols=cols)
    if verbose:
        print(f"Computing Q* for {rows}x{cols} grid world ...", end=" ", flush=True)
    q_star = value_iteration_q(env, gamma=gamma)
    if verbose:
        print("done.\n")

    tabular_params = rows * cols * NUM_ACTIONS

    sweep: dict = {}
    for rank in ranks:
        n_params = rank * (rows + cols + NUM_ACTIONS)
        if verbose:
            print(f"--- CP rank={rank}  params={n_params}  "
                  f"(tabular={tabular_params}, ratio={n_params/tabular_params:.2f}x) ---")
        sweep[rank] = run_experiment(
            rows=rows, cols=cols, rank=rank,
            n_episodes=n_episodes, n_seeds=n_seeds,
            lr=lr, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            q_star=q_star, verbose=verbose,
        )
        if verbose:
            rets = [float(np.mean(s["episode_returns"][-20:])) for s in sweep[rank]["seeds"]]
            errs = [s["value_errors"][-1] for s in sweep[rank]["seeds"]]
            print(f"  -> return {np.mean(rets):.3f} +/- {np.std(rets):.3f}  |  "
                  f"value error {np.mean(errs):.4f} +/- {np.std(errs):.4f}\n")

    return sweep


# ── Entry point ────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="CP Q-learning agent (M2/M3)")
    p.add_argument("--rows",      type=int,   default=5)
    p.add_argument("--cols",      type=int,   default=5)
    p.add_argument("--ranks",     type=int,   nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--episodes",  type=int,   default=500)
    p.add_argument("--seeds",     type=int,   default=5)
    p.add_argument("--lr",        type=float, default=0.4)
    p.add_argument("--gamma",     type=float, default=0.99)
    p.add_argument("--eps-start", type=float, default=1.0,   dest="epsilon_start")
    p.add_argument("--eps-min",   type=float, default=0.05,  dest="epsilon_min")
    p.add_argument("--eps-decay", type=float, default=0.995, dest="epsilon_decay")
    p.add_argument("--save", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'data', 'cp_rank_sweep_results.json'))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    tabular_params = args.rows * args.cols * NUM_ACTIONS
    print("=" * 65)
    print(f"CP Q-learning  ({args.rows}x{args.cols} grid)")
    print(f"Ranks: {args.ranks}   Episodes: {args.episodes}   Seeds: {args.seeds}")
    print(f"lr={args.lr}  gamma={args.gamma}  eps: {args.epsilon_start}->{args.epsilon_min}"
          f" (decay {args.epsilon_decay})")
    print(f"Tabular reference: {tabular_params} params")
    print(f"CP params = rank x ({args.rows} + {args.cols} + {NUM_ACTIONS})"
          f" = rank x {args.rows + args.cols + NUM_ACTIONS}")
    print("=" * 65 + "\n")

    sweep = run_rank_sweep(
        rows=args.rows, cols=args.cols, ranks=args.ranks,
        n_episodes=args.episodes, n_seeds=args.seeds,
        lr=args.lr, gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump({str(k): v for k, v in sweep.items()}, f, indent=2)

    print("\n" + "=" * 65)
    print("Rank sweep summary")
    print("=" * 65)
    print(f"{'Rank':<8} {'Params':<10} {'vs table':<10} {'Return':<20} {'Value error'}")
    print("-" * 65)
    for rank in args.ranks:
        res = sweep[rank]
        rets = [float(np.mean(s["episode_returns"][-20:])) for s in res["seeds"]]
        errs = [s["value_errors"][-1] for s in res["seeds"]]
        ratio = res["param_count"] / tabular_params
        print(f"{rank:<8} {res['param_count']:<10} {ratio:<10.2f}"
              f" {np.mean(rets):.3f} +/- {np.std(rets):.3f}   "
              f"{np.mean(errs):.4f} +/- {np.std(errs):.4f}")
    print(f"\n(tabular baseline: {tabular_params} params, "
          f"return ~ -7.37, value error ~ 0.073)")
    print(f"\nResults saved: {args.save}")

#!/usr/bin/env python3
"""
Plot the learned policy *on the grid itself*, for tabular vs CP Q-learning.

For every cell we draw:
  * an ARROW pointing in the greedy action's direction  (argmax_a Q(s,a))
        -> i.e. "which neighbouring cell the agent would move into"
  * the arrow is GREEN if that action is optimal (matches an argmax of the exact
    Q*), RED if it is sub-optimal. This green/red is the per-cell "accuracy".
  * the background HEATMAP colours each tile by the learned value
        V(s) = max_a Q(s,a)
    i.e. the strength of the learned Q-table at that cell (brighter = higher
    value = the agent thinks it is closer to the goal). This is literally the
    Q-table's strength painted onto the map.

We render three policies side by side -- Optimal (from Q*), Tabular, CP -- and
do it for both the random-start and fixed-start training regimes, so you can see
*where* on the map each method is right or wrong.

Run:
    python policy_map.py                 # 12x12 grid (arrows stay legible)
    python policy_map.py --grid 20       # matches the main experiment
"""

import argparse
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.tabular_baseline import value_iteration_q
from envs.gridworld import GridWorld, NUM_ACTIONS, _DELTA
from cp_speedup_experiment import (make_agent, RandomStartGridWorld,
                                   GAMMA, TAB_LR, CP_LR, EPSILON_DECAY)


# ── Training ─────────────────────────────────────────────────────────────────

def train(kind: str, rank: int, env: GridWorld, n_episodes: int,
          max_steps: int, seed: int):
    """Train one agent; return a get_q_row(r, c) accessor over the final Q."""
    random.seed(seed)
    np.random.seed(seed)
    agent, get_q_row = make_agent(kind, env.rows, env.cols, rank,
                                  TAB_LR, CP_LR, EPSILON_DECAY)
    for _ in range(n_episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = agent.select_action(state)
            ns, reward, done = env.step(action)
            agent.update(state, action, reward, ns, done)
            state = ns
            if done:
                break
        agent.decay_epsilon()
    return get_q_row


# ── One policy map onto a given axis ─────────────────────────────────────────

def plot_policy_map(ax, get_q_row, env: GridWorld, opt_mask: np.ndarray, title: str):
    G_r, G_c = env.rows, env.cols
    goal = env.goal

    V = np.full((G_r, G_c), np.nan)
    # arrow components, split into correct (green) and wrong (red)
    xs_ok, ys_ok, us_ok, vs_ok = [], [], [], []
    xs_no, ys_no, us_no, vs_no = [], [], [], []
    n_correct = n_total = 0

    for r in range(G_r):
        for c in range(G_c):
            q = np.asarray(get_q_row(r, c))
            V[r, c] = float(np.max(q))
            if (r, c) == goal:
                continue
            a = int(np.argmax(q))
            dr, dc = _DELTA[a]                       # (row, col) step
            ok = bool(opt_mask[r, c, a])
            n_correct += ok
            n_total += 1
            # display: x=col (right+), y=row (down+, matches imshow origin='upper')
            (xs_ok if ok else xs_no).append(c)
            (ys_ok if ok else ys_no).append(r)
            (us_ok if ok else us_no).append(dc)
            (vs_ok if ok else vs_no).append(dr)

    im = ax.imshow(V, cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="learned V(s)=max_a Q")

    arrow_len = 0.42
    common = dict(angles="xy", scale_units="xy", scale=1.0 / arrow_len,
                  width=0.012, pivot="mid")
    if xs_ok:
        ax.quiver(xs_ok, ys_ok, us_ok, vs_ok, color="#39ff14",
                  edgecolor="black", linewidth=0.4, **common)
    if xs_no:
        ax.quiver(xs_no, ys_no, us_no, vs_no, color="#ff1744",
                  edgecolor="black", linewidth=0.4, **common)

    # mark the goal
    ax.scatter([goal[1]], [goal[0]], marker="*", s=260, color="gold",
               edgecolors="black", linewidth=1.2, zorder=6)

    acc = n_correct / n_total if n_total else float("nan")
    ax.set_title(f"{title}\npolicy accuracy = {acc*100:.1f}%  (green=optimal, red=not)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    return acc


# ── Optimal-policy accessor from Q* ──────────────────────────────────────────

def q_star_accessor(q_star):
    return lambda r, c: q_star[r, c]


# ── Build a 3-panel figure for one start regime ──────────────────────────────

def make_figure(regime_name: str, env_train_cls, env_eval: GridWorld,
                q_star: np.ndarray, opt_mask: np.ndarray, rank: int,
                n_episodes: int, max_steps: int, seed: int, out_path: str):
    print(f"  [{regime_name}] training tabular ...", flush=True)
    tab_row = train("tabular", rank, env_train_cls(env_eval.rows, env_eval.cols),
                    n_episodes, max_steps, seed)
    print(f"  [{regime_name}] training CP rank={rank} ...", flush=True)
    cp_row = train("cp", rank, env_train_cls(env_eval.rows, env_eval.cols),
                   n_episodes, max_steps, seed)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    plot_policy_map(axes[0], q_star_accessor(q_star), env_eval, opt_mask,
                    "OPTIMAL (from Q*)")
    plot_policy_map(axes[1], tab_row, env_eval, opt_mask, "TABULAR Q-table")
    plot_policy_map(axes[2], cp_row, env_eval, opt_mask, f"CP rank={rank}")

    fig.suptitle(
        f"Learned policy on the grid  —  {regime_name}  "
        f"({env_eval.rows}x{env_eval.cols}, goal = gold star, bottom-right)",
        fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{regime_name}] saved {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    G = args.grid
    env_eval = GridWorld(G, G)
    print(f"Computing exact Q* for {G}x{G} grid ...", end=" ", flush=True)
    q_star = value_iteration_q(env_eval, gamma=GAMMA)
    opt_mask = q_star == q_star.max(axis=2, keepdims=True)
    print("done.")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    # Random start (the regime where CP shines)
    make_figure("RANDOM start", RandomStartGridWorld, env_eval, q_star, opt_mask,
                args.rank, args.episodes, args.max_steps, args.seed,
                os.path.join(out_dir, f"policy_map_random_{G}x{G}.png"))

    # Fixed start at (0,0) -- plain GridWorld -- where CP loses its edge
    make_figure("FIXED start (0,0)", GridWorld, env_eval, q_star, opt_mask,
                args.rank, args.episodes, args.max_steps, args.seed,
                os.path.join(out_dir, f"policy_map_fixed_{G}x{G}.png"))


if __name__ == "__main__":
    main()

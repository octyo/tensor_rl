#!/usr/bin/env python3
"""
Where CP *wins*: low-rank Q-learning learns FASTER than a tabular Q-table.

The companion experiment (tabular_experiments.py) showed CP matching a tabular
Q-table with far fewer parameters. This file goes one step further and looks for
an environment where the CP-decomposed Q-tensor does not just *tie* the tabular
baseline but actively *beats* it on sample efficiency — i.e. it learns the
optimal policy in fewer episodes.

Why it can win
--------------
On a large navigation grid the optimal value surface is almost separable:
the distance to a corner goal is roughly  (rows-1-r) + (cols-1-c), an *additive*
(hence near-rank-2) function of the two coordinates. A tabular agent has to learn
each of  rows x cols x actions  cells independently — it only improves a state it
has actually visited. The CP agent instead stores one small row/col/action factor
each, so a single update to row r or column c immediately *generalises* to every
other cell sharing that coordinate. With random start states (so both agents see
the whole grid) this generalisation translates directly into faster learning.

What we measure ("how each model learns during training")
---------------------------------------------------------
Per checkpoint we log, averaged over seeds:
  * policy accuracy  — fraction of states whose greedy action is optimal
                       (tie-aware: counts any action that is argmax of Q*)
  * greedy return    — mean return of the greedy policy from random starts
  * value error      — ||Q_hat - Q*||_F / ||Q*||_F

The headline plot is the policy-accuracy learning curve: CP vs tabular vs episode.

No SARSA here — only off-policy Q-learning, tabular vs CP.

Reuses TabularGridQAgent / CPGridQAgent / value_iteration_q from agents/, and
GridWorld from envs/ — only the random-start wrapper and the driver are new, so
nothing here touches the files your group is working on.

Run:
    python cp_speedup_experiment.py            # quick demo (3 seeds)
    python cp_speedup_experiment.py --full     # 5 seeds, more episodes
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.tabular_baseline import TabularGridQAgent, value_iteration_q
from agents.cp_agent import CPGridQAgent
from envs.gridworld import GridWorld, NUM_ACTIONS
from rank_analysis import cp_capture_curve, plot_capture_panel


# ============================================================================
# 1. NEW ENVIRONMENT: large grid with random start states
# ============================================================================

class RandomStartGridWorld(GridWorld):
    """GridWorld variant that resets to a uniformly random non-goal cell.

    Two reasons this is the right testbed for the CP-vs-tabular speed question:

    1. Random starts give *both* agents full state coverage, so any difference in
       learning speed is about how efficiently each one turns a transition into
       policy improvement — not about which states they happen to visit.
    2. The dense -1-per-step reward makes the optimal value surface smooth and
       near-rank-2, which is exactly the regime where a low-rank CP factorisation
       generalises across states and a flat table cannot.
    """

    def reset(self) -> tuple:
        while True:
            s = (random.randint(0, self.rows - 1),
                 random.randint(0, self.cols - 1))
            if s != self.goal:
                self._state = s
                return s


# ============================================================================
# 2. METRICS
# ============================================================================

def optimal_action_mask(q_star: np.ndarray) -> np.ndarray:
    """Boolean (rows, cols, actions): True where action is optimal under Q* (tie-aware)."""
    return q_star == q_star.max(axis=2, keepdims=True)


def policy_accuracy(get_q_row, env: GridWorld, opt_mask: np.ndarray) -> float:
    """Fraction of non-goal states whose greedy action is an optimal action.

    get_q_row(r, c) -> length-NUM_ACTIONS array of Q-values for that cell.
    Tie-aware: a state counts as correct if its greedy action is *any* argmax of Q*.
    """
    correct = total = 0
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) == env.goal:
                continue
            a = int(np.argmax(get_q_row(r, c)))
            correct += int(opt_mask[r, c, a])
            total += 1
    return correct / total


def greedy_return(agent, env: GridWorld, n_rollouts: int = 30,
                  max_steps: int = None) -> float:
    """Mean return of the greedy (evaluate=True) policy from random start states."""
    if max_steps is None:
        max_steps = 4 * (env.rows + env.cols)
    returns = []
    for _ in range(n_rollouts):
        state = env.reset()
        total = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state, evaluate=True)
            state, reward, done = env.step(action)
            total += reward
            if done:
                break
        returns.append(total)
    return float(np.mean(returns))


# ============================================================================
# 3. TRAINING WITH LEARNING-CURVE LOGGING
# ============================================================================

def make_agent(kind: str, rows: int, cols: int, rank: int,
               tab_lr: float, cp_lr: float, epsilon_decay: float):
    """kind == 'tabular' or 'cp'. Returns (agent, get_q_row_fn)."""
    if kind == "tabular":
        agent = TabularGridQAgent(rows, cols, lr=tab_lr, gamma=GAMMA,
                                  epsilon_decay=epsilon_decay)
        return agent, (lambda r, c: agent.Q[r, c])
    elif kind == "cp":
        agent = CPGridQAgent(rows, cols, rank=rank, lr=cp_lr, gamma=GAMMA,
                             epsilon_decay=epsilon_decay)
        return agent, (lambda r, c: agent._q_all_actions((r, c)))
    raise ValueError(kind)


def train_one(kind: str, rank: int, env: GridWorld, q_star: np.ndarray,
              opt_mask: np.ndarray, n_episodes: int, eval_every: int,
              max_steps: int, tab_lr: float, cp_lr: float,
              epsilon_decay: float, seed: int) -> dict:
    """Train a single agent, logging the learning curve every `eval_every` episodes."""
    random.seed(seed)
    np.random.seed(seed)
    agent, get_q_row = make_agent(kind, env.rows, env.cols, rank,
                                  tab_lr, cp_lr, epsilon_decay)
    q_star_norm = float(np.linalg.norm(q_star))

    checkpoints, accuracy, ret, verr = [], [], [], []

    for ep in range(n_episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.decay_epsilon()

        if (ep + 1) % eval_every == 0:
            checkpoints.append(ep + 1)
            accuracy.append(policy_accuracy(get_q_row, env, opt_mask))
            ret.append(greedy_return(agent, env))
            q_dense = agent.Q if kind == "tabular" else agent.to_dense()
            verr.append(float(np.linalg.norm(q_dense - q_star)) / (q_star_norm + 1e-12))

    return {"checkpoints": checkpoints, "accuracy": accuracy,
            "greedy_return": ret, "value_error": verr,
            "param_count": agent.param_count}


def run_config(kind: str, rank: int, env: GridWorld, q_star: np.ndarray,
               opt_mask: np.ndarray, n_episodes: int, eval_every: int,
               max_steps: int, tab_lr: float, cp_lr: float,
               epsilon_decay: float, n_seeds: int, label: str) -> dict:
    """Run one agent configuration over n_seeds and aggregate mean/std curves."""
    per_seed = []
    for seed in range(n_seeds):
        per_seed.append(train_one(kind, rank, env, q_star, opt_mask,
                                  n_episodes, eval_every, max_steps,
                                  tab_lr, cp_lr, epsilon_decay, seed))
        print(f"    {label:<14} seed {seed + 1}/{n_seeds}  "
              f"final acc={per_seed[-1]['accuracy'][-1]:.3f}  "
              f"return={per_seed[-1]['greedy_return'][-1]:.2f}")

    checkpoints = per_seed[0]["checkpoints"]
    stack = lambda key: np.array([s[key] for s in per_seed])  # (seeds, checkpoints)
    return {
        "label": label,
        "kind": kind,
        "rank": rank,
        "param_count": per_seed[0]["param_count"],
        "checkpoints": checkpoints,
        "acc_mean": stack("accuracy").mean(0).tolist(),
        "acc_std": stack("accuracy").std(0).tolist(),
        "ret_mean": stack("greedy_return").mean(0).tolist(),
        "ret_std": stack("greedy_return").std(0).tolist(),
        "verr_mean": stack("value_error").mean(0).tolist(),
    }


# ============================================================================
# 4. HYPERPARAMETERS
# ============================================================================

GRID = 20            # 20x20 grid -> 400 states, 1600-cell Q-table
GAMMA = 0.99
TAB_LR = 0.1         # tabular default (agents/tabular_baseline.py)
CP_LR = 0.4          # CP NLMS default (agents/cp_agent.py)
EPSILON_DECAY = 0.999
MAX_STEPS = 200      # per-episode step cap during training

# CP ranks to compare against the tabular baseline
RANKS = [2, 3, 6]


# ============================================================================
# 5. PLOTTING
# ============================================================================

def plot_results(configs: list, grid: int, out_path: str,
                 capture_curve: list = None, tabular_params: int = None,
                 trained_ranks: list = None, start_label: str = "random-start"):
    """6-panel figure; headline is the policy-accuracy learning curve."""
    tab_color = "#d62728"
    cp_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    def color_for(cfg, idx):
        return tab_color if cfg["kind"] == "tabular" else cp_colors[idx % len(cp_colors)]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # Panel 1: policy accuracy learning curve (the headline)
    ax = axes[0, 0]
    ci = 0
    for cfg in configs:
        x = cfg["checkpoints"]
        m = np.array(cfg["acc_mean"])
        s = np.array(cfg["acc_std"])
        col = color_for(cfg, ci)
        if cfg["kind"] == "cp":
            ci += 1
        ax.plot(x, m, label=cfg["label"], color=col, linewidth=2.2)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.15)
    ax.set_xlabel("Training episode", fontsize=11)
    ax.set_ylabel("Policy accuracy (frac. states with optimal action)", fontsize=11)
    ax.set_title("Learning speed: CP vs tabular Q-learning", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)

    # Panel 2: greedy return learning curve
    ax = axes[0, 1]
    ci = 0
    for cfg in configs:
        x = cfg["checkpoints"]
        m = np.array(cfg["ret_mean"])
        s = np.array(cfg["ret_std"])
        col = color_for(cfg, ci)
        if cfg["kind"] == "cp":
            ci += 1
        ax.plot(x, m, label=cfg["label"], color=col, linewidth=2.2)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.15)
    ax.set_xlabel("Training episode", fontsize=11)
    ax.set_ylabel("Greedy-policy return (higher = better)", fontsize=11)
    ax.set_title("Greedy return during training", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 3: episodes-to-threshold (sample efficiency) bar chart
    ax = axes[1, 0]
    threshold = 0.9
    labels, eps_to_thresh, bar_colors = [], [], []
    ci = 0
    for cfg in configs:
        col = color_for(cfg, ci)
        if cfg["kind"] == "cp":
            ci += 1
        acc = np.array(cfg["acc_mean"])
        x = np.array(cfg["checkpoints"])
        hit = np.where(acc >= threshold)[0]
        labels.append(cfg["label"])
        eps_to_thresh.append(int(x[hit[0]]) if len(hit) else np.nan)
        bar_colors.append(col)
    ymax = max([e for e in eps_to_thresh if not np.isnan(e)] + [1])
    plotted = [e if not np.isnan(e) else ymax * 1.15 for e in eps_to_thresh]
    bars = ax.bar(labels, plotted, color=bar_colors, alpha=0.8, edgecolor="black")
    for bar, e in zip(bars, eps_to_thresh):
        txt = f"{int(e)}" if not np.isnan(e) else "never"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                txt, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Episodes to reach {int(threshold*100)}% accuracy", fontsize=11)
    ax.set_title("Sample efficiency (lower = faster)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    # Panel 4: parameter count vs final accuracy
    ax = axes[1, 1]
    ci = 0
    for cfg in configs:
        col = color_for(cfg, ci)
        if cfg["kind"] == "cp":
            ci += 1
        ax.scatter(cfg["param_count"], cfg["acc_mean"][-1], s=220, color=col,
                   alpha=0.75, edgecolors="black", linewidth=1.5, zorder=3)
        ax.annotate(cfg["label"], (cfg["param_count"], cfg["acc_mean"][-1]),
                    textcoords="offset points", xytext=(0, 11), ha="center", fontsize=9)
    ax.set_xlabel("Parameter count", fontsize=11)
    ax.set_ylabel("Final policy accuracy", fontsize=11)
    ax.set_title("Parameters vs final accuracy", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 5: representation capacity — % of exact Q* captured vs CP rank
    if capture_curve is not None:
        plot_capture_panel(axes[0, 2], capture_curve, tabular_params,
                           trained_ranks=trained_ranks)
    else:
        axes[0, 2].axis("off")

    # Panel 6 unused
    axes[1, 2].axis("off")

    fig.suptitle(
        f"CP-decomposed vs tabular Q-learning  —  {grid}x{grid} {start_label} grid "
        f"({grid*grid} states, {grid*grid*NUM_ACTIONS}-cell table)",
        fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\n[OK] Figure saved to {out_path}")


# ============================================================================
# 6. DRIVER
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="5 seeds, 4000 episodes (default: 3 seeds, 2500 episodes)")
    parser.add_argument("--grid", type=int, default=GRID)
    parser.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--start", choices=["random", "fixed"], default="random",
                        help="random = spawn anywhere (CP shines); "
                             "fixed = always start at (0,0)")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds is not None else (5 if args.full else 3)
    n_episodes = args.episodes if args.episodes is not None else (4000 if args.full else 2500)
    grid = args.grid
    env_cls = RandomStartGridWorld if args.start == "random" else GridWorld
    start_label = "random-start" if args.start == "random" else "fixed-start (0,0)"

    print("=" * 72)
    print("CP vs TABULAR Q-LEARNING — learning-speed experiment")
    print("=" * 72)
    print(f"Environment : {grid}x{grid} {start_label} grid  "
          f"({grid*grid} states, dense -1/step reward)")
    print(f"Baseline    : tabular Q-table  ({grid*grid*NUM_ACTIONS} params)")
    print(f"CP ranks    : {args.ranks}  "
          f"(params = rank x ({grid}+{grid}+{NUM_ACTIONS}) = rank x {2*grid+NUM_ACTIONS})")
    print(f"Episodes    : {n_episodes}   Seeds : {n_seeds}   Eval every : {args.eval_every}")
    print("=" * 72 + "\n")

    env = env_cls(rows=grid, cols=grid)
    print("Computing exact Q* via value iteration ...", end=" ", flush=True)
    q_star = value_iteration_q(env, gamma=GAMMA)
    opt_mask = optimal_action_mask(q_star)
    print("done.\n")

    configs = []

    print("Tabular baseline:")
    configs.append(run_config(
        "tabular", 0, env, q_star, opt_mask, n_episodes, args.eval_every,
        MAX_STEPS, TAB_LR, CP_LR, EPSILON_DECAY, n_seeds, "Tabular"))

    for rank in args.ranks:
        print(f"\nCP rank={rank}:")
        configs.append(run_config(
            "cp", rank, env, q_star, opt_mask, n_episodes, args.eval_every,
            MAX_STEPS, TAB_LR, CP_LR, EPSILON_DECAY, n_seeds, f"CP rank={rank}"))

    # ── Summary table ──
    tab_params = grid * grid * NUM_ACTIONS
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Config':<16}{'Params':<10}{'vs table':<10}"
          f"{'Final acc':<12}{'Final return':<14}{'Eps to 90%':<10}")
    print("-" * 72)
    for cfg in configs:
        acc = np.array(cfg["acc_mean"])
        x = np.array(cfg["checkpoints"])
        hit = np.where(acc >= 0.9)[0]
        eps90 = str(int(x[hit[0]])) if len(hit) else "never"
        ratio = f"{cfg['param_count'] / tab_params:.2f}x"
        print(f"{cfg['label']:<16}{cfg['param_count']:<10}{ratio:<10}"
              f"{cfg['acc_mean'][-1]:<12.3f}{cfg['ret_mean'][-1]:<14.2f}{eps90:<10}")
    print("=" * 72)

    # ── Save artifacts ──
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Representation-capacity analysis: how much of exact Q* each rank can store
    print("\nComputing CP representation capacity (% of Q* captured per rank) ...",
          end=" ", flush=True)
    capture_ranks = sorted(set([1, 2, 3, 4, 6, 8, 10, 12] + list(args.ranks)))
    capture_curve = cp_capture_curve(q_star, capture_ranks)
    print("done.")
    print(f"{'rank':<6}{'% of Q* captured':<18}")
    for c in capture_curve:
        print(f"{c['rank']:<6}{c['pct_captured']:<18.3f}")

    png_path = os.path.join(out_dir, f"cp_speedup_{args.start}_{stamp}.png")
    plot_results(configs, grid, png_path,
                 capture_curve=capture_curve, tabular_params=tab_params,
                 trained_ranks=list(args.ranks), start_label=start_label)

    json_path = os.path.join(out_dir, f"cp_speedup_{args.start}_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({"grid": grid, "n_episodes": n_episodes, "n_seeds": n_seeds,
                   "tabular_params": tab_params, "configs": configs}, f, indent=2)
    print(f"[OK] Results saved to {json_path}")


if __name__ == "__main__":
    main()

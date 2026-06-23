#!/usr/bin/env python3
"""
Animate how the learned policy/value map converges during training:
Optimal | Tabular | CP, side by side, as an MP4.

Same per-tile encoding as policy_map.py:
  * arrow      = greedy action argmax_a Q(s,a)  (which cell it would step into)
  * arrow col  = green if optimal (matches argmax of exact Q*), red if not
  * heatmap    = learned value V(s)=max_a Q(s,a), on a FIXED colour scale
                 (vmin..0 from Q*) so you can watch the value gradient fill in
  * gold star  = goal (bottom-right)

We snapshot every `--snap-every` episodes and play the snapshots back, so equal
frame time = equal episodes -> the video honestly shows *who converges faster*.

Run (needs ffmpeg, which is present):
    python policy_map_video.py                      # 12x12, 10000 episodes
    python policy_map_video.py --grid 16 --episodes 6000
"""

import argparse
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.tabular_baseline import value_iteration_q
from envs.gridworld import GridWorld, NUM_ACTIONS, _DELTA
from cp_speedup_experiment import (make_agent, RandomStartGridWorld,
                                   GAMMA, TAB_LR, CP_LR, EPSILON_DECAY)

CMAP = "viridis"


# ── Training with periodic snapshots ─────────────────────────────────────────

def dense_q(kind, agent):
    """Full (rows, cols, actions) Q-tensor for either agent type."""
    return agent.Q if kind == "tabular" else agent.to_dense()


def train_with_snapshots(kind, rank, env, n_episodes, max_steps, snap_every, seed):
    """Train one agent; return list of (episode, V_grid, greedy_grid) snapshots."""
    random.seed(seed)
    np.random.seed(seed)
    agent, _ = make_agent(kind, env.rows, env.cols, rank,
                          TAB_LR, CP_LR, EPSILON_DECAY)
    snaps = []

    def snapshot(ep):
        Q = dense_q(kind, agent)
        snaps.append((ep, Q.max(axis=2), Q.argmax(axis=2)))

    snapshot(0)
    for ep in range(n_episodes):
        state = env.reset()
        for _ in range(max_steps):
            a = agent.select_action(state)
            ns, r, done = env.step(a)
            agent.update(state, a, r, ns, done)
            state = ns
            if done:
                break
        agent.decay_epsilon()
        if (ep + 1) % snap_every == 0:
            snapshot(ep + 1)
    return snaps


# ── Frame drawing ────────────────────────────────────────────────────────────

def quiver_arrays(greedy, opt_mask, goal):
    """Build green (optimal) and red (suboptimal) arrow component arrays."""
    ok = dict(x=[], y=[], u=[], v=[])
    no = dict(x=[], y=[], u=[], v=[])
    G_r, G_c = greedy.shape
    correct = total = 0
    for r in range(G_r):
        for c in range(G_c):
            if (r, c) == goal:
                continue
            a = int(greedy[r, c])
            dr, dc = _DELTA[a]
            d = ok if opt_mask[r, c, a] else no
            d["x"].append(c); d["y"].append(r); d["u"].append(dc); d["v"].append(dr)
            correct += int(opt_mask[r, c, a]); total += 1
    return ok, no, (correct / total if total else float("nan"))


def draw_policy(ax, V, greedy, opt_mask, goal, title, vmin, vmax):
    ax.clear()
    ax.imshow(V, cmap=CMAP, origin="upper", vmin=vmin, vmax=vmax)
    ok, no, acc = quiver_arrays(greedy, opt_mask, goal)
    common = dict(angles="xy", scale_units="xy", scale=1.0 / 0.42,
                  width=0.012, pivot="mid", edgecolor="black", linewidth=0.4)
    if ok["x"]:
        ax.quiver(ok["x"], ok["y"], ok["u"], ok["v"], color="#39ff14", **common)
    if no["x"]:
        ax.quiver(no["x"], no["y"], no["u"], no["v"], color="#ff1744", **common)
    ax.scatter([goal[1]], [goal[0]], marker="*", s=260, color="gold",
               edgecolors="black", linewidth=1.2, zorder=6)
    ax.set_title(f"{title}   accuracy = {acc*100:.1f}%", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    return acc


# ── Build the animation ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--snap-every", type=int, default=50)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", choices=["random", "fixed"], default="random",
                   help="random = spawn anywhere (CP shines); "
                        "fixed = always start at (0,0)")
    args = p.parse_args()

    G = args.grid
    start_cls = RandomStartGridWorld if args.start == "random" else GridWorld
    start_label = "random-start" if args.start == "random" else "fixed-start (0,0)"
    env_eval = GridWorld(G, G)
    print(f"Computing exact Q* for {G}x{G} grid ...", end=" ", flush=True)
    q_star = value_iteration_q(env_eval, gamma=GAMMA)
    opt_mask = q_star == q_star.max(axis=2, keepdims=True)
    vmin, vmax = float(q_star.min()), float(q_star.max())
    print("done.")

    print(f"Training TABULAR ({start_label}) for {args.episodes} episodes ...", flush=True)
    tab_snaps = train_with_snapshots("tabular", args.rank,
                                     start_cls(G, G), args.episodes,
                                     args.max_steps, args.snap_every, args.seed)
    print(f"Training CP rank={args.rank} ({start_label}) for {args.episodes} episodes ...", flush=True)
    cp_snaps = train_with_snapshots("cp", args.rank,
                                    start_cls(G, G), args.episodes,
                                    args.max_steps, args.snap_every, args.seed)

    n_frames = min(len(tab_snaps), len(cp_snaps))
    print(f"Rendering {n_frames} frames ...", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    # static optimal panel + one shared colourbar
    draw_policy(axes[0], q_star.max(axis=2), q_star.argmax(axis=2), opt_mask,
                env_eval.goal, "OPTIMAL (Q*)", vmin, vmax)
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=CMAP)
    fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02,
                 label="learned value V(s) = max_a Q(s,a)")
    suptitle = fig.suptitle("", fontsize=15, fontweight="bold")

    def update(i):
        ep_t, Vt, gt = tab_snaps[i]
        ep_c, Vc, gc = cp_snaps[i]
        at = draw_policy(axes[1], Vt, gt, opt_mask, env_eval.goal,
                         "TABULAR", vmin, vmax)
        ac = draw_policy(axes[2], Vc, gc, opt_mask, env_eval.goal,
                         f"CP rank={args.rank}", vmin, vmax)
        suptitle.set_text(
            f"Policy/value convergence on {G}x{G} {start_label} grid  —  "
            f"episode {ep_t:>5d} / {args.episodes}    "
            f"(tabular {at*100:.0f}%  vs  CP {ac*100:.0f}%)")
        return axes

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"policy_convergence_{args.start}_{G}x{G}.mp4")
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
    anim.save(out_path, writer=writer, dpi=90)
    plt.close(fig)
    print(f"\n[OK] Video saved to {out_path}  ({n_frames} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()

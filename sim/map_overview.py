#!/usr/bin/env python3
"""Overview figure of every map: layout (rows) x grid size (columns).

Each cell shows the maze for one (layout, size): walls in black, the value
surface V*(s) as the heatmap (so the geodesic-to-goal structure is visible),
start (cyan square), goal (gold star) and the optimal path (white line). Cells
where the layout is unsolvable at that size (swirl needs >=11, symmetric >=8)
are marked N/A.

    python map_overview.py                       # default sizes 8 12 24 48
    python map_overview.py --sizes 12 20 32 48
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.tabular_baseline import value_iteration_q
from envs.gridworld import _DELTA
from envs.minigrid import LAYOUTS

GAMMA = 0.99
LAYOUT_ORDER = ["open", "chicane", "symmetric", "islands", "swirl"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "_summary")


def greedy_path(q_star, env, max_len=600):
    state = (0, 0); path = [state]
    for _ in range(max_len):
        if state == env.goal:
            break
        a = int(np.argmax(q_star[state]))
        dr, dc = _DELTA[a]
        nr = max(0, min(env.rows - 1, state[0] + dr))
        nc = max(0, min(env.cols - 1, state[1] + dc))
        if (nr, nc) in env.walls:
            nr, nc = state
        state = (nr, nc); path.append(state)
        if len(path) > 3 and state == path[-3]:
            break
    return path


def draw_cell(ax, layout, size):
    env = LAYOUTS[layout](size, random_start=False)
    q = value_iteration_q(env, gamma=GAMMA)
    ax.set_xticks([]); ax.set_yticks([])
    if float(q[0, 0].max()) <= -90:                       # goal unreachable
        ax.text(0.5, 0.5, "N/A\n(needs larger grid)", ha="center", va="center",
                fontsize=9, color="#666", transform=ax.transAxes)
        ax.set_facecolor("#eee")
        return
    V = np.full((size, size), np.nan)
    for r in range(size):
        for c in range(size):
            if (r, c) not in env.walls:
                V[r, c] = float(q[r, c].max())
    vals = V[~np.isnan(V)]
    cmap = matplotlib.colormaps["viridis"].copy(); cmap.set_bad("#111111")
    ax.imshow(np.ma.masked_invalid(V), cmap=cmap, origin="upper",
              vmin=float(vals.min()), vmax=float(vals.max()))
    path = greedy_path(q, env)
    ax.plot([c for _, c in path], [r for r, _ in path], color="white",
            linewidth=1.6, alpha=0.85)
    ax.scatter([0], [0], marker="s", s=45, color="cyan", edgecolors="black", linewidth=0.8)
    ax.scatter([env.goal[1]], [env.goal[0]], marker="*", s=120, color="gold",
               edgecolors="black", linewidth=0.8)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 24, 48])
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows, cols = len(LAYOUT_ORDER), len(args.sizes)
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 2.7 * rows), squeeze=False)
    for i, layout in enumerate(LAYOUT_ORDER):
        for j, size in enumerate(args.sizes):
            print(f"  {layout} {size}x{size} ...", flush=True)
            draw_cell(axes[i][j], layout, size)
            if i == 0:
                axes[i][j].set_title(f"{size}x{size}", fontsize=13, fontweight="bold")
            if j == 0:
                axes[i][j].set_ylabel(layout, fontsize=13, fontweight="bold")

    fig.suptitle("Map overview — layout (rows) x grid size (columns)\n"
                 "walls=black, value surface=heatmap, start=cyan, goal=gold, white=optimal path",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "map_overview.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()

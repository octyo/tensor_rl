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
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.tabular_baseline import value_iteration_q
from envs.gridworld import _DELTA
from envs.minigrid import LAYOUTS, MINIGRID_ENVS

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


def _reachable(layout, size):
    """BFS from start to goal over free cells (cheap solvability check)."""
    env = LAYOUTS[layout](size, random_start=False)
    free = {(r, c) for r in range(size) for c in range(size) if (r, c) not in env.walls}
    if (0, 0) not in free:
        return False
    seen = {(0, 0)}; q = deque(seen)
    while q:
        r, c = q.popleft()
        if (r, c) == env.goal:
            return True
        for dr, dc in _DELTA.values():
            nb = (r + dr, c + dc)
            if nb in free and nb not in seen:
                seen.add(nb); q.append(nb)
    return env.goal in seen


def draw_minigrid_cell(ax, layout, size):
    """Render the real MiniGrid graphics for this (layout, size)."""
    env = MINIGRID_ENVS[layout](inner_size=size, render_mode="rgb_array")
    try:
        env.unwrapped.highlight = False          # cleaner full-map render (no agent-view tint)
    except Exception:
        pass
    env.reset()
    ax.imshow(env.render())
    ax.set_xticks([]); ax.set_yticks([])
    if not _reachable(layout, size):
        ax.text(0.5, 0.04, "unsolvable at this size", ha="center", va="bottom",
                fontsize=8, color="red", fontweight="bold", transform=ax.transAxes)


def _build_grid(sizes, draw_fn, title, out_name, cell=2.7):
    rows, cols = len(LAYOUT_ORDER), len(sizes)
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols, cell * rows), squeeze=False)
    for i, layout in enumerate(LAYOUT_ORDER):
        for j, size in enumerate(sizes):
            print(f"  [{out_name}] {layout} {size}x{size} ...", flush=True)
            draw_fn(axes[i][j], layout, size)
            if i == 0:
                axes[i][j].set_title(f"{size}x{size}", fontsize=13, fontweight="bold")
            if j == 0:
                axes[i][j].set_ylabel(layout, fontsize=13, fontweight="bold")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, out_name)
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 24, 48])
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    _build_grid(args.sizes, draw_cell,
                "Map overview (value surface) — layout (rows) x grid size (columns)\n"
                "walls=black, value heatmap, start=cyan, goal=gold, white=optimal path",
                "map_overview.png")
    _build_grid(args.sizes, draw_minigrid_cell,
                "Map overview (MiniGrid graphics) — layout (rows) x grid size (columns)\n"
                "real minigrid render: agent=red triangle, walls=grey, goal=green",
                "map_overview_minigrid.png")


if __name__ == "__main__":
    main()

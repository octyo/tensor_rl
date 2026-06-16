#!/usr/bin/env python3
"""Render two NEW preview maps (no experiments, no overwriting existing figures):
  00_map_jforce.png   J-shaped wall + second hook by the goal -> forces the backtrack
  00_map_islands.png  scattered random islands (connected so all cells reach goal)
"""

import os
import random
import sys
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.tabular_baseline import value_iteration_q
from envs.gridworld import _DELTA
from envs.obstacle_gridworld import ObstacleGridWorld, chicane_walls
from obstacle_lab import draw_policy_map, greedy_path, REPORT_DIR, GAMMA


def _connect4(a, b):
    (r0, c0), (r1, c1) = a, b
    cells = set(); r, c = r0, c0; cells.add((r, c))
    while (r, c) != (r1, c1):
        if abs(r1 - r) >= abs(c1 - c) and r != r1:
            r += 1 if r1 > r else -1
        elif c != c1:
            c += 1 if c1 > c else -1
        else:
            r += 1 if r1 > r else -1
        cells.add((r, c))
    return cells


def jforce_walls(G):
    """Proven chicane (forces down->across->UP->goal) dressed as a J: the left wall
    gets a hook curling off its bottom (the J), and the goal-side seal is the
    second, taller hook."""
    walls = set(chicane_walls(G, G))
    left_col = G - 4
    # hook curling left off the bottom of the left 'stem' -> makes it read as a J
    hook = [(G - 2, left_col), (G - 2, left_col - 3), (G - 5, left_col - 3)]
    for i in range(len(hook) - 1):
        walls |= _connect4(hook[i], hook[i + 1])
    walls.discard((0, 0)); walls.discard((G - 1, G - 1))
    return {(r, c) for (r, c) in walls if 0 <= r < G and 0 <= c < G}


def _connected_to_goal(walls, G):
    """All free cells reachable from the goal via 4-connected moves?"""
    free = {(r, c) for r in range(G) for c in range(G) if (r, c) not in walls}
    if (G - 1, G - 1) not in free:
        return False
    seen = {(G - 1, G - 1)}; q = deque(seen)
    while q:
        r, c = q.popleft()
        for dr, dc in _DELTA.values():
            nb = (r + dr, c + dc)
            if nb in free and nb not in seen:
                seen.add(nb); q.append(nb)
    return len(seen) == len(free)


def island_walls(G, n_islands=16, seed=0, max_block=2):
    """Scatter small rectangular islands; keep regenerating until fully connected."""
    forbidden = set()
    for s in [(0, 0), (G - 1, G - 1)]:
        forbidden.add(s)
        for dr, dc in _DELTA.values():
            forbidden.add((s[0] + dr, s[1] + dc))
    for attempt in range(seed, seed + 500):
        rng = random.Random(attempt)
        walls = set()
        for _ in range(n_islands):
            r = rng.randint(0, G - 1); c = rng.randint(0, G - 1)
            bh = rng.randint(1, max_block); bw = rng.randint(1, max_block)
            block = {(r + i, c + j) for i in range(bh) for j in range(bw)
                     if r + i < G and c + j < G}
            if block & forbidden:
                continue
            walls |= block
        walls -= forbidden
        if walls and _connected_to_goal(walls, G):
            return frozenset(walls)
    raise RuntimeError("no connected island layout found")


def render(env, title, fname, G):
    q_star = value_iteration_q(env, gamma=GAMMA)
    opt_mask = q_star == q_star.max(axis=2, keepdims=True)
    vals = np.array([q_star[r, c].max() for (r, c) in env.all_states()])
    vmin, vmax = float(vals.min()), float(vals.max())
    path = greedy_path(lambda r, c: q_star[r, c], env)
    ups = 0
    for (a, b) in zip(path, path[1:]):
        if b[0] < a[0] or b[1] < a[1]:
            ups += 1
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    draw_policy_map(ax, lambda r, c: q_star[r, c], env, opt_mask, title, vmin, vmax,
                    show_path=True)
    fig.suptitle(f"{title}  —  walls=black, start=cyan, goal=gold star, white=optimal path",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(REPORT_DIR, fname)
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {fname}: {len(env.walls)} walls, path len {len(path)-1}, "
          f"backward(up/left) moves {ups}, reaches goal {path[-1]==env.goal}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    Gj = 15
    render(ObstacleGridWorld(Gj, Gj, walls=jforce_walls(Gj)),
           f"J-maze that forces a backtrack ({Gj}x{Gj})", "00_map_jforce.png", Gj)
    Gi = 15
    render(ObstacleGridWorld(Gi, Gi, walls=island_walls(Gi, n_islands=16, seed=1)),
           f"Random islands ({Gi}x{Gi})", "00_map_islands.png", Gi)


if __name__ == "__main__":
    main()

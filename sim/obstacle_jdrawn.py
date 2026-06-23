#!/usr/bin/env python3
"""Reproduce the hand-drawn J/maze sketch as walls. Saves to a NEW file
data/obstacle_report/00_map_jdrawn.png (nothing else is overwritten).

Three strokes, matched to the sketch:
  A  upper cane/hook  : vertical stem curving over to the right at the top
  B  mid-right wall   : near-horizontal, gap before the right edge
  C  lower vertical   : curves down to the bottom edge
Coordinates are normalised (x=col fraction 0..1, y=row fraction 0..1).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.tabular_baseline import value_iteration_q
from envs.obstacle_gridworld import ObstacleGridWorld
from obstacle_lab import draw_policy_map, greedy_path, REPORT_DIR, GAMMA

G = 17

# (x, y) control points in [0,1], y grows downward
STROKE_A = [(0.24, 0.63), (0.235, 0.50), (0.235, 0.41), (0.25, 0.34),
            (0.30, 0.295), (0.37, 0.282), (0.45, 0.288), (0.52, 0.30)]
STROKE_B = [(0.47, 0.42), (0.58, 0.435), (0.70, 0.45), (0.83, 0.455), (0.88, 0.46)]
STROKE_C = [(0.40, 0.52), (0.405, 0.61), (0.42, 0.70), (0.42, 0.79),
            (0.41, 0.88), (0.40, 0.97)]


def _bres(a, b):
    (r0, c0), (r1, c1) = a, b
    cells = set()
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.add((r, c))
        if (r, c) == (r1, c1):
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc
    return cells


def strokes_to_walls(strokes, G):
    walls = set()
    for stroke in strokes:
        pts = [(round(y * (G - 1)), round(x * (G - 1))) for (x, y) in stroke]
        for i in range(len(pts) - 1):
            walls |= _bres(pts[i], pts[i + 1])
    walls.discard((0, 0)); walls.discard((G - 1, G - 1))
    return {(r, c) for (r, c) in walls if 0 <= r < G and 0 <= c < G}


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    walls = strokes_to_walls([STROKE_A, STROKE_B, STROKE_C], G)
    env = ObstacleGridWorld(G, G, walls=walls)
    q_star = value_iteration_q(env, gamma=GAMMA)
    opt_mask = q_star == q_star.max(axis=2, keepdims=True)
    vals = np.array([q_star[r, c].max() for (r, c) in env.all_states()])
    vmin, vmax = float(vals.min()), float(vals.max())
    path = greedy_path(lambda r, c: q_star[r, c], env)
    ups = sum(1 for a, b in zip(path, path[1:]) if b[0] < a[0] or b[1] < a[1])

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    draw_policy_map(ax, lambda r, c: q_star[r, c], env, opt_mask,
                    "Sketched maze (optimal policy)", vmin, vmax, show_path=True)
    fig.suptitle(f"Hand-drawn maze ({G}x{G}) — walls=black, start=cyan, "
                 f"goal=gold star, white=optimal path", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(REPORT_DIR, "00_map_jdrawn.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}: {len(env.walls)} walls, path len {len(path)-1}, "
          f"backward moves {ups}, reaches goal {path[-1]==env.goal}")


if __name__ == "__main__":
    main()

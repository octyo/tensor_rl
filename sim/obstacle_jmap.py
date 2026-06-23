#!/usr/bin/env python3
"""Render ONE J-shaped obstacle map (does not touch the chicane figures).

Saves to data/obstacle_report/00_map_jshape.png — a separate file, so the
existing 00_map_overview.png (chicane) is preserved.
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.tabular_baseline import value_iteration_q
from envs.obstacle_gridworld import ObstacleGridWorld
from obstacle_lab import draw_policy_map, REPORT_DIR, GAMMA


def _connect(a, b):
    (r0, c0), (r1, c1) = a, b
    cells = set()
    n = max(abs(r1 - r0), abs(c1 - c0)) or 1
    for i in range(n + 1):
        cells.add((round(r0 + (r1 - r0) * i / n), round(c0 + (c1 - c0) * i / n)))
    return cells


def j_walls(G):
    """A J: vertical stem from the top edge + a hook curling left-and-up at the bottom."""
    cs = round(G * 0.60)
    r_top, r_bot = 0, round(G * 0.60)
    rad = round(G * 0.20)
    cx_r, cx_c = r_bot, cs - rad
    pts = [(r, cs) for r in range(r_top, r_bot + 1)]          # stem
    for deg in range(0, 235, 3):                              # hook: down -> left -> curl up
        th = math.radians(deg)
        pts.append((round(cx_r + rad * math.sin(th)), round(cx_c + rad * math.cos(th))))
    cells = set()
    for i in range(len(pts) - 1):
        cells |= _connect(pts[i], pts[i + 1])
    cells = {(r, c) for (r, c) in cells if 0 <= r < G and 0 <= c < G}
    cells.discard((0, 0))
    cells.discard((G - 1, G - 1))
    return cells


def main():
    G = 15
    os.makedirs(REPORT_DIR, exist_ok=True)
    env = ObstacleGridWorld(G, G, walls=j_walls(G))
    print(f"J-maze {G}x{G}: {len(env.walls)} wall cells, "
          f"{len(env.nonterminal_states())} free non-goal cells.")
    print("Computing exact Q* ...", end=" ", flush=True)
    q_star = value_iteration_q(env, gamma=GAMMA)
    opt_mask = q_star == q_star.max(axis=2, keepdims=True)
    print("done.")

    vals = np.array([q_star[r, c].max() for (r, c) in env.all_states()])
    vmin, vmax = float(vals.min()), float(vals.max())

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    draw_policy_map(ax, lambda r, c: q_star[r, c], env, opt_mask,
                    "OPTIMAL policy on the J-maze", vmin, vmax, show_path=True)
    fig.suptitle(f"J-shaped obstacle ({G}x{G}) — walls=black, start=cyan, "
                 f"goal=gold star, white=optimal path", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(REPORT_DIR, "00_map_jshape.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[OK] saved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Accuracy-vs-parameters trade-off grid: layout (rows) x grid size (columns),
same matrix layout as map_overview / perf_overview.

Each cell plots final greedy-policy accuracy (y) against parameter count (x, log):
  * star      = tabular table  (S*S*A params, accuracy from final_acc["Tabular"])
  o-line      = CP at rank 3 -> 6 -> 12  (R*(2S+A) params each)
  blue = fixed start, orange = random start.

Reading it: a CP point that sits up-and-to-the-LEFT of the tabular star means CP
reached comparable accuracy with fewer parameters (a win); down-and-left means it
traded accuracy for compression.

    python tradeoff_grid.py                       # full sizes
    python tradeoff_grid.py --sizes 8 12 24 48 --fig tradeoff_grid_small.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.gridworld import NUM_ACTIONS as A
from perf_overview import index_runs, LAYOUT_ORDER, SPAWNS, SPAWN_COLOR, OUT

RANKS = [3, 6, 12]


def draw_cell(ax, runs, layout, size, fs=1.0):
    tabp = size * size * A
    ax.set_ylim(0, 1.06); ax.set_xscale("log")
    ax.tick_params(labelsize=7 * fs)
    ax.grid(True, which="both", alpha=0.25, lw=0.4)
    ax.axhline(1.0, color="#999", lw=0.8, ls="--", zorder=0)
    any_data = False
    for sp in SPAWNS:
        m = runs.get((layout, size, sp))
        if m is None:
            continue
        any_data = True
        fa = m.get("final_acc", {})
        col = SPAWN_COLOR[sp]
        xs = [r * (2 * size + A) for r in RANKS]
        ys = [fa.get(f"CP rank={r}", np.nan) for r in RANKS]
        ax.plot(xs, ys, "-o", color=col, ms=5 * fs, lw=1.4 * fs, label=f"CP ({sp})")
        for r, x, y in zip(RANKS, xs, ys):
            if y == y:
                ax.annotate(str(r), (x, y), textcoords="offset points",
                            xytext=(3, 3), fontsize=6 * fs, color=col)
        tacc = fa.get("Tabular", np.nan)
        if tacc == tacc:
            ax.scatter([tabp], [tacc], marker="*", s=140 * fs, color=col,
                       edgecolors="black", linewidth=0.6, zorder=5,
                       label=f"Table ({sp})")
    if not any_data:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=11,
                color="#666", transform=ax.transAxes)
        ax.set_facecolor("#eee")


def build_figure(sizes, runs, out_name="tradeoff_grid.png", cell=2.7, fs=1.0):
    rows, cols = len(LAYOUT_ORDER), len(sizes)
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols, cell * rows),
                             squeeze=False)
    for i, layout in enumerate(LAYOUT_ORDER):
        for j, size in enumerate(sizes):
            ax = axes[i][j]
            draw_cell(ax, runs, layout, size, fs=fs)
            if i == 0:
                ax.set_title(f"{size}x{size}", fontsize=13 * fs, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{layout}\naccuracy", fontsize=11 * fs, fontweight="bold")
            if i == rows - 1:
                ax.set_xlabel("# parameters (log)", fontsize=9 * fs)
    handles = [
        plt.Line2D([], [], color=SPAWN_COLOR["fixed"], marker="o", label="CP fixed"),
        plt.Line2D([], [], color=SPAWN_COLOR["random"], marker="o", label="CP random"),
        plt.Line2D([], [], color=SPAWN_COLOR["fixed"], marker="*", ls="",
                   ms=12, mec="black", label="Table fixed"),
        plt.Line2D([], [], color=SPAWN_COLOR["random"], marker="*", ls="",
                   ms=12, mec="black", label="Table random"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10, ncol=4)
    fig.suptitle("Accuracy vs parameters — Tabular (star) vs CP rank 3/6/12 (line)\n"
                 "layout (rows) x grid size (columns); up-and-left of the star = "
                 "CP wins (same accuracy, fewer params)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, out_name)
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=None)
    p.add_argument("--fig", default="tradeoff_grid.png")
    p.add_argument("--cell", type=float, default=2.7, help="subplot size (inches)")
    p.add_argument("--fs", type=float, default=1.0, help="font scale")
    args = p.parse_args()
    runs = index_runs()
    if not runs:
        sys.exit("no minigrid_* runs with metrics.json found in data/")
    sizes = args.sizes or sorted({k[1] for k in runs})
    print(f"layouts={LAYOUT_ORDER}  sizes={sizes}  ({len(runs)} runs)")
    build_figure(sizes, runs, out_name=args.fig, cell=args.cell, fs=args.fs)


if __name__ == "__main__":
    main()

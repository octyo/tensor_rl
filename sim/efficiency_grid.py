#!/usr/bin/env python3
"""Parameter-efficiency grid: layout (rows) x grid size (columns), same matrix
layout as perf_overview / map_overview.

Metric per method = final greedy-policy accuracy / number of parameters,
expressed as ACCURACY PER 1,000 PARAMETERS (acc / params * 1000) so the numbers
are readable. Higher = more accuracy squeezed out of each parameter.

  params: tabular  = S*S*A ;  CP rank R = R*(2S+A)   (A = 4 actions)

Each cell is a grouped bar chart over {Tabular, CP r3, CP r6, CP r12}, with two
bars per group = fixed vs random start. Note: each cell auto-scales its y-axis
(param counts differ by grid size), so compare WITHIN a cell, not across cells.

    python efficiency_grid.py                       # full sizes
    python efficiency_grid.py --sizes 8 12 24 48 --fig efficiency_grid_small.png
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
from perf_overview import (index_runs, LAYOUT_ORDER, SPAWNS, SPAWN_COLOR,
                           METHODS, SHORT, OUT)


def n_params(method, size):
    if method == "Tabular":
        return size * size * A
    r = int(method.split("=")[1])           # "CP rank=3" -> 3
    return r * (2 * size + A)


def draw_cell(ax, runs, layout, size, fs=1.0):
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(SHORT, fontsize=8 * fs)
    ax.tick_params(axis="y", labelsize=7 * fs)
    ax.grid(axis="y", lw=0.4, alpha=0.3)
    x = np.arange(len(METHODS)); w = 0.38
    any_data = False
    for i, sp in enumerate(SPAWNS):
        m = runs.get((layout, size, sp))
        if m is None:
            continue
        any_data = True
        fa = m.get("final_acc", {})
        eff = [(fa[meth] / n_params(meth, size) * 1000) if meth in fa else np.nan
               for meth in METHODS]
        ax.bar(x + (i - 0.5) * w, [0 if v != v else v for v in eff], w,
               color=SPAWN_COLOR[sp], label=sp,
               hatch="" if sp == "fixed" else "//",
               edgecolor="white", linewidth=0.4)
        for xi, v in zip(x + (i - 0.5) * w, eff):
            if v == v:
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=5.5 * fs, rotation=90, color="#222")
    if not any_data:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=11,
                color="#666", transform=ax.transAxes)
        ax.set_facecolor("#eee")
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)     # headroom for labels


def build_figure(sizes, runs, out_name="efficiency_grid.png", cell=2.7, fs=1.0):
    rows, cols = len(LAYOUT_ORDER), len(sizes)
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols, cell * rows),
                             squeeze=False)
    for i, layout in enumerate(LAYOUT_ORDER):
        for j, size in enumerate(sizes):
            draw_cell(axes[i][j], runs, layout, size, fs=fs)
            if i == 0:
                axes[i][j].set_title(f"{size}x{size}", fontsize=13 * fs, fontweight="bold")
            if j == 0:
                axes[i][j].set_ylabel(f"{layout}\nacc / 1k params", fontsize=11 * fs,
                                      fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SPAWN_COLOR["fixed"]),
               plt.Rectangle((0, 0), 1, 1, color=SPAWN_COLOR["random"], hatch="//")]
    fig.suptitle("Parameter efficiency", fontsize=16 * fs, fontweight="bold", y=0.995)
    fig.legend(handles, ["fixed start", "random start"], loc="upper center",
               bbox_to_anchor=(0.5, 0.965), fontsize=13 * fs, ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, out_name)
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=None)
    p.add_argument("--fig", default="efficiency_grid.png")
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

#!/usr/bin/env python3
"""Value-error grid: layout (rows) x grid size (columns), mirror of perf_overview
but for the final value error  ||Q_hat - Q*|| / ||Q*||  (lower = better).

Reads final_verr from each data/minigrid_*/metrics.json (produced by
obstacle_lab_verr.py). Each cell is a grouped bar chart over {Tabular, CP r3/6/12}
with fixed vs random start. Each cell auto-scales its y-axis (value-error
magnitude differs by size/layout), so compare WITHIN a cell.

    python value_error_grid.py
    python value_error_grid.py --sizes 8 24 48 --fig value_error_grid_small.png --cell 4.2 --fs 1.7
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_overview import (index_runs, LAYOUT_ORDER, SPAWNS, SPAWN_COLOR,
                           METHODS, SHORT, OUT)


def draw_cell(ax, runs, layout, size, fs=1.0):
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(SHORT, fontsize=8 * fs)
    ax.tick_params(axis="y", labelsize=7 * fs)
    ax.grid(axis="y", lw=0.4, alpha=0.3)
    x = np.arange(len(METHODS)); w = 0.38
    any_data = False
    for i, sp in enumerate(SPAWNS):
        m = runs.get((layout, size, sp))
        if m is None or "final_verr" not in m:
            continue
        any_data = True
        fv = m["final_verr"]
        vals = [fv.get(meth, np.nan) for meth in METHODS]
        ax.bar(x + (i - 0.5) * w, [0 if v != v else v for v in vals], w,
               color=SPAWN_COLOR[sp], label=sp,
               hatch="" if sp == "fixed" else "//",
               edgecolor="white", linewidth=0.4)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            if v == v:
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=5.5 * fs, rotation=90, color="#222")
    if not any_data:
        ax.text(0.5, 0.5, "N/A\n(no value error)", ha="center", va="center",
                fontsize=9, color="#666", transform=ax.transAxes)
        ax.set_facecolor("#eee")
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)


def build_figure(sizes, runs, out_name="value_error_grid.png", cell=2.7, fs=1.0):
    rows, cols = len(LAYOUT_ORDER), len(sizes)
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols, cell * rows),
                             squeeze=False)
    for i, layout in enumerate(LAYOUT_ORDER):
        for j, size in enumerate(sizes):
            draw_cell(axes[i][j], runs, layout, size, fs=fs)
            if i == 0:
                axes[i][j].set_title(f"{size}x{size}", fontsize=13 * fs, fontweight="bold")
            if j == 0:
                axes[i][j].set_ylabel(f"{layout}\nvalue error", fontsize=11 * fs,
                                      fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SPAWN_COLOR["fixed"]),
               plt.Rectangle((0, 0), 1, 1, color=SPAWN_COLOR["random"], hatch="//")]
    fig.suptitle("Final value error  ||Q-Q*|| / ||Q*||  (lower = better)",
                 fontsize=16 * fs, fontweight="bold", y=0.995)
    fig.legend(handles, ["fixed start", "random start"], loc="upper center",
               bbox_to_anchor=(0.5, 0.965), fontsize=13 * fs, ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, out_name)
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}")


def write_table(sizes, runs, out_name="value_error_table.md"):
    lines = ["# Final value error — Tabular vs CP", "",
             "||Q_hat - Q*|| / ||Q*|| over non-wall, non-goal cells (lower = better).", ""]
    lines += ["| layout | size | spawn | " + " | ".join(METHODS) + " |",
              "|" + "---|" * (3 + len(METHODS))]
    for layout in LAYOUT_ORDER:
        for size in sizes:
            for sp in SPAWNS:
                m = runs.get((layout, size, sp))
                fv = (m or {}).get("final_verr", {})
                cells = [f"{fv[k]:.3f}" if k in fv else "—" for k in METHODS]
                lines.append(f"| {layout} | {size} | {sp} | " + " | ".join(cells) + " |")
    out = os.path.join(OUT, out_name)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=None)
    p.add_argument("--fig", default="value_error_grid.png")
    p.add_argument("--table", default="value_error_table.md")
    p.add_argument("--cell", type=float, default=2.7)
    p.add_argument("--fs", type=float, default=1.0)
    args = p.parse_args()
    runs = index_runs()
    if not runs:
        sys.exit("no minigrid_* runs with metrics.json found in data/")
    sizes = args.sizes or sorted({k[1] for k in runs})
    print(f"layouts={LAYOUT_ORDER}  sizes={sizes}  ({len(runs)} runs)")
    build_figure(sizes, runs, out_name=args.fig, cell=args.cell, fs=args.fs)
    write_table(sizes, runs, out_name=args.table)


if __name__ == "__main__":
    main()

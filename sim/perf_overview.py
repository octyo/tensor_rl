#!/usr/bin/env python3
"""Performance overview: layout (rows) x grid size (columns), same matrix layout
as map_overview.py, but each cell compares the FINAL greedy-policy accuracy of
the tabular table vs the CP-decomposed Q at each rank, for BOTH start regimes.

Each cell is a grouped bar chart:
  x-groups = {Tabular, CP r=3, CP r=6, CP r=12}
  two bars per group = fixed (solid) vs random (hatched) start
  y = final policy accuracy in [0,1]  (1.0 = matches optimal policy everywhere)

Reads final_acc from every data/minigrid_*/metrics.json. Also writes a plain
numeric table (perf_table.md) with the same numbers.

    python perf_overview.py                 # auto-detect sizes present in data/
    python perf_overview.py --sizes 8 12 24 48
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(DATA, "_summary")
LAYOUT_ORDER = ["empty", "narrow", "chicane", "islands", "swirl"]
METHODS = ["Tabular", "CP rank=3", "CP rank=6", "CP rank=12"]
SHORT = ["Tab", "r3", "r6", "r12"]
SPAWNS = ["fixed", "random"]
SPAWN_COLOR = {"fixed": "#1f77b4", "random": "#ff7f0e"}

_PAT = re.compile(r"minigrid_(\d+)x\d+_(\d+)_([a-z]+)_(fixed|random)$")


def index_runs():
    """Map (layout, size, spawn) -> metrics dict for every run folder in data/."""
    runs = {}
    for d in glob.glob(os.path.join(DATA, "minigrid_*")):
        m = _PAT.search(os.path.basename(d))
        mj = os.path.join(d, "metrics.json")
        if not m or not os.path.exists(mj):
            continue
        size, _ep, layout, spawn = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        with open(mj) as f:
            runs[(layout, size, spawn)] = json.load(f)
    return runs


def draw_perf_cell(ax, runs, layout, size, fs=1.0):
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(SHORT, fontsize=8 * fs)
    ax.set_ylim(0, 1.06); ax.tick_params(axis="y", labelsize=7 * fs)
    ax.axhline(1.0, color="#999", lw=0.8, ls="--", zorder=0)
    ax.grid(axis="y", lw=0.4, alpha=0.3)
    x = np.arange(len(METHODS)); w = 0.38
    any_data = False
    for i, sp in enumerate(SPAWNS):
        m = runs.get((layout, size, sp))
        if m is None:
            continue
        any_data = True
        fa = m.get("final_acc", {})
        vals = [fa.get(meth, np.nan) for meth in METHODS]
        bars = ax.bar(x + (i - 0.5) * w, [0 if v != v else v for v in vals], w,
                      color=SPAWN_COLOR[sp], label=sp,
                      hatch="" if sp == "fixed" else "//",
                      edgecolor="white", linewidth=0.4)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            if v == v:
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=5.5 * fs, rotation=90, color="#222")
    if not any_data:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=11,
                color="#666", transform=ax.transAxes)
        ax.set_facecolor("#eee")


def build_figure(sizes, runs, out_name="perf_overview.png", cell=2.7, fs=1.0):
    rows, cols = len(LAYOUT_ORDER), len(sizes)
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols, cell * rows),
                             squeeze=False)
    for i, layout in enumerate(LAYOUT_ORDER):
        for j, size in enumerate(sizes):
            draw_perf_cell(axes[i][j], runs, layout, size, fs=fs)
            if i == 0:
                axes[i][j].set_title(f"{size}x{size}", fontsize=13 * fs, fontweight="bold")
            if j == 0:
                axes[i][j].set_ylabel(f"{layout}\nfinal accuracy", fontsize=11 * fs,
                                      fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SPAWN_COLOR["fixed"]),
               plt.Rectangle((0, 0), 1, 1, color=SPAWN_COLOR["random"], hatch="//")]
    fig.suptitle("Final accuracy", fontsize=16 * fs, fontweight="bold", y=0.995)
    fig.legend(handles, ["fixed start", "random start"], loc="upper center",
               bbox_to_anchor=(0.5, 0.965), fontsize=13 * fs, ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, out_name)
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}")


def write_table(sizes, runs, out_name="perf_table.md"):
    lines = ["# Final greedy-policy accuracy — Tabular vs CP",
             "", "acc in [0,1]; 1.0 = matches the optimal policy on every state.", ""]
    header = "| layout | size | spawn | " + " | ".join(METHODS) + " |"
    sep = "|" + "---|" * (3 + len(METHODS))
    lines += [header, sep]
    for layout in LAYOUT_ORDER:
        for size in sizes:
            for sp in SPAWNS:
                m = runs.get((layout, size, sp))
                if m is None:
                    cells = ["—"] * len(METHODS)
                else:
                    fa = m.get("final_acc", {})
                    cells = [f"{fa[k]:.3f}" if k in fa else "—" for k in METHODS]
                lines.append(f"| {layout} | {size} | {sp} | " + " | ".join(cells) + " |")
    out = os.path.join(OUT, out_name)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="grid sizes (default: every size found in data/)")
    p.add_argument("--fig", default="perf_overview.png", help="figure file name")
    p.add_argument("--table", default="perf_table.md", help="table file name")
    p.add_argument("--cell", type=float, default=2.7, help="subplot size (inches)")
    p.add_argument("--fs", type=float, default=1.0, help="font scale")
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    runs = index_runs()
    if not runs:
        sys.exit("no minigrid_* runs with metrics.json found in data/")
    sizes = args.sizes or sorted({k[1] for k in runs})
    print(f"layouts={LAYOUT_ORDER}  sizes={sizes}  ({len(runs)} runs indexed)")
    build_figure(sizes, runs, out_name=args.fig, cell=args.cell, fs=args.fs)
    write_table(sizes, runs, out_name=args.table)


if __name__ == "__main__":
    main()

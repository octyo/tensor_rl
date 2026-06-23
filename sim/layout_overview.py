#!/usr/bin/env python3
"""Per-layout breakdown: final accuracy by map variation (the dimension the
'_avg' figures average away). x = layout, grouped bars per method, averaged over
grid sizes (+/- std over sizes), fixed | random panels.

    python layout_overview.py
    python layout_overview.py --data data_nlms_backup
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "_summary")
LAYOUTS = ["empty", "narrow", "chicane", "islands", "swirl"]
METHODS = ["Tabular", "CP rank=3", "CP rank=6", "CP rank=12"]
COLORS = {"Tabular": "#000000", "CP rank=3": "#1f77b4",
          "CP rank=6": "#2ca02c", "CP rank=12": "#9467bd"}
SPAWNS = ["fixed", "random"]
PAT = re.compile(r"minigrid_(\d+)x\d+_(\d+)_([a-z]+)_(fixed|random)$")


def index_runs(data_dir):
    runs = {}
    for d in glob.glob(os.path.join(data_dir, "minigrid_*")):
        m = PAT.search(os.path.basename(d))
        mj = os.path.join(d, "metrics.json")
        if m and os.path.exists(mj):
            runs[(m.group(3), int(m.group(1)), m.group(4))] = json.load(open(mj))
    return runs


def cell(runs, layout, spawn, meth):
    """mean, std of final_acc over all sizes for one (layout, spawn, method)."""
    vals = [m["final_acc"][meth] for (l, s, sp), m in runs.items()
            if l == layout and sp == spawn and meth in m.get("final_acc", {})]
    return (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(HERE, "data"))
    p.add_argument("--fig", default="accuracy_by_layout.png")
    args = p.parse_args()
    data_dir = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    runs = index_runs(data_dir)
    if not runs:
        sys.exit(f"no runs under {data_dir}")

    x = np.arange(len(LAYOUTS)); w = 0.8 / len(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, spawn in zip(axes, SPAWNS):
        for i, meth in enumerate(METHODS):
            means = [cell(runs, lay, spawn, meth)[0] for lay in LAYOUTS]
            stds = [cell(runs, lay, spawn, meth)[1] for lay in LAYOUTS]
            ax.bar(x + (i - (len(METHODS) - 1) / 2) * w, means, w, yerr=stds,
                   capsize=2, color=COLORS[meth], label=meth,
                   error_kw={"elinewidth": 0.8, "alpha": 0.5})
        ax.set_xticks(x); ax.set_xticklabels(LAYOUTS, fontsize=10)
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{spawn} start", fontsize=12, fontweight="bold")
        ax.set_xlabel("map layout", fontsize=11)
    axes[0].set_ylabel("final accuracy (mean over sizes)", fontsize=11)
    axes[1].legend(fontsize=9)
    fig.suptitle("Final accuracy by map layout  (averaged over grid sizes, +/-1 std)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, args.fig)
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()

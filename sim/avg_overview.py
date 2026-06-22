#!/usr/bin/env python3
"""Compact 'averaged over layouts' summaries: how the metric varies with grid
size, per method, with the 5 layouts collapsed into mean +/- std.

Two figures (each: fixed | random panels, x = grid size, line per method):
  accuracy_avg.png    final greedy-policy accuracy
  efficiency_avg.png  accuracy per 1,000 parameters

    python avg_overview.py                      # reads data/
    python avg_overview.py --data data_nlms_backup
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
from envs.gridworld import NUM_ACTIONS as A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "_summary")
METHODS = ["Tabular", "CP rank=3", "CP rank=6", "CP rank=12"]
COLORS = {"Tabular": "#000000", "CP rank=3": "#1f77b4",
          "CP rank=6": "#2ca02c", "CP rank=12": "#9467bd"}
SPAWNS = ["fixed", "random"]
PAT = re.compile(r"minigrid_(\d+)x\d+_(\d+)_([a-z]+)_(fixed|random)$")


def n_params(method, size):
    if method == "Tabular":
        return size * size * A
    return int(method.split("=")[1]) * (2 * size + A)


def index_runs(data_dir):
    runs = {}
    for d in glob.glob(os.path.join(data_dir, "minigrid_*")):
        m = PAT.search(os.path.basename(d))
        mj = os.path.join(d, "metrics.json")
        if not m or not os.path.exists(mj):
            continue
        size, layout, spawn = int(m.group(1)), m.group(3), m.group(4)
        runs[(layout, size, spawn)] = json.load(open(mj))
    return runs


def collect(runs, spawn, metric):
    """-> sizes, {method: (mean_over_layouts, std_over_layouts)} for the metric."""
    sizes = sorted({s for (_l, s, sp) in runs if sp == spawn})
    layouts = sorted({l for (l, _s, sp) in runs if sp == spawn})
    out = {}
    for meth in METHODS:
        means, stds = [], []
        for s in sizes:
            vals = []
            for lay in layouts:
                m = runs.get((lay, s, spawn))
                if not m or meth not in m.get("final_acc", {}):
                    continue
                acc = m["final_acc"][meth]
                vals.append(acc if metric == "acc" else acc / n_params(meth, s) * 1000)
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else np.nan)
        out[meth] = (np.array(means), np.array(stds))
    return sizes, out


def make_fig(runs, metric, ylabel, title, out_name, ylim=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, spawn in zip(axes, SPAWNS):
        sizes, data = collect(runs, spawn, metric)
        x = np.arange(len(sizes))
        for meth in METHODS:
            m, s = data[meth]
            style = "--" if meth == "Tabular" else "-"
            ax.plot(x, m, style, color=COLORS[meth], lw=2.2, marker="o", ms=5,
                    label=meth)
            ax.fill_between(x, m - s, m + s, color=COLORS[meth], alpha=0.12, linewidth=0)
        ax.set_xticks(x); ax.set_xticklabels([f"{s}x{s}" for s in sizes])
        ax.set_xlabel("grid size", fontsize=11)
        ax.set_title(f"{spawn} start", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel, fontsize=11)
    axes[1].legend(fontsize=9)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, out_name)
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}")


def avg_training(runs, spawn, t_grid):
    """Mean accuracy-over-training per method for one spawn, averaged over all
    layouts+sizes after normalising each curve's x to 0..1 of its episode budget."""
    out = {}
    for meth in METHODS:
        curves = []
        for (lay, s, sp), m in runs.items():
            if sp != spawn:
                continue
            cfg = next((c for c in m.get("configs", []) if c["label"] == meth), None)
            if not cfg or len(cfg["checkpoints"]) < 2:
                continue
            cp = np.array(cfg["checkpoints"], float)
            acc = np.array(cfg["acc_mean"], float)
            curves.append(np.interp(t_grid, cp / cp[-1], acc))
        out[meth] = np.nanmean(np.vstack(curves), axis=0) if curves else None
    return out


def make_training_fig(runs, out_name="training_avg.png"):
    t = np.linspace(0, 1, 41); x = t * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, spawn in zip(axes, SPAWNS):
        series = avg_training(runs, spawn, t)
        for meth in METHODS:
            ys = series[meth]
            if ys is None:
                continue
            style = "--" if meth == "Tabular" else "-"
            ax.plot(x, ys, style, color=COLORS[meth], lw=2.2, label=meth)
        ax.set_ylim(0, 1.03); ax.set_xlim(0, 100); ax.grid(True, alpha=0.3)
        ax.set_xlabel("training progress (% of episodes)", fontsize=11)
        ax.set_title(f"{spawn} start", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("accuracy (mean over layouts+sizes)", fontsize=11)
    axes[1].legend(fontsize=9)
    fig.suptitle("Average training curve  (accuracy vs % of training)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, out_name)
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(HERE, "data"),
                   help="data dir to read (default: data/)")
    args = p.parse_args()
    data_dir = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    runs = index_runs(data_dir)
    if not runs:
        sys.exit(f"no minigrid_* runs with metrics.json under {data_dir}")
    n_lay = len({l for (l, _s, _sp) in runs})
    print(f"read {len(runs)} runs from {data_dir}  ({n_lay} layouts averaged)")
    make_fig(runs, "acc",
             "final accuracy  (mean over layouts)",
             f"Final accuracy vs grid size  (averaged over {n_lay} layouts, +/-1 std)",
             "accuracy_avg.png", ylim=(0, 1.03))
    make_fig(runs, "eff",
             "accuracy per 1,000 params  (mean over layouts)",
             f"Parameter efficiency vs grid size  (averaged over {n_lay} layouts, +/-1 std)",
             "efficiency_avg.png")
    make_training_fig(runs)


if __name__ == "__main__":
    main()

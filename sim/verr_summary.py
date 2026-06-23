#!/usr/bin/env python3
"""Value-error summary suite: the same family of graphs as the accuracy summaries,
but measured by the value error  ||Q_hat - Q*|| / ||Q*||  (lower = better).

Reads a data dir (default sim/data_hpc), uses ONLY folders that have value-error
data (final_verr / configs.verr_mean -- i.e. the obstacle_lab_verr runs), and
writes into <data>/_summary:

  value_error_grid.png (+_medium/_small)  layout x size grid, bars per method
  value_error_avg.png                     value error vs grid size (fixed|random)
  value_error_by_layout.png               value error by map layout (fixed|random)
  value_error_training.png                value error vs % of training (fixed|random)

    python verr_summary.py                       # sim/data_hpc
    python verr_summary.py --data data_hpc
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
LAYOUTS = ["empty", "narrow", "chicane", "islands", "swirl"]
METHODS = ["Tabular", "CP rank=3", "CP rank=6", "CP rank=12"]
SHORT = ["Tab", "r3", "r6", "r12"]
COLORS = {"Tabular": "#000000", "CP rank=3": "#1f77b4",
          "CP rank=6": "#2ca02c", "CP rank=12": "#9467bd"}
SPAWNS = ["fixed", "random"]
SPAWN_COLOR = {"fixed": "#1f77b4", "random": "#ff7f0e"}
A = 4
PAT = re.compile(r"minigrid_(\d+)x\d+_(\d+)_([a-z]+)_(fixed|random)$")


def n_params(meth, size):
    if meth == "Tabular":
        return size * size * A
    return int(meth.split("=")[1]) * (2 * size + A)


def index_runs(data):
    """(layout, size, spawn) -> metrics, ONLY for folders that have value error."""
    runs = {}
    for d in glob.glob(os.path.join(data, "minigrid_*")):
        m = PAT.search(os.path.basename(d))
        mj = os.path.join(d, "metrics.json")
        if not m or not os.path.exists(mj):
            continue
        j = json.load(open(mj))
        if "final_verr" not in j:
            continue
        runs[(m.group(3), int(m.group(1)), m.group(4))] = j
    return runs


# ── layout x size grid of final value error ──────────────────────────────────

def _grid_cell(ax, runs, layout, size, fs):
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(SHORT, fontsize=8 * fs)
    ax.tick_params(axis="y", labelsize=7 * fs); ax.grid(axis="y", lw=0.4, alpha=0.3)
    x = np.arange(len(METHODS)); w = 0.38; any_data = False
    for i, sp in enumerate(SPAWNS):
        m = runs.get((layout, size, sp))
        if m is None:
            continue
        any_data = True
        fv = m["final_verr"]
        vals = [fv.get(me, np.nan) for me in METHODS]
        ax.bar(x + (i - 0.5) * w, [0 if v != v else v for v in vals], w,
               color=SPAWN_COLOR[sp], hatch="" if sp == "fixed" else "//",
               edgecolor="white", linewidth=0.4, label=sp)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            if v == v:
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=5.5 * fs, rotation=90, color="#222")
    if not any_data:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10,
                color="#666", transform=ax.transAxes); ax.set_facecolor("#eee")
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)


def fig_grid(runs, sizes, out, cell=2.7, fs=1.0):
    rows, cols = len(LAYOUTS), len(sizes)
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols, cell * rows), squeeze=False)
    for i, lay in enumerate(LAYOUTS):
        for j, s in enumerate(sizes):
            _grid_cell(axes[i][j], runs, lay, s, fs)
            if i == 0:
                axes[i][j].set_title(f"{s}x{s}", fontsize=13 * fs, fontweight="bold")
            if j == 0:
                axes[i][j].set_ylabel(f"{lay}\nvalue error", fontsize=11 * fs, fontweight="bold")
    h = [plt.Rectangle((0, 0), 1, 1, color=SPAWN_COLOR["fixed"]),
         plt.Rectangle((0, 0), 1, 1, color=SPAWN_COLOR["random"], hatch="//")]
    fig.suptitle("Final value error  ||Q-Q*|| / ||Q*||  (lower = better)",
                 fontsize=16 * fs, fontweight="bold", y=0.995)
    fig.legend(h, ["fixed start", "random start"], loc="upper center",
               bbox_to_anchor=(0.5, 0.965), fontsize=13 * fs, ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig); print(f"[OK] {out}")


# ── value error vs grid size (averaged over layouts) ─────────────────────────

def fig_avg(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ymax = 0
    for ax, spawn in zip(axes, SPAWNS):
        sizes = sorted({s for (_l, s, sp) in runs if sp == spawn})
        layouts = sorted({l for (l, _s, sp) in runs if sp == spawn})
        x = np.arange(len(sizes))
        for meth in METHODS:
            mean, std = [], []
            for s in sizes:
                v = [runs[(l, s, spawn)]["final_verr"][meth] for l in layouts
                     if (l, s, spawn) in runs and meth in runs[(l, s, spawn)]["final_verr"]]
                mean.append(np.mean(v) if v else np.nan); std.append(np.std(v) if v else np.nan)
            mean, std = np.array(mean), np.array(std)
            ymax = max(ymax, np.nanmax(mean + std) if np.isfinite(mean).any() else 0)
            ls = "--" if meth == "Tabular" else "-"
            ax.plot(x, mean, ls, color=COLORS[meth], lw=2.2, marker="o", ms=5, label=meth)
            ax.fill_between(x, mean - std, mean + std, color=COLORS[meth], alpha=0.12, lw=0)
        ax.set_xticks(x); ax.set_xticklabels([f"{s}x{s}" for s in sizes])
        ax.set_xlabel("grid size", fontsize=11); ax.grid(True, alpha=0.3)
        ax.set_title(f"{spawn} start", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_ylim(0, ymax * 1.05)
    axes[0].set_ylabel("final value error (mean over layouts)", fontsize=11)
    axes[1].legend(fontsize=9)
    fig.suptitle("Value error vs grid size  (averaged over layouts, +/-1 std, lower=better)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig); print(f"[OK] {out}")


# ── value error by layout (averaged over sizes) ──────────────────────────────

def fig_by_layout(runs, out):
    x = np.arange(len(LAYOUTS)); w = 0.8 / len(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, spawn in zip(axes, SPAWNS):
        for i, meth in enumerate(METHODS):
            means, stds = [], []
            for lay in LAYOUTS:
                v = [m["final_verr"][meth] for (l, s, sp), m in runs.items()
                     if l == lay and sp == spawn and meth in m["final_verr"]]
                means.append(np.mean(v) if v else np.nan); stds.append(np.std(v) if v else np.nan)
            ax.bar(x + (i - (len(METHODS) - 1) / 2) * w, means, w, yerr=stds, capsize=2,
                   color=COLORS[meth], label=meth, error_kw={"elinewidth": 0.8, "alpha": 0.5})
        ax.set_xticks(x); ax.set_xticklabels(LAYOUTS, fontsize=10); ax.grid(axis="y", alpha=0.3)
        ax.set_xlabel("map layout", fontsize=11)
        ax.set_title(f"{spawn} start", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("final value error (mean over sizes)", fontsize=11)
    axes[1].legend(fontsize=9)
    fig.suptitle("Value error by map layout  (averaged over grid sizes, +/-1 std, lower=better)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig); print(f"[OK] {out}")


# ── value error vs % of training (averaged over layouts+sizes) ───────────────

def fig_training(runs, out):
    t = np.linspace(0, 1, 41); xp = t * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ymax = 0
    for ax, spawn in zip(axes, SPAWNS):
        for meth in METHODS:
            curves = []
            for (l, s, sp), m in runs.items():
                if sp != spawn:
                    continue
                cfg = next((c for c in m.get("configs", []) if c["label"] == meth), None)
                if not cfg or "verr_mean" not in cfg or len(cfg["checkpoints"]) < 2:
                    continue
                cp = np.array(cfg["checkpoints"], float)
                curves.append(np.interp(t, cp / cp[-1], np.array(cfg["verr_mean"], float)))
            if not curves:
                continue
            mean = np.nanmean(np.vstack(curves), axis=0)
            ymax = max(ymax, np.nanmax(mean))
            ls = "--" if meth == "Tabular" else "-"
            ax.plot(xp, mean, ls, color=COLORS[meth], lw=2.2, label=meth)
        ax.set_xlim(0, 100); ax.grid(True, alpha=0.3)
        ax.set_xlabel("training progress (% of episodes)", fontsize=11)
        ax.set_title(f"{spawn} start", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_ylim(0, ymax * 1.05)
    axes[0].set_ylabel("value error (mean over layouts+sizes)", fontsize=11)
    axes[1].legend(fontsize=9)
    fig.suptitle("Value error vs % of training  (averaged, lower=better)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig); print(f"[OK] {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(HERE, "data_hpc"))
    args = p.parse_args()
    data = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    runs = index_runs(data)
    if not runs:
        sys.exit(f"no value-error runs (final_verr) found under {data}")
    out_dir = os.path.join(data, "_summary")
    os.makedirs(out_dir, exist_ok=True)
    sizes = sorted({s for (_l, s, _sp) in runs})
    print(f"{len(runs)} value-error runs | layouts present: "
          f"{sorted({l for (l, _s, _sp) in runs})} | sizes: {sizes}")
    O = lambda n: os.path.join(out_dir, n)
    fig_grid(runs, sizes, O("value_error_grid.png"))
    fig_grid(runs, [s for s in (8, 12, 24, 48) if s in sizes], O("value_error_grid_medium.png"))
    fig_grid(runs, [s for s in (8, 24, 48) if s in sizes], O("value_error_grid_small.png"),
             cell=4.2, fs=1.7)
    fig_avg(runs, O("value_error_avg.png"))
    fig_by_layout(runs, O("value_error_by_layout.png"))
    fig_training(runs, O("value_error_training.png"))
    print(f"\nValue-error summary in {out_dir}")


if __name__ == "__main__":
    main()

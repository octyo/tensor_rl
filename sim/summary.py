#!/usr/bin/env python3
"""Aggregate every minigrid run in sim/data into cross-size x cross-layout figures.

Discovers folders named  minigrid_{S}x{S}_{episodes}_{layout}_{spawn}  and builds:

  summary_rank_capture.png   how low-rank each layout's Q* is, vs grid size
                             (recomputed from Q* directly -- no training needed,
                             so this works on every folder immediately)
  summary_accuracy.png       final CP/tabular policy accuracy vs size, per layout
                             and spawn (read from metrics.json; folders without it
                             -- e.g. older local runs -- are listed and skipped)

Outputs go to sim/data/_summary/.

    python summary.py
"""

import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.tabular_baseline import value_iteration_q
from envs.minigrid import LAYOUTS
from rank_analysis import cp_capture_curve

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.path.join(DATA_ROOT, "_summary")
GAMMA = 0.99
LAYOUT_ORDER = ["open", "chicane", "symmetric", "islands", "swirl"]
LAYOUT_COLORS = {"open": "#1f77b4", "chicane": "#2ca02c", "symmetric": "#9467bd",
                 "islands": "#ff7f0e", "swirl": "#d62728"}
NAME_RE = re.compile(r"minigrid_(\d+)x\d+_(\d+)_([a-z]+)_(random|fixed)$")


def discover():
    runs = []
    if not os.path.isdir(DATA_ROOT):
        return runs
    for name in sorted(os.listdir(DATA_ROOT)):
        m = NAME_RE.match(name)
        if not m:
            continue
        size, episodes, layout, spawn = int(m[1]), int(m[2]), m[3], m[4]
        path = os.path.join(DATA_ROOT, name)
        mpath = os.path.join(path, "metrics.json")
        metrics = json.load(open(mpath)) if os.path.exists(mpath) else None
        runs.append({"size": size, "episodes": episodes, "layout": layout,
                     "spawn": spawn, "path": path, "metrics": metrics})
    return runs


# ── rank capture (recomputed from Q*, cached per (layout,size)) ──────────────

_CAP_CACHE = {}


def capture_for(layout, size, ranks=(1, 2, 3, 6, 12, 20, 30)):
    key = (layout, size)
    if key in _CAP_CACHE:
        return _CAP_CACHE[key]
    env = LAYOUTS[layout](size, random_start=False)
    q_star = value_iteration_q(env, gamma=GAMMA)
    curve = cp_capture_curve(q_star, list(ranks))
    _CAP_CACHE[key] = curve
    return curve


def pct_at_rank(curve, rank):
    for c in curve:
        if c["rank"] == rank:
            return c["pct_captured"]
    return np.nan


def rank_for_pct(curve, target=99.0):
    for c in curve:
        if c["pct_captured"] >= target:
            return c["rank"]
    return np.nan


def plot_rank_capture(runs):
    sizes_by_layout = {}
    for r in runs:
        sizes_by_layout.setdefault(r["layout"], set()).add(r["size"])
    layouts = [l for l in LAYOUT_ORDER if l in sizes_by_layout]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for lay in layouts:
        sizes = sorted(sizes_by_layout[lay])
        col = LAYOUT_COLORS[lay]
        p6 = [pct_at_rank(capture_for(lay, s), 6) for s in sizes]
        r99 = [rank_for_pct(capture_for(lay, s), 99.0) for s in sizes]
        axes[0].plot(sizes, p6, "o-", color=col, label=lay, linewidth=2)
        axes[1].plot(sizes, r99, "o-", color=col, label=lay, linewidth=2)
    axes[0].set_ylabel("% of Q* captured at CP rank 6")
    axes[0].set_title("How low-rank is the value function?", fontweight="bold")
    axes[1].set_ylabel("CP rank needed for 99% capture")
    axes[1].set_title("Rank required to (nearly) represent Q*", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("grid size (S, for SxS)"); ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Rank-capture vs grid size, per layout  (recomputed from exact Q*)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "summary_rank_capture.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}")


def plot_accuracy(runs):
    have = [r for r in runs if r["metrics"]]
    missing = [r for r in runs if not r["metrics"]]
    if missing:
        print(f"[note] {len(missing)} folder(s) lack metrics.json (older runs) -> "
              f"skipped in accuracy fig; re-run them or use the HPC sweep to fill in.")
    if not have:
        print("[note] no metrics.json anywhere yet -> skipping accuracy figure.")
        return

    layouts = [l for l in LAYOUT_ORDER if any(r["layout"] == l for r in have)]
    fig, axes = plt.subplots(2, len(layouts), figsize=(4.2 * len(layouts), 8),
                             squeeze=False)
    for j, lay in enumerate(layouts):
        for i, spawn in enumerate(["random", "fixed"]):
            ax = axes[i][j]
            rs = sorted([r for r in have if r["layout"] == lay and r["spawn"] == spawn],
                        key=lambda r: r["size"])
            if rs:
                sizes = [r["size"] for r in rs]
                fa = [r["metrics"]["final_acc"] for r in rs]
                labels = list(fa[0].keys())
                for lbl in labels:
                    ys = [d.get(lbl, np.nan) for d in fa]
                    style = "k--o" if lbl == "Tabular" else "-o"
                    ax.plot(sizes, ys, style, label=lbl, linewidth=2, markersize=4)
            ax.set_ylim(0, 1.03); ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(lay, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{spawn} start\nfinal policy accuracy")
            ax.set_xlabel("grid size")
            if i == 0 and j == len(layouts) - 1:
                ax.legend(fontsize=7)
    fig.suptitle("Final policy accuracy vs grid size  (tabular vs CP ranks)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "summary_accuracy.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = discover()
    if not runs:
        print(f"No minigrid_* run folders found in {DATA_ROOT}"); return
    print(f"Found {len(runs)} run folders "
          f"({sum(1 for r in runs if r['metrics'])} with metrics.json).")
    plot_rank_capture(runs)
    plot_accuracy(runs)
    print(f"\nSummary figures in {OUT_DIR}")


if __name__ == "__main__":
    main()

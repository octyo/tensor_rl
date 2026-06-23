#!/usr/bin/env python3
"""For every run folder that has value-error data, render a single
value-error-over-training plot as value_error_curve.png, titled

    CP vs tabular value error - {layout} {S}x{S} {spawn}

Reads configs -> checkpoints / verr_mean / verr_std from each metrics.json
(only present for folders produced by obstacle_lab_verr.py). Lower = better.

    python value_error_curves.py
"""

import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PAT = re.compile(r"minigrid_(\d+)x\d+_(\d+)_([a-z]+)_(fixed|random)$")
COLORS = {"Tabular": "#000000", "CP rank=3": "#1f77b4",
          "CP rank=6": "#2ca02c", "CP rank=12": "#9467bd"}


def render(folder, meta):
    size, layout, spawn = meta
    with open(os.path.join(folder, "metrics.json")) as f:
        data = json.load(f)
    cfgs = [c for c in data.get("configs", []) if "verr_mean" in c]
    if not cfgs:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in cfgs:
        lbl = cfg["label"]
        x = cfg["checkpoints"]
        y = np.array(cfg["verr_mean"]); s = np.array(cfg.get("verr_std", np.zeros_like(y)))
        col = COLORS.get(lbl)
        style = "--" if cfg.get("kind") == "tabular" else "-"
        ax.plot(x, y, style, color=col, lw=2, label=lbl)
        ax.fill_between(x, y - s, y + s, color=col, alpha=0.15, linewidth=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("training episodes", fontsize=12)
    ax.set_ylabel("value error  ||Q-Q*|| / ||Q*||", fontsize=12)
    ax.set_title(f"CP vs tabular value error - {layout} {size}x{size} {spawn}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(folder, "value_error_curve.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    n = 0
    for d in sorted(glob.glob(os.path.join(DATA, "minigrid_*"))):
        m = PAT.search(os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "metrics.json")):
            continue
        size, _ep, layout, spawn = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        if render(d, (size, layout, spawn)):
            n += 1
    print(f"[OK] wrote {n} value_error_curve.png files")


if __name__ == "__main__":
    main()

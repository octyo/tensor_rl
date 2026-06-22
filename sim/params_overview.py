#!/usr/bin/env python3
"""Parameter-reduction diagram: tabular Q-table vs CP-decomposed Q.

The Q-tensor is Q[row, col, action] with row=col=S, |action|=A=4.
  tabular params = S * S * A            (grows QUADRATICALLY with S)
  CP rank-R params = R * (S + S + A)    (grows LINEARLY with S)
so the compression factor tab/CP = S*S*A / (R*(2S+A)) improves as the grid grows.
This is layout-independent (it depends only on the dimensions and rank), so one
diagram covers every maze.

Left panel : parameter count vs grid size (log y), tabular + CP at each rank.
Right panel: compression factor (x fewer params) vs grid size, per rank.
Also writes params_reduction.md with the exact numbers.

    python params_overview.py                       # sizes 8 12 16 24 32 48
    python params_overview.py --sizes 8 12 24 48 --ranks 3 6 12
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "_summary")
RANK_COLORS = {3: "#1f77b4", 6: "#2ca02c", 12: "#9467bd"}


def tab_params(s):
    return s * s * A


def cp_params(s, r):
    return r * (2 * s + A)


def build_figure(sizes, ranks, out_name="params_reduction.png", log=True):
    fig, axL = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(sizes))

    # ── absolute parameter counts ────────────────────────────────────────────
    tab = [tab_params(s) for s in sizes]
    axL.plot(x, tab, "o-", color="#d62728", lw=2.5, ms=8, label="Tabular")
    for r in ranks:
        cp = [cp_params(s, r) for s in sizes]
        axL.plot(x, cp, "s--", color=RANK_COLORS.get(r), lw=2, ms=6,
                 label=f"CP rank {r}")
    if log:
        axL.set_yscale("log")
    axL.set_xticks(x); axL.set_xticklabels([f"{s}x{s}" for s in sizes])
    axL.set_xlabel("grid size", fontsize=12)
    axL.set_ylabel("number of parameters", fontsize=12)
    axL.set_title("Parameter count by different map sizes", fontsize=13,
                  fontweight="bold")
    axL.grid(True, which="both", alpha=0.25)
    axL.legend(fontsize=10)
    for xi, t in zip(x, tab):
        axL.annotate(f"{t:,}", (xi, t), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#d62728")

    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, out_name)
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] {out}")


def write_table(sizes, ranks, out_name="params_reduction.md"):
    lines = ["# Parameter reduction — tabular vs CP",
             "", f"Q[S, S, {A}].  tabular = S·S·{A};  CP rank R = R·(2S+{A}).  "
             "Compression = tabular / CP.  Layout-independent.", ""]
    head = "| size | tabular | " + " | ".join(
        f"CP r{r} (params / x)" for r in ranks) + " |"
    lines += [head, "|" + "---|" * (2 + len(ranks))]
    for s in sizes:
        t = tab_params(s)
        cells = [f"{cp_params(s, r):,} / {t / cp_params(s, r):.1f}x" for r in ranks]
        lines.append(f"| {s}x{s} | {t:,} | " + " | ".join(cells) + " |")
    out = os.path.join(OUT, out_name)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 24, 32, 48])
    p.add_argument("--ranks", type=int, nargs="+", default=[3, 6, 12])
    args = p.parse_args()
    print(f"sizes={args.sizes}  ranks={args.ranks}  actions={A}")
    build_figure(args.sizes, args.ranks, "params_reduction.png", log=True)
    build_figure(args.sizes, args.ranks, "params_reduction_linear.png", log=False)
    write_table(args.sizes, args.ranks)


if __name__ == "__main__":
    main()

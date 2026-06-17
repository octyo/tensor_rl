#!/usr/bin/env python3
"""
How much of the optimal Q-tensor can each CP rank represent?

This answers a *representation* question, separate from learning: if you handed
the CP model the best possible factors, what fraction of the exact Q* could a
rank-R decomposition reproduce? It quantifies why low rank is "enough" on the
grid navigation task — the value surface is near-additive, so a couple of CP
components already capture ~99% of it.

% captured := 1 - ||Q* - Q_R||_F / ||Q*||_F          (relative Frobenius)

where Q_R is the best rank-R CP fit (via plain ALS, best of a few random inits).

Reusable:  cp_als(), cp_capture_curve()  are imported by cp_speedup_experiment.py
Standalone: prints a table and saves a "% captured vs rank" figure.
    python rank_analysis.py --grid 20 --ranks 1 2 3 4 6 8 10 12
"""


import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.tabular_baseline import value_iteration_q
from envs.gridworld import GridWorld, NUM_ACTIONS


# ── CP-ALS fit of a 3-way tensor ────────────────────────────────────────────

def cp_als(T: np.ndarray, rank: int, iters: int = 300, seed: int = 0) -> np.ndarray:
    """Best rank-R CP reconstruction of 3-way tensor T via alternating least squares.

    CP model:  T[i,j,k] ~= sum_r A0[i,r] * A1[j,r] * A2[k,r].
    Returns the dense reconstruction (same shape as T). ALS is non-convex, so
    callers typically take the best of several seeds (see cp_capture_curve).
    """
    rng = np.random.default_rng(seed)
    dims = T.shape
    A = [rng.standard_normal((d, rank)) for d in dims]

    def unfold(X, n):
        return np.moveaxis(X, n, 0).reshape(dims[n], -1)

    for _ in range(iters):
        for n in range(3):
            others = [A[m] for m in range(3) if m != n]
            # Khatri-Rao of the two other factors (higher mode varies fastest,
            # matching the C-order mode-n unfolding below).
            kr = (others[0][:, None, :] * others[1][None, :, :]).reshape(-1, rank)
            V = np.ones((rank, rank))
            for m in range(3):
                if m != n:
                    V *= A[m].T @ A[m]
            A[n] = unfold(T, n) @ kr @ np.linalg.pinv(V)

    return np.einsum("ir,jr,kr->ijk", A[0], A[1], A[2])


def cp_capture_curve(q_star: np.ndarray, ranks: list, n_init: int = 4,
                     iters: int = 300) -> list:
    """For each rank, fit CP to q_star and report representation quality.

    Returns a list of dicts: {rank, params, rel_error, pct_captured}.
    params is the CP factor count for q_star's shape, pct_captured = 1 - rel_error.
    """
    norm = float(np.linalg.norm(q_star))
    axis_sizes = sum(q_star.shape)
    out = []
    for R in ranks:
        best = None
        for seed in range(n_init):
            rec = cp_als(q_star, R, iters=iters, seed=seed)
            err = float(np.linalg.norm(q_star - rec)) / norm
            best = err if best is None else min(best, err)
        out.append({"rank": R, "params": R * axis_sizes,
                    "rel_error": best, "pct_captured": (1.0 - best) * 100.0})
    return out


# ── Plot helper (also called from cp_speedup_experiment.py) ──────────────────

def plot_capture_panel(ax, curve: list, tabular_params: int,
                       trained_ranks: list | None = None):
    """Draw a '% of Q* captured vs CP rank' curve onto a given matplotlib axis."""
    ranks = [c["rank"] for c in curve]
    pct = [c["pct_captured"] for c in curve]
    ax.plot(ranks, pct, "o-", color="#2ca02c", linewidth=2.2, markersize=7,
            label="CP fit to exact Q*")
    if trained_ranks:
        for c in curve:
            if c["rank"] in trained_ranks:
                ax.scatter([c["rank"]], [c["pct_captured"]], s=180,
                           facecolors="none", edgecolors="#d62728",
                           linewidth=2.2, zorder=5)
        ax.scatter([], [], s=120, facecolors="none", edgecolors="#d62728",
                   linewidth=2.2, label="ranks trained in experiment")
    ax.axhline(99.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(ranks[-1], 99.1, "99%", ha="right", va="bottom", fontsize=8, color="gray")
    for c in curve:
        ax.annotate(f"{c['pct_captured']:.1f}%", (c["rank"], c["pct_captured"]),
                    textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7.5)
    ax.set_xlabel("CP rank", fontsize=11)
    ax.set_ylabel("% of exact Q* captured", fontsize=11)
    ax.set_title("Representation capacity vs rank", fontsize=13, fontweight="bold")
    ax.set_xticks(ranks)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)


# ── Standalone ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, default=20)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10, 12])
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-init", type=int, default=4)
    args = p.parse_args()

    env = GridWorld(rows=args.grid, cols=args.grid)
    print(f"Computing exact Q* for {args.grid}x{args.grid} grid ...", end=" ", flush=True)
    q_star = value_iteration_q(env, gamma=args.gamma)
    print("done.")
    tab_params = q_star.size

    curve = cp_capture_curve(q_star, args.ranks, n_init=args.n_init)

    print("\n" + "=" * 60)
    print(f"CP representation of exact Q*  ({args.grid}x{args.grid} grid, "
          f"tabular table = {tab_params} params)")
    print("=" * 60)
    print(f"{'rank':<6}{'CP params':<12}{'% of table':<13}{'% Q* captured':<14}")
    print("-" * 60)
    for c in curve:
        print(f"{c['rank']:<6}{c['params']:<12}{100*c['params']/tab_params:<13.1f}"
              f"{c['pct_captured']:<14.3f}")
    print("=" * 60)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rank_capture_{args.grid}x{args.grid}.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_capture_panel(ax, curve, tab_params)
    fig.suptitle(f"How much of Q* a CP rank can represent ({args.grid}x{args.grid} grid)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\n[OK] Figure saved to {out_path}")


if __name__ == "__main__":
    main()

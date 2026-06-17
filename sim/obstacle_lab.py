#!/usr/bin/env python3
"""
Obstacle-maze lab: runs the full CP-vs-tabular battery on the chicane maze and
drops every artifact into data/obstacle_report/.

Same tests as the open-grid experiments, obstacle-aware:
  00_map_overview.png        maze + optimal policy + optimal path
  01_rank_capture.png        % of maze Q* captured vs CP rank
  02_curves_random.png       accuracy & greedy-return over training (random start)
  03_curves_fixed.png        accuracy & greedy-return over training (fixed start)
  04_policymaps_random.png   Optimal | Tabular | CP policy maps (random start)
  05_policymaps_fixed.png    Optimal | Tabular | CP policy maps (fixed start)
  policy_convergence_random.mp4 / _fixed.mp4   side-by-side convergence videos

Run:
    python obstacle_lab.py --map     # just the map (quick check-in)
    python obstacle_lab.py --all     # everything (default)
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.tabular_baseline import value_iteration_q
from envs.gridworld import NUM_ACTIONS, _DELTA
from envs.obstacle_gridworld import ObstacleGridWorld
from rank_analysis import cp_capture_curve, plot_capture_panel
from cp_speedup_experiment import make_agent, GAMMA, TAB_LR, CP_LR, EPSILON_DECAY
from envs.minigrid import LAYOUTS

# new convention: all data under sim/data, one self-contained folder per config
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "data", "obstacle_report")
ENV_NAME = "minigrid"
CMAP = "viridis"
TAB_COLOR = "#d62728"
CP_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]


# ── env factories ────────────────────────────────────────────────────────────

# set in main() per chosen layout; None -> ObstacleGridWorld's default chicane walls
WALLS_OVERRIDE = None
OUT_DIR = REPORT_DIR


def make_env(grid, random_start):
    return ObstacleGridWorld(grid, grid, walls=WALLS_OVERRIDE, random_start=random_start)


# ── metrics (obstacle-aware: skip wall cells and goal) ───────────────────────

def policy_accuracy(get_q_row, env, opt_mask):
    correct = total = 0
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in env.walls or (r, c) == env.goal:
                continue
            a = int(np.argmax(get_q_row(r, c)))
            correct += int(opt_mask[r, c, a]); total += 1
    return correct / total if total else float("nan")


def greedy_return(agent, env, n_rollouts=20, max_steps=200):
    returns = []
    for _ in range(n_rollouts):
        state = env.reset(); total = 0.0
        for _ in range(max_steps):
            a = agent.select_action(state, evaluate=True)
            state, reward, done = env.step(a); total += reward
            if done:
                break
        returns.append(total)
    return float(np.mean(returns))


def greedy_path(get_q_row, env, max_len=400):
    state = (0, 0); path = [state]
    for _ in range(max_len):
        if state == env.goal:
            break
        a = int(np.argmax(get_q_row(*state)))
        dr, dc = _DELTA[a]
        nr = max(0, min(env.rows - 1, state[0] + dr))
        nc = max(0, min(env.cols - 1, state[1] + dc))
        if (nr, nc) in env.walls:
            nr, nc = state
        state = (nr, nc); path.append(state)
        if len(path) > 3 and state == path[-3]:
            break
    return path


# ── drawing ──────────────────────────────────────────────────────────────────

def draw_policy_map(ax, get_q_row, env, opt_mask, title, vmin, vmax, show_path=False):
    ax.clear()
    V = np.full((env.rows, env.cols), np.nan)
    ok = dict(x=[], y=[], u=[], v=[]); no = dict(x=[], y=[], u=[], v=[])
    correct = total = 0
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in env.walls:
                continue
            q = np.asarray(get_q_row(r, c)); V[r, c] = float(np.max(q))
            if (r, c) == env.goal:
                continue
            a = int(np.argmax(q)); dr, dc = _DELTA[a]
            d = ok if opt_mask[r, c, a] else no
            d["x"].append(c); d["y"].append(r); d["u"].append(dc); d["v"].append(dr)
            correct += int(opt_mask[r, c, a]); total += 1

    cmap = matplotlib.colormaps[CMAP].copy(); cmap.set_bad("#111111")
    ax.imshow(np.ma.masked_invalid(V), cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    common = dict(angles="xy", scale_units="xy", scale=1.0 / 0.42,
                  width=0.012, pivot="mid", edgecolor="black", linewidth=0.4)
    if ok["x"]:
        ax.quiver(ok["x"], ok["y"], ok["u"], ok["v"], color="#39ff14", **common)
    if no["x"]:
        ax.quiver(no["x"], no["y"], no["u"], no["v"], color="#ff1744", **common)
    if show_path:
        path = greedy_path(get_q_row, env)
        ax.plot([c for _, c in path], [r for r, _ in path],
                color="white", linewidth=2.0, alpha=0.85, zorder=5)
    ax.scatter([0], [0], marker="s", s=110, color="cyan",
               edgecolors="black", linewidth=1.1, zorder=6)
    ax.scatter([env.goal[1]], [env.goal[0]], marker="*", s=300, color="gold",
               edgecolors="black", linewidth=1.1, zorder=7)
    acc = correct / total if total else float("nan")
    ax.set_title(f"{title}   accuracy = {acc*100:.1f}%", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    return acc


# ── training with learning curve ─────────────────────────────────────────────

def _dense_q(agent):
    return agent.Q if hasattr(agent, "Q") else agent.to_dense()


def value_error(agent, env, q_star):
    """Mads' yardstick: ||Q_hat - Q*||_F / ||Q*||_F over non-wall, non-goal cells."""
    dense = _dense_q(agent)
    mask = np.ones((env.rows, env.cols), dtype=bool)
    for (r, c) in env.walls:
        mask[r, c] = False
    mask[env.goal] = False
    d, q = dense[mask], q_star[mask]
    return float(np.linalg.norm(d - q) / (np.linalg.norm(q) + 1e-12))


def train_curve(kind, rank, grid, random_start, q_star, opt_mask,
                n_episodes, eval_every, max_steps, seed):
    random.seed(seed); np.random.seed(seed)
    env = make_env(grid, random_start)
    eval_env = make_env(grid, random_start)
    agent, get_q_row = make_agent(kind, grid, grid, rank, TAB_LR, CP_LR, EPSILON_DECAY)
    cp, acc, ret, verr = [], [], [], []
    for ep in range(n_episodes):
        state = env.reset()
        for _ in range(max_steps):
            a = agent.select_action(state)
            ns, r, done = env.step(a); agent.update(state, a, r, ns, done); state = ns
            if done:
                break
        agent.decay_epsilon()
        if (ep + 1) % eval_every == 0:
            cp.append(ep + 1)
            acc.append(policy_accuracy(get_q_row, eval_env, opt_mask))
            ret.append(greedy_return(agent, eval_env))
            verr.append(value_error(agent, eval_env, q_star))
    return {"checkpoints": cp, "acc": acc, "ret": ret, "verr": verr,
            "params": agent.param_count}, agent, get_q_row


def run_config(kind, rank, grid, random_start, q_star, opt_mask,
               n_episodes, eval_every, max_steps, n_seeds, label):
    per_seed = []
    keep_agent = keep_row = None
    for seed in range(n_seeds):
        d, agent, get_q_row = train_curve(kind, rank, grid, random_start, q_star,
                                          opt_mask, n_episodes, eval_every, max_steps, seed)
        per_seed.append(d)
        if seed == 0:
            keep_agent, keep_row = agent, get_q_row
        print(f"    {label:<12} seed {seed+1}/{n_seeds}  "
              f"final acc={d['acc'][-1]:.3f}  valueErr={d['verr'][-1]:.3f}")
    stack = lambda k: np.array([s[k] for s in per_seed])
    agg = {"label": label, "kind": kind, "rank": rank,
           "params": per_seed[0]["params"], "checkpoints": per_seed[0]["checkpoints"],
           "acc_mean": stack("acc").mean(0), "acc_std": stack("acc").std(0),
           "ret_mean": stack("ret").mean(0), "ret_std": stack("ret").std(0),
           "verr_mean": stack("verr").mean(0), "verr_std": stack("verr").std(0)}
    return agg, keep_row


_RANK_COLORS = {1: "#17becf", 2: "#8c564b", 3: "#1f77b4", 6: "#2ca02c",
                12: "#9467bd", 20: "#ff7f0e", 30: "#e377c2"}


def _curve_style(cfg):
    """Tabular = black dotted; CP (nlms) = solid; CP-target = dashed; colour by rank."""
    if cfg["kind"] == "tabular":
        return {"color": "black", "linestyle": ":", "linewidth": 2.4}
    ls = "--" if cfg["kind"] == "cp_target" else "-"
    return {"color": _RANK_COLORS.get(cfg["rank"], "#555"), "linestyle": ls, "linewidth": 2.0}


def plot_curves(configs, opt_return, out_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    panels = [("acc_mean", "acc_std"), ("ret_mean", "ret_std"), ("verr_mean", "verr_std")]
    for cfg in configs:
        st = _curve_style(cfg)
        x = cfg["checkpoints"]
        for ax, (mkey, skey) in zip(axes, panels):
            m, s = np.array(cfg[mkey]), np.array(cfg[skey])
            ax.plot(x, m, label=cfg["label"], **st)
            ax.fill_between(x, m - s, m + s, color=st["color"], alpha=0.12)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Policy accuracy (frac. optimal actions)")
    axes[0].set_title("Accuracy over training", fontweight="bold")
    axes[1].axhline(opt_return, color="gray", linestyle="--", linewidth=1.2,
                    label="optimal")
    axes[1].set_ylabel("Greedy-policy return")
    axes[1].set_title("Greedy return over training", fontweight="bold")
    axes[2].set_ylabel("value error  ||Q-Q*|| / ||Q*||")
    axes[2].set_title("Value error vs Q* (lower=better)", fontweight="bold")
    axes[2].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("Training episode"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] {out_path}")


# ── convergence video ────────────────────────────────────────────────────────

def _video_snapshots(grid, kind, rank, random_start, n_episodes, snap_every, max_steps):
    """Train one agent, snapshotting its dense Q every snap_every episodes."""
    random.seed(0); np.random.seed(0)
    env = make_env(grid, random_start)
    agent, _ = make_agent(kind, grid, grid, rank, TAB_LR, CP_LR, EPSILON_DECAY)
    dense = lambda: agent.Q if kind == "tabular" else agent.to_dense()
    snaps = [dense().copy()]
    for ep in range(n_episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = agent.select_action(s); ns, r, d = env.step(a)
            agent.update(s, a, r, ns, d); s = ns
            if d:
                break
        agent.decay_epsilon()
        if (ep + 1) % snap_every == 0:
            snaps.append(dense().copy())
    return snaps


def _render_video(panels, q_star, opt_mask, grid, vmin, vmax,
                  snap_every, n_episodes, fps, out_path, title):
    """panels: list of (label, snaps). Renders OPTIMAL + each panel as a heatmap."""
    eval_env = make_env(grid, False)
    n = min(len(s) for _, s in panels)
    ncol = 1 + len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(6.5 * ncol, 7))
    draw_policy_map(axes[0], lambda r, c: q_star[r, c], eval_env, opt_mask,
                    "OPTIMAL", vmin, vmax, show_path=True)
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=CMAP)
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, label="learned V(s)")
    sup = fig.suptitle("", fontsize=14, fontweight="bold")

    def update(i):
        for ax, (lbl, snaps) in zip(axes[1:], panels):
            draw_policy_map(ax, lambda r, c: snaps[i][r, c], eval_env, opt_mask,
                            lbl, vmin, vmax, show_path=True)
        sup.set_text(f"{title}  —  episode {i*snap_every} / {n_episodes}")
        return axes
    anim = animation.FuncAnimation(fig, update, frames=n, blit=False)
    anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=90)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(out_path)}  ({n} frames, {ncol} panels)")


def make_videos(out, grid, random_start, q_star, opt_mask, vmin, vmax,
                ranks, video_rank, n_episodes, snap_every, max_steps, fps, title,
                variants=("nlms",)):
    """Two videos from a single training pass per agent, comparing CP variants:
       video.mp4        OPTIMAL | TABULAR | <each variant at video_rank>
       video_multi.mp4  OPTIMAL | TABULAR | <each variant at every rank>
    """
    variant_kind = {"nlms": "cp", "target": "cp_target"}
    variant_tag = {"nlms": "CP", "target": "CP-tgt"}
    snaps = {("tabular", 0): _video_snapshots(grid, "tabular", 0, random_start,
                                              n_episodes, snap_every, max_steps)}
    for v in variants:
        for rk in sorted(set(ranks) | {video_rank}):
            snaps[(v, rk)] = _video_snapshots(grid, variant_kind[v], rk, random_start,
                                              n_episodes, snap_every, max_steps)
    easy = [("TABULAR", snaps[("tabular", 0)])]
    easy += [(f"{variant_tag[v]} r{video_rank}", snaps[(v, video_rank)]) for v in variants]
    _render_video(easy, q_star, opt_mask, grid, vmin, vmax, snap_every, n_episodes, fps,
                  os.path.join(out, "video.mp4"), title)
    multi = [("TABULAR", snaps[("tabular", 0)])]
    multi += [(f"{variant_tag[v]} r{rk}", snaps[(v, rk)]) for v in variants for rk in ranks]
    _render_video(multi, q_star, opt_mask, grid, vmin, vmax, snap_every, n_episodes,
                  fps, os.path.join(out, "video_multi.mp4"), title)


# ── driver ───────────────────────────────────────────────────────────────────

def _save_map(q_star, env, opt_mask, vmin, vmax, layout, grid, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    draw_policy_map(ax, lambda r, c: q_star[r, c], env, opt_mask,
                    "OPTIMAL policy", vmin, vmax, show_path=True)
    fig.suptitle(f"{layout} grid ({grid}x{grid}) — walls=black, start=cyan, "
                 f"goal=gold star, white=optimal path", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def _save_capture(cap, tab_params, ranks, layout, grid, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    plot_capture_panel(ax, cap, tab_params, trained_ranks=ranks)
    fig.suptitle(f"How much of the {layout} Q* a CP rank can represent ({grid}x{grid})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layout", choices=list(LAYOUTS), default="chicane",
                   help="which layout to run the battery on")
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--ranks", type=int, nargs="+", default=[3, 6, 12])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--snap-every", type=int, default=50)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--video-rank", type=int, default=6)
    p.add_argument("--map", action="store_true", help="only render the map overview")
    p.add_argument("--videos-only", action="store_true",
                   help="regenerate only the videos in existing folders")
    p.add_argument("--spawn", choices=["random", "fixed", "both"], default="both",
                   help="which start regime(s) to run (one per HPC task)")
    p.add_argument("--cp-variants", nargs="+", choices=["nlms", "target"],
                   default=["nlms", "target"],
                   help="CP agents to compare: nlms (original) and/or target (deadly-triad fix)")
    p.add_argument("--no-videos", action="store_true",
                   help="skip video rendering (e.g. if ffmpeg is unavailable)")
    args = p.parse_args()

    spawn_list = [("random", True), ("fixed", False)]
    if args.spawn != "both":
        spawn_list = [(s, r) for (s, r) in spawn_list if s == args.spawn]

    global WALLS_OVERRIDE
    grid, layout, E = args.size, args.layout, args.episodes
    base_env = LAYOUTS[layout](grid, random_start=False)
    WALLS_OVERRIDE = base_env.walls          # make_env() picks this up for every fresh env
    print(f"Layout '{layout}' {grid}x{grid}: {len(base_env.walls)} walls, "
          f"{len(base_env.nonterminal_states())} free non-goal cells.")
    print("Computing exact Q* ...", end=" ", flush=True)
    q_star = value_iteration_q(base_env, gamma=GAMMA)
    opt_mask = q_star == q_star.max(axis=2, keepdims=True)
    vals = np.array([q_star[r, c].max() for (r, c) in base_env.all_states()])
    vmin, vmax = float(vals.min()), float(vals.max())
    tab_params = grid * grid * NUM_ACTIONS
    print("done.")

    def folder(spawn):
        d = os.path.join(DATA_ROOT, f"{ENV_NAME}_{grid}x{grid}_{E}_{layout}_{spawn}")
        os.makedirs(d, exist_ok=True)
        return d

    if args.map:
        _save_map(q_star, base_env, opt_mask, vmin, vmax, layout, grid,
                  os.path.join(folder("random"), "map.png"))
        print("[OK] map only"); return

    # videos-only: regenerate both videos in the existing folders, skip everything else
    if args.videos_only:
        for spawn, rstart in spawn_list:
            out = folder(spawn)
            print(f"\n=== {layout} / {spawn} videos -> {out} ===", flush=True)
            make_videos(out, grid, rstart, q_star, opt_mask, vmin, vmax,
                        args.ranks, args.video_rank, E, args.snap_every,
                        args.max_steps, args.fps, f"{layout} {grid}x{grid}, {spawn}",
                        variants=args.cp_variants)
        return

    # spawn-independent: rank-capture curve (compute once, drop into both folders)
    print("Rank capture ...", flush=True)
    cap = cp_capture_curve(q_star, sorted(set([1, 2, 3, 6, 12, 20, 30] + args.ranks)))

    for spawn, rstart in spawn_list:
        out = folder(spawn)
        print(f"\n=== {layout} / {spawn} -> {out} ===")
        eval_env = make_env(grid, rstart)

        # self-contained: map + capture in every folder
        _save_map(q_star, base_env, opt_mask, vmin, vmax, layout, grid,
                  os.path.join(out, "map.png"))
        _save_capture(cap, tab_params, args.ranks, layout, grid,
                      os.path.join(out, "rank_capture.png"))

        # optimal greedy return under this start distribution (reference line)
        class _Opt:
            def select_action(self, s, evaluate=True): return int(np.argmax(q_star[s]))
        random.seed(123); np.random.seed(123)
        opt_ret = greedy_return(_Opt(), eval_env, n_rollouts=50)

        configs, map_rows = [], []
        print("  tabular:")
        agg, row = run_config("tabular", 0, grid, rstart, q_star, opt_mask,
                              E, args.eval_every, args.max_steps, args.seeds, "Tabular")
        configs.append(agg); map_rows.append(("TABULAR", row))
        # CP variants: "nlms" = original CP agent, "target" = deadly-triad fix
        variant_kind = {"nlms": "cp", "target": "cp_target"}
        variant_tag = {"nlms": "CP", "target": "CP-tgt"}
        for variant in args.cp_variants:
            for rk in args.ranks:
                print(f"  {variant_tag[variant]} rank={rk}:")
                agg, row = run_config(variant_kind[variant], rk, grid, rstart, q_star,
                                      opt_mask, E, args.eval_every, args.max_steps,
                                      args.seeds, f"{variant_tag[variant]} r{rk}")
                configs.append(agg); map_rows.append((f"{variant_tag[variant]} r{rk}", row))

        # cp-speedup-style learning curves (accuracy + greedy return over training)
        plot_curves(configs, opt_ret, os.path.join(out, "curves.png"),
                    f"CP vs tabular — {layout} {grid}x{grid}, {spawn} start "
                    f"({grid*grid} cells, {tab_params}-cell table)")

        # learned policy maps: Optimal | Tabular | CP ranks
        n_panels = 2 + len(args.ranks)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
        draw_policy_map(axes[0], lambda r, c: q_star[r, c], eval_env, opt_mask,
                        "OPTIMAL", vmin, vmax, show_path=True)
        for ax, (lbl, prow) in zip(axes[1:], map_rows):
            draw_policy_map(ax, prow, eval_env, opt_mask, lbl, vmin, vmax, show_path=True)
        fig.suptitle(f"Learned policy — {layout} {grid}x{grid}, {spawn} start",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out, "policymaps.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] policymaps.png")

        # machine-readable metrics for cross-run aggregation (summary.py)
        def _ser(c):
            return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in c.items()}
        metrics = {
            "layout": layout, "size": grid, "episodes": E, "spawn": spawn,
            "tab_params": tab_params, "opt_return": opt_ret,
            "rank_capture": cap, "ranks": list(args.ranks),
            "configs": [_ser(c) for c in configs],
            "final_acc": {c["label"]: float(c["acc_mean"][-1]) for c in configs},
        }
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  [OK] metrics.json")

        # convergence videos (3-panel + multi-rank)
        if not args.no_videos:
            print(f"  rendering videos ...", flush=True)
            make_videos(out, grid, rstart, q_star, opt_mask, vmin, vmax,
                        args.ranks, args.video_rank, E, args.snap_every,
                        args.max_steps, args.fps, f"{layout} {grid}x{grid}, {spawn}",
                        variants=args.cp_variants)
        print(f"  [OK] all artifacts in {out}")


if __name__ == "__main__":
    main()

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(directory="sim/data"):
    """Load all JSON run files. Returns (runs, agg_runs) where each is a list of dicts
    with keys 'name' and 'metrics'. Aggregate files (*_agg.json) are separated out."""
    runs, agg_runs = [], []
    if not os.path.isdir(directory):
        return runs, agg_runs
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(directory, fname)
        with open(path, "r") as f:
            content = json.load(f)
        entry = {"name": fname.replace(".json", ""), "metrics": content}
        if fname.endswith("_agg.json"):
            agg_runs.append(entry)
        else:
            runs.append(entry)
    return runs, agg_runs


def _parse_run_name(name: str) -> dict:
    """Extract (algo, network, tensorize, rank, seed) from a run name string."""
    m = re.search(r"_(dqn|double_dqn|dueling_dqn|tabular_q|sarsa)_", name)
    algo = m.group(1) if m else "unknown"

    m = re.search(r"_(standard|cp|tucker|tt)_", name)
    network = m.group(1) if m else "unknown"

    m = re.search(r"_tz(all|hidden|none)", name)
    tensorize = m.group(1) if m else "all"

    m = re.search(r"_rank(\d+)", name)
    rank = int(m.group(1)) if m else 0

    m = re.search(r"_seed(\d+)", name)
    seed = int(m.group(1)) if m else 0

    return dict(algo=algo, network=network, tensorize=tensorize, rank=rank, seed=seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth(values, window=10):
    """Simple moving average. Returns array of same length, edges handled by shrinking window."""
    values = np.array(values, dtype=float)
    out = np.empty_like(values)
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out[i] = values[lo:hi].mean()
    return out


def _group_label(info: dict) -> str:
    tz = f"/tz:{info['tensorize']}" if info["tensorize"] != "all" else ""
    return f"{info['algo']} | {info['network']}{tz} rank={info['rank']}"


# ---------------------------------------------------------------------------
# Plot: learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(data, agg_data=None, metric="rewards", smoothing=10,
                         save_path=None):
    """Plot smoothed training curves. Uses aggregated (mean±std) data when available,
    otherwise plots individual runs grouped by (algo, network, tensorize)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if agg_data:
        for d in agg_data:
            info = _parse_run_name(d["name"])
            m = d["metrics"]
            eps = m["episodes"]
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key not in m:
                continue
            mean = smooth(m[mean_key], smoothing)
            std = np.array(m[std_key])
            label = _group_label(info)
            ax.plot(eps, mean, label=label)
            ax.fill_between(eps, mean - std, mean + std, alpha=0.2)
    else:
        # Group individual runs
        groups: dict[str, list] = {}
        for d in data:
            info = _parse_run_name(d["name"])
            label = _group_label(info)
            groups.setdefault(label, []).append(d["metrics"].get(metric, []))
        for label, series_list in groups.items():
            min_len = min(len(s) for s in series_list)
            arr = np.array([s[:min_len] for s in series_list])
            mean = smooth(arr.mean(axis=0), smoothing)
            std = arr.std(axis=0)
            eps = list(range(1, min_len + 1))
            ax.plot(eps, mean, label=label)
            ax.fill_between(eps, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Learning Curves — {metric}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Plot: parameter efficiency
# ---------------------------------------------------------------------------

def plot_param_efficiency(data, save_path=None):
    """Scatter: x=param count, y=mean eval_reward over last 50 episodes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in data:
        m = d["metrics"]
        params = m.get("params", None)
        eval_rewards = m.get("eval_reward", [])
        if params is None or not eval_rewards:
            continue
        final_perf = float(np.mean(eval_rewards[-50:])) if len(eval_rewards) >= 50 else float(np.mean(eval_rewards))
        info = _parse_run_name(d["name"])
        label = _group_label(info)
        ax.scatter(params, final_perf, label=label, s=80, zorder=3)

    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Mean Eval Reward (last 50 eps)")
    ax.set_title("Parameter Efficiency")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Plot: stability (grad norm + Q values)
# ---------------------------------------------------------------------------

def plot_stability(data, metric="grad_norm", save_path=None):
    """Two-panel plot: gradient norm and mean_q over training, grouped by network type."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    groups_gn: dict[str, list] = {}
    groups_mq: dict[str, list] = {}
    for d in data:
        info = _parse_run_name(d["name"])
        label = _group_label(info)
        gn = d["metrics"].get("grad_norm", [])
        mq = d["metrics"].get("mean_q", [])
        if gn:
            groups_gn.setdefault(label, []).append(gn)
        if mq:
            groups_mq.setdefault(label, []).append(mq)

    for ax, groups, title, ylabel in [
        (axes[0], groups_gn, "Gradient Norm", "Grad Norm"),
        (axes[1], groups_mq, "Mean Max Q-Value", "Mean Q"),
    ]:
        for label, series_list in groups.items():
            min_len = min(len(s) for s in series_list)
            arr = np.array([s[:min_len] for s in series_list])
            mean = smooth(arr.mean(axis=0), 10)
            eps = list(range(1, min_len + 1))
            ax.plot(eps, mean, label=label)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(data):
    """Mann-Whitney U + Cohen's d on final-20-episode eval_reward per network type pair."""
    from scipy.stats import mannwhitneyu

    # Collect final-20 eval rewards per network type
    by_network: dict[str, list] = {}
    for d in data:
        info = _parse_run_name(d["name"])
        net = info["network"]
        eval_rewards = d["metrics"].get("eval_reward", [])
        if not eval_rewards:
            continue
        final = eval_rewards[-20:] if len(eval_rewards) >= 20 else eval_rewards
        by_network.setdefault(net, []).extend(final)

    networks = sorted(by_network.keys())
    results = {}

    header = f"{'Network A':<16} {'Network B':<16} {'U-stat':>10} {'p-value':>10} {'Sig?':>6} {'Cohen d':>9}"
    print(header)
    print("-" * len(header))

    for i in range(len(networks)):
        for j in range(i + 1, len(networks)):
            a, b = networks[i], networks[j]
            xa, xb = np.array(by_network[a]), np.array(by_network[b])
            u_stat, p_val = mannwhitneyu(xa, xb, alternative="two-sided")

            # Cohen's d
            pooled_std = np.sqrt((xa.std() ** 2 + xb.std() ** 2) / 2)
            cohens_d = (xa.mean() - xb.mean()) / pooled_std if pooled_std > 0 else 0.0

            sig = "Yes" if p_val < 0.05 else "No"
            print(f"{a:<16} {b:<16} {u_stat:>10.1f} {p_val:>10.4f} {sig:>6} {cohens_d:>9.3f}")
            results[f"{a}_vs_{b}"] = dict(u_stat=u_stat, p_value=p_val, significant=sig == "Yes",
                                           cohens_d=cohens_d)

    return results


# ---------------------------------------------------------------------------
# Legacy helper (kept for compatibility)
# ---------------------------------------------------------------------------

def quick_plot(data):
    """Basic reward curve plot for all runs."""
    plt.figure()
    for d in data:
        m = d["metrics"]
        plt.plot(m.get("episodes", range(len(m["rewards"]))), m["rewards"],
                 label=d["name"], alpha=0.8)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("RL Training Progress")
    plt.legend(fontsize=7)
    plt.grid()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    runs, agg_runs = load_data("sim/data")

    if not runs and not agg_runs:
        print("No data found in sim/data/. Run some training first.")
    else:
        plot_learning_curves(runs, agg_data=agg_runs, metric="rewards",
                             save_path="figures/learning_curves.png")
        plot_learning_curves(runs, agg_data=agg_runs, metric="eval_reward",
                             save_path="figures/eval_curves.png")
        plot_param_efficiency(runs, save_path="figures/param_efficiency.png")
        plot_stability(runs, save_path="figures/stability.png")

        if len(runs) >= 2:
            print("\n--- Statistical Tests ---")
            run_statistical_tests(runs)

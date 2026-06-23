"""Matplotlib figures for sim/designs/."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from designs.shared.training import env_slug


def _rolling_mean(arr, window=10):
    out = np.convolve(arr, np.ones(window)/window, mode='valid')
    return out


def plot_learning_curves(results, baseline, env_id, save_dir):
    """Learning curves: rolling mean reward per episode, shaded std."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if baseline:
        per_seed = baseline['per_seed_rewards']
        mat = np.array([_rolling_mean(s) for s in per_seed])
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, '--', color='black', label='baseline', linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color='black', alpha=0.1)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))
    for i, (label, agg) in enumerate(results.items()):
        per_seed = agg['per_seed_rewards']
        mat = np.array([_rolling_mean(s) for s in per_seed])
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, color=colors[i], label=label, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=colors[i], alpha=0.1)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (rolling mean, window=10)')
    ax.set_title(f'Learning Curves — {env_id}')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'learning_curves_{env_slug(env_id)}.png'), dpi=150)
    plt.close(fig)


def plot_param_efficiency(results, baseline, env_id, save_dir):
    """Scatter: params vs solve rate."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if baseline:
        ax.axhline(baseline['solve_mean'], color='black', linestyle='--',
                    label=f"baseline ({baseline['n_params']} params)", linewidth=1.5)

    labels, params, solves = [], [], []
    for label, agg in results.items():
        labels.append(label)
        params.append(agg['n_params'])
        solves.append(agg['solve_mean'])

    ax.scatter(params, solves, c=range(len(params)), cmap='tab10', s=60, zorder=5)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (params[i], solves[i]), fontsize=6, textcoords='offset points',
                    xytext=(4, 4))

    ax.set_xlabel('Parameters')
    ax.set_ylabel('Solve Rate (last 50 eps)')
    ax.set_title(f'Parameter Efficiency — {env_id}')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'param_efficiency_{env_slug(env_id)}.png'), dpi=150)
    plt.close(fig)


def plot_convergence_heatmap(results, env_id, save_dir):
    """Heatmap: configs x 10% episode buckets."""
    labels = list(results.keys())
    data = np.array([results[l]['avg_quantiles'] for l in labels])

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.4 + 1)))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xticks(range(10))
    ax.set_xticklabels([f'{(i+1)*10}%' for i in range(10)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Training Progress')
    ax.set_title(f'Convergence Heatmap — {env_id}')

    for i in range(len(labels)):
        for j in range(10):
            ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center', fontsize=6)

    fig.colorbar(im, label='Mean Reward')
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'convergence_heatmap_{env_slug(env_id)}.png'), dpi=150)
    plt.close(fig)


def generate_design_figures(results, design_dir, baseline_loader=None):
    """Generate all standard figures for a design."""
    from designs.shared.config import ENVS
    fig_dir = os.path.join(design_dir, 'figures')

    for env_id in ENVS:
        if env_id not in results:
            continue
        baseline = baseline_loader(env_id) if baseline_loader else None
        env_results = results[env_id]

        plot_learning_curves(env_results, baseline, env_id, fig_dir)
        plot_param_efficiency(env_results, baseline, env_id, fig_dir)
        plot_convergence_heatmap(env_results, env_id, fig_dir)

    print(f"  Figures saved to {fig_dir}/")

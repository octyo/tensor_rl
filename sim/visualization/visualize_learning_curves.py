#!/usr/bin/env python3
"""
Visualize learning curves comparing Tabular Baseline and CP Agent at different ranks.

Plots episode returns (total reward per episode) vs. episodes, smoothed and averaged over seeds.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from sim.envs.gridworld import GridWorld


def load_tabular_results(path: str) -> dict:
    """Load tabular baseline results."""
    with open(path, 'r') as f:
        return json.load(f)


def load_cp_results(path: str) -> dict:
    """Load CP rank sweep results."""
    with open(path, 'r') as f:
        return json.load(f)


def smooth_curve(curve: np.ndarray, window_size: int = 20) -> np.ndarray:
    """Smooth a curve using uniform filter."""
    return uniform_filter1d(curve, size=window_size, mode='nearest')


def visualize_learning_curves(tabular_data: dict, cp_data: dict, save_path: str = None):
    """Create learning curve comparison plot."""

    # Extract curves
    fig, ax = plt.subplots(figsize=(12, 6))

    # Tabular baseline
    tabular_curves = [np.array(seed['episode_returns']) for seed in tabular_data['seeds']]
    tabular_mean = np.mean(tabular_curves, axis=0)
    tabular_std = np.std(tabular_curves, axis=0)

    episodes = np.arange(len(tabular_mean))

    # Plot tabular
    tabular_smooth = smooth_curve(tabular_mean, window_size=20)
    ax.plot(episodes, tabular_smooth, linewidth=2.5, label='Tabular', color='black', zorder=5)
    ax.fill_between(episodes,
                     smooth_curve(tabular_mean - tabular_std, window_size=20),
                     smooth_curve(tabular_mean + tabular_std, window_size=20),
                     alpha=0.15, color='black')

    # Colors for CP ranks
    colors = {
        1: '#d62728',  # red
        2: '#ff7f0e',  # orange
        4: '#2ca02c',  # green
        8: '#1f77b4',  # blue
        16: '#9467bd'  # purple
    }

    ranks_to_plot = [1, 2, 4, 8, 16]

    # Plot CP ranks
    for rank in ranks_to_plot:
        rank_str = str(rank)
        if rank_str not in cp_data:
            continue

        rank_data = cp_data[rank_str]
        rank_curves = [np.array(seed['episode_returns']) for seed in rank_data['seeds']]
        rank_mean = np.mean(rank_curves, axis=0)
        rank_std = np.std(rank_curves, axis=0)

        rank_smooth = smooth_curve(rank_mean, window_size=20)
        ax.plot(episodes, rank_smooth, linewidth=2, label=f'CP Rank {rank}',
               color=colors[rank], alpha=0.8)
        ax.fill_between(episodes,
                       smooth_curve(rank_mean - rank_std, window_size=20),
                       smooth_curve(rank_mean + rank_std, window_size=20),
                       alpha=0.1, color=colors[rank])

    ax.set_xlabel('Episode', fontsize=12, weight='bold')
    ax.set_ylabel('Total Episode Return (Reward)', fontsize=12, weight='bold')
    ax.set_title('Learning Curves: Tabular Baseline vs. CP Agent', fontsize=14, weight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2000)

    # Add horizontal reference line at final tabular performance
    final_tabular = tabular_smooth[-1]
    ax.axhline(y=final_tabular, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(len(episodes) * 0.02, final_tabular + 0.5, f'Tabular final: {final_tabular:.2f}',
           fontsize=9, color='black', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Learning curves saved: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    tabular_path = os.path.join(script_dir, '..', 'data', 'tabular_baseline_results.json')
    cp_path = os.path.join(script_dir, '..', 'data', 'cp_rank_sweep_results.json')

    print("Loading results...")
    tabular_data = load_tabular_results(tabular_path)
    cp_data = load_cp_results(cp_path)

    output_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    print("Generating learning curve plot...")
    learning_curve_path = os.path.join(output_dir, 'learning_curves.png')
    visualize_learning_curves(tabular_data, cp_data, save_path=learning_curve_path)

    print(f"\n[OK] Visualization saved to {output_dir}/")


if __name__ == "__main__":
    main()

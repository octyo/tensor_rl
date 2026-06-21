"""Generate all visualizations for designs and cases."""

import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

DESIGNS_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.join(DESIGNS_DIR, '..', 'cases')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_design_data(design_dir):
    """Load all JSON results from a design's data/ directory."""
    data_dir = os.path.join(design_dir, 'data')
    if not os.path.exists(data_dir):
        return {}
    results = {}
    for f in sorted(glob.glob(os.path.join(data_dir, '*.json'))):
        with open(f) as fh:
            d = json.load(fh)
        key = d.get('config', os.path.basename(f).split('_seed')[0])
        env = d.get('env', 'unknown')
        if env not in results:
            results[env] = {}
        if key not in results[env]:
            results[env][key] = []
        results[env][key].append(d)
    return results


def rolling_mean(arr, window=10):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode='valid')


# ---------------------------------------------------------------------------
# 1. Regenerate per-design figures (Designs 0-4 with baseline overlay)
# ---------------------------------------------------------------------------

def regenerate_design_figures():
    """Regenerate all per-design figures from cached JSON data."""
    from designs.shared.training import load_baseline_data, aggregate_seeds, env_slug
    from designs.shared.plotting import (plot_learning_curves, plot_param_efficiency,
                                          plot_convergence_heatmap)
    from designs.shared.config import ENVS

    for design_num in range(5):
        design_names = ['0_baseline', '1_cp_qfunction', '2_weight_decomp',
                        '3_input_reduction', '4_mps_network']
        design_dir = os.path.join(DESIGNS_DIR, design_names[design_num])
        fig_dir = os.path.join(design_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        raw = load_design_data(design_dir)
        if not raw:
            continue

        for env_id in ENVS:
            if env_id not in raw:
                continue

            env_results = {}
            for config_label, seed_list in raw[env_id].items():
                agg = aggregate_seeds_from_json(seed_list)
                env_results[config_label] = agg

            baseline = None
            if design_num > 0:
                try:
                    baseline = load_baseline_data(env_id)
                except FileNotFoundError:
                    pass

            plot_learning_curves(env_results, baseline, env_id, fig_dir)
            plot_param_efficiency(env_results, baseline, env_id, fig_dir)
            plot_convergence_heatmap(env_results, env_id, fig_dir)

        print(f"  Design {design_num}: {fig_dir}/")


def aggregate_seeds_from_json(seed_list):
    return {
        'n_params': seed_list[0]['n_params'],
        'reward_mean': float(np.mean([s['reward_mean_final'] for s in seed_list])),
        'solve_mean': float(np.mean([s['solve_rate_final'] for s in seed_list])),
        'per_seed_rewards': [s['all_rewards'] for s in seed_list],
        'avg_quantiles': [float(np.mean([s['quantile_means'][i] for s in seed_list]))
                          for i in range(len(seed_list[0]['quantile_means']))],
    }


# ---------------------------------------------------------------------------
# 2. Cross-design comparison figures
# ---------------------------------------------------------------------------

def cross_design_comparison():
    """Best config per design per environment — side-by-side comparison."""
    from designs.shared.config import ENVS
    from designs.shared.training import env_slug

    design_names = ['0_baseline', '1_cp_qfunction', '2_weight_decomp',
                    '3_input_reduction', '4_mps_network']
    design_labels = ['0: Baseline', '1: CP Q-func', '2: Weight Decomp',
                     '3: Input Embed', '4: MPS Network']

    fig_dir = os.path.join(DESIGNS_DIR, 'comparison_figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Load best config per design per env
    all_data = {}
    for dn in design_names:
        all_data[dn] = load_design_data(os.path.join(DESIGNS_DIR, dn))

    # --- Figure: Best solve rate per design per env ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(design_names)))

    for ei, env_id in enumerate(ENVS):
        ax = axes[ei]
        best_solves = []
        best_labels = []
        best_params = []
        for di, dn in enumerate(design_names):
            if env_id not in all_data[dn]:
                best_solves.append(0)
                best_labels.append('')
                best_params.append(0)
                continue
            configs = all_data[dn][env_id]
            best_config = None
            best_solve = -1
            for cfg, seeds in configs.items():
                solve = float(np.mean([s['solve_rate_final'] for s in seeds]))
                if solve > best_solve:
                    best_solve = solve
                    best_config = cfg
                    best_params_val = seeds[0]['n_params']
            best_solves.append(best_solve)
            best_labels.append(f"{best_config}\n({best_params_val}p)")
            best_params.append(best_params_val)

        bars = ax.bar(range(len(design_names)), best_solves, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(design_names)))
        ax.set_xticklabels([dl.split(': ')[1] for dl in design_labels], fontsize=7, rotation=30, ha='right')
        ax.set_title(env_id.replace('MiniGrid-', ''), fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        for i, (b, lbl) in enumerate(zip(bars, best_labels)):
            if best_solves[i] > 0:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                        lbl, ha='center', va='bottom', fontsize=5.5)

    axes[0].set_ylabel('Best Solve Rate')
    fig.suptitle('Cross-Design Comparison: Best Config Per Design', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'cross_design_best_solve.png'), dpi=150)
    plt.close(fig)

    # --- Figure: Learning curves of best config per design (one plot per env) ---
    for env_id in ENVS:
        fig, ax = plt.subplots(figsize=(10, 6))
        for di, dn in enumerate(design_names):
            if env_id not in all_data[dn]:
                continue
            configs = all_data[dn][env_id]
            best_config = max(configs.keys(),
                              key=lambda c: np.mean([s['solve_rate_final'] for s in configs[c]]))
            seeds = configs[best_config]
            all_rewards = [s['all_rewards'] for s in seeds]
            mat = np.array([rolling_mean(r) for r in all_rewards])
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            x = np.arange(len(mean))
            label = f"{design_labels[di]}: {best_config} ({seeds[0]['n_params']}p)"
            ax.plot(x, mean, color=colors[di], label=label, linewidth=1.5)
            ax.fill_between(x, mean - std, mean + std, color=colors[di], alpha=0.1)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (rolling mean)')
        ax.set_title(f'Best Config Per Design — {env_id}')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'cross_design_curves_{env_slug(env_id)}.png'), dpi=150)
        plt.close(fig)

    # --- Figure: Parameter efficiency across all designs ---
    for env_id in ENVS:
        fig, ax = plt.subplots(figsize=(10, 7))
        for di, dn in enumerate(design_names):
            if env_id not in all_data[dn]:
                continue
            configs = all_data[dn][env_id]
            for cfg, seeds in configs.items():
                params = seeds[0]['n_params']
                solve = float(np.mean([s['solve_rate_final'] for s in seeds]))
                ax.scatter(params, solve, color=colors[di], s=40, zorder=5, alpha=0.8)
            # Add invisible point for legend
            ax.scatter([], [], color=colors[di], s=40, label=design_labels[di])

        ax.set_xlabel('Parameters (log)')
        ax.set_ylabel('Solve Rate')
        ax.set_title(f'All Configs Across All Designs — {env_id}')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'all_designs_param_efficiency_{env_slug(env_id)}.png'), dpi=150)
        plt.close(fig)

    print(f"  Cross-design: {fig_dir}/")


# ---------------------------------------------------------------------------
# 3. High-dimensional designs (5, 6) figures
# ---------------------------------------------------------------------------

def highdim_figures():
    """Figures for Designs 5 and 6."""
    for design_name, title_suffix in [('5_highdim', '10D Nav (3^10)'),
                                       ('6_crossover', '8D Nav (3^8)')]:
        design_dir = os.path.join(DESIGNS_DIR, design_name)
        data_dir = os.path.join(design_dir, 'data')
        fig_dir = os.path.join(design_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        if not os.path.exists(data_dir):
            continue

        results = {}
        for f in sorted(glob.glob(os.path.join(data_dir, '*.json'))):
            with open(f) as fh:
                d = json.load(fh)
            results[d['config']] = d

        if not results:
            continue

        # Learning curves
        fig, ax = plt.subplots(figsize=(12, 7))
        for label, res in results.items():
            rews = res['all_rewards']
            smooth = rolling_mean(rews, 20)
            if len(smooth) > 0:
                ax.plot(smooth, label=f"{label} ({res['n_params']}p)", linewidth=1.2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (rolling mean, window=20)')
        ax.set_title(f'Learning Curves — {title_suffix}')
        ax.legend(fontsize=6, loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'learning_curves.png'), dpi=150)
        plt.close(fig)

        # Param efficiency
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = list(results.keys())
        params = [results[l]['n_params'] for l in labels]
        solves = [results[l]['solve_rate_final'] for l in labels]
        colors_sc = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        ax.scatter(params, solves, c=range(len(params)), cmap='tab10', s=60, zorder=5)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (params[i], solves[i]), fontsize=6,
                        textcoords='offset points', xytext=(4, 4))
        ax.set_xlabel('Parameters (log)')
        ax.set_ylabel('Solve Rate')
        ax.set_title(f'Parameter Efficiency — {title_suffix}')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'param_efficiency.png'), dpi=150)
        plt.close(fig)

        # Convergence heatmap
        labels_sorted = list(results.keys())
        qm = [results[l]['quantile_means'] for l in labels_sorted]
        if qm and all(len(q) > 0 for q in qm):
            data = np.array(qm)
            fig, ax = plt.subplots(figsize=(10, max(3, len(labels_sorted) * 0.4 + 1)))
            im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
            ax.set_xticks(range(data.shape[1]))
            ax.set_xticklabels([f'{(i+1)*10}%' for i in range(data.shape[1])])
            ax.set_yticks(range(len(labels_sorted)))
            ax.set_yticklabels(labels_sorted, fontsize=7)
            ax.set_xlabel('Training Progress')
            ax.set_title(f'Convergence Heatmap — {title_suffix}')
            for i in range(len(labels_sorted)):
                for j in range(data.shape[1]):
                    ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center', fontsize=5)
            fig.colorbar(im, label='Mean Reward')
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'convergence_heatmap.png'), dpi=150)
            plt.close(fig)

        print(f"  {design_name}: {fig_dir}/")


# ---------------------------------------------------------------------------
# 4. Cross-dimensionality comparison (Designs 5 vs 6)
# ---------------------------------------------------------------------------

def dimensionality_comparison():
    """Compare method families across 8D and 10D."""
    fig_dir = os.path.join(DESIGNS_DIR, 'comparison_figures')
    os.makedirs(fig_dir, exist_ok=True)

    dim_data = {}
    for design_name, n_dim in [('5_highdim', 10), ('6_crossover', 8)]:
        data_dir = os.path.join(DESIGNS_DIR, design_name, 'data')
        if not os.path.exists(data_dir):
            continue
        dim_data[n_dim] = {}
        for f in sorted(glob.glob(os.path.join(data_dir, '*.json'))):
            with open(f) as fh:
                d = json.load(fh)
            dim_data[n_dim][d['config']] = d

    if not dim_data:
        return

    # Group methods into families
    families = {
        'baseline': lambda c: c == 'baseline',
        'cp_flat': lambda c: c.startswith('cp_flat'),
        'tucker_flat': lambda c: c.startswith('tucker_flat'),
        'cp_qfunc': lambda c: c.startswith('cp_qfunc'),
        'embed_tt': lambda c: c.startswith('embed_tt'),
        'embed_cp': lambda c: c.startswith('embed_cp'),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(families)))
    x_dims = sorted(dim_data.keys())

    for fi, (family, matcher) in enumerate(families.items()):
        best_per_dim = []
        for nd in x_dims:
            matching = {c: d for c, d in dim_data[nd].items() if matcher(c)}
            if matching:
                best = max(matching.values(), key=lambda d: d['solve_rate_final'])
                best_per_dim.append(best['solve_rate_final'])
            else:
                best_per_dim.append(0)
        ax.plot(x_dims, best_per_dim, 'o-', color=colors[fi], label=family,
                linewidth=2, markersize=8)

    ax.set_xlabel('Number of Dimensions')
    ax.set_ylabel('Best Solve Rate')
    ax.set_title('Method Performance vs Dimensionality (NDim Navigation)')
    ax.set_xticks(x_dims)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'dimensionality_crossover.png'), dpi=150)
    plt.close(fig)
    print(f"  Dimensionality comparison: {fig_dir}/")


# ---------------------------------------------------------------------------
# 5. Case study figures
# ---------------------------------------------------------------------------

def parse_case_output(filepath):
    """Parse a case output.txt into structured data."""
    rows = []
    with open(filepath) as f:
        lines = f.readlines()

    in_table = False
    headers = None
    for line in lines:
        line = line.rstrip()
        if line.startswith('---'):
            if headers is not None:
                in_table = True
            continue
        if not in_table and ('Config' in line or 'config' in line) and ('Params' in line or 'params' in line):
            headers = line.split()
            in_table = False
            continue
        if in_table and line and not line.startswith('=') and not line.startswith('Training') and not line.startswith('Standard'):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    config = parts[0]
                    params = int(parts[1])
                    # Find solve percentage
                    solve_str = [p for p in parts if '%' in p and '+' not in p]
                    if solve_str:
                        solve = int(solve_str[0].replace('%', '')) / 100
                    else:
                        solve = 0
                    # Find reward
                    reward_candidates = [p for p in parts if '.' in p and '%' not in p and '+' not in p]
                    reward = float(reward_candidates[-1]) if reward_candidates else 0
                    rows.append({'config': config, 'params': params, 'solve': solve, 'reward': reward})
                except (ValueError, IndexError):
                    continue
    return rows


def case_figures():
    """Generate summary figures from case output files."""
    fig_dir = os.path.join(CASES_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # --- Case 0/0a: Baseline comparison across envs ---
    case_files = {
        'Empty-5x5 (DQN)': 'case0_baseline_output.txt',
        'Empty-5x5 (DDQN)': 'case0a_baseline_ddqn_output.txt',
        'Empty-Random-5x5': 'case0a_emptyrandom5x5_output.txt',
        'DoorKey-5x5 (300ep)': 'case0a_doorkey5x5_output.txt',
        'DoorKey-5x5 (1000ep)': 'case0a_doorkey5x5_1000eps_output.txt',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = 0
    tick_positions = []
    tick_labels = []
    group_starts = []

    for case_label, fname in case_files.items():
        path = os.path.join(CASES_DIR, fname)
        if not os.path.exists(path):
            continue
        rows = parse_case_output(path)
        if not rows:
            continue
        group_starts.append(x_pos)
        for r in rows:
            color = plt.cm.Set2(hash(r['config']) % 8 / 8)
            ax.bar(x_pos, r['solve'], color=color, edgecolor='black', linewidth=0.3, width=0.8)
            ax.text(x_pos, r['solve'] + 0.02, f"{r['config']}\n{r['params']}p",
                    ha='center', fontsize=5, rotation=45)
            x_pos += 1
        tick_positions.append((group_starts[-1] + x_pos - 1) / 2)
        tick_labels.append(case_label)
        x_pos += 1

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=15, ha='right')
    ax.set_ylabel('Solve Rate')
    ax.set_title('Case Studies: Baseline Network Size Sweep Across Environments')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'baselines_across_envs.png'), dpi=150)
    plt.close(fig)

    # --- Case 1/2a/5: Tensor layer type comparison on Empty-5x5 ---
    tensor_cases = {
        'Case 1: CP Layers': 'case1_cp_layers_output.txt',
        'Case 2a: Decomp Init': 'case2a_decomp_init_output.txt',
        'Case 5: All Types': 'case5_tensorized_output.txt',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ci, (case_label, fname) in enumerate(tensor_cases.items()):
        path = os.path.join(CASES_DIR, fname)
        if not os.path.exists(path):
            continue
        rows = parse_case_output(path)
        if not rows:
            continue
        ax = axes[ci]
        params = [r['params'] for r in rows]
        solves = [r['solve'] for r in rows]
        configs = [r['config'] for r in rows]

        # Color by method type
        for i, r in enumerate(rows):
            c = r['config'].lower()
            if 'standard' in c:
                color = 'black'
            elif 'tucker' in c:
                color = '#e74c3c'
            elif 'cp' in c:
                color = '#3498db'
            elif 'tt' in c:
                color = '#2ecc71'
            elif 'mps' in c:
                color = '#9b59b6'
            else:
                color = 'gray'
            ax.scatter(r['params'], r['solve'], color=color, s=50, zorder=5)
            ax.annotate(r['config'], (r['params'], r['solve']), fontsize=5,
                        textcoords='offset points', xytext=(3, 3))

        ax.set_xscale('log')
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Solve Rate')
        ax.set_title(case_label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)

    # Manual legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Standard'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='CP'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Tucker'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='TT'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9b59b6', markersize=8, label='MPS'),
    ]
    axes[2].legend(handles=legend_elements, fontsize=7, loc='lower right')
    fig.suptitle('Tensor Layer Types: Parameter Efficiency on Empty-5x5', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'tensor_types_comparison.png'), dpi=150)
    plt.close(fig)

    # --- Case 4: CNN tensorization ---
    path = os.path.join(CASES_DIR, 'case4_cnn_output.txt')
    if os.path.exists(path):
        rows = parse_case_output(path)
        if rows:
            fig, ax = plt.subplots(figsize=(10, 6))
            for r in rows:
                c = r['config'].lower()
                if 'backbone' in c:
                    color, marker = '#3498db', 's'
                elif 'tensorized' in c:
                    color, marker = '#e74c3c', '^'
                else:
                    color, marker = 'black', 'o'
                ax.scatter(r['params'], r['solve'], color=color, marker=marker, s=60, zorder=5)
                ax.annotate(r['config'], (r['params'], r['solve']), fontsize=6,
                            textcoords='offset points', xytext=(4, 4))
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Standard CNN'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db', markersize=8, label='Backbone Decomp'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='#e74c3c', markersize=8, label='Full Tensorization'),
            ]
            ax.legend(handles=legend_elements, fontsize=8)
            ax.set_xscale('log')
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Solve Rate')
            ax.set_title('Case 4: CNN Tensorization — Backbone vs Full')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'cnn_tensorization.png'), dpi=150)
            plt.close(fig)

    # --- Case 6/8: MPS comparison ---
    mps_cases = {
        'Case 6: MPS Layers': 'case6_mps_output.txt',
        'Case 8: True MPS': 'case8_true_mps_output.txt',
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ci, (case_label, fname) in enumerate(mps_cases.items()):
        path = os.path.join(CASES_DIR, fname)
        if not os.path.exists(path):
            continue
        rows = parse_case_output(path)
        if not rows:
            continue
        ax = axes[ci]
        for r in rows:
            c = r['config'].lower()
            if 'standard' in c:
                color = 'black'
            elif 'true_mps' in c:
                color = '#e74c3c'
            elif 'mps' in c:
                color = '#9b59b6'
            elif 'tt_embed' in c:
                color = '#2ecc71'
            elif 'cp' in c:
                color = '#3498db'
            else:
                color = 'gray'
            ax.scatter(r['params'], r['solve'], color=color, s=50, zorder=5)
            ax.annotate(r['config'], (r['params'], r['solve']), fontsize=5,
                        textcoords='offset points', xytext=(3, 3))
        ax.set_xscale('log')
        ax.set_xlabel('Parameters')
        ax.set_title(case_label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)
    axes[0].set_ylabel('Solve Rate')
    fig.suptitle('MPS Methods: Weight Layers vs True Tensor Networks', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'mps_comparison.png'), dpi=150)
    plt.close(fig)

    # --- Case 2b: Post-hoc compression ---
    path = os.path.join(CASES_DIR, 'case2b_compress_output.txt')
    if os.path.exists(path):
        rows = parse_case_output(path)
        if rows:
            fig, ax = plt.subplots(figsize=(10, 5))
            configs = [r['config'] for r in rows]
            solves = [r['solve'] for r in rows]
            colors_bar = ['black' if 'standard' in c else '#3498db' if 'cp' in c
                          else '#e74c3c' if 'tucker' in c else '#9b59b6' for c in configs]
            ax.barh(range(len(configs)), solves, color=colors_bar, edgecolor='black', linewidth=0.3)
            ax.set_yticks(range(len(configs)))
            ax.set_yticklabels([f"{c} ({r['params']}p)" for c, r in zip(configs, rows)], fontsize=7)
            ax.set_xlabel('Solve Rate (after fine-tuning)')
            ax.set_title('Case 2b: Post-Hoc Compression — Solve Rate After Fine-Tuning')
            ax.grid(axis='x', alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'posthoc_compression.png'), dpi=150)
            plt.close(fig)

    # --- Case 7: MPS efficiency across envs ---
    mps_eff_cases = {
        'Empty-5x5': 'case7_mps_efficiency_output.txt',
        'Empty-Random-5x5': 'case7_mps_efficiency_emptyrandom5x5_output.txt',
        'DoorKey-5x5': 'case7_mps_efficiency_doorkey5x5_output.txt',
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ci, (env_label, fname) in enumerate(mps_eff_cases.items()):
        path = os.path.join(CASES_DIR, fname)
        if not os.path.exists(path):
            continue
        rows = parse_case_output(path)
        if not rows:
            continue
        ax = axes[ci]
        ranks = [int(r['config'].replace('mps_r', '')) for r in rows if r['config'].startswith('mps_r')]
        solves = [r['solve'] for r in rows if r['config'].startswith('mps_r')]
        params = [r['params'] for r in rows if r['config'].startswith('mps_r')]
        ax.plot(ranks, solves, 'o-', color='#9b59b6', linewidth=2, markersize=8)
        for i, (rk, s, p) in enumerate(zip(ranks, solves, params)):
            ax.annotate(f'{p}p', (rk, s), fontsize=6, textcoords='offset points', xytext=(4, 4))
        ax.set_xlabel('MPS Rank')
        ax.set_title(env_label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)
    axes[0].set_ylabel('Solve Rate')
    fig.suptitle('Case 7: MPS Rank Efficiency Across Environments', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'mps_rank_efficiency.png'), dpi=150)
    plt.close(fig)

    print(f"  Cases: {fig_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating all figures...\n")

    print("1. Per-design figures (Designs 0-4):")
    regenerate_design_figures()

    print("\n2. Cross-design comparison:")
    cross_design_comparison()

    print("\n3. High-dimensional designs (5, 6):")
    highdim_figures()

    print("\n4. Dimensionality comparison (8D vs 10D):")
    dimensionality_comparison()

    print("\n5. Case study figures:")
    case_figures()

    print("\nDone.")


if __name__ == '__main__':
    main()

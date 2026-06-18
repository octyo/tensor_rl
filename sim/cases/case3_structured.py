"""
Case 3: Structured Input Embedding (Input Space Reduction)
==========================================================
Instead of flattening the (7,7,3) MiniGrid observation, a tensor-network
embedding layer maps the raw 3D observation to a hidden vector while
preserving the spatial structure. Uses TTEmbedding / CPEmbedding /
TuckerEmbedding as the first layer.

Sweep: embedding_type in [tt, cp, tucker]  x  rank in [2, 4, 8, 16]
Baseline: standard DQN on flattened obs
Seeds: 7  |  Episodes: 200  |  Algo: DQN
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import (ENV_ID, N_EPISODES, SEEDS, ALGO,
                   train_flat, train_structured,
                   run_seeds, get_flat_dims, get_structured_dims,
                   print_table, build_output, save_output)
from models.q_networks import QNetwork, StructuredQNetwork

METHODS    = ['tt', 'cp', 'tucker']
RANKS      = [2, 4, 8, 16]
SEEDS      = [42, 43, 44, 45, 46, 47, 48]
N_EPISODES = 200
OUT_FILE   = os.path.join(os.path.dirname(__file__), 'case3_structured_output.txt')


def main():
    state_dim, action_dim = get_flat_dims()
    mode_dims, _          = get_structured_dims()   # (7, 7, 3)

    ref_net    = QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                          network_type='standard', rank=4)
    std_params = sum(p.numel() for p in ref_net.parameters())

    rows      = []
    conv_rows = []
    t0        = time.time()

    configs      = [('standard', None)] + [(m, r) for m in METHODS for r in RANKS]
    n_configs    = len(configs)
    config_idx   = 0

    # Baseline: standard flat DQN
    config_idx += 1
    print(f'\n  [config {config_idx}/{n_configs}] standard (flat baseline)')

    def std_factory():
        return QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                        network_type='standard', rank=4)

    res = run_seeds(train_flat, std_factory, label='standard',
                    n_episodes=N_EPISODES, seeds=SEEDS,
                    config_idx=config_idx, config_total=n_configs, experiment_t0=t0)
    rows.append(['standard', res['n_params'], '1.00',
                 f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                 f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
    conv_rows.append(['standard'] + [f'{q:.3f}' for q in res['avg_quantiles']])

    # Structured embedding sweep
    for method in METHODS:
        for rank in RANKS:
            config_idx += 1
            label = f'{method}_r{rank}'
            print(f'\n  [config {config_idx}/{n_configs}] {label}  mode_dims={mode_dims}')

            def factory(md=mode_dims, m=method, r=rank):
                return StructuredQNetwork(md, action_dim,
                                         hidden_size=128,
                                         embedding_type=m,
                                         rank=r)

            try:
                res = run_seeds(train_structured, factory, label=label,
                                n_episodes=N_EPISODES, seeds=SEEDS,
                                config_idx=config_idx, config_total=n_configs,
                                experiment_t0=t0)
                vs  = f"{res['n_params'] / std_params:.3f}"
                rows.append([label, res['n_params'], vs,
                             f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                             f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
                conv_rows.append([label] + [f'{q:.3f}' for q in res['avg_quantiles']])
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                rows.append([label, 'ERR', '-', '-', str(e)[:40]])
                conv_rows.append([label] + ['-'] * 10)

    pct_headers  = [f'{(i+1)*10}%' for i in range(10)]
    conv_headers = ['Config'] + pct_headers
    headers = ['Config', 'Params', 'vs_std', 'Solve%', 'Reward']
    title   = (f"Case 3: Structured Input Embedding  "
               f"(env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})")
    print_table(rows, headers, title=title)
    print(f"\n  Standard baseline: {std_params} params")
    print(f"  mode_dims={mode_dims}  action_dim={action_dim}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

    notes = [f"Standard baseline: {std_params} params (flat obs)",
             f"Structured input: mode_dims={mode_dims}",
             f"Total runtime: {(time.time()-t0)/60:.1f} min"]
    lines = build_output(title, f"env={ENV_ID}", rows, headers, notes,
                         convergence_rows=conv_rows, convergence_headers=conv_headers)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()

"""
Case 5: All Tensorized Layer Types — Head-to-Head Comparison
=============================================================
Compares all custom tensor layer types (CP, Tucker, TT, MPS) against the
standard DQN at matched rank values. This is the full head-to-head grid
showing which decomposition achieves the best performance / parameter tradeoff.

Sub-experiments:
  A) Fixed rank=8:  [standard, cp, tucker, tt, mps]
  B) Rank sweep:    all tensor types at rank in [2, 4, 8, 16]

Seeds: 3  |  Episodes: 200  |  Algo: DQN
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import (ENV_ID, N_EPISODES, SEEDS, ALGO,
                   train_flat, run_seeds, get_flat_dims,
                   print_table, build_output, save_output)
from models.q_networks import QNetwork

TYPES      = ['standard', 'cp', 'tucker', 'tt', 'mps']
FIXED_RANK = 8
RANKS      = [2, 4, 8, 16]
OUT_FILE   = os.path.join(os.path.dirname(__file__), 'case5_tensorized_output.txt')


def make_net(state_dim, action_dim, net_type, rank):
    return QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                    network_type=net_type, rank=rank, tensorize_layers='all')


def main():
    state_dim, action_dim = get_flat_dims()
    ref_net    = make_net(state_dim, action_dim, 'standard', 4)
    std_params = sum(p.numel() for p in ref_net.parameters())

    all_rows      = []
    all_conv_rows = []
    t0            = time.time()

    # --- Sub-experiment A: all types at fixed rank ---
    print(f'\n{"="*60}')
    print(f'A) All types at rank={FIXED_RANK}')
    print('='*60)

    rows_a = []
    for net_type in TYPES:
        rank  = 4 if net_type == 'standard' else FIXED_RANK
        label = f'{net_type}_r{rank}'
        print(f'\n  Config: {label}')

        def factory(nt=net_type, r=rank):
            return make_net(state_dim, action_dim, nt, r)

        res = run_seeds(train_flat, factory, label=label)
        vs  = f"{res['n_params'] / std_params:.3f}"
        rows_a.append([label, res['n_params'], vs,
                       f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                       f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
        all_conv_rows.append([label] + [f'{q:.3f}' for q in res['avg_quantiles']])
    all_rows.extend(rows_a)

    headers = ['Config', 'Params', 'vs_std', 'Solve%', 'Reward']
    print_table(rows_a, headers,
                title=f'A) All types at rank={FIXED_RANK}')

    # --- Sub-experiment B: rank sweep for all tensor types ---
    print(f'\n{"="*60}')
    print('B) Rank sweep for all tensor types')
    print('='*60)

    rows_b = []
    for net_type in ['cp', 'tucker', 'tt', 'mps']:
        for rank in RANKS:
            label = f'{net_type}_r{rank}'
            print(f'\n  Config: {label}')

            def factory(nt=net_type, r=rank):
                return make_net(state_dim, action_dim, nt, r)

            try:
                res = run_seeds(train_flat, factory, label=label)
                vs  = f"{res['n_params'] / std_params:.3f}"
                rows_b.append([label, res['n_params'], vs,
                               f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                               f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
                all_conv_rows.append([label] + [f'{q:.3f}' for q in res['avg_quantiles']])
            except Exception as e:
                print(f"    ERROR: {e}")
                rows_b.append([label, 'ERR', '-', '-', str(e)[:40]])
                all_conv_rows.append([label] + ['-'] * 10)
    all_rows.extend(rows_b)

    print_table(rows_b, headers,
                title='B) Rank sweep for all tensor types')

    elapsed = (time.time() - t0) / 60
    print(f"\n  Standard baseline: {std_params} params")
    print(f"  Total time: {elapsed:.1f} min")

    pct_headers  = [f'{(i+1)*10}%' for i in range(10)]
    conv_headers = ['Config'] + pct_headers
    title = (f"Case 5: All Tensorized Types  "
             f"(env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})")
    notes = [f"Standard baseline: {std_params} params",
             f"Total runtime: {elapsed:.1f} min"]
    lines = build_output(title, f"env={ENV_ID}", all_rows, headers, notes,
                         convergence_rows=all_conv_rows, convergence_headers=conv_headers)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()

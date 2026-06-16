"""
Case 2a: Pre-Init Decomposition (Decomposed from the Start)
============================================================
Initialises the DQN with decomposed weight layers from episode 1.
Compares CP / Tucker / TT / MPS at a fixed rank and at a rank sweep.

Sub-experiments:
  A) Method comparison: [standard, cp, tucker, tt, mps] at rank=8
  B) Rank sweep:        cp and tucker at rank in [2, 4, 8, 16]

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

FIXED_RANK = 8
RANKS      = [2, 4, 8, 16]
OUT_FILE   = os.path.join(os.path.dirname(__file__), 'case2a_decomp_init_output.txt')


def make_net(state_dim, action_dim, network_type, rank):
    return QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                    network_type=network_type, rank=rank, tensorize_layers='all')


def main():
    state_dim, action_dim = get_flat_dims()
    ref_net    = make_net(state_dim, action_dim, 'standard', 4)
    std_params = sum(p.numel() for p in ref_net.parameters())

    all_rows      = []
    all_conv_rows = []
    t0            = time.time()

    # --- Sub-experiment A: method comparison at rank=FIXED_RANK ---
    print(f'\n{"="*60}')
    print(f'A) Method comparison at rank={FIXED_RANK}')
    print('='*60)

    methods = ['standard', 'cp', 'tucker', 'tt', 'mps']
    rows_a  = []
    for method in methods:
        rank  = 4 if method == 'standard' else FIXED_RANK
        label = f'{method}_r{rank}'
        print(f'\n  Config: {label}')

        def factory(m=method, r=rank):
            return make_net(state_dim, action_dim, m, r)

        res = run_seeds(train_flat, factory, label=label)
        vs  = f"{res['n_params'] / std_params:.3f}"
        rows_a.append([label, res['n_params'], vs,
                       f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                       f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
        all_conv_rows.append([label] + [f'{q:.3f}' for q in res['avg_quantiles']])
    all_rows.extend(rows_a)

    headers = ['Config', 'Params', 'vs_std', 'Solve%', 'Reward']
    print_table(rows_a, headers,
                title=f'A) Method comparison at rank={FIXED_RANK}')

    # --- Sub-experiment B: rank sweep for cp and tucker ---
    print(f'\n{"="*60}')
    print('B) Rank sweep: cp and tucker')
    print('='*60)

    rows_b = []
    for method in ['cp', 'tucker']:
        for rank in RANKS:
            label = f'{method}_r{rank}'
            print(f'\n  Config: {label}')

            def factory(m=method, r=rank):
                return make_net(state_dim, action_dim, m, r)

            res = run_seeds(train_flat, factory, label=label)
            vs  = f"{res['n_params'] / std_params:.3f}"
            rows_b.append([label, res['n_params'], vs,
                           f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                           f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
            all_conv_rows.append([label] + [f'{q:.3f}' for q in res['avg_quantiles']])
    all_rows.extend(rows_b)

    print_table(rows_b, headers,
                title='B) Rank sweep: cp and tucker')

    elapsed = (time.time() - t0) / 60
    print(f"\n  Standard baseline: {std_params} params")
    print(f"  Total time: {elapsed:.1f} min")

    pct_headers  = [f'{(i+1)*10}%' for i in range(10)]
    conv_headers = ['Config'] + pct_headers
    title = (f"Case 2a: Pre-Init Decomposition  "
             f"(env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})")
    notes = [f"Standard baseline: {std_params} params",
             f"Total runtime: {elapsed:.1f} min"]
    lines = build_output(title, f"env={ENV_ID}", all_rows, headers, notes,
                         convergence_rows=all_conv_rows, convergence_headers=conv_headers)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()

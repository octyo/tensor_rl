"""
Case 6: True MPS / Tensor-Train Layers
=======================================
Uses MPSLinear — a genuine N-core Tensor Train decomposition of the weight
matrix (not a 2-factor rank-R approximation like CPLinear or the old TTLinear).
The weight matrix is reshaped into a higher-order tensor and factored into a
chain of TT-cores with bond dimension = rank.

Sweep: rank in [1, 2, 4, 8, 16]
Compare: vs standard DQN and vs CP at the same rank
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

RANKS    = [1, 2, 4, 8, 16]
OUT_FILE = os.path.join(os.path.dirname(__file__), 'case6_mps_output.txt')


def make_net(state_dim, action_dim, net_type, rank):
    return QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                    network_type=net_type, rank=rank, tensorize_layers='all')


def main():
    state_dim, action_dim = get_flat_dims()
    ref_net    = make_net(state_dim, action_dim, 'standard', 4)
    std_params = sum(p.numel() for p in ref_net.parameters())

    rows = []
    t0   = time.time()

    # Standard baseline
    print('\n  Config: standard (baseline)')

    def std_factory():
        return make_net(state_dim, action_dim, 'standard', 4)

    res = run_seeds(train_flat, std_factory, label='standard')
    rows.append(['standard', res['n_params'], '1.00',
                 f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                 f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])

    # CP and MPS at each rank for direct comparison
    for rank in RANKS:
        for net_type in ['cp', 'mps']:
            label = f'{net_type}_r{rank}'
            print(f'\n  Config: {label}')

            def factory(nt=net_type, r=rank):
                return make_net(state_dim, action_dim, nt, r)

            try:
                res = run_seeds(train_flat, factory, label=label)
                vs  = f"{res['n_params'] / std_params:.3f}"
                rows.append([label, res['n_params'], vs,
                             f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                             f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
            except Exception as e:
                print(f"    ERROR: {e}")
                rows.append([label, 'ERR', '-', '-', str(e)[:40]])

    headers = ['Config', 'Params', 'vs_std', 'Solve%', 'Reward']
    title   = (f"Case 6: MPS / True TT Layers  "
               f"(env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})")
    print_table(rows, headers, title=title)
    print(f"\n  Standard baseline: {std_params} params")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

    notes = [f"Standard baseline: {std_params} params",
             f"env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS}, algo={ALGO}",
             f"MPSLinear: N-core TT with bond-dim=rank, auto-factored weight shapes",
             f"Total runtime: {(time.time()-t0)/60:.1f} min"]
    lines = build_output(title, f"env={ENV_ID}", rows, headers, notes)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()

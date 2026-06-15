"""
Case 1: CP Tensor Layers
========================
Replaces all nn.Linear layers in a DQN with CPLinear (rank-R factorized weights).
This is the simplest tensorization: W ≈ factor_out · factor_in^T.

Sweep: rank in [1, 2, 4, 8, 16, 32]
Baseline: standard h128x128 DQN (36359 params)
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

RANKS    = [1, 2, 4, 8, 16, 32]
OUT_FILE = os.path.join(os.path.dirname(__file__), 'case1_cp_layers_output.txt')


def main():
    state_dim, action_dim = get_flat_dims()

    ref_net    = QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                          network_type='standard', rank=4)
    std_params = sum(p.numel() for p in ref_net.parameters())

    rows = []
    t0   = time.time()

    # Baseline row first
    print('\n  Config: standard (baseline)')

    def std_factory():
        return QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                        network_type='standard', rank=4)

    res = run_seeds(train_flat, std_factory, label='standard')
    rows.append(['standard', res['n_params'], '1.00',
                 f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                 f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])

    # CP rank sweep
    for rank in RANKS:
        label = f'cp_r{rank}'
        print(f'\n  Config: CP rank={rank}')

        def factory(r=rank):
            return QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                            network_type='cp', rank=r, tensorize_layers='all')

        res = run_seeds(train_flat, factory, label=label)
        vs  = f"{res['n_params'] / std_params:.3f}"
        rows.append([label, res['n_params'], vs,
                     f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                     f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])

    headers = ['Config', 'Params', 'vs_std', 'Solve%', 'Reward']
    title   = (f"Case 1: CP Layers  "
               f"(env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})")
    print_table(rows, headers, title=title)
    print(f"\n  Standard baseline: {std_params} params")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

    notes = [f"Standard baseline: {std_params} params",
             f"env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS}, algo={ALGO}",
             f"Total runtime: {(time.time()-t0)/60:.1f} min"]
    lines = build_output(title, f"env={ENV_ID}", rows, headers, notes)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()

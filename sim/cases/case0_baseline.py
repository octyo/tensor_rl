"""
Case 0: Baseline DQN
====================
Sweeps hidden layer sizes to establish the reference performance.
All other cases are compared against the h128x128 config.

Sweep: hidden_sizes in [(64,), (128,), (64,64), (128,128)]
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

HIDDEN_CONFIGS = [
    ((64,),        'h64'),
    ((128,),       'h128'),
    ((64, 64),     'h64x64'),
    ((128, 128),   'h128x128'),
]

OUT_FILE = os.path.join(os.path.dirname(__file__), 'case0_baseline_output.txt')


def main():
    state_dim, action_dim = get_flat_dims()

    ref_net = QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                       network_type='standard', rank=4)
    std_params = sum(p.numel() for p in ref_net.parameters())

    rows = []
    t0 = time.time()

    for hidden_sizes, label in HIDDEN_CONFIGS:
        print(f'\n  Config: {label}  hidden_sizes={hidden_sizes}')

        def factory(h=hidden_sizes):
            return QNetwork(state_dim, action_dim, hidden_sizes=h,
                            network_type='standard', rank=4)

        res = run_seeds(train_flat, factory, label=label)

        vs_std      = f"{res['n_params'] / std_params:.2f}"
        reward_str  = f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"
        solve_str   = f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}"
        rows.append([label, res['n_params'], vs_std, solve_str, reward_str])

    headers = ['Config', 'Params', 'vs_std', 'Solve%', 'Reward']
    title   = f"Case 0: Baseline DQN  (env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})"
    print_table(rows, headers, title=title)
    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")

    notes = [f"Standard baseline: h128x128 = {std_params} params",
             f"env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS}, algo={ALGO}",
             f"Total runtime: {(time.time()-t0)/60:.1f} min"]
    lines = build_output(title, f"env={ENV_ID}", rows, headers, notes)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()

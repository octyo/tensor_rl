"""
Case 0a (Empty-Random-5x5): Baseline Double DQN on randomised goal map
=======================================================================
Replicates Case 0a (hidden-layer sweep with Double DQN) on
MiniGrid-Empty-Random-5x5-v0. The goal position is randomised every episode,
forcing the agent to learn a general spatial policy rather than a fixed route.

Sweep: hidden_sizes in [(64,), (128,), (64,64), (128,128)]
Seeds: 3  |  Episodes: 200  |  Algo: double_dqn
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import (train_flat, run_seeds, get_flat_dims,
                   print_table, build_output, save_output)
from models.q_networks import QNetwork

ENV_ID     = 'MiniGrid-Empty-Random-5x5-v0'
ALGO       = 'double_dqn'
SEEDS      = [42, 43, 44]
N_EPISODES = 200

HIDDEN_CONFIGS = [
    ((64,),       'h64'),
    ((128,),      'h128'),
    ((64, 64),    'h64x64'),
    ((128, 128),  'h128x128'),
]

OUT_FILE = os.path.join(os.path.dirname(__file__), 'case0a_emptyrandom5x5_output.txt')


def main():
    # Override utils ENV_ID for this case
    import utils
    utils.ENV_ID = ENV_ID

    state_dim, action_dim = get_flat_dims()

    ref_net    = QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                          network_type='standard', rank=4)
    std_params = sum(p.numel() for p in ref_net.parameters())

    rows      = []
    conv_rows = []
    t0        = time.time()
    n_configs = len(HIDDEN_CONFIGS)

    print(f'\n  Environment: {ENV_ID}')
    print(f'  state_dim={state_dim}  action_dim={action_dim}  '
          f'episodes={N_EPISODES}  seeds={SEEDS}')

    for config_idx, (hidden_sizes, label) in enumerate(HIDDEN_CONFIGS, start=1):
        print(f'\n  [config {config_idx}/{n_configs}] {label}  '
              f'hidden_sizes={hidden_sizes}')

        def factory(h=hidden_sizes):
            return QNetwork(state_dim, action_dim, hidden_sizes=h,
                            network_type='standard', rank=4)

        res = run_seeds(train_flat, factory, label=label, algo=ALGO,
                        n_episodes=N_EPISODES, seeds=SEEDS,
                        config_idx=config_idx, config_total=n_configs,
                        experiment_t0=t0)

        vs_std     = f"{res['n_params'] / std_params:.2f}"
        reward_str = f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"
        solve_str  = f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}"
        rows.append([label, res['n_params'], vs_std, solve_str, reward_str])
        conv_rows.append([label] + [f'{q:.3f}' for q in res['avg_quantiles']])

    pct_headers  = [f'{(i+1)*10}%' for i in range(10)]
    conv_headers = ['Config'] + pct_headers
    headers      = ['Config', 'Params', 'vs_std', 'Solve%', 'Reward']
    title        = (f"Case 0a (Empty-Random-5x5): Baseline Double DQN  "
                    f"(env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})")

    print_table(rows, headers, title=title)
    elapsed = (time.time() - t0) / 60
    print(f"\n  Standard baseline: h128x128 = {std_params} params")
    print(f"  Total time: {elapsed:.1f} min")

    notes = [
        f"Standard baseline: h128x128 = {std_params} params",
        f"Algorithm: {ALGO} (Polyak target update, SmoothL1 loss)",
        f"Task: reach randomised goal position (new position each episode)",
        f"Total runtime: {elapsed:.1f} min",
    ]
    lines = build_output(title, f"env={ENV_ID}", rows, headers, notes,
                         convergence_rows=conv_rows,
                         convergence_headers=conv_headers)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()

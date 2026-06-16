"""
Case 4: Tensorized CNN
======================
Uses a CNN to process the raw (3,7,7) MiniGrid image instead of the flat (147,)
observation vector. Two sub-modes:
  backbone   — standard Conv2d backbone + tensorized (CP/Tucker) linear head
  tensorized — CPConv2d / TuckerConv2d backbone + standard linear head

Sweep: cnn_mode in [backbone, tensorized]
       network_type in [cp, tucker]
       rank in [2, 4, 8]
Baseline: standard CNN (standard Conv2d + standard linear head)
Seeds: 3  |  Episodes: 200  |  Algo: DQN
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import (ENV_ID, N_EPISODES, SEEDS, ALGO,
                   train_flat, train_cnn,
                   run_seeds, get_flat_dims, get_cnn_dims,
                   print_table, build_output, save_output)
from models.q_networks import QNetwork
from models.cnn_networks import CNNQNetwork

CNN_MODES    = ['backbone', 'tensorized']
NET_TYPES    = ['cp', 'tucker']
RANKS        = [2, 4, 8]
OUT_FILE     = os.path.join(os.path.dirname(__file__), 'case4_cnn_output.txt')


def main():
    state_dim, action_dim = get_flat_dims()
    obs_shape, _          = get_cnn_dims()   # (3, 7, 7)

    ref_net    = QNetwork(state_dim, action_dim, hidden_sizes=(128, 128),
                          network_type='standard', rank=4)
    std_params = sum(p.numel() for p in ref_net.parameters())

    rows      = []
    conv_rows = []
    t0        = time.time()

    # Baseline: standard CNN (backbone + standard linear)
    print('\n  Config: cnn_standard (baseline)')

    def cnn_std_factory(obs=obs_shape):
        return CNNQNetwork(obs, action_dim, rank=4, network_type='standard',
                           cnn_mode='backbone', hidden_sizes=(128, 128))

    res = run_seeds(train_cnn, cnn_std_factory, label='cnn_standard')
    vs  = f"{res['n_params'] / std_params:.3f}"
    rows.append(['cnn_standard', res['n_params'], vs,
                 f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                 f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
    conv_rows.append(['cnn_standard'] + [f'{q:.3f}' for q in res['avg_quantiles']])

    # Tensorized CNN sweep
    for cnn_mode in CNN_MODES:
        for net_type in NET_TYPES:
            for rank in RANKS:
                label = f'{cnn_mode}_{net_type}_r{rank}'
                print(f'\n  Config: {label}  obs_shape={obs_shape}')

                def factory(obs=obs_shape, cm=cnn_mode, nt=net_type, r=rank):
                    return CNNQNetwork(obs, action_dim, rank=r,
                                      network_type=nt, cnn_mode=cm,
                                      hidden_sizes=(128, 128))

                try:
                    res = run_seeds(train_cnn, factory, label=label)
                    vs  = f"{res['n_params'] / std_params:.3f}"
                    rows.append([label, res['n_params'], vs,
                                 f"{res['solve_mean']:.0%} +/- {res['solve_std']:.0%}",
                                 f"{res['reward_mean']:.3f} +/- {res['reward_std']:.3f}"])
                    conv_rows.append([label] + [f'{q:.3f}' for q in res['avg_quantiles']])
                except Exception as e:
                    print(f"    ERROR: {e}")
                    rows.append([label, 'ERR', '-', '-', str(e)[:40]])
                    conv_rows.append([label] + ['-'] * 10)

    pct_headers  = [f'{(i+1)*10}%' for i in range(10)]
    conv_headers = ['Config'] + pct_headers
    headers = ['Config', 'Params', 'vs_std', 'Solve%', 'Reward']
    title   = (f"Case 4: Tensorized CNN  "
               f"(env={ENV_ID}, episodes={N_EPISODES}, seeds={SEEDS})")
    print_table(rows, headers, title=title)
    print(f"\n  Standard flat baseline: {std_params} params")
    print(f"  CNN obs_shape: {obs_shape}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

    notes = [f"Standard flat baseline: {std_params} params",
             f"CNN obs_shape: {obs_shape}",
             f"Total runtime: {(time.time()-t0)/60:.1f} min"]
    lines = build_output(title, f"env={ENV_ID}", rows, headers, notes,
                         convergence_rows=conv_rows, convergence_headers=conv_headers)
    save_output(lines, OUT_FILE)


if __name__ == '__main__':
    main()
